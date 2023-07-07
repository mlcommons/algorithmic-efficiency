"""Transformer-based machine translation model.

Reference https://github.com/google/flax/tree/main/examples/wmt.
"""

from typing import Any, Callable, Optional

from flax import linen as nn
from flax import struct
from jax import lax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  share_embeddings: bool = True
  dtype: Any = jnp.float32
  vocab_size: int = 32000
  emb_dim: int = 1024
  num_heads: int = 16
  num_layers: int = 6
  qkv_dim: int = 1024
  mlp_dim: int = 1024
  max_len: int = 256
  #If None, defaults to 0.1.
  dropout_rate: Optional[float] = 0.1
  #If None, defaults to 0.1.
  attention_dropout_rate: Optional[float] = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2:2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """
  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    cfg = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              f' but it is: {inputs.ndim}')
    length = inputs.shape[1]
    pos_emb_shape = (1, cfg.max_len, inputs.shape[-1])
    if cfg.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=cfg.max_len)(None,
                                                           pos_emb_shape,
                                                           None)
    else:
      pos_embedding = self.param('pos_embedding',
                                 cfg.posemb_init,
                                 pos_emb_shape)
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable('cache',
                                  'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.uint32))
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """
  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    cfg = self.config
    actual_out_dim = (
        inputs.shape[-1] if self.out_dim is None else self.out_dim)
    x = nn.Dense(
        cfg.mlp_dim,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init)(
            inputs)
    x = nn.relu(x)
    if cfg.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = cfg.dropout_rate
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=cfg.deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=cfg.dtype,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init)(
            x)
    output = nn.Dropout(rate=dropout_rate)(
        output, deterministic=cfg.deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self, inputs, encoder_mask=None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    cfg = self.config

    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
    if cfg.attention_dropout_rate is None:
      attention_dropout_rate = 0.1
    else:
      attention_dropout_rate = cfg.attention_dropout_rate
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=cfg.deterministic)(x, encoder_mask)

    if cfg.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = cfg.dropout_rate
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=cfg.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = MlpBlock(config=cfg)(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  @nn.compact
  def __call__(self,
               targets,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    cfg = self.config

    # Decoder block.
    assert targets.ndim == 3
    x = nn.LayerNorm(dtype=cfg.dtype)(targets)

    if cfg.attention_dropout_rate is None:
      attention_dropout_rate = 0.1
    else:
      attention_dropout_rate = cfg.attention_dropout_rate
    x = nn.SelfAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=cfg.deterministic,
        decode=cfg.decode)(x, decoder_mask)
    if cfg.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = cfg.dropout_rate
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=cfg.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(dtype=cfg.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        qkv_features=cfg.qkv_dim,
        kernel_init=cfg.kernel_init,
        bias_init=cfg.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=attention_dropout_rate,
        deterministic=cfg.deterministic)(y, encoded, encoder_decoder_mask)

    y = nn.Dropout(rate=dropout_rate)(y, deterministic=cfg.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=cfg.dtype)(y)
    z = MlpBlock(config=cfg)(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self, inputs, inputs_positions=None, encoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    cfg = self.config
    assert inputs.ndim == 2  # (batch, len)

    # Input Embedding
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(
        config=cfg, decode=False, name='posembed_input')(
            x, inputs_positions=inputs_positions)
    if cfg.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = cfg.dropout_rate
    x = nn.Dropout(rate=dropout_rate)(x, deterministic=cfg.deterministic)

    x = x.astype(cfg.dtype)

    # Input Encoder
    for lyr in range(cfg.num_layers):
      x = Encoder1DBlock(
          config=cfg, name=f'encoderblock_{lyr}')(x, encoder_mask)

    encoded = nn.LayerNorm(dtype=cfg.dtype, name='encoder_layernorm')(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """
  config: TransformerConfig
  shared_embedding: Any = None

  @nn.compact
  def __call__(self,
               encoded,
               targets,
               targets_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    cfg = self.config

    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    # Target Embedding
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if not cfg.decode:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        config=cfg, decode=cfg.decode, name='posembed_output')(
            y, inputs_positions=targets_positions)
    if cfg.dropout_rate is None:
      dropout_rate = 0.1
    else:
      dropout_rate = cfg.dropout_rate
    y = nn.Dropout(rate=dropout_rate)(y, deterministic=cfg.deterministic)

    y = y.astype(cfg.dtype)

    # Target-Input Decoder
    for lyr in range(cfg.num_layers):
      y = EncoderDecoder1DBlock(
          config=cfg, name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask)
    y = nn.LayerNorm(dtype=cfg.dtype, name='encoderdecoder_layernorm')(y)

    # Use the transpose of embedding matrix for logit transform.
    logits = output_embed.attend(y.astype(jnp.float32))
    # Correctly normalize pre-softmax logits for this shared case.
    logits = logits / jnp.sqrt(y.shape[-1])
    return logits


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """
  config: TransformerConfig

  def setup(self):
    cfg = self.config

    if cfg.share_embeddings:
      if cfg.vocab_size is not None:
        assert cfg.vocab_size == cfg.vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          num_embeddings=cfg.vocab_size,
          features=cfg.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      self.shared_embedding = None

    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

  def encode(self, inputs, inputs_positions=None, inputs_segmentation=None):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      encoded feature array from the transformer encoder.
    """
    cfg = self.config
    # Make padding attention mask.
    encoder_mask = nn.make_attention_mask(
        inputs > 0, inputs > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if inputs_segmentation is not None:
      encoder_mask = nn.combine_masks(
          encoder_mask,
          nn.make_attention_mask(
              inputs_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
    return self.encoder(
        inputs, inputs_positions=inputs_positions, encoder_mask=encoder_mask)

  def decode(
      self,
      encoded,
      inputs,  # only needed for masks
      targets,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    cfg = self.config

    # Make padding attention masks.
    if cfg.decode:
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0, dtype=cfg.dtype)
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=cfg.dtype),
          nn.make_causal_mask(targets, dtype=cfg.dtype))
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs > 0, dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              targets_segmentation,
              targets_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
      encoder_decoder_mask = nn.combine_masks(
          encoder_decoder_mask,
          nn.make_attention_mask(
              targets_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=cfg.dtype))
    logits = self.decoder(
        encoded,
        targets,
        targets_positions=targets_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask)
    return logits.astype(self.config.dtype)

  def __call__(self,
               inputs,
               targets,
               inputs_positions=None,
               targets_positions=None,
               inputs_segmentation=None,
               targets_segmentation=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    return self.decode(
        encoded,
        inputs,  # only used for masks
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation)
