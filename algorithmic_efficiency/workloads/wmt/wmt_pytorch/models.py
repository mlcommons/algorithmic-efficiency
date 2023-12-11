import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_


def make_causal_mask(x: Tensor, device: str = 'cuda:0') -> Tensor:
  """Make a causal mask for self-attention.

  Args:
    x: input array of shape `[batch..., len]`
    device: device to store the idxs

  Returns:
    A `[batch..., len, len]` shaped causal attention mask.
  """
  idxs = torch.broadcast_to(
      torch.arange(x.shape[-1], dtype=torch.int32, device=device), x.shape)
  return torch.greater_equal(idxs.unsqueeze(-1), idxs.unsqueeze(-2))


def make_src_mask(src, inputs_segmentation, nhead):
  """Utility for creating src mask and adjust it for PyTorch Transformer API."""
  src_mask = torch.mul((src > 0).unsqueeze(-1), (src > 0).unsqueeze(-2))
  # Add segmentation block-diagonal attention mask if using segmented data.
  if inputs_segmentation is not None:
    src_mask = torch.logical_and(
        src_mask,
        torch.eq(
            inputs_segmentation.unsqueeze(-1),
            inputs_segmentation.unsqueeze(-2)))
  # Flip values and ensure numerical stability.
  src_mask = torch.repeat_interleave(
      torch.logical_not(src_mask), repeats=nhead, dim=0)
  new_src_mask = torch.zeros_like(src_mask, dtype=torch.float32)
  new_src_mask.masked_fill_(src_mask, -1e10)
  return new_src_mask


def make_tgt_and_memory_mask(tgt,
                             src,
                             inputs_segmentation,
                             targets_segmentation,
                             decode,
                             nhead):
  """ Utility for creating target and memory mask and adjust them for PyTorch
  Transformer API."""
  if not decode:
    tgt_mask = torch.logical_and(
        torch.mul((tgt > 0).unsqueeze(-1), (tgt > 0).unsqueeze(-2)),
        make_causal_mask(tgt, device=tgt.device))
    memory_mask = torch.mul((tgt > 0).unsqueeze(-1), (src > 0).unsqueeze(-2))
  else:
    tgt_mask = None
    memory_mask = torch.mul((torch.ones_like(tgt) > 0).unsqueeze(-1),
                            (src > 0).unsqueeze(-2))
  # Add segmentation block-diagonal attention masks if using segmented data.
  if inputs_segmentation is not None:
    tgt_mask = torch.logical_and(
        tgt_mask,
        torch.eq(
            targets_segmentation.unsqueeze(-1),
            targets_segmentation.unsqueeze(-2)))
    memory_mask = torch.logical_and(
        memory_mask,
        torch.eq(
            targets_segmentation.unsqueeze(-1),
            inputs_segmentation.unsqueeze(-2)))
  # Flip values and ensure numerical stability.
  memory_mask = torch.repeat_interleave(
      torch.logical_not(memory_mask), repeats=nhead, dim=0)
  new_memory_mask = torch.zeros_like(memory_mask, dtype=torch.float32)
  new_memory_mask.masked_fill_(memory_mask, -1e10)
  if tgt_mask is not None:
    tgt_mask = torch.repeat_interleave(
        torch.logical_not(tgt_mask), repeats=nhead, dim=0)
    new_tgt_mask = torch.zeros_like(tgt_mask, dtype=torch.float32)
    new_tgt_mask.masked_fill_(tgt_mask, -1e10)
    tgt_mask = new_tgt_mask
  return tgt_mask, new_memory_mask


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  pad_widths = tuple(t for tup in reversed(pad_widths) for t in tup)
  padded = F.pad(x, pad_widths, mode='constant')
  return padded[:, :-1]


class Transformer(nn.Module):
  """Transformer architecture based on the model from the WMT Jax workload."""

  def __init__(self,
               ntoken: int = 32000,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: Optional[float] = 0.1,
               attention_dropout_rate: Optional[float] = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               glu: bool = False,
               layer_norm_eps: float = 1e-6,
               attention_temp: float = 1.0,
               norm_first: bool = True):
    super().__init__()
    if dropout_rate is None:
      dropout_rate = 0.1
    if attention_dropout_rate is None:
      attention_dropout_rate = 0.1
    self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
    self.shared_embedding = nn.Embedding(ntoken, d_model)
    self.encoder = Encoder(d_model,
                           nhead,
                           d_hid,
                           nlayers,
                           dropout_rate,
                           attention_dropout_rate,
                           activation,
                           glu,
                           layer_norm_eps,
                           attention_temp,
                           norm_first)
    self.decoder = Decoder(d_model,
                           nhead,
                           d_hid,
                           nlayers,
                           dropout_rate,
                           attention_dropout_rate,
                           activation,
                           glu,
                           layer_norm_eps,
                           attention_temp,
                           norm_first)
    # Share positional encoding and embedding between encoder and decoder.
    self.encoder.pos_encoder = self.pos_encoder
    self.encoder.shared_embedding = self.shared_embedding
    self.decoder.pos_encoder = self.pos_encoder
    self.decoder.shared_embedding = self.shared_embedding

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
          normal_(module.bias, std=1e-6)

  def forward(self,
              src: Tensor,
              tgt: Tensor,
              inputs_positions: Optional[Tensor] = None,
              targets_positions: Optional[Tensor] = None,
              inputs_segmentation: Optional[Tensor] = None,
              targets_segmentation: Optional[Tensor] = None,
              decode: bool = False) -> Tensor:
    """
    Args:
      src: Tensor, shape [batch_size, seq_len]
      tgt: Tensor, shape [batch_size, seq_len]
      inputs_positions: Optional[Tensor], shape [batch_size, seq_len]
      targets_positions: Optional[Tensor], shape [batch_size, seq_len]
      inputs_segmentation: Optional[Tensor], shape [batch_size, seq_len]
      targets_segmentation: Optional[Tensor], shape [batch_size, seq_len]
      decode: bool

    Returns:
      output Tensor of shape [batch_size, seq_len, ntoken]
    """
    if src.size(0) != tgt.size(0):
      raise RuntimeError('The batch size of src and tgt must be equal.')
    memory = self.encoder(
        src,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)
    output = self.decoder(
        tgt,
        memory,
        src,  # just for calculating the padding mask
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        decode=decode)
    return output


class TransformerEncoder(nn.Module):
  r"""TransformerEncoder is a stack of N encoder layers. Users can build the
  BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

  Args:
      encoder_layer: an instance of the TransformerEncoderLayer() class.
      num_layers: the number of sub-encoder-layers in the encoder.
      norm: the layer normalization component (optional).
      enable_nested_tensor: if True, input will automatically convert to
        nested tensor (and convert back on output). This will improve
        the overall performance of TransformerEncoder when padding
        rate is high.

  Examples::
    >>> encoder_layer = nn.TransformerEncoderLayer(12, 8)
    >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, 6)
    >>> src = torch.rand(10, 32, 512)
    >>> out = transformer_encoder(src)
  """
  __constants__ = ['norm']

  def __init__(self,
               encoder_layer,
               num_layers,
               norm=None,
               enable_nested_tensor=True,
               mask_check=True):
    super().__init__()
    self.layers = nn.ModuleList(
        [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
    self.num_layers = num_layers
    self.norm = norm
    self.enable_nested_tensor = enable_nested_tensor
    self.mask_check = mask_check

  def forward(self,
              src: Tensor,
              mask: Optional[Tensor] = None) -> Tensor:
    """Pass the input through the encoder layers in turn.

    Args:
        src: the sequence to the encoder (required).
        mask: the mask for the src sequence (optional).

    Shape:
        see the docs in Transformer class.
    """
    output = src
    convert_to_nested = False

    for mod in self.layers:
      output = mod(output, src_mask=mask)

    if convert_to_nested:
      output = output.to_padded_tensor(0.)

    if self.norm is not None:
      output = self.norm(output)

    return output


class Encoder(nn.Module):

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               glu: bool = False,
               layer_norm_eps: float = 1e-6,
               attention_temp: float = 1.0,
               norm_first: bool = True):
    super().__init__()
    self.nhead = nhead
    self.shared_embedding = None
    self.pos_encoder = None
    encoder_layer = TransformerEncoderLayer(
        d_model,
        nhead,
        d_hid,
        dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        activation=activation,
        glu=glu,
        layer_norm_eps=layer_norm_eps,
        attention_temp=attention_temp,
        norm_first=norm_first)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    self.encoder = TransformerEncoder(encoder_layer, nlayers, encoder_norm)

  def forward(self,
              src: Tensor,
              inputs_positions: Optional[Tensor] = None,
              inputs_segmentation: Optional[Tensor] = None) -> Tensor:
    src = src.to(torch.int)
    src_mask = make_src_mask(src, inputs_segmentation, self.nhead)
    src = self.shared_embedding(src)
    src = self.pos_encoder(src, inputs_positions)
    memory = self.encoder(src, mask=src_mask)
    return memory


class Decoder(nn.Module):

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               d_hid: int = 1024,
               nlayers: int = 6,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               glu: bool = False,
               layer_norm_eps: float = 1e-6,
               attention_temp: float = 1.0,
               norm_first: bool = True):
    super().__init__()
    self.nhead = nhead
    self.shared_embedding = None
    self.pos_encoder = None
    self.decoder = TransformerDecoder(d_model,
                                      nhead,
                                      d_hid,
                                      dropout_rate,
                                      attention_dropout_rate,
                                      activation,
                                      glu,
                                      layer_norm_eps,
                                      nlayers,
                                      attention_temp,
                                      norm_first)

  def forward(
      self,
      tgt: Tensor,
      memory: Tensor,
      src: Tensor,  # just for calculating the padding mask
      targets_positions: Optional[Tensor] = None,
      inputs_segmentation: Optional[Tensor] = None,
      targets_segmentation: Optional[Tensor] = None,
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None) -> Any:
    tgt = tgt.to(torch.int)
    tgt_mask, memory_mask = make_tgt_and_memory_mask(
        tgt, src, inputs_segmentation, targets_segmentation,
        decode, self.nhead)
    if not decode:
      tgt = shift_right(tgt)
    tgt = self.shared_embedding(tgt)
    tgt = self.pos_encoder(tgt, targets_positions, decode=decode, cache=cache)
    if decode:
      tgt, cache = tgt
    output = self.decoder(
        tgt,
        memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        decode=decode,
        max_len=max_len,
        cache=cache)
    if decode:
      output, cache = output
    normalize = math.sqrt(output.shape[-1])
    output = torch.matmul(output, self.shared_embedding.weight.T) / normalize
    if decode:
      return output, cache
    return output


class PositionalEncoding(nn.Module):

  def __init__(self,
               d_model: int,
               dropout_rate: float = 0.1,
               max_len: int = 256):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout_rate)

    position = torch.arange(max_len).unsqueeze(1)
    scale_factor = -math.log(10000.0) / (d_model // 2 - 1)
    div_term = torch.exp(torch.arange(d_model // 2) * scale_factor)
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, :d_model // 2] = torch.sin(position * div_term)
    pe[0, :, d_model // 2:2 * (d_model // 2)] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(
      self,
      x: Tensor,
      inputs_positions: Optional[Tensor] = None,
      decode: bool = False,
      cache: Optional[Dict[str, Dict[str, Tensor]]] = None
  ) -> Union[Tensor, Tuple[Tensor, Dict[str, Dict[str, Tensor]]]]:
    """
    Args:
      x: Tensor (shape [batch_size, seq_len, embedding_dim])
      inputs_positions: Tensor (shape [batch_size, seq_len]) or None
      decode: bool
      cache: Dict[str, Dict[str, Tensor]] or None
    Returns:
      Tensor or Tuple[Tensor, Dict[str, Dict[str, Tensor]]]
    """
    # We use a cache position index for tracking decoding position.
    if decode:
      name = self._get_name()
      if cache is None:
        cache = {
            name: {
                'cache_index':
                    torch.tensor(0, dtype=torch.long, device=self.pe.device),
            },
        }
      pe = self.pe[0, cache[name]['cache_index'], :]
      cache[name]['cache_index'] += 1
      return self.dropout(x + pe), cache
    if inputs_positions is None:
      # normal unpacked case:
      pe = self.pe[:, :x.size(1), :]
    else:
      # for packed data we need to use known position indices:
      pe = self.pe[0, inputs_positions, :]
    return self.dropout(x + pe)


# TransformerEncoderLayer and TransformerDecoderLayer are taken from:
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# Main difference is the use of custom MultiheadAttention modules.
class TransformerEncoderLayer(nn.Module):
  r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
  This standard encoder layer is based on the paper "Attention Is All You Need".
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
  you need. In Advances in Neural Information Processing Systems,
  pages 6000-6010. Users may modify or implement in a different way during
  application.
  Args:
    d_model: the number of expected features in the input (default=1024).
    nhead: the number of heads in the multiheadattention models (default=16).
    dim_feedforward: the dimension of the feedforward network model
        (default=1024).
    dropout_rate: the dropout_rate value (default=0.1).
    activation: the activation function of the intermediate layer, can be a
       string ("relu" or "gelu") or a unary callable (default=F.relu).
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    norm_first: if ``True``, layer norm is done prior to attention and
        feedforward operations, respectivaly. Otherwise it's done after.
        Default: ``True``.
  Examples::
    >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    >>> src = torch.rand(32, 10, 512)
    >>> out = encoder_layer(src)
  """
  __constants__ = ['norm_first']

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               dim_feedforward: int = 1024,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               glu: bool = False,
               layer_norm_eps: float = 1e-6,
               attention_temp: float = 1.0,
               norm_first: bool = True,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.self_attn = MultiheadAttention(
        d_model,
        nhead,
        self_attn=True,
        dropout_rate=attention_dropout_rate,
        attention_temp=attention_temp,
        bias=False,
        **factory_kwargs)

    # Implementation of Feedforward model.
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.glu = glu
    if self.glu:
      self.linear_glu = nn.Linear(dim_feedforward,
                                  dim_feedforward,
                                  **factory_kwargs)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)

    self.activation = activation

  def forward(self,
              src: Tensor,
              src_mask: Optional[Tensor] = None) -> Tensor:
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).

    Shape:
        see the docs in Transformer class.
    """
    x = src
    if self.norm_first:
      x = x + self._sa_block(self.norm1(x), src_mask)
      x = x + self._ff_block(self.norm2(x))
    else:
      x = self.norm1(x + self._sa_block(x, src_mask))
      x = self.norm2(x + self._ff_block(x))

    return x

  # Self-attention block:
  def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tensor:
    x, _ = self.self_attn(x, attn_mask=attn_mask)
    return self.dropout1(x)

  # Feed forward block:
  def _ff_block(self, x: Tensor) -> Tensor:
    x = self.activation(self.linear1(x))
    if self.glu:
      y = self.linear_glu(x)
      x = x * y
    x = self.linear2(self.dropout(x))
    return self.dropout2(x)


# Modified to use cache for autoregressive decoding and custom
# MultiheadAttention modules.
class TransformerDecoder(nn.Module):
  r"""TransformerDecoder is a stack of N decoder layers
  Args:
    d_model: the number of expected features in the input (default=1024)
    nhead: the number of heads in the multiheadattention models (default=16)
    d_hid: the dimension of the feedforward network model
        (default=1024)
    dropout_rate: the dropout_rate value (default=0.1)
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    decoder_layer: an instance of the TransformerDecoderLayer() class
    num_layers: the number of sub-decoder-layers in the decoder
  Examples::
    >>> decoder_layer = nn.TransformerDecoderLayer(12, 8)
    >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, 6)
    >>> memory = torch.rand(10, 32, 512)
    >>> tgt = torch.rand(20, 32, 512)
    >>> out = transformer_decoder(tgt, memory)
  """
  __constants__ = ['norm']

  def __init__(self,
               d_model,
               nhead,
               d_hid,
               dropout_rate,
               attention_dropout_rate,
               activation,
               glu,
               layer_norm_eps,
               num_layers,
               attention_temp,
               norm_first):
    super().__init__()
    self.layers = nn.ModuleList([
        TransformerDecoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout_rate,
            attention_dropout_rate,
            activation,
            glu,
            layer_norm_eps=layer_norm_eps,
            attention_temp=attention_temp,
            norm_first=norm_first) for _ in range(num_layers)
    ])
    self.num_layers = num_layers
    self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

  def forward(self,
              tgt: Tensor,
              memory: Tensor,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              decode: bool = False,
              max_len: Optional[int] = None,
              cache: Optional[dict] = None) -> Any:
    r"""Pass the inputs (and mask) through the decoder layer in turn.
    Args:
      tgt: the sequence to the decoder (required).
      memory: the sequence from the last layer of the encoder (required).
      tgt_mask: the mask for the tgt sequence (optional).
      memory_mask: the mask for the memory sequence (optional).
      decode: whether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
    Shape:
      see the docs in Transformer class.
    """
    output = tgt

    for idx, mod in enumerate(self.layers):
      output, cache = mod(
          output,
          memory,
          tgt_mask=tgt_mask,
          memory_mask=memory_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=idx)

    if self.norm is not None:
      output = self.norm(output)

    if decode:
      return output, cache
    return output


# Modified to use cache for autoregressive decoding and custom
# MultiheadAttention modules.
class TransformerDecoderLayer(nn.Module):
  r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and
  feedforward network.
  This standard decoder layer is based on the paper "Attention Is All You Need".
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
  you need. In Advances in Neural Information Processing Systems,
  pages 6000-6010. Users may modify or implement in a different way during
  application.
  Args:
    d_model: the number of expected features in the input (default=1024).
    nhead: the number of heads in the multiheadattention models (default=16).
    dim_feedforward: the dimension of the feedforward network model
        (default=1024).
    dropout_rate: the dropout_rate value (default=0.1).
    activation: the activation function of the intermediate layer, can be a
        string ("relu" or "gelu") or a unary callable (default=F.relu).
    layer_norm_eps: the eps value in layer normalization components
        (default=1e-6).
    norm_first: if ``True``, layer norm is done prior to self attention,
        multihead attention and feedforward operations, respectivaly.
        Otherwise it's done after. Default: ``True``.
  Examples::
    >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    >>> memory = torch.rand(32, 10, 512)
    >>> tgt = torch.rand(32, 20, 512)
    >>> out = decoder_layer(tgt, memory)
  """
  __constants__ = ['norm_first']

  def __init__(self,
               d_model: int = 1024,
               nhead: int = 16,
               dim_feedforward: int = 1024,
               dropout_rate: float = 0.1,
               attention_dropout_rate: float = 0.1,
               activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
               glu: bool = False,
               layer_norm_eps: float = 1e-6,
               norm_first: bool = True,
               attention_temp: float = 1.0,
               device=None,
               dtype=None) -> None:
    factory_kwargs = {'device': device, 'dtype': dtype}
    super().__init__()
    self.self_attn = MultiheadAttention(
        d_model,
        nhead,
        self_attn=True,
        dropout_rate=attention_dropout_rate,
        attention_temp=attention_temp,
        bias=False,
        **factory_kwargs)
    self.multihead_attn = MultiheadAttention(
        d_model,
        nhead,
        self_attn=False,
        dropout_rate=attention_dropout_rate,
        attention_temp=attention_temp,
        bias=False,
        **factory_kwargs)

    # Implementation of Feedforward model.
    self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
    self.glu = glu
    if self.glu:
      self.linear_glu = nn.Linear(dim_feedforward,
                                  dim_feedforward,
                                  **factory_kwargs)
    self.dropout = nn.Dropout(dropout_rate)
    self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

    self.norm_first = norm_first
    self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
    self.dropout1 = nn.Dropout(dropout_rate)
    self.dropout2 = nn.Dropout(dropout_rate)
    self.dropout3 = nn.Dropout(dropout_rate)

    self.activation = activation

  def forward(  # pylint: disable=arguments-renamed
      self,
      tgt: Tensor,
      memory: Tensor,
      tgt_mask: Optional[Tensor] = None,
      memory_mask: Optional[Tensor] = None,
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None,
      index: Optional[int] = None) -> Any:
    r"""Pass the inputs (and mask) through the decoder layer.
    Args:
      tgt: the sequence to the decoder layer (required).
      memory: the sequence from the last layer of the encoder (required).
      tgt_mask: the mask for the tgt sequence (optional).
      memory_mask: the mask for the memory sequence (optional).
      decode: wether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
    Shape:
      see the docs in Transformer class.
    """
    # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

    x = tgt
    if self.norm_first:
      sa_out, cache = self._sa_block(
          self.norm1(x),
          tgt_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=index)
      x = x + sa_out
      x = x + self._mha_block(self.norm2(x), memory, memory_mask)
      x = x + self._ff_block(self.norm3(x))
    else:
      sa_out, cache = self._sa_block(
          x,
          tgt_mask,
          decode=decode,
          max_len=max_len,
          cache=cache,
          index=index)
      x = self.norm1(x + sa_out)
      x = self.norm2(x + self._mha_block(x, memory, memory_mask))
      x = self.norm3(x + self._ff_block(x))

    return x, cache

  # Self-attention block:
  def _sa_block(  # pylint: disable=arguments-renamed
      self,
      x: Tensor,
      attn_mask: Optional[Tensor],
      decode: bool = False,
      max_len: Optional[int] = None,
      cache: Optional[dict] = None,
      index: Optional[int] = None) -> Any:
    x, cache = self.self_attn(
        x,
        attn_mask=attn_mask,
        decode=decode,
        max_len=max_len,
        cache=cache,
        index=index)
    return self.dropout1(x), cache

  # Multihead attention block:
  def _mha_block(self, x: Tensor, mem: Tensor,
                 attn_mask: Optional[Tensor]) -> Tensor:
    x, _ = self.multihead_attn(x, mem, attn_mask=attn_mask)
    return self.dropout2(x)

  # Feed forward block.
  def _ff_block(self, x: Tensor) -> Tensor:
    x = self.activation(self.linear1(x))
    if self.glu:
      y = self.linear_glu(x)
      x = x * y
    x = self.linear2(self.dropout(x))
    return self.dropout3(x)


class MultiheadAttention(nn.Module):
  r"""Allows the model to jointly attend to information
  from different representation subspaces. Supports self-attention and
  encoder-decoder attention.
  See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
  .. math::
      \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
  where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
  Args:
    embed_dim: Total dimension of the model.
    num_heads: Number of parallel attention heads. Note that ``embed_dim`` will
        be split across ``num_heads`` (i.e. each head will have dimension
        ``embed_dim // num_heads``).
    self_attn: Whether self attention or encoder-decoder attention is used.
        Default: ``True``.
    dropout_rate: Dropout probability on ``attn_output_weights``.
        Default: ``0.0`` (no dropout_rate).
    bias: If specified, adds bias to input / output projection layers.
       Default: ``False``.
    device: The device of the module.
    dtype: The dtype of the module.
  Examples::
    >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
    >>> attn_output, cache = multihead_attn(x)
  """

  def __init__(self,
               embed_dim: int,
               num_heads: int,
               self_attn: bool = True,
               dropout_rate: float = 0.,
               attention_temp: float = 1.0,
               bias: bool = False,
               device: Optional[torch.device] = None,
               dtype: Optional[torch.dtype] = None) -> None:
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.self_attn = self_attn
    self.dropout = dropout_rate
    self.head_dim = embed_dim // num_heads
    self.attention_temp = attention_temp
    assert self.head_dim * num_heads == self.embed_dim, \
        'embed_dim must be divisible by num_heads.'

    factory_kwargs = {'device': device, 'dtype': dtype}
    if self_attn:
      # Self-attention.
      self.in_proj = nn.Linear(
          embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
    else:
      # Encoder-decoder attention.
      self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
      self.kv_proj = nn.Linear(
          embed_dim, 2 * embed_dim, bias=bias, **factory_kwargs)
    self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the MultiheadAttention module."""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
          normal_(module.bias, std=1e-6)

  def forward(self,
              x: Tensor,
              mem: Optional[Tensor] = None,
              attn_mask: Optional[Tensor] = None,
              decode: bool = False,
              max_len: Optional[int] = None,
              cache: Optional[dict] = None,
              index: Optional[int] = None) -> Any:
    r"""
    Args:
      x: Batch of input sequences of shape
          (batch size, sequence length, embedding dimensionality) for self
          attention mechanism. See "Attention Is All You Need" for more details.
      mem: Batch of input sequences of shape
          (batch size, sequence length, embedding dimensionality) for
          encoder-decoder attention. See "Attention Is All You Need" for more 
          details.
      attn_mask: If specified, a 2D or 3D mask preventing attention to certain
          positions. Must be of shape :math:`(L, S)` or
          :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the
          batch size, :math:`L` is the target sequence length, and :math:`S`
          is the source sequence length. A 2D mask will be broadcasted across
          the batch while a 3D mask allows for a different mask for each entry
          in the batch. Binary, byte, and float masks are supported.
          For a binary mask, a ``True`` value indicates that the
          corresponding position is not allowed to attend. For a byte mask,
          a non-zero value indicates that the corresponding position is not
          allowed to attend. For a float mask, the mask values will be added to
          the attention weight.
      decode: wether to use cache for autoregressive decoding or not.
      max_len: maximum sequence length, necessary for decoding cache.
      cache: cache dictionary for autoregressive decoding.
      index: index of the current decoding step, necessary for decoding cache.
    Outputs:
      - **attn_output** - Attention outputs of shape :math:`(N, L, E)`, where
        :math:`L` is the target sequence length, :math:`N` is the batch size,
        and :math:`E` is the embedding dimension ``embed_dim``.
      - **cache** - For autoregressive decoding.
    """
    # Shape: (batch size, sequence length, embedding dimensionality)
    bsz, seq_len, embed_dim = x.size()
    # In projection.
    if self.self_attn:
      q, k, v = self.in_proj(x).split(self.embed_dim, dim=2)
    else:
      q = self.q_proj(x)
      k, v = self.kv_proj(mem).split(self.embed_dim, dim=2)
    # This is 1 (!= seq_len) during autoreregressive decoding.
    tgt_len = q.size(1)

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    name = f'decoder.layers.{index}.self_attn'
    loc_cache = cache[name] if decode and name in cache else None
    if decode:
      if loc_cache is None:
        loc_cache = {
            'cached_key':
                torch.zeros((bsz, max_len, embed_dim),
                            dtype=k.dtype,
                            device=k.device),
            'cached_value':
                torch.zeros((bsz, max_len, embed_dim),
                            dtype=v.dtype,
                            device=v.device),
            'cache_index':
                torch.tensor(0, dtype=torch.long, device=k.device),
        }
      cached_key = loc_cache['cached_key']
      cached_value = loc_cache['cached_value']
      cache_index = loc_cache['cache_index']
      # Shape check of cached keys against query input.
      expected_shape = (bsz, 1, embed_dim)
      if expected_shape != x.shape:
        raise ValueError('Autoregressive cache shape error, expected query '
                         f'shape {expected_shape} instead got {x.shape}.')
      # Update key, value caches with our new 1d spatial slices.
      cached_key[:, cache_index:cache_index + 1, :] = k
      cached_value[:, cache_index:cache_index + 1, :] = v
      k = cached_key
      v = cached_value
      cache_index += 1
      # Causal mask for cached decoder self-attention:
      # our single query position should only attend to those key
      # positions that have already been generated and cached,
      # not the remaining zero elements.
      if attn_mask is not None:
        raise ValueError('Attention mask has to be None for decode == True.')
      attn_mask = (torch.arange(max_len, device=k.device) >=
                   cache_index).reshape(1, max_len)

    # Update sequence length to account for complete sequence.
    seq_len = k.size(1)

    # Rearrange q, k, v for multihead attention.
    q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    # Check dtype and shape of attention mask.
    if not decode and attn_mask is not None:
      assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
            f'Float and bool dtypes are supported, not {attn_mask.dtype}.'
      # Ensure attn_mask's dim is 3.
      if attn_mask.dim() == 3:
        correct_3d_size = (bsz * self.num_heads, tgt_len, seq_len)
        if attn_mask.shape != correct_3d_size:
          raise RuntimeError(f'The shape of attn_mask is {attn_mask.shape}, '
                             f'but should be {correct_3d_size}.')
      else:
        raise RuntimeError(
            f"attn_mask's dimension {attn_mask.dim()} is not supported")
      # Reshape attention mask to be consistent with q, k, v.
      attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, seq_len)

    # Convert attention mask to float.
    if attn_mask is not None and attn_mask.dtype == torch.bool:
      new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
      new_attn_mask.masked_fill_(attn_mask, -1e10)
      attn_mask = new_attn_mask

    # Adjust dropout_rate probability.
    dropout_rate = self.dropout if self.training else 0.0

    # Calculate attention.
    q = self.attention_temp * q
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_rate)
    # Rearrange for output projection.
    attn_output = attn_output.transpose(1, 2).contiguous().view(
        bsz, tgt_len, embed_dim)
    # Output projection.
    attn_output = self.out_proj(attn_output)

    if decode:
      cache[name] = loc_cache

    return attn_output, cache
