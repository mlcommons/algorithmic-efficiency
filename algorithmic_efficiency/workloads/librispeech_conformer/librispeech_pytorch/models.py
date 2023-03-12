"""This is a pytorch implementation mirroring:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/conformer.py.
"""

from dataclasses import dataclass
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch import \
    preprocessor
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.spectrum_augmenter import \
    SpecAug


@dataclass
class ConformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int = 1024
  encoder_dim: int = 512
  num_attention_heads: int = 8
  num_encoder_layers: int = 4
  attention_dropout_rate: float = 0.0
  attention_residual_dropout_rate: float = 0.1
  conv_residual_dropout_rate: float = 0.0
  feed_forward_dropout_rate: float = 0.0
  feed_forward_residual_dropout_rate: float = 0.1
  convolution_kernel_size: int = 5
  feed_forward_expansion_factor: int = 4
  freq_mask_count: int = 2
  freq_mask_max_bins: int = 27
  time_mask_count: int = 10
  time_mask_max_frames: int = 40
  time_mask_max_ratio: float = 0.05
  time_masks_per_frame: float = 0.0
  use_dynamic_time_mask_max_frames: bool = True
  input_dropout_rate: float = 0.1
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  use_specaug: bool = True


def initialize(m):
  if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
    init.xavier_uniform_(m.weight)
    if m.bias is not None:
      init.constant_(m.bias, 0)
  elif isinstance(m, nn.MultiheadAttention):
    init.xavier_uniform_(m.in_proj_weight)
  for i in m.children():
    initialize(i)


class LayerNorm(nn.Module):

  def __init__(self, dim, epsilon=1e-6):
    super().__init__()
    self.dim = dim

    self.scale = nn.Parameter(torch.zeros(self.dim))
    self.bias = nn.Parameter(torch.zeros(self.dim))
    self.epsilon = epsilon

  def forward(self, x):
    return F.layer_norm(x, (self.dim,), 1 + self.scale, self.bias, self.epsilon)


class Subsample(nn.Module):

  def __init__(self, encoder_dim: int = 0, input_dropout_rate: float = 0.0):
    super().__init__()
    self.encoder_dim = encoder_dim
    self.input_dropout_rate = input_dropout_rate

    self.conv1 = Conv2dSubsampling(
        input_channels=1, output_channels=encoder_dim)
    self.conv2 = Conv2dSubsampling(
        input_channels=encoder_dim, output_channels=encoder_dim)

    self.linear = nn.LazyLinear(out_features=self.encoder_dim, bias=True)
    self.pos_encode = AddPositionalEmbedding(embedding_dim=self.encoder_dim)
    self.dropout = nn.Dropout(p=self.input_dropout_rate)

  def forward(self, inputs, input_paddings):
    output_paddings = input_paddings
    outputs = inputs[:, None, :, :]

    outputs, output_paddings = self.conv1(outputs, output_paddings)
    outputs, output_paddings = self.conv2(outputs, output_paddings)

    batch_size, channels, subsampled_lengths, subsampled_dims = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).reshape(batch_size,
                                                  subsampled_lengths,
                                                  subsampled_dims * channels)

    outputs = self.linear(outputs)
    outputs = outputs + self.pos_encode(seq_length=outputs.shape[1])
    outputs = self.dropout(outputs)

    return outputs, output_paddings


class Conv2dSubsampling(nn.Module):

  def __init__(self,
               input_channels: int,
               output_channels: int,
               filter_stride: Tuple[int] = (2, 2),
               padding: str = 'SAME'):
    super().__init__()

    self.input_channels = input_channels
    self.output_channels = output_channels
    self.filter_stride = filter_stride
    self.padding = padding

    self.filter_shape = (output_channels, input_channels, 3, 3)

    self.kernel = nn.Parameter(
        torch.nn.init.xavier_uniform_(torch.empty(*self.filter_shape)))
    self.bias = nn.Parameter(torch.zeros(output_channels))

  def get_same_padding(self, input_shape):
    in_height, in_width = input_shape[2:]
    stride_height, stride_width = self.filter_stride
    filter_height, filter_width = 3, 3
    if in_height % stride_height == 0:
      pad_along_height = max(filter_height - stride_height, 0)
    else:
      pad_along_height = max(filter_height - (in_height % stride_height), 0)
    if in_width % stride_width == 0:
      pad_along_width = max(filter_width - stride_width, 0)
    else:
      pad_along_width = max(filter_width - (in_width % stride_width), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

  def forward(self, inputs, paddings):
    groups = inputs.shape[1] // self.input_channels

    if self.padding == 'SAME':
      in_ = F.pad(inputs, self.get_same_padding(inputs.shape))
    else:
      in_ = inputs
    outputs = F.conv2d(
        input=in_,
        weight=self.kernel,
        bias=self.bias,
        stride=self.filter_stride,
        dilation=(1, 1),
        groups=groups)

    outputs = F.relu(outputs)

    input_length = paddings.shape[1]
    stride = self.filter_stride[0]
    pad_len = (input_length + stride - 1) // stride * stride - input_length
    padded_paddings = torch.cat([
        paddings[:, None, :],
        torch.zeros(
            size=(paddings.shape[0], 1, pad_len), device=paddings.device)
    ],
                                dim=2)
    out_padding = F.conv1d(
        input=padded_paddings,
        weight=torch.ones([1, 1, 1], device=paddings.device),
        stride=self.filter_stride[:1])
    out_padding = out_padding.squeeze(dim=1)
    outputs = outputs * (1 - out_padding[:, None, :, None])
    return outputs, out_padding


class FeedForwardModule(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    self.config = config

    self.ln = LayerNorm(dim=config.encoder_dim)
    self.linear1 = nn.LazyLinear(
        out_features=config.encoder_dim * config.feed_forward_expansion_factor,
        bias=True)
    self.dropout1 = nn.Dropout(p=config.feed_forward_dropout_rate)
    self.linear2 = nn.LazyLinear(out_features=config.encoder_dim, bias=True)

    if config.feed_forward_residual_dropout_rate is None:
      feed_forward_residual_dropout_rate = 0.1
    else:
      feed_forward_residual_dropout_rate = (
          config.feed_forward_residual_dropout_rate)
    self.dropout2 = nn.Dropout(p=feed_forward_residual_dropout_rate)

  def forward(self, inputs, padding_mask):
    inputs = self.ln(inputs)
    inputs = self.linear1(inputs)
    inputs = F.silu(inputs)
    inputs = self.dropout1(inputs)
    inputs = inputs * padding_mask
    inputs = self.linear2(inputs)
    inputs = inputs * padding_mask
    inputs = self.dropout2(inputs)

    return inputs


class AddPositionalEmbedding(nn.Module):

  def __init__(self,
               min_timescale: int = 1,
               max_timescale: int = 10_000,
               embedding_dim: int = 512):
    super().__init__()
    self.min_timescale = min_timescale
    self.max_timescale = max_timescale
    self.embedding_dim = embedding_dim
    num_timescales = self.embedding_dim // 2
    log_timescale_increment = math.log(
        float(self.max_timescale) / float(self.min_timescale)) / (
            num_timescales - 1)
    inv_timescales = self.min_timescale * \
        torch.exp(torch.arange(num_timescales, dtype=torch.float32)
                  * -log_timescale_increment)
    self.register_buffer('inv_timescales', inv_timescales[None, None, :])

  def forward(self, seq_length):
    position = torch.arange(
        end=seq_length, dtype=torch.float32, device=self.inv_timescales.device)
    scaled_time = position[None, :, None] * self.inv_timescales
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
    if self.embedding_dim % 2:
      signal = torch.cat(
          [signal, torch.zeros(signal.shape[0], signal.shape[1], 1)], dim=2)
    return signal


class QueryScaler(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.scale = nn.Parameter(torch.zeros(self.dim))

  def forward(self, inputs):
    r_softplus_0 = 1.442695041
    scale = r_softplus_0 * F.softplus(self.scale)
    return inputs * scale


class MHSAwithQS(nn.MultiheadAttention):
  # pylint: disable=locally-disabled, use-a-generator, line-too-long, invalid-name
  def __init__(self, config: ConformerConfig):
    super().__init__(
        embed_dim=config.encoder_dim,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout_rate,
        bias=True,
        batch_first=True)
    self.qs = QueryScaler(dim=config.encoder_dim // config.num_attention_heads)

  def forward(self,
              query,
              key,
              value,
              key_padding_mask=None,
              need_weights: bool = True,
              attn_mask=None,
              average_attn_weights: bool = True):
    r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
    is_batched = query.dim() == 3
    if key_padding_mask is not None:
      _kpm_dtype = key_padding_mask.dtype
      if _kpm_dtype != torch.bool and not torch.is_floating_point(
          key_padding_mask):
        raise AssertionError(
            "only bool and floating types of key_padding_mask are supported")
    why_not_fast_path = ''
    if not is_batched:
      why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    elif query is not key or key is not value:
      # When lifting this restriction, don't forget to either
      # enforce that the dtypes all match or test cases where
      # they don't!
      why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
      why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
    elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
      # this case will fail anyway, but at least they'll get a useful error message.
      why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
    elif self.training:
      why_not_fast_path = "training is enabled"
    elif not self.batch_first:
      why_not_fast_path = "batch_first was not True"
    elif self.bias_k is not None:
      why_not_fast_path = "self.bias_k was not None"
    elif self.bias_v is not None:
      why_not_fast_path = "self.bias_v was not None"
    elif self.dropout:
      why_not_fast_path = f"dropout was {self.dropout}, required zero"
    elif self.add_zero_attn:
      why_not_fast_path = "add_zero_attn was enabled"
    elif not self._qkv_same_embed_dim:
      why_not_fast_path = "_qkv_same_embed_dim was not True"
    elif attn_mask is not None:
      why_not_fast_path = "attn_mask was not None"
    elif query.is_nested and key_padding_mask is not None:
      why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
    elif self.num_heads % 2 == 1:
      why_not_fast_path = "num_heads is odd"
    elif torch.is_autocast_enabled():
      why_not_fast_path = "autocast is enabled"

    if not why_not_fast_path:
      tensor_args = (
          query,
          key,
          value,
          self.in_proj_weight,
          self.in_proj_bias,
          self.out_proj.weight,
          self.out_proj.bias,
      )
      # We have to use list comprehensions below because TorchScript does not support
      # generator expressions.
      if torch.overrides.has_torch_function(tensor_args):
        why_not_fast_path = "some Tensor argument has_torch_function"
      elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device))
                    for x in tensor_args]):
        why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
      elif torch.is_grad_enabled() and any(
          [x is not None and x.requires_grad for x in tensor_args]):
        why_not_fast_path = (
            "grad is enabled and at least one of query or the "
            "input/output projection weights or biases requires_grad")
      if not why_not_fast_path:
        # Scale the query bias parameter and the query vector
        query = self.qs(
            query.view(query.shape[0],
                       query.shape[1],
                       self.num_heads,
                       self.embed_dim // self.num_heads)).view(*query.shape)
        in_proj_bias = self.in_proj_bias + 0
        in_proj_bias[:self.embed_dim] = self.qs(
            self.in_proj_bias[:self.embed_dim].view(
                self.num_heads, self.embed_dim // self.num_heads)).view(-1)
        return torch._native_multi_head_attention(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            in_proj_bias,
            self.out_proj.weight,
            self.out_proj.bias,
            key_padding_mask if key_padding_mask is not None else attn_mask,
            need_weights,
            average_attn_weights,
            1 if key_padding_mask is not None else
            0 if attn_mask is not None else None)
    any_nested = query.is_nested or key.is_nested or value.is_nested
    assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                            f"The fast path was not hit because {why_not_fast_path}")

    if self.batch_first and is_batched:
      # make sure that the transpose op does not affect the "is" property
      if key is value:
        if query is key:
          query = key = value = query.transpose(1, 0)
        else:
          query, key = [x.transpose(1, 0) for x in (query, key)]
          value = key
      else:
        query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    if not self._qkv_same_embed_dim:
      attn_output, attn_output_weights = F.multi_head_attention_forward(
          query, key, value, self.embed_dim, self.num_heads,
          self.in_proj_weight, self.in_proj_bias,
          self.bias_k, self.bias_v, self.add_zero_attn,
          self.dropout, self.out_proj.weight, self.out_proj.bias,
          training=self.training,
          key_padding_mask=key_padding_mask, need_weights=need_weights,
          attn_mask=attn_mask, use_separate_proj_weight=True,
          q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
          v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
    else:
      # Scale the query bias parameter and the query vector
      query = self.qs(
          query.view(query.shape[0],
                     query.shape[1],
                     self.num_heads,
                     self.embed_dim // self.num_heads)).view(*query.shape)
      in_proj_bias = self.in_proj_bias + 0
      in_proj_bias[:self.embed_dim] = self.qs(
          self.in_proj_bias[:self.embed_dim].view(
              self.num_heads, self.embed_dim // self.num_heads)).view(-1)
      attn_output, attn_output_weights = F.multi_head_attention_forward(
          query, key, value, self.embed_dim, self.num_heads,
          self.in_proj_weight, in_proj_bias,
          self.bias_k, self.bias_v, self.add_zero_attn,
          self.dropout, self.out_proj.weight, self.out_proj.bias,
          training=self.training,
          key_padding_mask=key_padding_mask, need_weights=need_weights,
          attn_mask=attn_mask, average_attn_weights=average_attn_weights)
    if self.batch_first and is_batched:
      return attn_output.transpose(1, 0), attn_output_weights
    else:
      return attn_output, attn_output_weights


class MultiHeadedSelfAttention(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()

    self.config = config

    self.ln = LayerNorm(dim=config.encoder_dim)
    self.self_attention = MHSAwithQS(config)
    if config.attention_residual_dropout_rate is None:
      attention_residual_dropout_rate = 0.1
    else:
      attention_residual_dropout_rate = config.attention_residual_dropout_rate
    self.dropout = nn.Dropout(p=attention_residual_dropout_rate)

  def forward(self, outputs, paddings):
    outputs = self.ln(outputs)
    outputs, _ = self.self_attention(
        query=outputs,
        key=outputs,
        value=outputs,
        key_padding_mask=paddings==1,
        need_weights=False,
    )
    outputs = self.dropout(outputs)
    return outputs


class BatchNorm(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    running_mean = torch.zeros(config.encoder_dim)
    running_var = torch.ones(config.encoder_dim)
    self.register_buffer('running_mean', running_mean)
    self.register_buffer('running_var', running_var)
    self.scale = nn.Parameter(torch.zeros(config.encoder_dim))
    self.bias = nn.Parameter(torch.zeros(config.encoder_dim))
    self.register_buffer('momentum',
                         torch.FloatTensor([config.batch_norm_momentum]))
    self.register_buffer('epsilon',
                         torch.FloatTensor([config.batch_norm_epsilon]))
    self.register_buffer('dim', torch.FloatTensor([config.encoder_dim]))
    # self.momentum = config.batch_norm_momentum
    # self.epsilon = config.batch_norm_epsilon
    # self.dim = config.encoder_dim

  def forward(self, inputs, input_paddings):
    #inputs: NHD
    #padding: NH
    mask = 1 - input_paddings[:, :, None]
    if self.training:
      count = mask.sum()
      masked_inp = inputs.masked_fill(mask == 0, 0)
      mean = (masked_inp).sum(dim=(0, 1)) / count
      var = (torch.square(masked_inp - mean) * mask).sum(dim=(0, 1)) / count

      self.running_mean = self.momentum * self.running_mean + (
          1 - self.momentum) * mean.detach()
      self.running_var = self.momentum * self.running_var + (
          1 - self.momentum) * var.detach()
    else:
      mean = self.running_mean
      var = self.running_var
    v = (1 + self.scale) * torch.rsqrt(var + self.epsilon)
    bn = (inputs - mean) * v + self.bias
    output = bn.masked_fill(mask == 0, 0)
    return output


class ConvolutionBlock(nn.Module):

  def __init__(self, config):
    super().__init__()

    self.config = config
    self.ln = LayerNorm(dim=config.encoder_dim)
    self.lin1 = nn.Linear(
        in_features=config.encoder_dim, out_features=config.encoder_dim)
    self.lin2 = nn.Linear(
        in_features=config.encoder_dim, out_features=config.encoder_dim)

    self.conv1 = nn.Conv1d(
        in_channels=config.encoder_dim,
        out_channels=config.encoder_dim,
        kernel_size=(config.convolution_kernel_size,),
        stride=(1,),
        padding='same',
        bias=False,
        groups=config.encoder_dim)
    self.bn = BatchNorm(config)
    self.lin3 = nn.Linear(config.encoder_dim, config.encoder_dim)
    if config.conv_residual_dropout_rate is None:
      conv_residual_dropout_rate = 0.0
    else:
      conv_residual_dropout_rate = config.conv_residual_dropout_rate
    self.dropout = nn.Dropout(p=conv_residual_dropout_rate)

  def forward(self, inputs, input_paddings):
    inputs = self.ln(inputs)

    inputs = F.glu(torch.cat([self.lin1(inputs), self.lin2(inputs)], dim=2))
    inputs = inputs * (1 - input_paddings[:, :, None])

    inputs = inputs.permute(0, 2, 1)
    inputs = self.conv1(inputs)
    inputs = inputs.permute(0, 2, 1)

    inputs = self.bn(inputs, input_paddings)
    inputs = F.silu(inputs)
    inputs = self.lin3(inputs)

    inputs = self.dropout(inputs)
    return inputs


class ConformerBlock(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()

    self.ff1 = FeedForwardModule(config)
    self.mhsa = MultiHeadedSelfAttention(config)
    self.conv = ConvolutionBlock(config)
    self.ff2 = FeedForwardModule(config)
    self.ln = LayerNorm(dim=config.encoder_dim)

  def forward(self, inputs, input_paddings):
    padding_mask = 1 - input_paddings[:, :, None]
    inputs = inputs + 0.5 * self.ff1(inputs, padding_mask)
    inputs = inputs + self.mhsa(inputs, input_paddings)
    inputs = inputs + self.conv(inputs, input_paddings)
    inputs = inputs + 0.5 * self.ff2(inputs, padding_mask)
    inputs = self.ln(inputs)
    return inputs


class ConformerEncoderDecoder(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    self.config = config
    preprocessing_config = preprocessor.PreprocessorConfig()
    self.preprocessor = preprocessor.MelFilterbankFrontend(
        preprocessing_config,
        per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
        per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR)
    self.specaug = SpecAug(
        freq_mask_count=config.freq_mask_count,
        freq_mask_max_bins=config.freq_mask_max_bins,
        time_mask_count=config.time_mask_count,
        time_mask_max_frames=config.time_mask_max_frames,
        time_mask_max_ratio=config.time_mask_max_ratio,
        time_masks_per_frame=config.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=config.use_dynamic_time_mask_max_frames
    )
    if config.input_dropout_rate is None:
      input_dropout_rate = 0.1
    else:
      input_dropout_rate = config.input_dropout_rate
    self.subsample = Subsample(
        encoder_dim=config.encoder_dim, input_dropout_rate=input_dropout_rate)
    self.conformers = nn.ModuleList(
        [ConformerBlock(config) for _ in range(config.num_encoder_layers)])

    self.ln = LayerNorm(config.encoder_dim)
    self.lin = nn.Linear(config.encoder_dim, config.vocab_size)

  def forward(self, inputs, input_paddings):
    outputs = inputs
    output_paddings = input_paddings
    outputs, output_paddings = self.preprocessor(outputs, output_paddings)
    if self.training and self.config.use_specaug:
      outputs, output_paddings = self.specaug(outputs, output_paddings)
    outputs, output_paddings = self.subsample(outputs, output_paddings)
    for conformer in self.conformers:
      outputs = conformer(outputs, output_paddings)
    outputs = self.ln(outputs)
    outputs = self.lin(outputs)
    return outputs, output_paddings
