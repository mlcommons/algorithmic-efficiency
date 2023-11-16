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
  attention_temperature: float = 1.0
  activation_function_name: str = 'swish'
  use_post_layer_norm: bool = True


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

  def __init__(self,
               encoder_dim: int = 0,
               input_dropout_rate: float = 0.0,
               num_bins: int = 80):
    super().__init__()
    self.encoder_dim = encoder_dim
    self.input_dropout_rate = input_dropout_rate

    self.conv1 = Conv2dSubsampling(
        input_channels=1, output_channels=encoder_dim)
    self.conv2 = Conv2dSubsampling(
        input_channels=encoder_dim, output_channels=encoder_dim)

    self.linear = nn.Linear(
        in_features=self.encoder_dim * num_bins // 4,
        out_features=self.encoder_dim,
        bias=True)
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
    self.register_buffer('paddings_kernel', torch.ones([1, 1, 1]))

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
    padded_paddings = F.pad(
        paddings[:, None, :], (0, pad_len), mode='constant', value=0)
    out_padding = F.conv1d(
        input=padded_paddings,
        weight=self.paddings_kernel,
        stride=self.filter_stride[:1])
    out_padding = out_padding.squeeze(dim=1)
    outputs = outputs * (1 - out_padding[:, None, :, None])
    return outputs, out_padding


class FeedForwardModule(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    self.config = config

    self.ln = LayerNorm(dim=config.encoder_dim)
    self.linear1 = nn.Linear(
        in_features=config.encoder_dim,
        out_features=config.encoder_dim * config.feed_forward_expansion_factor,
        bias=True)
    self.dropout1 = nn.Dropout(p=config.feed_forward_dropout_rate)
    self.linear2 = nn.Linear(
        in_features=config.encoder_dim * config.feed_forward_expansion_factor,
        out_features=config.encoder_dim,
        bias=True)

    if config.feed_forward_residual_dropout_rate is None:
      feed_forward_residual_dropout_rate = 0.1
    else:
      feed_forward_residual_dropout_rate = (
          config.feed_forward_residual_dropout_rate)
    self.dropout2 = nn.Dropout(p=feed_forward_residual_dropout_rate)

  def forward(self, inputs, padding_mask):
    inputs = self.ln(inputs)
    inputs = self.linear1(inputs)
    if self.config.activation_function_name == 'swish':
      activation_fn = F.silu
    elif self.config.activation_function_name == 'gelu':
      activation_fn = F.gelu
    else:
      raise ValueError(
          'Only "swish" and "gelu" are supported '
          'config.activation_function_name values, recieved '
          f'{self.config.activation_function_name}')
    inputs = activation_fn(inputs)
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


class MHSAwithQS(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    self.embed_dim = config.encoder_dim
    self.num_heads = config.num_attention_heads
    self.dropout = config.attention_dropout_rate
    self.in_proj = nn.Linear(config.encoder_dim, 3 * config.encoder_dim)
    self.out_proj = nn.Linear(config.encoder_dim, config.encoder_dim)
    self.qs = QueryScaler(dim=config.encoder_dim // config.num_attention_heads)
    self.attention_temperature = config.attention_temperature

  def forward(self, inputs, key_padding_mask=None):
    batch_size, seq_len, embed_dim = inputs.shape
    q, k, v = self.in_proj(inputs).split(self.embed_dim, dim=2)
    q = self.qs(q.view(batch_size, seq_len, self.num_heads, -1)).transpose(1, 2)
    k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=~key_padding_mask[:, None, None],
        dropout_p=self.dropout,
    ).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
    out = out * attention_temperature
    out = self.out_proj(out)
    return out


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
    outputs = self.self_attention(
        outputs,
        key_padding_mask=paddings == 1,
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

    self.register_buffer('dim', torch.FloatTensor([config.encoder_dim]))
    self.momentum = config.batch_norm_momentum
    self.epsilon = config.batch_norm_epsilon

  def forward(self, inputs, input_paddings):
    #inputs: NHD
    #padding: NH
    """
    Alternatively:
    inputs[input_paddings==0] = F.batch_norm(
      input = inputs[input_paddings==0],
      running_mean = self.running_mean,
      running_var = self.running_var,
      weight = 1+self.scale,
      bias = self.bias,
      training = self.training,
      momentum=1-self.momentum,
      eps=self.epsilon
    )
    inputs.masked_fill(input_paddings[...,None] != 0, 0)
    return inputs
    """
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
    if self.config.activation_function_name == 'swish':
      activation_fn = F.silu
    elif self.config.activation_function_name == 'gelu':
      activation_fn = F.gelu
    else:
      raise ValueError(
          'Only "swish" and "gelu" are supported '
          'config.activation_function_name values, recieved '
          f'{self.config.activation_function_name}')
    inputs = activation_fn(inputs)
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
    self.ln = None
    if config.use_post_layer_norm:
      self.ln = LayerNorm(dim=config.encoder_dim)

  def forward(self, inputs, input_paddings):
    padding_mask = 1 - input_paddings[:, :, None]
    inputs = inputs + 0.5 * self.ff1(inputs, padding_mask)
    inputs = inputs + self.mhsa(inputs, input_paddings)
    inputs = inputs + self.conv(inputs, input_paddings)
    inputs = inputs + 0.5 * self.ff2(inputs, padding_mask)
    if self.ln:
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
        encoder_dim=config.encoder_dim,
        input_dropout_rate=input_dropout_rate,
        num_bins=preprocessing_config.num_bins)
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
