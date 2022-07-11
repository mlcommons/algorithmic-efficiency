"""
This is a pytorch implementation mirroring:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/conformer.py
"""
from collections import namedtuple
import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

ConformerConfig = namedtuple(
    "ConformerConfig",
    field_names=[
        "vocab_size",
        "dtype",
        "encoder_dim",
        "num_attention_heads",
        "num_encoder_layers",
        "attention_dropout_rate",
        "attention_residual_dropout_rate",
        "input_dropout_rate",
        "conv_residual_dropout_rate",
        "feed_forward_dropout_rate",
        "feed_forward_residual_dropout_rate",
        "convolution_kernel_size",
        "feed_forward_expansion_factor",
        "conv_expansion_factor",
        "conv_subsampling_factor",
        "conv_subsampling_layers",
        "train",
        "use_specaug",
        "freq_mask_count",
        "freq_mask_max_bins",
        "time_mask_count",
        "time_mask_max_frames",
        "time_mask_max_ratio",
        "time_masks_per_frame",
        "use_dynamic_time_mask_max_frames",
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "enable_conformer_post_layer_norm",
        "enable_decoder_pre_layer_norm",
        "use_lingvo_attention",
    ])


class LayerNorm(nn.Module):

  def __init__(self, dim, epsilon=1e-6):
    super().__init__()
    self.dim = dim

    self.scale = nn.Parameter(torch.zeros(self.dim))
    self.bias = nn.Parameter(torch.zeros(self.dim))
    self.epsilon = epsilon

  def forward(self, x):
    mean = x.mean(dim=-1, keepdims=True)
    var = x.var(dim=-1, unbiased=False, keepdims=True)

    normed_x = (x - mean) * torch.rsqrt(var + self.epsilon)
    normed_x *= (1 + self.scale)
    normed_x += self.bias

    return normed_x


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
    out_padding = F.conv1d(
        input=torch.cat([
            paddings[:, None, :],
            torch.zeros(size=(paddings.shape[0], 1, pad_len))
        ],
                        dim=2),
        weight=torch.ones([1, 1, 1]),
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
    self.dropout2 = nn.Dropout(p=config.feed_forward_residual_dropout_rate)

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
    self.inv_timescales = inv_timescales[None, None, :]

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


class MultiHeadedSelfAttention(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()

    self.config = config

    self.ln = LayerNorm(dim=config.encoder_dim)
    self.qs = QueryScaler(dim=config.encoder_dim)
    self.self_attention = nn.MultiheadAttention(
        embed_dim=config.encoder_dim,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout_rate,
        bias=True,
        batch_first=True)
    self.dropout = nn.Dropout(p=config.attention_residual_dropout_rate)

  def forward(self, outputs, paddings):
    outputs = self.ln(outputs)
    outputs, _ = self.self_attention(
        query=self.qs(outputs),
        key=outputs,
        value=outputs,
        key_padding_mask=paddings,
        need_weights=False,
    )
    outputs = self.dropout(outputs)
    return outputs


class BatchNorm(nn.Module):

  def __init__(self, config: ConformerConfig):
    super().__init__()
    self.bn = nn.BatchNorm1d(
        num_features=config.encoder_dim,
        momentum=config.batch_norm_momentum,
        eps=config.batch_norm_epsilon)

  def forward(self, inputs, input_paddings):
    #inputs: NHD
    #padding: NH
    n, h, d = inputs.shape
    inputs = inputs.reshape(n * h, d)
    input_paddings = input_paddings.reshape(n * h)
    bn_inp = self.bn(inputs[input_paddings == 0])
    output = torch.zeros(n * h, d, device=inputs.device)
    output[input_paddings == 0] = bn_inp

    return output.reshape(n, h, d)


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
        padding="same",
        bias=False,
        groups=config.encoder_dim)
    self.bn = BatchNorm(config)
    self.lin3 = nn.Linear(config.encoder_dim, config.encoder_dim)
    self.dropout = nn.Dropout(p=config.conv_residual_dropout_rate)

  def forward(self, inputs, input_paddings):
    inputs = self.ln(inputs)

    input_gated1 = self.lin1(inputs)
    input_gated2 = self.lin2(inputs)

    inputs = input_gated1 * torch.sigmoid(input_gated2)
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

    if config.enable_conformer_post_layer_norm:
      self.ln = LayerNorm(dim=config.encoder_dim)
    else:
      self.ln = nn.Identity()

  def forward(self, inputs, input_paddings):
    padding_mask = 1 - input_paddings[:, :, None]
    inputs = inputs + 0.5 * self.ff1(inputs, padding_mask)
    inputs = inputs + self.mhsa(inputs, input_paddings)
    inputs = inputs + self.conv(inputs, input_paddings)
    inputs = inputs + 0.5 * self.ff2(inputs, padding_mask)
    inputs = self.ln(inputs)
    return inputs
