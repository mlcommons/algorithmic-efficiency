"""
This is a pytorch implementation mirroring:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/conformer.py
"""

from dataclasses import dataclass
import functools
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch import \
    preprocessor
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.spectrum_augmenter import \
    SpecAug

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


@dataclass
class DeepspeechConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int = 1024
  encoder_dim: int = 512
  num_lstm_layers: int = 4
  num_ffn_layers: int = 3
  conv_subsampling_factor: int = 2
  conv_subsampling_layers: int = 2
  use_specaug: bool = True
  freq_mask_count: int = 2
  freq_mask_max_bins: int = 27
  time_mask_count: int = 10
  time_mask_max_frames: int = 40
  time_mask_max_ratio: float = 0.05
  time_masks_per_frame: float = 0.0
  use_dynamic_time_mask_max_frames: bool = True
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  # If None, defaults to 0.1.
  input_dropout_rate: Optional[float] = 0.1
  # If None, defaults to 0.1.
  feed_forward_dropout_rate: Optional[float] = 0.1
  enable_residual_connections: bool = True
  enable_decoder_layer_norm: bool = True
  bidirectional: bool = True


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

  def __init__(self, config: DeepspeechConfig):
    super().__init__()
    encoder_dim = config.encoder_dim

    self.encoder_dim = encoder_dim

    self.conv1 = Conv2dSubsampling(
        input_channels=1, output_channels=encoder_dim)
    self.conv2 = Conv2dSubsampling(
        input_channels=encoder_dim, output_channels=encoder_dim)

    self.lin = nn.LazyLinear(out_features=self.encoder_dim, bias=True)

    if config.input_dropout_rate is None:
      input_dropout_rate = 0.1
    else:
      input_dropout_rate = config.input_dropout_rate
    self.dropout = nn.Dropout(p=input_dropout_rate)

  def forward(self, inputs, input_paddings):
    output_paddings = input_paddings
    outputs = inputs[:, None, :, :]

    outputs, output_paddings = self.conv1(outputs, output_paddings)
    outputs, output_paddings = self.conv2(outputs, output_paddings)

    batch_size, channels, subsampled_lengths, subsampled_dims = outputs.shape
    outputs = outputs.permute(0, 2, 3, 1).reshape(batch_size,
                                                  subsampled_lengths,
                                                  subsampled_dims * channels)

    outputs = self.lin(outputs)
    outputs = self.dropout(outputs)

    return outputs, output_paddings


class Conv2dSubsampling(nn.Module):

  def __init__(self,
               input_channels: int,
               output_channels: int,
               filter_stride: Tuple[int] = (2, 2),
               padding: str = 'SAME',
               batch_norm_momentum: float = 0.999,
               batch_norm_epsilon: float = 0.001):
    super().__init__()

    self.input_channels = input_channels
    self.output_channels = output_channels
    self.filter_stride = filter_stride
    self.padding = padding

    self.filter_shape = (output_channels, input_channels, 3, 3)

    self.kernel = nn.Parameter(
        nn.init.xavier_uniform_(torch.empty(*self.filter_shape)))
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
            torch.zeros(
                size=(paddings.shape[0], 1, pad_len), device=paddings.device)
        ],
                        dim=2),
        weight=torch.ones([1, 1, 1], device=paddings.device),
        stride=self.filter_stride[:1])
    out_padding = out_padding.squeeze(dim=1)
    outputs = outputs * (1 - out_padding[:, None, :, None])
    return outputs, out_padding


class FeedForwardModule(nn.Module):

  def __init__(self, config: DeepspeechConfig):
    super().__init__()
    self.config = config

    self.bn = BatchNorm(
        dim=config.encoder_dim,
        batch_norm_momentum=config.batch_norm_momentum,
        batch_norm_epsilon=config.batch_norm_epsilon)
    self.lin = nn.LazyLinear(out_features=config.encoder_dim, bias=True)
    if config.feed_forward_dropout_rate is None:
      feed_forward_dropout_rate = 0.1
    else:
      feed_forward_dropout_rate = config.feed_forward_dropout_rate
    self.dropout = nn.Dropout(p=feed_forward_dropout_rate)

  def forward(self, inputs, input_paddings):
    padding_mask = (1 - input_paddings)[:, :, None]
    inputs = self.bn(inputs, input_paddings)
    inputs = self.lin(inputs)
    inputs = F.relu(inputs)
    inputs = inputs * padding_mask
    inputs = self.dropout(inputs)

    return inputs


class CustomBatchNorm1d(nn.SyncBatchNorm):
  # pylint: disable=locally-disabled, use-a-generator, line-too-long, redefined-builtin, pointless-string-statement
  def __init__(self, num_features, momentum, eps):
    super().__init__(num_features=num_features, momentum=momentum, eps=eps)

  def reset_parameters(self) -> None:
    self.reset_running_stats()
    if self.affine:
      init.zeros_(self.weight)
      init.zeros_(self.bias)

  def forward(self, input):
    # currently only GPU input is supported: added check below
    # if not input.is_cuda:
    #     raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

    self._check_input_dim(input)
    self._check_non_zero_input_channels(input)

    # exponential_average_factor is set to self.momentum
    # (when it is available) only so that it gets updated
    # in ONNX graph when this node is exported to ONNX.
    if self.momentum is None:
      exponential_average_factor = 0.0
    else:
      exponential_average_factor = self.momentum

    if self.training and self.track_running_stats:
      assert self.num_batches_tracked is not None
      self.num_batches_tracked.add_(1)
      if self.momentum is None:  # use cumulative moving average
        exponential_average_factor = 1.0 / self.num_batches_tracked.item()
      else:  # use exponential moving average
        exponential_average_factor = self.momentum
    r"""
    Decide whether the mini-batch stats should be used for normalization rather than the buffers.
    Mini-batch stats are used in training mode, and in eval mode when buffers are None.
    """
    if self.training:
      bn_training = True
    else:
      bn_training = (self.running_mean is None) and (self.running_var is None)
    r"""
    Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
    passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
    used for normalization (i.e. in eval mode when buffers are not None).
    """
    # If buffers are not to be tracked, ensure that they won't be updated
    running_mean = (
        self.running_mean
        if not self.training or self.track_running_stats else None)
    running_var = (
        self.running_var
        if not self.training or self.track_running_stats else None)

    # Don't sync batchnorm stats in inference mode (model.eval()).
    need_sync = (
        bn_training and self.training and USE_PYTORCH_DDP and input.is_cuda)
    if need_sync:
      process_group = torch.distributed.group.WORLD
      if self.process_group:
        process_group = self.process_group
      world_size = torch.distributed.get_world_size(process_group)
      need_sync = world_size > 1

    # fallback to framework BN when synchronization is not necessary
    if not need_sync:
      return F.batch_norm(
          input,
          running_mean,
          running_var,
          1 + self.weight,
          self.bias,
          bn_training,
          exponential_average_factor,
          self.eps,
      )
    else:
      assert bn_training
      return sync_batch_norm.apply(
          input,
          1 + self.weight,
          self.bias,
          running_mean,
          running_var,
          self.eps,
          exponential_average_factor,
          process_group,
          world_size,
      )


class BatchNorm(nn.Module):

  def __init__(self, dim, batch_norm_momentum, batch_norm_epsilon):
    super().__init__()
    self.bn = CustomBatchNorm1d(
        num_features=dim,
        momentum=1 - batch_norm_momentum,
        eps=batch_norm_epsilon)

  def forward(self, inputs, input_paddings=None):
    # inputs: ...D
    #padding: ...
    *b, d = inputs.shape
    reduce_size = functools.reduce(lambda x, y: x * y, b)

    inputs = inputs.reshape(reduce_size, d)
    if input_paddings is None:
      input_paddings = torch.zeros(size=b, device=inputs.device)
    input_paddings = input_paddings.reshape(reduce_size)
    bn_inp = self.bn(inputs[input_paddings == 0])
    output = torch.zeros(reduce_size, d, device=inputs.device)
    output[input_paddings == 0] = bn_inp

    return output.reshape(*b, d)


class BatchRNN(nn.Module):

  def __init__(self, config: DeepspeechConfig):
    super().__init__()
    self.config = config
    hidden_size = config.encoder_dim
    input_size = config.encoder_dim
    bidirectional = config.bidirectional
    self.bidirectional = bidirectional

    self.bn = BatchNorm(config.encoder_dim,
                        config.batch_norm_momentum,
                        config.batch_norm_epsilon)

    if bidirectional:
      self.lstm = nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size // 2,
          bidirectional=True,
          batch_first=True)
    else:
      self.lstm = nn.LSTM(
          input_size=input_size, hidden_size=hidden_size, batch_first=True)

  def forward(self, inputs, input_paddings):
    inputs = self.bn(inputs, input_paddings)
    lengths = torch.sum(1 - input_paddings, dim=1).detach().cpu()
    packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
        inputs, lengths, batch_first=True, enforce_sorted=False)
    packed_outputs, _ = self.lstm(packed_inputs)
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed_outputs, batch_first=True)
    if outputs.shape[1] < inputs.shape[1]:
      outputs = torch.cat([
          outputs,
          torch.zeros(
              size=(outputs.shape[0],
                    inputs.shape[1] - outputs.shape[1],
                    outputs.shape[2]),
              device=outputs.device)
      ],
                          dim=1)
    return outputs


class DeepspeechEncoderDecoder(nn.Module):

  def __init__(self, config: DeepspeechConfig):
    super().__init__()
    self.config = config

    self.specaug = SpecAug(
        freq_mask_count=config.freq_mask_count,
        freq_mask_max_bins=config.freq_mask_max_bins,
        time_mask_count=config.time_mask_count,
        time_mask_max_frames=config.time_mask_max_frames,
        time_mask_max_ratio=config.time_mask_max_ratio,
        time_masks_per_frame=config.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=config.use_dynamic_time_mask_max_frames
    )
    preprocessing_config = preprocessor.PreprocessorConfig()
    self.preprocessor = preprocessor.MelFilterbankFrontend(
        preprocessing_config,
        per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
        per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR)

    self.subsample = Subsample(config=config)

    self.lstms = nn.ModuleList(
        [BatchRNN(config) for _ in range(config.num_lstm_layers)])
    self.ffns = nn.ModuleList(
        [FeedForwardModule(config) for _ in range(config.num_ffn_layers)])

    if config.enable_decoder_layer_norm:
      self.ln = LayerNorm(config.encoder_dim)
    else:
      self.ln = nn.Identity()

    self.lin = nn.Linear(config.encoder_dim, config.vocab_size)

  def forward(self, inputs, input_paddings):
    outputs = inputs
    output_paddings = input_paddings

    outputs, output_paddings = self.preprocessor(outputs, output_paddings)
    if self.training:
      outputs, output_paddings = self.specaug(outputs, output_paddings)
    outputs, output_paddings = self.subsample(outputs, output_paddings)
    for idx in range(self.config.num_lstm_layers):
      if self.config.enable_residual_connections:
        outputs = outputs + self.lstms[idx](outputs, output_paddings)
      else:
        outputs = self.lstms[idx](outputs, output_paddings)

    for idx in range(self.config.num_ffn_layers):
      if self.config.enable_residual_connections:
        outputs = outputs + self.ffns[idx](outputs, output_paddings)
      else:
        outputs = self.ffns[idx](outputs, output_paddings)

    if self.config.enable_decoder_layer_norm:
      outputs = self.ln(outputs)

    outputs = self.lin(outputs)

    return outputs, output_paddings
