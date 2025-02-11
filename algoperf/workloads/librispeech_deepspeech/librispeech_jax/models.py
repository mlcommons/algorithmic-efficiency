r"""Deepspeech.

This model uses a deepspeech2 network to convert speech to text.
paper : https://arxiv.org/abs/1512.02595

# BiLSTM code contributed by bastings@
# github : https://github.com/bastings
# webpage : https://bastings.github.io/
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from flax import linen as nn
from flax import struct
import jax
from jax.experimental import rnn
import jax.numpy as jnp

from algoperf.workloads.librispeech_conformer.librispeech_jax import \
    librispeech_preprocessor as preprocessor
from algoperf.workloads.librispeech_conformer.librispeech_jax import \
    spectrum_augmenter

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Carry = Any
CarryHistory = Any
Output = Any


@struct.dataclass
class DeepspeechConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int = 1024
  dtype: Any = jnp.float32
  encoder_dim: int = 512
  num_lstm_layers: int = 6
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
  use_tanh: bool = False
  layernorm_everywhere: bool = False


class Subsample(nn.Module):
  """Module to perform strided convolution in order to subsample inputs.

  Attributes:
    encoder_dim: model dimension of conformer.
    input_dropout_rate: dropout rate for inputs.
  """
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, output_paddings, train):
    config = self.config
    outputs = jnp.expand_dims(inputs, axis=-1)

    outputs, output_paddings = Conv2dSubsampling(
        encoder_dim=config.encoder_dim,
        dtype=config.dtype,
        batch_norm_momentum=config.batch_norm_momentum,
        batch_norm_epsilon=config.batch_norm_epsilon,
        input_channels=1,
        output_channels=config.encoder_dim,
        use_tanh=config.use_tanh
        )(outputs, output_paddings, train)

    outputs, output_paddings = Conv2dSubsampling(
        encoder_dim=config.encoder_dim,
        dtype=config.dtype,
        batch_norm_momentum=config.batch_norm_momentum,
        batch_norm_epsilon=config.batch_norm_epsilon,
        input_channels=config.encoder_dim,
        output_channels=config.encoder_dim,
        use_tanh=config.use_tanh)(outputs, output_paddings, train)

    batch_size, subsampled_lengths, subsampled_dims, channels = outputs.shape

    outputs = jnp.reshape(
        outputs, (batch_size, subsampled_lengths, subsampled_dims * channels))

    outputs = nn.Dense(
        config.encoder_dim,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(
            outputs)

    if config.input_dropout_rate is None:
      input_dropout_rate = 0.1
    else:
      input_dropout_rate = config.input_dropout_rate
    outputs = nn.Dropout(
        rate=input_dropout_rate, deterministic=not train)(
            outputs)

    return outputs, output_paddings


class Conv2dSubsampling(nn.Module):
  """Helper module used in Subsample layer.

  1) Performs strided convolution over inputs and then applies non-linearity.
  2) Also performs strided convolution over input_paddings to return the correct
  paddings for downstream layers.
  """
  input_channels: int = 0
  output_channels: int = 0
  filter_stride: List[int] = (2, 2)
  padding: str = 'SAME'
  encoder_dim: int = 0
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001
  use_tanh: bool = False

  def setup(self):
    self.filter_shape = (3, 3, self.input_channels, self.output_channels)
    self.kernel = self.param('kernel',
                             nn.initializers.xavier_uniform(),
                             self.filter_shape)
    self.bias = self.param(
        'bias', lambda rng, s: jnp.zeros(s, jnp.float32), self.output_channels)

  @nn.compact
  def __call__(self, inputs, paddings, train):
    # Computing strided convolution to subsample inputs.
    feature_group_count = inputs.shape[3] // self.filter_shape[2]
    outputs = jax.lax.conv_general_dilated(
        lhs=inputs,
        rhs=self.kernel,
        window_strides=self.filter_stride,
        padding=self.padding,
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=feature_group_count)

    outputs += jnp.reshape(self.bias, (1,) * (outputs.ndim - 1) + (-1,))

    if self.use_tanh:
      outputs = nn.tanh(outputs)
    else:
      outputs = nn.relu(outputs)

    # Computing correct paddings post input convolution.
    input_length = paddings.shape[1]
    stride = self.filter_stride[0]

    pad_len = (input_length + stride - 1) // stride * stride - input_length
    out_padding = jax.lax.conv_general_dilated(
        lhs=paddings[:, :, None],
        rhs=jnp.ones([1, 1, 1]),
        window_strides=self.filter_stride[:1],
        padding=[(0, pad_len)],
        dimension_numbers=('NHC', 'HIO', 'NHC'))
    out_padding = jnp.squeeze(out_padding, axis=-1)

    # Mask outputs by correct paddings to ensure padded elements in inputs map
    # to padded value in outputs.
    outputs = outputs * (1.0 -
                         jnp.expand_dims(jnp.expand_dims(out_padding, -1), -1))

    return outputs, out_padding


class FeedForwardModule(nn.Module):
  """Feedforward block of conformer layer."""
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, input_paddings=None, train=False):
    padding_mask = jnp.expand_dims(1 - input_paddings, -1)
    config = self.config

    if config.layernorm_everywhere:
      inputs = LayerNorm(config.encoder_dim)(inputs)
    else:
      inputs = BatchNorm(config.encoder_dim,
                         config.dtype,
                         config.batch_norm_momentum,
                         config.batch_norm_epsilon)(inputs,
                                                    input_paddings,
                                                    train)
    inputs = nn.Dense(
        config.encoder_dim,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(
            inputs)
    if config.use_tanh:
      inputs = nn.tanh(inputs)
    else:
      inputs = nn.relu(inputs)
    inputs *= padding_mask

    if config.feed_forward_dropout_rate is None:
      feed_forward_dropout_rate = 0.1
    else:
      feed_forward_dropout_rate = config.feed_forward_dropout_rate
    inputs = nn.Dropout(rate=feed_forward_dropout_rate)(
        inputs, deterministic=not train)

    return inputs


class LayerNorm(nn.Module):
  """Module implementing layer normalization.

  This implementation is same as in this paper:
  https://arxiv.org/pdf/1607.06450.pdf.

  note: we multiply normalized inputs by (1 + scale) and initialize scale to
  zeros, this differs from default flax implementation of multiplying by scale
  and initializing to ones.
  """
  dim: int = 0
  epsilon: float = 1e-6

  def setup(self):
    self.scale = self.param('scale', nn.initializers.zeros, [self.dim])
    self.bias = self.param('bias', nn.initializers.zeros, [self.dim])

  @nn.compact
  def __call__(self, inputs):
    mean = jnp.mean(inputs, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(inputs - mean), axis=-1, keepdims=True)

    normed_inputs = (inputs - mean) * jax.lax.rsqrt(var + self.epsilon)
    normed_inputs *= (1 + self.scale)
    normed_inputs += self.bias

    return normed_inputs


class BatchNorm(nn.Module):
  """Implements batch norm respecting input paddings.

  This implementation takes into account input padding by masking inputs before
  computing mean and variance.

  This is inspired by lingvo jax implementation of BatchNorm:
  https://github.com/tensorflow/lingvo/blob/84b85514d7ad3652bc9720cb45acfab08604519b/lingvo/jax/layers/normalizations.py#L92

  and the corresponding defaults for momentum and epsilon have been copied over
  from lingvo.
  """
  encoder_dim: int = 0
  dtype: Any = jnp.float32
  batch_norm_momentum: float = 0.999
  batch_norm_epsilon: float = 0.001

  def setup(self):
    dim = self.encoder_dim
    dtype = self.dtype

    self.ra_mean = self.variable('batch_stats',
                                 'mean',
                                 lambda s: jnp.zeros(s, dtype),
                                 dim)
    self.ra_var = self.variable('batch_stats',
                                'var',
                                lambda s: jnp.ones(s, dtype),
                                dim)

    self.gamma = self.param('scale', nn.initializers.zeros, dim, dtype)
    self.beta = self.param('bias', nn.initializers.zeros, dim, dtype)

  def _get_default_paddings(self, inputs):
    """Gets the default paddings for an input."""
    in_shape = list(inputs.shape)
    in_shape[-1] = 1

    return jnp.zeros(in_shape, dtype=inputs.dtype)

  @nn.compact
  def __call__(self, inputs, input_paddings=None, train=False):
    rank = inputs.ndim
    reduce_over_dims = list(range(0, rank - 1))

    if input_paddings is None:
      padding = self._get_default_paddings(inputs)
    else:
      padding = jnp.expand_dims(input_paddings, -1)

    momentum = self.batch_norm_momentum
    epsilon = self.batch_norm_epsilon

    if train:
      mask = 1.0 - padding
      sum_v = jnp.sum(inputs * mask, axis=reduce_over_dims, keepdims=True)
      count_v = jnp.sum(
          jnp.ones_like(inputs) * mask, axis=reduce_over_dims, keepdims=True)

      sum_v = jax.lax.psum(sum_v, axis_name='batch')
      count_v = jax.lax.psum(count_v, axis_name='batch')

      count_v = jnp.maximum(count_v, 1.0)
      mean = sum_v / count_v
      variance = (inputs - mean) * (inputs - mean) * mask

      sum_vv = jnp.sum(variance, axis=reduce_over_dims, keepdims=True)

      sum_vv = jax.lax.psum(sum_vv, axis_name='batch')
      var = sum_vv / count_v

      self.ra_mean.value = momentum * self.ra_mean.value + (1 - momentum) * mean
      self.ra_var.value = momentum * self.ra_var.value + (1 - momentum) * var
    else:
      mean = self.ra_mean.value
      var = self.ra_var.value

    inv = (1 + self.gamma) / jnp.sqrt(var + epsilon)

    bn_output = (inputs - mean) * inv + self.beta
    bn_output *= 1.0 - padding

    return bn_output
    # return inputs


class CudnnLSTM(nn.Module):
  features: int
  num_layers: int = 1
  dropout_rate: float = 0.0
  bidirectional: bool = False

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      segmentation_mask: Optional[Array] = None,
      return_carry: Optional[bool] = None,
      deterministic: bool = False,
      initial_states: Optional[Tuple[Array, Array]] = None,
      use_cuda: bool = True,
  ) -> Union[Array, Tuple[Array, Carry]]:

    if jax.devices()[0].platform != 'gpu':
      use_cuda = False

    batch_size = inputs.shape[0]
    input_size = inputs.shape[2]
    num_directions = 2 if self.bidirectional else 1
    dropout = 0.0 if deterministic else self.dropout_rate

    weights = self.param(
        'weights',
        rnn.init_lstm_weight,
        input_size,
        self.features,
        self.num_layers,
        self.bidirectional,
    )

    if initial_states is None:
      h_0 = jnp.zeros(
          (num_directions * self.num_layers, batch_size, self.features),
          jnp.float32,
      )
      c_0 = jnp.zeros(
          (num_directions * self.num_layers, batch_size, self.features),
          jnp.float32,
      )
    else:
      h_0, c_0 = initial_states

    if segmentation_mask is not None:
      seq_lengths = jnp.sum(1 - segmentation_mask, axis=1, dtype=jnp.int32)
    else:
      seq_lengths = jnp.full((batch_size,), inputs.shape[1], dtype=jnp.int32)

    if use_cuda:
      y, h, c = rnn.lstm(
          x=inputs, h_0=h_0, c_0=c_0, weights=weights,
          seq_lengths=seq_lengths, input_size=input_size,
          hidden_size=self.features, num_layers=self.num_layers,
          dropout=dropout, bidirectional=self.bidirectional,
      )
    else:
      weight_ih, weight_hh, bias_ih, bias_hh = self.unpack_weights(
          weights, input_size)
      y, h, c = rnn.lstm_ref(
          x=inputs, h_0=h_0, c_0=c_0, W_ih=weight_ih, W_hh=weight_hh,
          b_ih=bias_ih, b_hh=bias_hh, seq_lengths=seq_lengths,
          input_size=input_size, hidden_size=self.features,
          num_layers=self.num_layers, dropout=dropout,
          bidirectional=self.bidirectional,
      )

    if return_carry:
      return y, (h, c)

    return y

  @nn.nowrap
  def unpack_weights(
      self, weights: Array, input_size: int
  ) -> Tuple[
      Dict[int, Array], Dict[int, Array], Dict[int, Array], Dict[int, Array]]:
    return jax.experimental.rnn.unpack_lstm_weights(
        weights,
        input_size,
        self.features,
        self.num_layers,
        self.bidirectional,
    )


class BatchRNN(nn.Module):
  """Implements a single deepspeech encoder layer.
  """
  config: DeepspeechConfig

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config

    if config.layernorm_everywhere:
      inputs = LayerNorm(config.encoder_dim)(inputs)
    else:
      inputs = BatchNorm(config.encoder_dim,
                         config.dtype,
                         config.batch_norm_momentum,
                         config.batch_norm_epsilon)(inputs,
                                                    input_paddings,
                                                    train)
    output = CudnnLSTM(
        features=config.encoder_dim // 2,
        bidirectional=config.bidirectional,
        num_layers=1)(inputs, input_paddings)

    return output


class Deepspeech(nn.Module):
  """Conformer (encoder + decoder) block.

  Takes audio input signals and outputs probability distribution over vocab size
  for each time step. The output is then fed into a CTC loss which eliminates
  the need for alignment with targets.
  """
  config: DeepspeechConfig

  def setup(self):
    config = self.config
    self.specaug = spectrum_augmenter.SpecAug(
        freq_mask_count=config.freq_mask_count,
        freq_mask_max_bins=config.freq_mask_max_bins,
        time_mask_count=config.time_mask_count,
        time_mask_max_frames=config.time_mask_max_frames,
        time_mask_max_ratio=config.time_mask_max_ratio,
        time_masks_per_frame=config.time_masks_per_frame,
        use_dynamic_time_mask_max_frames=config.use_dynamic_time_mask_max_frames
    )

  @nn.compact
  def __call__(self, inputs, input_paddings, train):
    config = self.config

    outputs = inputs
    output_paddings = input_paddings

    # Compute normalized log mel spectrograms from input audio signal.
    preprocessing_config = preprocessor.LibrispeechPreprocessingConfig()
    outputs, output_paddings = preprocessor.MelFilterbankFrontend(
        preprocessing_config,
        per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
        per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR)(outputs,
                                                            output_paddings)

    # Ablate random parts of input along temporal and frequency dimension
    # following the specaug procedure in https://arxiv.org/abs/1904.08779.
    if config.use_specaug and train:
      outputs, output_paddings = self.specaug(outputs, output_paddings)

    # Subsample input by a factor of 4 by performing strided convolutions.
    outputs, output_paddings = Subsample(
        config=config)(outputs, output_paddings, train)

    # Run the lstm layers.
    for _ in range(config.num_lstm_layers):
      if config.enable_residual_connections:
        outputs = outputs + BatchRNN(config)(outputs, output_paddings, train)
      else:
        outputs = BatchRNN(config)(outputs, output_paddings, train)

    for _ in range(config.num_ffn_layers):
      if config.enable_residual_connections:
        outputs = outputs + FeedForwardModule(config=self.config)(
            outputs, output_paddings, train)
      else:
        outputs = FeedForwardModule(config=self.config)(outputs,
                                                        output_paddings,
                                                        train)

    # Run the decoder which in this case is a trivial projection layer.
    if config.enable_decoder_layer_norm:
      outputs = LayerNorm(config.encoder_dim)(outputs)

    outputs = nn.Dense(
        config.vocab_size,
        use_bias=True,
        kernel_init=nn.initializers.xavier_uniform())(
            outputs)

    return outputs, output_paddings
