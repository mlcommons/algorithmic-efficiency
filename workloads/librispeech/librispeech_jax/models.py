"""DeepSpeech Models modified from https://github.com/lsari/librispeech_100/blob/main/models.py."""

import functools
import math
from typing import Sequence, Optional, Iterable, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.normalization import _canonicalize_axes, _normalize
from jax import lax

Axes = Union[int, Iterable[int]]


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BatchNorm(nn.BatchNorm):
    momentum: float = 0.9

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.

        NOTE:
        During initialization (when parameters are mutable) the running average
        of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats
            will be used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = nn.merge_param(
            'use_running_average', self.use_running_average, use_running_average)
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection('params')

        ra_mean = self.variable('batch_stats', 'mean',
                                lambda s: jnp.zeros(s, jnp.float32),
                                feature_shape)
        ra_var = self.variable('batch_stats', 'var',
                               lambda s: jnp.ones(s, jnp.float32),
                               feature_shape)

        if use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            x: jnp.ndarray = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
            non_zeros = jnp.sum(mask, reduction_axes)
            mean = jnp.sum(x, reduction_axes) / non_zeros
            mean2 = jnp.sum(lax.square(x), reduction_axes) / non_zeros
            # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
            # to floating point round-off errors.
            var = jnp.maximum(0., mean2 - lax.square(mean))

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        out = _normalize(
            self, x, mean, var, reduction_axes, feature_axes,
            self.dtype, self.param_dtype, self.epsilon,
            self.use_bias, self.use_scale,
            self.bias_init, self.scale_init)
        return out * mask


class SequenceWise(nn.Module):
    """Collapses input of dim T*N*H to (T*N)*H, and applies to a module.

      Allows handling of variable sequence lengths and minibatch sizes.
      Args:
        module: Module to apply input to.
      """
    module: nn.Module

    def __call__(self, x, mask, training=False):
        # [Seq, Batch, Feature]
        t, n = x.shape[0], x.shape[1]
        x = jnp.reshape(x, (t * n, -1))  # [Seq, Batch, Feature] -> [Seq * Batch, Feature]
        x = self.module(x, mask, not training)
        x = jnp.reshape(x, (t, n, -1))  # [Seq, Batch, Feature] -> [Seq, Batch, Feature]
        return x


class MaskConv(nn.Module):
    """Adds padding to the output of the module based on the given lengths.
      This is to ensure that the results of the model do not change when batch
      sizes change during inference. Input needs to be in the shape of (BxCxDxT)
      Args:
        seq_module: The sequential module containing the conv stack.
      """
    seq_module: Sequence[nn.Module]

    @nn.compact
    def __call__(self, x, mask, training):
        """Forward pass.
        Args:
          x: The input (before transposing it to channels-last) is of Shape[Batch, Channels, "D", TimeSteps]
          lengths: The actual length of each sequence in the batch
        Returns:
          Masked output from the module
        """

        x = x.transpose(0, 2, 3, 1)
        for module in self.seq_module:
            x = module(x)
            x = BatchNorm()(x, mask, training)
            x = hard_tanh(x, 0, 20)
        x = x.transpose(0, 3, 1, 2)
        return x


@jax.vmap
def flip_sequences(inputs, lengths):
    """Flips a sequence of inputs along the time dimension.
    This function can be used to prepare inputs for the reverse direction of a
    bidirectional LSTM. It solves the issue that, when naively flipping multiple
    padded sequences stored in a matrix, the first elements would be padding
    values for those sequences that were padded. This function keeps the padding
    at the end, while flipping the rest of the elements.
    Example:
    ```python
    inputs = [[1, 0, 0],
              [2, 3, 0]
              [4, 5, 6]]
    lengths = [1, 2, 3]
    flip_sequences(inputs, lengths) = [[1, 0, 0],
                                       [3, 2, 0],
                                       [6, 5, 4]]
    ```
    Args:
      inputs: An array of input IDs <int>[batch_size, seq_length].
      lengths: The length of each sequence <int>[batch_size].
    Returns:
      An ndarray with the flipped inputs.
    """
    # Note: since this function is vmapped, the code below is effectively for
    # a single example.
    max_length = inputs.shape[0]
    return jnp.flip(jnp.roll(inputs, max_length - lengths, axis=0), axis=0)


class SimpleLSTM(nn.Module):
    """A simple unidirectional LSTM."""

    @functools.partial(
        nn.transforms.scan,
        variable_broadcast='params',
        in_axes=1, out_axes=1,
        split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        return nn.OptimizedLSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(batch_dims, hidden_size):
        # Use fixed random key since default state init fn is just zeros.
        return nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(0), batch_dims, hidden_size)


class SimpleBiLSTM(nn.Module):
    """A simple bi-directional LSTM."""
    hidden_size: int

    def setup(self):
        self.forward_lstm = SimpleLSTM()
        self.backward_lstm = SimpleLSTM()

    def __call__(self, embedded_inputs, lengths):
        # [Batch, Seq, Feature]
        batch_size = embedded_inputs.shape[0]

        # Forward LSTM.
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, forward_outputs = self.forward_lstm(initial_state, embedded_inputs)

        # Backward LSTM.
        reversed_inputs = flip_sequences(embedded_inputs, lengths)
        initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
        _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
        backward_outputs = flip_sequences(backward_outputs, lengths)

        # Add the forward and backward representations.
        return forward_outputs + backward_outputs


class BatchRNN(nn.Module):
    input_size: int
    hidden_size: int
    use_batch_norm: bool = True
    train: bool = False

    def setup(self):
        if self.use_batch_norm:
            self.batch_norm = SequenceWise(BatchNorm())
        else:
            self.batch_norm = None
        self.rnn = SimpleBiLSTM(self.hidden_size)

    def __call__(self, inputs, lengths, mask, training=False):
        # [Seq, Batch, Feature]
        if self.batch_norm is not None:
            inputs = self.batch_norm(inputs, mask)
        inputs = inputs.transpose(1, 0, 2)  # [Seq, Batch, Feature] -> [Batch, Seq, Feature]
        inputs = self.rnn(inputs, lengths)
        inputs = inputs.transpose(1, 0, 2)  # [Batch, Seq, Feature] -> [Seq, Batch, Feature]
        return inputs


def hard_tanh(x, min_value=-1., max_value=1.):
    return jnp.clip(x, min_value, max_value)


def get_seq_lens(input_length, conv_seq_module):
    """Get a 1D tensor or variable containing the size sequences that will be output by the network.

    Args:
      input_length: 1D Tensor

    Returns:
      1D Tensor scaled by model
    """
    seq_len = input_length
    for m in conv_seq_module:
        if isinstance(m, nn.Conv):
            seq_len = (seq_len + 2 * m.padding[1][1] - m.kernel_dilation * (m.kernel_size[1] - 1) - 1) // m.strides[
                1] + 1
    return seq_len


class CNNLSTM(nn.Module):
    num_classes = 29
    hidden_size = 768
    hidden_layers = 5
    context = 20

    def setup(self):
        sequential = [
            nn.Conv(
                features=32,
                kernel_size=(41, 11),
                strides=(2, 2),
                padding=((20, 20), (5, 5)),
            ),
            nn.Conv(
                features=32,
                kernel_size=(21, 11),
                strides=(2, 1),
                padding=((10, 10), (5, 5)),
            ),
        ]
        self.conv = MaskConv(sequential)

        rnn_input_size = 161
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = [BatchRNN(input_size=rnn_input_size, hidden_size=self.hidden_size,
                         use_batch_norm=False)]
        rnns.extend([BatchRNN(input_size=self.hidden_size, hidden_size=self.hidden_size)
                     for _ in range(self.hidden_layers - 1)])
        self.rnns = rnns
        self.out_norm = SequenceWise(BatchNorm())
        self.fc = nn.Dense(self.num_classes, use_bias=False)

    def __call__(self, inputs, lengths, training=False):
        output_lengths = get_seq_lens(lengths, self.conv.seq_module)

        mask: jnp.ndarray = jnp.arange(inputs.shape[1]).reshape(1, -1, 1, 1) >= output_lengths.reshape(-1, 1, 1, 1)
        mask = mask.astype(jnp.float32)

        x = self.conv(inputs, mask, training)

        sizes = x.shape
        x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(2, 0, 1)  # [Batch, Feature, Seq] -> [Seq, Batch, Feature]

        for rnn in self.rnns:
            x = rnn(x, output_lengths, mask, training=training)
        x = self.out_norm(x, mask, training=training)
        x = self.fc(x)
        log_probs = jax.nn.log_softmax(x, axis=-1)
        log_probs = log_probs.transpose(1, 0, 2)  # [Seq, Batch, Feature] -> [Batch, Seq, Feature]

        return log_probs, output_lengths
