"""DeepSpeech Models modified from https://github.com/lsari/librispeech_100/blob/main/models.py."""

import functools
import math
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class Sequential(nn.Module):
  layers: Sequence[nn.Module]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class SequenceWise(nn.Module):
  """Collapses input of dim T*N*H to (T*N)*H, and applies to a module.

    Allows handling of variable sequence lengths and minibatch sizes.
    Args:
      module: Module to apply input to.
    """
  module: nn.Module

  def __call__(self, x, training=False):
    t, n = x.shape[0], x.shape[1]
    x = jnp.reshape(x, (t * n, -1))
    x = self.module(x)
    x = jnp.reshape(x, (t, n, -1))
    return x

class MaskConv(nn.Module):
  """Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch
    sizes change during inference. Input needs to be in the shape of (BxCxDxT)
    Args:
      seq_module: The sequential module containing the conv stack.
    """
  seq_module: Sequence[nn.Module]

  def __call__(self, x, lengths, training=False):
    """Forward pass.
    Args:
      x: The input (before transposing it to channels-last) is of size BxCxDxT
      lengths: The actual length of each sequence in the batch
    Returns:
      Masked output from the module
    """
    x = x.transpose(0, 2, 3, 1)
    for module in self.seq_module:
      if isinstance(module, nn.BatchNorm):
        x = module(x, use_running_average=training)
      else:
        x = module(x)
      mask = jnp.arange(x.shape[1]).reshape(1, -1, 1, 1) >= lengths.reshape(-1, 1, 1, 1)
      mask = mask.astype(jnp.float32)
      x *= mask
    x = x.transpose(0, 3, 1, 2)
    return x, lengths
  
  
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
    batch_size = embedded_inputs.shape[0]

    # Forward LSTM.
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, forward_outputs = self.forward_lstm(initial_state, embedded_inputs)

    # Backward LSTM.
    reversed_inputs = flip_sequences(embedded_inputs, lengths)
    initial_state = SimpleLSTM.initialize_carry((batch_size,), self.hidden_size)
    _, backward_outputs = self.backward_lstm(initial_state, reversed_inputs)
    backward_outputs = flip_sequences(backward_outputs, lengths)

    # Concatenate the forward and backward representations.
    outputs = jnp.concatenate([forward_outputs, backward_outputs], -1)
    return outputs


class BatchRNN(nn.Module):

  input_size: int
  hidden_size: int
  use_batch_norm: bool = True
  train: bool = False

  def setup(self):
    if self.use_batch_norm:
      self.batch_norm = SequenceWise(nn.BatchNorm(use_running_average=True))
    else:
      self.batch_norm = None
    self.rnn = SimpleBiLSTM(self.hidden_size)

  def __call__(self, inputs, lengths, training=False):
    if self.batch_norm is not None:
      inputs = self.batch_norm(inputs)
    inputs = inputs.transpose(1, 0, 2)
    inputs = self.rnn(inputs, lengths)
    inputs = inputs.transpose(1, 0, 2)
    # (TxNxH*2) -> (TxNxH) by sum
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1).sum(2)
    return inputs


def hard_tanh(x, min_value=-1., max_value=1.):

  return jnp.where(x > max_value, max_value, jnp.where(x < min_value, min_value, x))


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
      nn.BatchNorm(use_running_average=True),
      functools.partial(hard_tanh, min_value=0, max_value=20),
      nn.Conv(
        features=32,
        kernel_size=(21, 11),
        strides=(2, 1),
        padding=((10, 10), (5, 5)),
      ),
      nn.BatchNorm(use_running_average=True),
      functools.partial(hard_tanh, min_value=0, max_value=20),
    ]
    self.conv = MaskConv(sequential)

    rnn_input_size = 161
    rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
    rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
    rnn_input_size *= 32

    rnns = []
    rnn = BatchRNN(
        input_size=rnn_input_size,
        hidden_size=self.hidden_size,
        use_batch_norm=False,
    )
    rnns.append(rnn)
    for _ in range(self.hidden_layers - 1):
      rnn = BatchRNN(
          input_size=self.hidden_size,
          hidden_size=self.hidden_size
          )
      rnns.append(rnn)

    self.rnns = rnns

    fully_connected = Sequential([
      nn.BatchNorm(use_running_average=True),
      nn.Dense(self.num_classes, use_bias=False)
    ])

    self.fc = SequenceWise(fully_connected)

  def get_seq_lens(self, input_length):
    """Get a 1D tensor or variable containing the size sequences that will be output by the network.

    Args:
      input_length: 1D Tensor

    Returns:
      1D Tensor scaled by model
    """
    seq_len = input_length
    for m in self.conv.seq_module:
      if isinstance(m, nn.Conv):
        seq_len = (seq_len + 2 * m.padding[1][1] - m.kernel_dilation * (m.kernel_size[1] - 1) - 1) // m.strides[1] + 1
    return seq_len
  
  def __call__(self, inputs, lengths, training=False):
    output_lengths = self.get_seq_lens(lengths)

    x, _ = self.conv(inputs, lengths)

    sizes = x.shape
    x = x.reshape(sizes[0], sizes[1] * sizes[2],
               sizes[3]) 
    x = x.transpose(0, 2, 1).transpose(1, 0, 2)

    for rnn in self.rnns:
      x = rnn(x, output_lengths, training=training)
    
    x = self.fc(x, training=training)
    log_probs = jax.nn.log_softmax(x, axis=-1).transpose(1, 0, 2)

    return log_probs, output_lengths
