import functools
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
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
  # Compute the indices to put the inputs in flipped order as per above example.
  max_length = inputs.shape[0]
  idxs = (jnp.arange(max_length - 1, -1, -1) + lengths) % max_length
  return inputs[idxs]


class GenericRNNSequenceEncoder(nn.Module):
  """Encodes a single sequence using any RNN cell, for example `nn.LSTMCell`.

  The sequence can be encoded left-to-right (default) or right-to-left (by
  calling the module with reverse=True). Regardless of encoding direction,
  outputs[i, j, ...] is the representation of inputs[i, j, ...].

  Attributes:
    hidden_size: The hidden size of the RNN cell.
    cell_type: The RNN cell module to use, for example, `nn.LSTMCell`.
    cell_kwargs: Optional keyword arguments for the recurrent cell.
    recurrent_dropout_rate: The dropout to apply across time steps. If this is
      greater than zero, you must use an RNN cell that implements
      `RecurrentDropoutCell` such as RecurrentDropoutOptimizedLSTMCell.
  """
  hidden_size: int
  cell_type: Type[nn.RNNCellBase]
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()
  recurrent_dropout_rate: float = 0.0

  def setup(self):
    self.cell = self.cell_type(features=self.hidden_size, **self.cell_kwargs)

  @functools.partial(  # Repeatedly calls the below method to encode the inputs.
      nn.transforms.scan,
      variable_broadcast='params',
      in_axes=(1, flax.core.axes_scan.broadcast, flax.core.axes_scan.broadcast),
      out_axes=1,
      split_rngs={'params': False})
  def unroll_cell(self, cell_state: StateType, inputs: Array,
                  recurrent_dropout_mask: Optional[Array], deterministic: bool):
    """Unrolls a recurrent cell over an input sequence.

    Args:
      cell_state: The initial cell state, shape: <float32>[batch_size,
        hidden_size] (or an n-tuple thereof).
      inputs: The input sequence. <float32>[batch_size, seq_len, input_dim].
      recurrent_dropout_mask: An optional recurrent dropout mask to apply in
        between time steps. <float32>[batch_size, hidden_size].
      deterministic: Disables recurrent dropout when set to True.

    Returns:
      The cell state after processing the complete sequence (including padding),
      and a tuple with all intermediate cell states and cell outputs.
    """
    # We do not directly scan the cell itself, since it only returns the output.
    # This returns both the state and the output, so we can slice out the
    # correct final states later.
    new_cell_state, output = self.cell(cell_state, inputs)
    return new_cell_state, (new_cell_state, output)

  def __call__(self,
               inputs: Array,
               lengths: Array,
               initial_state: StateType,
               reverse: bool = False,
               deterministic: bool = False):
    """Unrolls the RNN cell over the inputs.

    Arguments:
      inputs: A batch of sequences. Shape: <float32>[batch_size, seq_len,
        input_dim].
      lengths: The lengths of the input sequences.
      initial_state: The initial state for the RNN cell. Shape: [batch_size,
        hidden_size].
      reverse: Process the inputs in reverse order, and reverse the outputs.
        This means that the outputs still correspond to the order of the inputs,
        but their contexts come from the right, and not from the left.
      deterministic: Disables recurrent dropout if set to True.

    Returns:
      The encoded sequence of inputs, shaped <float32>[batch_size, seq_len,
        hidden_size], as well as the final hidden states of the RNN cell. For an
        LSTM cell the final states are a tuple (c, h), each shaped <float32>[
          batch_size, hidden_size].
    """
    if reverse:
      inputs = flip_sequences(inputs, lengths)

    recurrent_dropout_mask = None
    _, (cell_states, outputs) = self.unroll_cell(initial_state, inputs,
                                                 recurrent_dropout_mask,
                                                 deterministic)
    final_state = jax.tree.map(
        lambda x: x[jnp.arange(inputs.shape[0]), lengths - 1], cell_states)

    if reverse:
      outputs = flip_sequences(outputs, lengths)

    return outputs, final_state


class GenericRNN(nn.Module):
  """Generic RNN class.

  This provides generic RNN functionality to encode sequences with any RNN cell.
  The class provides unidirectional and bidirectional layers, and these are
  stacked when asking for multiple layers.

  This class be used to create a specific RNN class such as LSTM or GRU.

  Attributes:
    cell_type: An RNN cell class to use, e.g., `flax.linen.LSTMCell`.
    hidden_size: The size of each recurrent cell.
    num_layers: The number of stacked recurrent layers. The output of the first
      layer, with optional dropout applied, feeds into the next layer.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  cell_type: Type[nn.RNNCellBase]
  hidden_size: int
  num_layers: int = 1
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      lengths: Array,
      initial_states: Optional[Sequence[StateType]] = None,
      deterministic: bool = False
  ) -> Tuple[Array, Sequence[StateType]]:
    """Processes the input sequence using the recurrent cell.

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`).
        These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
      deterministic: Disables dropout between layers when set to True.
    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states for each cell and direction. Directions are alternated (first
      forward, then backward, if bidirectional). For example for a bidirectional
      cell this would be: layer 1 forward, layer 1 backward, layer 2 forward,
      layer 2 backward, etc..
      For some cells like LSTMCell a state consists of an (c, h) tuple, while
      for others cells it only contains a single vector (h,).
    """
    batch_size = inputs.shape[0]
    final_states = []
    num_directions = 2 if self.bidirectional else 1
    num_cells = self.num_layers * num_directions

    # Construct initial states.
    if initial_states is None:  # Initialize with zeros.
      rng = jax.random.PRNGKey(0)
      initial_states = [
          self.cell_type(self.hidden_size).initialize_carry(
              rng, (batch_size, 1)
          )
          for _ in range(num_cells)
      ]
    if len(initial_states) != num_cells:
      raise ValueError(
          f'Please provide {self.num_cells} (`num_layers`, *2 if bidirectional)'
          'initial states.'
      )

    # For each layer, apply the forward and optionally the backward RNN cell.
    cell_idx = 0
    for _ in range(self.num_layers):
      # Unroll an RNN cell (forward direction) for this layer.
      outputs, final_state = GenericRNNSequenceEncoder(
          cell_type=self.cell_type,
          cell_kwargs=self.cell_kwargs,
          hidden_size=self.hidden_size,
          recurrent_dropout_rate=self.recurrent_dropout_rate,
          name=f'{self.name}SequenceEncoder_{cell_idx}')(
              inputs,
              lengths,
              initial_state=initial_states[cell_idx],
              deterministic=deterministic)
      final_states.append(final_state)
      cell_idx += 1

      # Unroll an RNN cell (backward direction) for this layer.
      if self.bidirectional:
        backward_outputs, backward_final_state = GenericRNNSequenceEncoder(
            cell_type=self.cell_type,
            cell_kwargs=self.cell_kwargs,
            hidden_size=self.hidden_size,
            recurrent_dropout_rate=self.recurrent_dropout_rate,
            name=f'{self.name}SequenceEncoder_{cell_idx}')(
                inputs,
                lengths,
                initial_state=initial_states[cell_idx],
                reverse=True,
                deterministic=deterministic)
        outputs = jnp.concatenate([outputs, backward_outputs], axis=-1)
        final_states.append(backward_final_state)
        cell_idx += 1

      inputs = outputs

    return outputs, final_states


class LSTM(nn.Module):
  """LSTM.

  Attributes:
    hidden_size: The size of each recurrent cell.
    num_layers: The number of stacked recurrent layers. The output of the first
      layer, with optional dropout applied, feeds into the next layer.
    dropout_rate: Dropout rate to be applied between LSTM layers. Only applies
      when num_layers > 1.
    recurrent_dropout_rate: Dropout rate to be applied on the hidden state at
      each time step repeating the same dropout mask.
    bidirectional: Process the sequence left-to-right and right-to-left and
      concatenate the outputs from the two directions.
    cell_type: The LSTM cell class to use. Default:
      `flax.linen.OptimizedLSTMCell`. If you use hidden_size of >2048, consider
      using `flax.linen.LSTMCell` instead, since the optimized LSTM cell works
      best for hidden sizes up to 2048.
    cell_kwargs: Optional keyword arguments to instantiate the cell with.
  """
  hidden_size: int
  num_layers: int = 1
  dropout_rate: float = 0.
  recurrent_dropout_rate: float = 0.
  bidirectional: bool = False
  cell_type: Any = nn.OptimizedLSTMCell
  cell_kwargs: Mapping[str, Any] = flax.core.FrozenDict()

  @nn.compact
  def __call__(
      self,
      inputs: Array,
      lengths: Array,
      initial_states: Optional[Sequence[StateType]] = None,
      deterministic: bool = False) -> Tuple[Array, Sequence[StateType]]:
    """Processes an input sequence with an LSTM cell.

    Example usage:
    ```
      inputs = np.random.normal(size=(2, 3, 4))
      lengths = np.array([1, 3])
      outputs, final_states = LSTM(hidden_size=10).apply(rngs, inputs, lengths)
    ```

    Args:
      inputs: The input sequence <float32>[batch_size, sequence_length, ...]
      lengths: The lengths of each sequence in the batch. <int64>[batch_size]
      initial_states: The initial states for the cells. You must provide
        `num_layers` initial states (when using bidirectional, `num_layers *
        2`). These must be ordered in the following way: (layer_0_forward,
          layer_0_backward, layer_1_forward, layer_1_backward, ...). If None,
          all initial states will be initialized with zeros.
      deterministic: Disables dropout between layers when set to True.

    Returns:
      The sequence of all outputs for the final layer, and a list of final
      states (h, c) for each cell and direction, ordered first by layer number
      and then by direction (first forward, then backward, if bidirectional).
    """
    return GenericRNN(
        cell_type=self.cell_type,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout_rate=self.dropout_rate,
        recurrent_dropout_rate=self.recurrent_dropout_rate,
        bidirectional=self.bidirectional,
        cell_kwargs=self.cell_kwargs,
        name='LSTM')(
            inputs,
            lengths,
            initial_states=initial_states,
            deterministic=deterministic)
