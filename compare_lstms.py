#!/usr/bin/env python3

"""
Comparison between CudnnLSTM (GPU-optimized) and Classic LSTM implementations.

This script isolates both LSTM implementations and provides a comprehensive
comparison in terms of performance, API usage, and output consistency.
"""

import functools
import time
from typing import Any, Mapping, Optional, Sequence, Tuple, Type, Union

import flax
from flax import linen as nn
import jax
from jax.experimental import rnn
import jax.numpy as jnp
import numpy as np
import optax

Array = jnp.ndarray
StateType = Union[Array, Tuple[Array, ...]]
PRNGKey = Any

from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
from jax.experimental.shard_map import shard_map


# =============================================================================
# CudnnLSTM Implementation (from Deepspeech)
# =============================================================================

class CudnnLSTM(nn.Module):
    """GPU-optimized LSTM using JAX's experimental RNN functions."""

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
    ) -> Union[Array, Tuple[Array, Tuple[Array, Array]]]:

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
        dict[int, Array], dict[int, Array], dict[int, Array], dict[int, Array]]:
        return jax.experimental.rnn.unpack_lstm_weights(
            weights,
            input_size,
            self.features,
            self.num_layers,
            self.bidirectional,
        )


# =============================================================================
# Classic LSTM Implementation (from Flax-based)
# =============================================================================
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
      def make_initial_state():
        state = self.cell_type(self.hidden_size).initialize_carry(
            rng, (batch_size, 1)
        )
        # Apply pvary to mark states as varying over batch axis
        if isinstance(state, tuple):
          # For LSTM: (c, h) tuple
          return tuple(jax.lax.pvary(s, ('batch',)) for s in state)
        else:
          # For other RNN types: single state
          return jax.lax.pvary(state, ('batch',))
      
      initial_states = [make_initial_state() for _ in range(num_cells)]
    else:
      # If initial_states are provided, ensure they have correct VMA annotation
      def apply_pvary_to_state(state):
        if isinstance(state, tuple):
          # For LSTM: (c, h) tuple
          return tuple(jax.lax.pvary(s, ('batch',)) for s in state)
        else:
          # For other RNN types: single state
          return jax.lax.pvary(state, ('batch',))
      
      initial_states = [apply_pvary_to_state(state) for state in initial_states]
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

class ClassicLSTM(nn.Module):
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

# =============================================================================
# Gradient Computation Functions
# =============================================================================

def compute_gradients_cudnn(lstm_module, params, inputs, target_output):
    """Compute gradients for CudnnLSTM."""
    def loss_fn(params):
        output = lstm_module.apply(params, inputs, deterministic=True)
        return jnp.mean((output - target_output) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    return grad_fn(params)


def compute_gradients_classic(lstm_module, params, inputs, lengths, target_output):
    """Compute gradients for Classic LSTM."""
    def loss_fn(params):
        output, _ = lstm_module.apply(params, inputs, lengths, deterministic=True)
        return jnp.mean((output - target_output) ** 2)
    
    grad_fn = jax.grad(loss_fn)
    return grad_fn(params)


def compute_gradient_norm(gradients):
    """Compute the L2 norm of gradients."""
    flat_grads = jax.tree.leaves(gradients)
    squared_norms = [jnp.sum(grad ** 2) for grad in flat_grads]
    return jnp.sqrt(jnp.sum(jnp.array(squared_norms)))


def compute_gradient_norms_by_layer(gradients):
    """Compute gradient norms for each layer/parameter group."""
    norms = {}
    for key, value in gradients.items():
        if isinstance(value, dict):
            norms[key] = compute_gradient_norms_by_layer(value)
        else:
            norms[key] = jnp.sqrt(jnp.sum(value ** 2))
    return norms


# =============================================================================
# Training Functions
# =============================================================================

def create_training_data(batch_size: int, seq_len: int, input_dim: int, output_dim: int, rng_key):
    """Create training data with inputs and targets."""
    rng1, rng2 = jax.random.split(rng_key)
    
    # Generate random input sequences
    inputs = jax.random.normal(rng1, (batch_size, seq_len, input_dim))
    
    # Generate random target sequences (for sequence-to-sequence learning)
    targets = jax.random.normal(rng2, (batch_size, seq_len, output_dim))
    
    # Generate lengths (all sequences same length for simplicity)
    lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)
    
    return inputs, targets, lengths


def train_step_cudnn(lstm_module, params, opt_state, optimizer, inputs, targets):
    """Single training step for CudnnLSTM."""
    def loss_fn(params):
        output = lstm_module.apply(params, inputs, deterministic=False)
        return jnp.mean((output - targets) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    grad_norm = compute_gradient_norm(grads)
    return params, opt_state, loss, grad_norm, grads


def train_step_classic(lstm_module, params, opt_state, optimizer, inputs, targets, lengths):
    """Single training step for Classic LSTM."""
    def loss_fn(params):
        output, _ = lstm_module.apply(params, inputs, lengths, deterministic=False)
        return jnp.mean((output - targets) ** 2)
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    grad_norm = compute_gradient_norm(grads)
    return params, opt_state, loss, grad_norm, grads


def run_training_comparison(cudnn_lstm, classic_lstm, cudnn_params, classic_params, 
                          inputs, targets, lengths, num_steps: int = 10, 
                          learning_rate: float = 0.001):
    """Run training for N steps and compare gradient norms."""
    
    # Create mesh for data parallelism
    devices = jax.devices()
    num_devices = len(devices)
    mesh = jax.make_mesh((num_devices,), ('batch',))
    batch_sharding = P('batch')
    replicated_sharding = P()
    
    # Initialize optimizers
    optimizer = optax.adam(learning_rate)
    cudnn_opt_state = optimizer.init(cudnn_params)
    classic_opt_state = optimizer.init(classic_params)
    
    # Shard data across devices, replicate params
    inputs_sharded = jax.device_put(inputs, NamedSharding(mesh, batch_sharding))
    targets_sharded = jax.device_put(targets, NamedSharding(mesh, batch_sharding))
    lengths_sharded = jax.device_put(lengths, NamedSharding(mesh, batch_sharding))
    
    cudnn_params = jax.device_put(cudnn_params, NamedSharding(mesh, replicated_sharding))
    classic_params = jax.device_put(classic_params, NamedSharding(mesh, replicated_sharding))
    # Don't shard optimizer states
    
    # Define model functions for shard_map
    def cudnn_model_fn(params, inputs_local):
        return cudnn_lstm.apply(params, inputs_local, deterministic=False)
    
    def classic_model_fn(params, inputs_local, lengths_local):
        output, _ = classic_lstm.apply(params, inputs_local, lengths_local, deterministic=False)
        return output
    
    # Wrap model functions with shard_map
    cudnn_model_sharded = shard_map(
        cudnn_model_fn,
        mesh=mesh,
        in_specs=(replicated_sharding, batch_sharding),
        out_specs=batch_sharding
    )
    
    classic_model_sharded = shard_map(
        classic_model_fn,
        mesh=mesh,
        in_specs=(replicated_sharding, batch_sharding, batch_sharding),
        out_specs=batch_sharding
    )
    
    # Define training step functions (no shard_map here)
    def cudnn_train_step(params, opt_state, inputs_sharded, targets_sharded):
        def loss_fn(params):
            output = cudnn_model_sharded(params, inputs_sharded)
            return jnp.mean((output - targets_sharded) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        grad_norm = compute_gradient_norm(grads)
        return params, opt_state, loss, grad_norm, grads
    
    def classic_train_step(params, opt_state, inputs_sharded, targets_sharded, lengths_sharded):
        def loss_fn(params):
            output = classic_model_sharded(params, inputs_sharded, lengths_sharded)
            return jnp.mean((output - targets_sharded) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        grad_norm = compute_gradient_norm(grads)
        return params, opt_state, loss, grad_norm, grads
    
    # Storage for metrics
    cudnn_losses = []
    classic_losses = []
    cudnn_grad_norms = []
    classic_grad_norms = []
    cudnn_step_times = []
    classic_step_times = []
    
    print(f"Training for {num_steps} steps with learning rate {learning_rate}...")
    print(f"{'Step':<6} {'CudnnLSTM Loss':<15} {'Classic Loss':<15} {'CudnnLSTM Grad':<15} {'Classic Grad':<15} {'Ratio':<10} {'CudnnLSTM Time':<15} {'Classic Time':<15}")
    print("-" * 125)
    
    for step in range(num_steps):
        # CudnnLSTM training step
        cudnn_start_time = time.time()
        cudnn_params, cudnn_opt_state, cudnn_loss, cudnn_grad_norm, cudnn_grads = cudnn_train_step(
            cudnn_params, cudnn_opt_state, inputs_sharded, targets_sharded)
        # Block until computation is complete
        cudnn_loss.block_until_ready()
        cudnn_step_time = time.time() - cudnn_start_time
        
        # Classic LSTM training step  
        classic_start_time = time.time()
        classic_params, classic_opt_state, classic_loss, classic_grad_norm, classic_grads = classic_train_step(
            classic_params, classic_opt_state, inputs_sharded, targets_sharded, lengths_sharded)
        # Block until computation is complete
        classic_loss.block_until_ready()
        classic_step_time = time.time() - classic_start_time
        
        # Store metrics
        cudnn_losses.append(float(cudnn_loss))
        classic_losses.append(float(classic_loss))
        cudnn_grad_norms.append(float(cudnn_grad_norm))
        classic_grad_norms.append(float(classic_grad_norm))
        cudnn_step_times.append(cudnn_step_time)
        classic_step_times.append(classic_step_time)
        
        # Print progress
        ratio = float(cudnn_grad_norm) / float(classic_grad_norm)
        print(f"{step:<6} {cudnn_loss:<15.6f} {classic_loss:<15.6f} {cudnn_grad_norm:<15.6f} {classic_grad_norm:<15.6f} {ratio:<10.2f} {cudnn_step_time*1000:<15.2f} {classic_step_time*1000:<15.2f}")
    
    return {
        'cudnn_losses': cudnn_losses,
        'classic_losses': classic_losses,
        'cudnn_grad_norms': cudnn_grad_norms,
        'classic_grad_norms': classic_grad_norms,
        'final_cudnn_params': cudnn_params,
        'final_classic_params': classic_params
    }


# =============================================================================
# Comparison and Testing
# =============================================================================

def create_test_data(batch_size: int = 4, seq_len: int = 20, input_dim: int = 16):
    """Create test data for LSTM comparison."""
    rng = jax.random.PRNGKey(42)

    # Generate random input sequences
    inputs = jax.random.normal(rng, (batch_size, seq_len, input_dim))

    # Generate random sequence lengths (but keep them all the same for simplicity)
    lengths = jnp.full((batch_size,), seq_len, dtype=jnp.int32)

    return inputs, lengths


def benchmark_lstm(lstm_module, module_name, params, inputs, lengths, num_runs: int = 10):
    """Benchmark LSTM performance."""

    # Compile the function
    @jax.jit
    def run_lstm():
        if module_name == "classic":
            return lstm_module.apply(params, inputs, lengths, deterministic=True)
        else:
            return lstm_module.apply(params, inputs, deterministic=True)

    # Warmup
    _ = run_lstm()

    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        outputs = run_lstm()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time, outputs


def compare_outputs(output1, output2, tolerance: float = 1e-4):
    """Compare outputs from two LSTM implementations."""
    if isinstance(output1, tuple) and isinstance(output2, tuple):
        # Compare sequences
        seq_diff = jnp.max(jnp.abs(output1[0] - output2[0]))
        print(f"  Max sequence difference: {seq_diff:.2e}")

        # Compare final states (if available)
        if len(output1) > 1 and len(output2) > 1:
            if isinstance(output1[1], (list, tuple)) and isinstance(output2[1], (list, tuple)):
                for i, (state1, state2) in enumerate(zip(output1[1], output2[1])):
                    if isinstance(state1, tuple) and isinstance(state2, tuple):
                        # LSTM states (h, c)
                        h_diff = jnp.max(jnp.abs(state1[0] - state2[0]))
                        c_diff = jnp.max(jnp.abs(state1[1] - state2[1]))
                        print(f"  Layer {i} - h difference: {h_diff:.2e}, c difference: {c_diff:.2e}")

        return seq_diff < tolerance
    else:
        # Just sequences
        diff = jnp.max(jnp.abs(output1 - output2))
        print(f"  Max difference: {diff:.2e}")
        return diff < tolerance


def main():
    """Main comparison function."""
    print("=" * 80)
    print("LSTM IMPLEMENTATIONS COMPARISON")
    print("=" * 80)

    # Configuration
    batch_size = 32
    seq_len = 100
    input_dim = 64
    hidden_size = 128
    num_layers = 2
    bidirectional = False

    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Number of layers: {num_layers}")
    print(f"  Bidirectional: {bidirectional}")

    # Create test data
    inputs, lengths = create_test_data(batch_size, seq_len, input_dim)
    print(f"\nInput shape: {inputs.shape}")
    print(f"\nLenghts shape: {lengths.shape}")
    print(f"Device: {jax.devices()[0]}")

    # Initialize models
    rng = jax.random.PRNGKey(123)

    # CudnnLSTM
    cudnn_lstm = CudnnLSTM(
        features=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    # Classic LSTM
    classic_lstm = ClassicLSTM(
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    print(f"\n" + "="*50)
    print("INITIALIZATION")
    print("="*50)

    classic_params = classic_lstm.init(rng, inputs, lengths)
    print("✓ Classic LSTM initialized successfully")

    # Count parameters
    classic_param_count = sum(x.size for x in jax.tree.leaves(classic_params))
    print(f"  Classic LSTM parameters: {classic_param_count:,}")

    cudnn_params = cudnn_lstm.init(rng, inputs)
    print("✓ CudnnLSTM initialized successfully")

    # Count parameters
    cudnn_param_count = sum(x.size for x in jax.tree.leaves(cudnn_params))
    print(f"  CudnnLSTM parameters: {cudnn_param_count:,}")

    print(f"\n" + "="*50)
    print("FORWARD PASS")
    print("="*50)

    # Test forward pass
    try:
        cudnn_output = cudnn_lstm.apply(cudnn_params, inputs, deterministic=True)
        print("✓ CudnnLSTM forward pass successful")
        print(f"  Output shape: {cudnn_output.shape}")
        expected_hidden = hidden_size * 2 if bidirectional else hidden_size
        print(f"  Expected hidden dimension: {expected_hidden}")
    except Exception as e:
        print(f"✗ CudnnLSTM forward pass failed: {e}")
        return

    try:
        classic_output = classic_lstm.apply(classic_params, inputs, lengths, deterministic=True)
        print("✓ Classic LSTM forward pass successful")
        print(f"  Output shape: {classic_output[0].shape}")
        print(f"  Number of final states: {len(classic_output[1])}")
    except Exception as e:
        print(f"✗ Classic LSTM forward pass failed: {e}")
        return

    print(f"\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)

    # Benchmark performance
    num_runs = 20
    print(f"Running {num_runs} iterations...")

    try:
        classic_time, classic_result = benchmark_lstm(
            classic_lstm, "classic", classic_params, inputs, lengths, num_runs)
        print(f"✓ Classic LSTM average time: {classic_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ Classic LSTM benchmark failed: {e}")
        return

    try:
        cudnn_time, cudnn_result = benchmark_lstm(
            cudnn_lstm, "cudnn", cudnn_params, inputs, lengths, num_runs)
        print(f"✓ CudnnLSTM average time: {cudnn_time*1000:.2f} ms")
    except Exception as e:
        print(f"✗ CudnnLSTM benchmark failed: {e}")
        return
    # Speed comparison
    speedup = classic_time / cudnn_time
    print(f"\nSpeedup (CudnnLSTM vs Classic): {speedup:.2f}x")

    print(f"\n" + "="*50)
    print("GRADIENT COMPUTATION")
    print("="*50)

    # Create dummy target output for gradient computation
    target_output = jnp.zeros_like(cudnn_output)
    
    try:
        # Compute gradients for CudnnLSTM
        cudnn_gradients = compute_gradients_cudnn(cudnn_lstm, cudnn_params, inputs, target_output)
        cudnn_grad_norm = compute_gradient_norm(cudnn_gradients)
        cudnn_grad_norms_by_layer = compute_gradient_norms_by_layer(cudnn_gradients)
        
        print("✓ CudnnLSTM gradients computed successfully")
        print(f"  Total gradient norm: {cudnn_grad_norm:.6f}")
        print(f"  Gradient norms by layer: {cudnn_grad_norms_by_layer}")
        
    except Exception as e:
        print(f"✗ CudnnLSTM gradient computation failed: {e}")
        return

    try:
        # Create target output for classic LSTM (same shape as its output)
        classic_target_output = jnp.zeros_like(classic_output[0])
        
        # Compute gradients for Classic LSTM
        classic_gradients = compute_gradients_classic(classic_lstm, classic_params, inputs, lengths, classic_target_output)
        classic_grad_norm = compute_gradient_norm(classic_gradients)
        classic_grad_norms_by_layer = compute_gradient_norms_by_layer(classic_gradients)
        
        print("✓ Classic LSTM gradients computed successfully")
        print(f"  Total gradient norm: {classic_grad_norm:.6f}")
        print(f"  Gradient norms by layer: {classic_grad_norms_by_layer}")
        
    except Exception as e:
        print(f"✗ Classic LSTM gradient computation failed: {e}")
        return

    # Compare gradient norms
    grad_norm_ratio = classic_grad_norm / cudnn_grad_norm
    print(f"\nGradient norm ratio (Classic/CudnnLSTM): {grad_norm_ratio:.6f}")

    print(f"\n" + "="*50)
    print("TRAINING COMPARISON")
    print("="*50)

    # Create training data
    train_rng = jax.random.PRNGKey(456)
    train_inputs, train_targets, train_lengths = create_training_data(
        batch_size, seq_len, input_dim, hidden_size, train_rng)
    
    print(f"Training data shapes:")
    print(f"  Inputs: {train_inputs.shape}")
    print(f"  Targets: {train_targets.shape}")
    print(f"  Lengths: {train_lengths.shape}")
    
    # Run training comparison
    num_training_steps = 20
    learning_rate = 0.001
    
    try:
        training_results = run_training_comparison(
            cudnn_lstm, classic_lstm, cudnn_params, classic_params,
            train_inputs, train_targets, train_lengths, 
            num_steps=num_training_steps, learning_rate=learning_rate)
        
        print(f"\nTraining Summary:")
        print(f"  Average CudnnLSTM gradient norm: {np.mean(training_results['cudnn_grad_norms']):.6f}")
        print(f"  Average Classic gradient norm: {np.mean(training_results['classic_grad_norms']):.6f}")
        print(f"  Average gradient norm ratio: {np.mean(np.array(training_results['cudnn_grad_norms']) / np.array(training_results['classic_grad_norms'])):.6f}")
        
        print(f"  Final CudnnLSTM loss: {training_results['cudnn_losses'][-1]:.6f}")
        print(f"  Final Classic loss: {training_results['classic_losses'][-1]:.6f}")
        
        # Check for gradient explosion
        max_cudnn_grad = max(training_results['cudnn_grad_norms'])
        max_classic_grad = max(training_results['classic_grad_norms'])
        print(f"  Max CudnnLSTM gradient norm: {max_cudnn_grad:.6f}")
        print(f"  Max Classic gradient norm: {max_classic_grad:.6f}")
        
        if max_cudnn_grad > 10.0:
            print(f"  ⚠️  WARNING: CudnnLSTM shows potential gradient explosion!")
        if max_classic_grad > 10.0:
            print(f"  ⚠️  WARNING: Classic LSTM shows potential gradient explosion!")
            
    except Exception as e:
        print(f"✗ Training comparison failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n" + "="*50)
    print("MEMORY USAGE")
    print("="*50)

    # Memory usage (approximate)
    def estimate_memory(params):
        return sum(x.nbytes for x in jax.tree.leaves(params)) / (1024**2)  # MB

    cudnn_memory = estimate_memory(cudnn_params)
    classic_memory = estimate_memory(classic_params)

    print(f"Parameter count difference: {abs(cudnn_param_count - classic_param_count):,}")
    print(f"Performance gain: {speedup:.2f}x faster (CudnnLSTM)")
    print(f"Memory efficiency: {classic_memory/cudnn_memory:.2f}x more memory (Classic)")


if __name__ == "__main__":
    main()
