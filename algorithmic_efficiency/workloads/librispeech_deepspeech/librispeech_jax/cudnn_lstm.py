import jax
import flax.linen as nn
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union, TypeVar, cast
from jax.experimental import rnn 
import jax.numpy as jnp

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array
Carry = Any
CarryHistory = Any
Output = Any

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
        input_size, self.features,
        self.num_layers, self.bidirectional,
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
      seq_lengths = jnp.sum(1-segmentation_mask, axis=1, dtype=jnp.int32)
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
      W_ih, W_hh, b_ih, b_hh = self.unpack_weights(weights, input_size)
      y, h, c = rnn.lstm_ref(
        x=inputs, h_0=h_0, c_0=c_0, W_ih=W_ih, W_hh=W_hh,
        b_ih=b_ih, b_hh=b_hh, seq_lengths=seq_lengths,
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
  ) -> Tuple[Dict[int, Array], Dict[int, Array], Dict[int, Array], Dict[int, Array]]:
    return jax.experimental.rnn.unpack_lstm_weights(
      weights, input_size, self.features, self.num_layers, self.bidirectional,
    )
