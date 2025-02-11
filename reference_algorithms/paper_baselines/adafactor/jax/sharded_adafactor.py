# coding=utf-8
# Copyright 2023 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PAX/Praxis implementation of Adafactor.

Copied from Praxis's `sharded_adafactor`, removing unnecessary sharding-related
code and dependencies on Praxis.

Code:
https://github.com/google/praxis/blob/516a96bce6f03090c5903531038f8f8af6212250/praxis/optimizers.py#L2308

Forked from:
https://github.com/google/init2winit/master/init2winit/optimizer_lib/pax_adafactor.py
"""

import dataclasses
import functools
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import jax
from jax import numpy as jnp
import optax

JTensor = Any
NestedJTensor = Any
NestedHParams = Any


def to_quantized(fvalue: JTensor,
                 quantized_dtype: jnp.dtype) -> Tuple[JTensor, JTensor]:
  """Converts floating point values `fvalues` to quantized values.

  We use a very simple quantization scheme where the range is symmetric around
  0.0, and we simply map 0 to 0.0.

  Let x = bucket_size
  We map [-0.5x, 0.5x] to 0
         [-1.5x, -0.5x] to -1
         [0.5x, 1.5x] to 1
         and so on so forth.

  Some properties:
    a1, a2 = to_quantized(x, quantized_dtype)
    b1 = to_float(a1, a2)
    c1, c2 = to_quantized(b1, quantized_dtype)

    then a1 == c1, a2 == c2

  Args:
    fvalue: Values in floating point.
    quantized_dtype: Quantized dtype, can be either jnp.int8, or jnp.int16.

  Returns:
    A (quantized_values, bucket_size) 2-tuple.
    `quantized_values * bucket_size[jnp.newaxis, ...]` are the quantized
    values
    on the floating value axis.
  """
  float_dtype = fvalue.dtype
  if quantized_dtype == jnp.int8:
    # value -128 is not used.
    num_buckets = jnp.array(127.0, dtype=float_dtype)
  elif quantized_dtype == jnp.int16:
    # value -32768 is not used.
    num_buckets = jnp.array(32767.0, dtype=float_dtype)
  else:
    raise ValueError(f'Quantized dtype {quantized_dtype} not supported.')
  # max value is mapped to num_buckets

  # We first decide the scale.
  if fvalue.ndim < 1:
    raise ValueError(
        f'Input array {fvalue} must have a strictly positive number of '
        'dimensions.')

  max_abs = jnp.max(jnp.abs(fvalue), axis=0)
  bucket_size = max_abs / num_buckets
  bs_expanded = bucket_size[jnp.newaxis, ...]
  # To avoid divide by 0.0
  bs_nonzero = jnp.where(bs_expanded > 0.0,
                         bs_expanded,
                         jnp.ones_like(bs_expanded))
  ratio = fvalue / bs_nonzero
  # We use rounding to remove bias.
  quantized = jnp.round(ratio)
  return quantized.astype(quantized_dtype), bucket_size


def to_float(quantized: JTensor, bucket_size: JTensor) -> JTensor:
  """Converts quantized values to float values.

  Args:
    quantized: Quantized values, of type either jnp.int8 or jnp.int16.
    bucket_size: The size of each bucket on the floating-point axis. bucket_size
      is of rank tf.rank(quantized) - 1. For example, if quantized is of shape
      [x, ...], bucket_size is of shape [...].

  Returns:
    Unquantized values of type bucket_size.dtype.
  """
  float_dtype = bucket_size.dtype
  bucket_size = bucket_size[jnp.newaxis, ...]
  return quantized.astype(float_dtype) * bucket_size


def adafactor_decay_rate_adam(beta2: float, step_counter: JTensor) -> JTensor:
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  beta2 = jnp.array(beta2, dtype=jnp.float32)
  t = step + 1.
  return beta2 * (1. - jnp.power(beta2, t - 1.)) / (1. - jnp.power(beta2, t))


def adafactor_decay_rate_pow(exponent: float, step_counter: JTensor) -> JTensor:
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: A floating point value between 0 and 1.
    step_counter: A scalar tensor keeping track of the number of steps
      performed.

  Returns:
    The decay rate as a scalar JTensor.
  """
  step = step_counter
  exponent = jnp.array(exponent, dtype=jnp.float32)
  return 1. - jnp.power((step + 1.), -exponent)


def reduce_mean(array: JTensor) -> JTensor:
  """Computes the mean of `array` in a more numerically stable way.

  Args:
    array: Input array.

  Returns:
    The mean of the input array as a scalar array.
  """
  num_elements = array.size
  if num_elements > 1e8:
    # When x is too large, simple jnp.mean() can result in nan or inf values.
    # TODO(bf-jax): The following code snippet is consistent with the TensorFlow
    # implementation. This can be simplified into `jnp.mean(jnp.mean(x, -1))`.
    # Update to using mean() after verifying consistency.
    array_sum = jnp.sum(array, axis=-1)
    array_sum = jnp.sum(array_sum)
    return array_sum / jnp.array(num_elements, dtype=array_sum.dtype)
  else:
    return jnp.mean(array)


def reduce_rms(array: JTensor) -> JTensor:
  """Computes the RMS of `array` (in a numerically stable way).

  Args:
    array: Input array.

  Returns:
    The root mean square of the input array as a scalar array.
  """
  sq = jnp.square(array)
  sq_mean = reduce_mean(sq)
  return jnp.sqrt(sq_mean)


@dataclasses.dataclass(frozen=True)
class _ShardedAdafactorUpdateResult:
  """Structure containing per-variable info for Adafactor."""
  update: Optional[Any]
  m: Optional[Any]
  m_scale: Optional[Any]
  vr: Optional[Any]
  vc: Optional[Any]
  v: Optional[Any]


class ShardedAdafactorState(NamedTuple):
  """Overall state of the ShardedAdafactor optimizer."""
  count: JTensor
  m: Optional[NestedJTensor]
  m_scale: Optional[NestedJTensor]
  vr: Optional[NestedJTensor]
  vc: Optional[NestedJTensor]
  v: Optional[NestedJTensor]


class _ShardedAdafactorHelper:
  """Helper class to implement optax-based sharded Adafactor."""

  def __init__(self,
               learning_rate: optax.Schedule,
               weight_decay: Optional[float],
               layerwise_adaptation: bool,
               decay_method: str,
               decay_adam: float,
               decay_pow: float,
               beta1: float,
               clip_threshold: Optional[float],
               factored: bool,
               epsilon1_grad_sq_reg: float,
               quantized_dtype: jnp.dtype,
               respect_skip_lp_regularization: bool,
               exclude_from_layerwise_adaptation: Optional[List[str]],
               per_var_learning_summary: bool,
               sort_factored_second_moment_dims: bool,
               min_dim_size_to_factor: int,
               multiply_by_parameter_scale: bool,
               epsilon2_param_scale_reg: float,
               maybe_inf_to_nan: bool,
               nesterov: bool) -> None:
    """Constructor. See ShardedAdafactor() below."""

    self._learning_rate = learning_rate
    self._weight_decay = weight_decay
    self._layerwise_adaptation = layerwise_adaptation
    self._decay_method = decay_method
    self._decay_adam = decay_adam
    self._decay_pow = decay_pow
    self._beta1 = beta1
    self._clip_threshold = clip_threshold
    self._factored = factored
    self._epsilon1 = epsilon1_grad_sq_reg
    self._quantized_dtype = quantized_dtype
    self._respect_skip_lp_regularization = respect_skip_lp_regularization
    self._exclude_from_layerwise_adaptation = exclude_from_layerwise_adaptation
    self._per_var_learning_summary = per_var_learning_summary
    self._sort_factored_second_moment_dims = sort_factored_second_moment_dims
    self._min_dim_size_to_factor = min_dim_size_to_factor
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    self._epsilon2 = epsilon2_param_scale_reg
    self._maybe_inf_to_nan = maybe_inf_to_nan
    self._nesterov = nesterov

  def should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.

    Args:
      shape: a list of integers.

    Returns:
      A boolean.
    """
    return self.factored_second_moment_dims(shape) is not None

  def factored_second_moment_dims(self, shape):
    """Should we use a factored second moment estimator.

    We select largest and second largest var dims as row and colum dims.

    Default list of factored dims is -1, -2.

    Args:
      shape: a list of integers.

    Returns:
      either a list of 2 Dimension indices for row and col or None
    """
    if not self._factored:
      return None
    if len(shape) < 2:
      return None
    if not self._sort_factored_second_moment_dims:
      return len(shape) - 1, len(shape) - 2

    def largest_two_dim_indices():
      s = [(s, i) for i, s in enumerate(shape)]
      sorted_dims = sorted(s, key=lambda d: -d[0])
      return sorted_dims[0][1], sorted_dims[1][1]

    r_idx, c_idx = largest_two_dim_indices()
    if shape[c_idx] < self._min_dim_size_to_factor:
      return None
    return r_idx, c_idx

  def should_store_momentum_in_qint(self, shape):
    """Should we store momentum as quantized integers.

    Based on the shape of the variable.

    Args:
      shape: a list of integers

    Returns:
      A boolean.
    """
    if jnp.issubdtype(self._quantized_dtype, jnp.floating):
      return False
    if self._quantized_dtype is None:
      return False
    return len(shape) >= 1

  def to_state(self, count, result_tree):
    """Maps from a tree of (factored) values to separate trees of values."""
    return ShardedAdafactorState(
        count=count,
        m=jax.tree.map(lambda o: o.m, result_tree),
        m_scale=jax.tree.map(lambda o: o.m_scale, result_tree),
        vr=jax.tree.map(lambda o: o.vr, result_tree),
        vc=jax.tree.map(lambda o: o.vc, result_tree),
        v=jax.tree.map(lambda o: o.v, result_tree))

  def init(self, param):
    """Initializes the optimizer state for a given param."""
    # The actually value that will be added to a variable for updating it.
    output_update = jnp.zeros((1,))
    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))
    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))
    shape = param.shape
    if self._beta1:
      if jnp.issubdtype(self._quantized_dtype, jnp.floating):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
      elif self.should_store_momentum_in_qint(shape):
        output_m = jnp.zeros(shape, dtype=self._quantized_dtype)
        scale_shape = shape[1:]
        output_m_scale = jnp.zeros(scale_shape, dtype=jnp.float32)
      else:
        output_m = jnp.zeros(shape, dtype=jnp.float32)
    if self.should_use_factored_second_moment_estimate(shape):
      factored_dims = self.factored_second_moment_dims(shape)
      vr_axis, vc_axis = factored_dims
      output_vr_shape = list(shape).copy()
      del output_vr_shape[vr_axis]
      output_vc_shape = list(shape).copy()
      del output_vc_shape[vc_axis]
      output_vr = jnp.zeros(output_vr_shape, dtype=jnp.float32)
      output_vc = jnp.zeros(output_vc_shape, dtype=jnp.float32)
    else:
      output_v = jnp.zeros(shape, dtype=jnp.float32)
    return _ShardedAdafactorUpdateResult(
        update=output_update,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)

  def inf_to_nan(self, array):
    """Converting Infinity values to the more sticky NaN."""
    # For example, when we have y = 1.0 / x in code and x == inf, y will become
    # 0. Therefore the infinite value of x is hidden in the calculation,
    # leading to silent omission of numerical issues.
    if not self._maybe_inf_to_nan:
      return array
    return jnp.nan_to_num(array, nan=jnp.nan, posinf=jnp.nan, neginf=jnp.nan)

  def parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.

    Returns:
      a Scalar
    """
    return jnp.maximum(reduce_rms(var), jnp.asarray(self._epsilon2, var.dtype))

  def compute_var_and_slot_update(self,
                                  count,
                                  grad,
                                  m,
                                  m_scale,
                                  vr,
                                  vc,
                                  v,
                                  param,
                                  var_name=None):
    """Computes the var and optimizer slots updates for a single variable."""
    # We can probably skip this step
    grad = grad.astype(jnp.float32)
    grad = self.inf_to_nan(grad)
    grad_squared = jnp.square(grad)

    # Add epsilon1_grad_sq_reg as per Algorithm 4
    # of https://arxiv.org/pdf/1804.04235.pdf
    grad_squared += self._epsilon1
    grad_squared_mean = self.inf_to_nan(reduce_mean(grad_squared))
    if self._decay_method == 'adam':
      assert self._decay_adam > 0
      decay_rate = adafactor_decay_rate_adam(self._decay_adam, count)
    elif self._decay_method == 'pow':
      assert self._decay_pow > 0
      decay_rate = adafactor_decay_rate_pow(self._decay_pow, count)
    else:
      raise ValueError(f'decay_method {self._decay_method} not supported.')

    learning_rate = self._learning_rate
    if callable(learning_rate):
      learning_rate = learning_rate(count)

    update_scale = learning_rate
    old_val = param

    if self._multiply_by_parameter_scale:
      update_scale *= self.parameter_scale(old_val).astype(update_scale.dtype)

    # Q(yonghui): Can we remove the hack now?
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK

    mixing_rate = 1. - decay_rate
    shape = param.shape

    output_m = jnp.zeros((1,))
    output_m_scale = jnp.zeros((1,))
    output_vr = jnp.zeros((1,))
    output_vc = jnp.zeros((1,))
    output_v = jnp.zeros((1,))

    factored_second_moment_dims = self.factored_second_moment_dims(shape)
    if factored_second_moment_dims is not None:
      # Q(shafey): Should we use the more numerically stable version
      # reduce_mean().
      vr_axis, vc_axis = factored_second_moment_dims
      grad_squared_row_mean = self.inf_to_nan(
          jnp.mean(grad_squared, axis=vr_axis))
      grad_squared_col_mean = self.inf_to_nan(
          jnp.mean(grad_squared, axis=vc_axis))
      new_vr = decay_rate * vr + mixing_rate * grad_squared_row_mean
      new_vc = decay_rate * vc + mixing_rate * grad_squared_col_mean
      output_vr = new_vr
      output_vc = new_vc
      long_term_mean = jnp.mean(new_vr, axis=-1, keepdims=True)
      r_factor = 1. / jnp.sqrt(new_vr / long_term_mean)
      c_factor = 1. / jnp.sqrt(new_vc)
      x = grad * jnp.expand_dims(r_factor, vr_axis) * jnp.expand_dims(
          c_factor, vc_axis)
    else:
      # v with sharding annotation.
      new_v = decay_rate * v + mixing_rate * grad_squared
      output_v = new_v
      x = grad / jnp.sqrt(new_v)

    if self._clip_threshold is not None:
      clipping_denom = jnp.maximum(1., reduce_rms(x) / self._clip_threshold)
      clipping_denom = self.inf_to_nan(clipping_denom)
      x /= clipping_denom

    subtrahend = update_scale * x
    if self._beta1:
      if jnp.issubdtype(self._quantized_dtype, jnp.floating):
        m = m.astype(jnp.float32)
      elif self.should_store_momentum_in_qint(shape):
        m_init_dtype = m.dtype
        m = to_float(m, m_scale)
      if self._nesterov:
        subtrahend_original = subtrahend
      subtrahend = self._beta1 * m + (1. - self._beta1) * subtrahend
      subtrahend = self.inf_to_nan(subtrahend)
      if self._quantized_dtype == jnp.bfloat16:
        new_m = subtrahend.astype(jnp.bfloat16)
        output_m = new_m
      elif self.should_store_momentum_in_qint(shape):
        # Update the momentum values.
        new_m_val, new_m_scale = to_quantized(subtrahend, m_init_dtype)
        output_m = new_m_val
        output_m_scale = new_m_scale
      else:
        output_m = subtrahend

      if self._nesterov:
        subtrahend = (
            self._beta1 * subtrahend +
            (1.0 - self._beta1) * subtrahend_original)

    if self._weight_decay is not None:
      # Apply decoupled weight decay to be consistent with AdamW.
      var_weight_decay = None
      if isinstance(self._weight_decay, dict):
        for scope_pattern in self._weight_decay.keys():
          regex_pattern = re.compile(scope_pattern)
          if regex_pattern.match(var_name):
            var_weight_decay = self._weight_decay[scope_pattern]
      else:
        var_weight_decay = self._weight_decay

      if var_weight_decay is not None:
        weight_decay = var_weight_decay * learning_rate
        subtrahend += weight_decay * old_val

    if self._layerwise_adaptation:
      include = True
      if self._exclude_from_layerwise_adaptation is not None:
        for scope_pattern in self._exclude_from_layerwise_adaptation:
          regex_pattern = re.compile(scope_pattern)
          if regex_pattern.match(var_name):
            include = False
            break
      if include:
        w_norm = reduce_rms(old_val)
        g_norm = reduce_rms(subtrahend / update_scale) + self._epsilon1
        ratio = w_norm / g_norm
        ratio = jnp.where(
            jnp.greater(w_norm, 0),
            jnp.where(jnp.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0)
        subtrahend *= ratio

    return _ShardedAdafactorUpdateResult(
        update=-subtrahend,
        m=output_m,
        m_scale=output_m_scale,
        vr=output_vr,
        vc=output_vc,
        v=output_v)


def sharded_adafactor(
    learning_rate: optax.Schedule,
    weight_decay: Optional[Union[float, Dict[str, float]]] = None,
    layerwise_adaptation: bool = False,
    decay_method: str = 'adam',
    decay_adam: float = 0.99,
    decay_pow: float = 0.,
    beta1: float = 0.9,
    clip_threshold: Optional[float] = 1.,
    factored: bool = True,
    epsilon1_grad_sq_reg: float = 1e-30,
    quantized_dtype: jnp.dtype = jnp.int8,
    respect_skip_lp_regularization: bool = False,
    exclude_from_layerwise_adaptation: Optional[List[str]] = None,
    per_var_learning_summary: bool = False,
    sort_factored_second_moment_dims: bool = False,
    # min_dim_size_to_factor is only used when
    # sort_factored_second_moment_dims=True.
    min_dim_size_to_factor: int = 128,
    multiply_by_parameter_scale: bool = False,
    epsilon2_param_scale_reg: float = 1e-3,
    maybe_inf_to_nan: bool = True,
    nesterov: bool = False,
) -> optax.GradientTransformation:
  """AdaFactor optimizer that supports SPMD sharding.

  Reference:
    Shazeer et al, 2018: https://arxiv.org/abs/1804.04235

  Adafactor is very similar to Adam (Kingma and Ba, 2019), the major
  differences being:

  1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
     parameters to maintain the second-moment estimator, instead of AB.
     This is advantageous on memory-limited systems.  In addition, beta1
     (momentum) is set to zero by default, saving an additional auxiliary
     parameter per weight.  Variables with >=3 dimensions are treated as
     collections of two-dimensional matrices - factorization is over the final
     two dimensions.

  2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
     gradient clipping.  This improves stability.

  3. Adafactor does not require an external "learning rate".  By default, it
     incorporates a relative-update-scale schedule, corresponding to
     inverse-square-root learning-rate-decay in Adam.  We hope this works well
     for most applications.

  Args:
    learning_rate: a callable that given the current training step, returns the
      learning rate to apply.
    weight_decay: an optional float tensor as decoupled weight decay value, or a
      dictionary with key as regex scope pattern and value as corresponding
      weight decay float tensor. The value will apply to all variables under
      that scope name.
    layerwise_adaptation: a boolean, whether or not to use layer-wise adaptive
      moments (LAMB) https://arxiv.org/abs/1904.00962.
    decay_method: a string, deciding how decay_rate should be computed.
      Permitted values are 'adam' and 'pow'.
    decay_adam: a float, decay if decay_method == 'adam'.
    decay_pow: a float, decay if decay_method == 'pow'.
    beta1: a float value between 0 and 1 for momentum.
    clip_threshold: an optional float >= 1
    factored: a boolean, whether or not to use factored second order momentum.
    epsilon1_grad_sq_reg: Regularization constant for squared gradient.
    quantized_dtype: type of the quantized input. Allowed options are jnp.int8,
      jnp.int16, jnp.bfloat16 and jnp.float32. If floating-point type is
      specified, accumulators are stored as such type, instead of quantized
      integers.
    respect_skip_lp_regularization: whether or not to respect lingvo
      SKIP_LP_REGULARIZATION var collection that skips decoupled weight decay.
    exclude_from_layerwise_adaptation: A dictionary with key as regex scope
      pattern for variables to be skipped.
    per_var_learning_summary: a bool, whether or not to export per-var learning
      summaries.
    sort_factored_second_moment_dims: a bool, whether to select dims to factor
      by size, for the factored second moment.
    min_dim_size_to_factor: an integer, only factor the statistics if two array
      dimensions have at least this size. NOTE min_dim_size_to_factor is only
      used when sort_factored_second_moment_dims=True.
    multiply_by_parameter_scale: a boolean, if True, then scale learning_rate by
      parameter scale. if False provided learning_rate is absolute step size.
      NOTE False by default.
    epsilon2_param_scale_reg: Regularization constant for parameter scale. Only
      used when multiply_by_parameter_scale is True.
    maybe_inf_to_nan: Will use jax.nan_to_num during update when True.
    nesterov: Will use Nesterov momentum when True.

  Returns:
    A `ShardedGradientTransformation`.
  """

  assert not respect_skip_lp_regularization
  assert decay_adam >= 0
  assert decay_pow >= 0
  assert learning_rate is not None
  assert decay_method == 'adam' or decay_method == 'pow', (
      f'decay_method: {decay_method} not supported. Supported methods are '
      '"pow", or "adam".')

  sharded_adafactor_helper = _ShardedAdafactorHelper(
      learning_rate=learning_rate,
      weight_decay=weight_decay,
      layerwise_adaptation=layerwise_adaptation,
      decay_method=decay_method,
      decay_adam=decay_adam,
      decay_pow=decay_pow,
      beta1=beta1,
      clip_threshold=clip_threshold,
      factored=factored,
      epsilon1_grad_sq_reg=epsilon1_grad_sq_reg,
      quantized_dtype=quantized_dtype,
      respect_skip_lp_regularization=respect_skip_lp_regularization,
      exclude_from_layerwise_adaptation=exclude_from_layerwise_adaptation,
      per_var_learning_summary=per_var_learning_summary,
      sort_factored_second_moment_dims=sort_factored_second_moment_dims,
      min_dim_size_to_factor=min_dim_size_to_factor,
      multiply_by_parameter_scale=multiply_by_parameter_scale,
      epsilon2_param_scale_reg=epsilon2_param_scale_reg,
      maybe_inf_to_nan=maybe_inf_to_nan,
      nesterov=nesterov)

  def init_fn(params):
    """Initializes the optimizer's state."""
    return sharded_adafactor_helper.to_state(
        jnp.zeros([], jnp.int32),
        jax.tree.map(sharded_adafactor_helper.init, params))

  def update_fn(updates, state, params=None):
    if params is None:
      raise ValueError(
          'You are using a transformation that requires the current value of '
          'parameters, but you are not passing `params` when calling `update`.')

    compute_var_and_slot_update_fn = functools.partial(
        sharded_adafactor_helper.compute_var_and_slot_update, state.count)
    output = jax.tree.map(compute_var_and_slot_update_fn,
                          updates,
                          state.m,
                          state.m_scale,
                          state.vr,
                          state.vc,
                          state.v,
                          params)
    updates = jax.tree.map(lambda o: o.update, output)
    count_plus_one = state.count + jnp.array(1, jnp.int32)
    updated_states = sharded_adafactor_helper.to_state(count_plus_one, output)
    return updates, updated_states

  return optax.GradientTransformation(init=init_fn, update=update_fn)
