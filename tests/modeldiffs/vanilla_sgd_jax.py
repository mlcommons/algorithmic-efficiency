from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.jax_submission_base import \
    update_params  # pylint: disable=unused-import


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Vanilla SGD Optimizer."""
  del model_params
  del model_state
  del rng

  # Create optimizer.
  params_zeros_like = jax.tree.map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)

  opt_init_fn, opt_update_fn = optax.sgd(learning_rate=0.001)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn
