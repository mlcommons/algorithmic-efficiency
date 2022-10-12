"""Submission file for an AdamW optimizer with warmup+cosine LR in Jax."""
from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from reference_algorithms.target_setting_algorithms import cosine_warmup
from reference_algorithms.target_setting_algorithms.data_selection import \
    data_selection  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.get_batch_size import \
    get_batch_size  # pylint: disable=unused-import
from reference_algorithms.target_setting_algorithms.jax_submission_base import \
    update_params  # pylint: disable=unused-import


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  lr_schedule_fn = cosine_warmup.jax_cosine_warmup(workload.step_hint,
                                                   hyperparameters)

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  epsilon = (
      hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8)
  opt_init_fn, opt_update_fn = optax.adamw(
      learning_rate=lr_schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=epsilon,
      weight_decay=hyperparameters.weight_decay)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn
