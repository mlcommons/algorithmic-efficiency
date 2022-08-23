"""Submission file for an AdamW optimizer in Jax."""

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  # Create learning rate schedule.
  warmup_fn = optax.linear_schedule(
      init_value=0.,
      end_value=hyperparameters.learning_rate,
      transition_steps=hyperparameters.warmup_steps)
  cosine_steps = max(hyperparameters.num_steps - hyperparameters.warmup_steps,
                     1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=hyperparameters.learning_rate, decay_steps=cosine_steps)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[hyperparameters.warmup_steps])

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = opt_init_fn, opt_update_fn = optax.adamw(
      learning_rate=schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=hyperparameters.epsilon,
      weight_decay=hyperparameters.l2)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn
