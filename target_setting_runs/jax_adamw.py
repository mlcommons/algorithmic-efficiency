"""Submission file for an AdamW optimizer with warmup+cosine LR in Jax."""
import functools
from typing import Dict, List, Tuple

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from target_setting_runs.data_selection import \
    data_selection  # pylint: disable=unused-import


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb_dlrm':
    return 524288



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
  epsilon = (
      hyperparameters.epsilon if hasattr(hyperparameters, 'epsilon') else 1e-8)
  opt_init_fn, opt_update_fn = optax.adamw(
      learning_rate=schedule_fn,
      b1=hyperparameters.beta1,
      b2=hyperparameters.beta2,
      eps=epsilon,
      weight_decay=hyperparameters.l2)
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


@functools.partial(
    jax.pmap,
    axis_name='batch',
    in_axes=(None, None, 0, 0, 0, 0, 0, None),
    static_broadcasted_argnums=(0, 1))
def pmapped_train_step(workload,
                       opt_update_fn,
                       model_state,
                       optimizer_state,
                       current_param_container,
                       batch,
                       rng,
                       label_smoothing):

  def _loss_fn(params):
    """Loss function used for training."""
    logits, new_model_state = workload.model_fn(
        params,
        batch,
        model_state,
        spec.ForwardPassMode.TRAIN,
        rng,
        dropout_rate=None,
        aux_dropout_rate=None,
        update_batch_norm=False)
    loss = jnp.mean(
        workload.loss_fn(
            batch['targets'], logits, label_smoothing=label_smoothing))
    return loss, (new_model_state, logits)

  grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (new_model_state, _), grad = grad_fn(current_param_container)
  grad = lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_model_state, new_optimizer_state, updated_params


def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del global_step

  optimizer_state, opt_update_fn = optimizer_state
  per_device_rngs = jax.random.split(rng, jax.local_device_count())
  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  new_model_state, new_optimizer_state, new_params = pmapped_train_step(
      workload, opt_update_fn, model_state, optimizer_state,
      current_param_container, batch, per_device_rngs, label_smoothing)

  return (new_optimizer_state, opt_update_fn), new_params, new_model_state
