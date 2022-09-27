"""Jax submission for the target-setting run on WMT with AdamW."""

import functools
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec
from target_setting_runs.data_selection import \
    data_selection  # pylint: disable=unused-import
from target_setting_runs.jax_adamw import \
    init_optimizer_state  # pylint: disable=unused-import


def get_batch_size(workload_name):
  # Return the global batch size.
  del workload_name
  return 256


@functools.partial(
    jax.pmap,
    in_axes=(None, None, 0, 0, 0, 0, None),
    axis_name='batch',
    static_broadcasted_argnums=(0, 1))
def pmapped_train_step(workload,
                       opt_update_fn,
                       optimizer_state,
                       current_param_container,
                       batch,
                       dropout_rng,
                       label_smoothing):
  """Perform a single training step."""

  def _loss_fn(params):
    """Loss function used for training."""
    logits, _ = workload.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.TRAIN,
        rng=dropout_rng,
        dropout_prob=0.1,  # Default.
        aux_dropout_prob=0.1,  # Default.
        update_batch_norm=False)
    targets = batch['targets']
    weights = jnp.where(targets > 0, 1.0, 0.0)
    loss = (workload.loss_fn(targets, logits, label_smoothing=label_smoothing) *
            weights).sum() / weights.sum()
    return loss

  grad_fn = jax.value_and_grad(_loss_fn)
  _, grad = grad_fn(current_param_container)
  grad = jax.lax.pmean(grad, axis_name='batch')
  updates, new_optimizer_state = opt_update_fn(
      grad, optimizer_state, current_param_container)
  updated_params = optax.apply_updates(current_param_container, updates)
  return new_optimizer_state, updated_params


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
  del eval_results
  del global_step
  del model_state
  del loss_type

  optimizer_state, opt_update_fn = optimizer_state
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  new_optimizer_state, updated_params = pmapped_train_step(
      workload,
      opt_update_fn,
      optimizer_state,
      current_param_container,
      batch,
      dropout_rngs,
      label_smoothing)
  return (new_optimizer_state, opt_update_fn), updated_params, None
