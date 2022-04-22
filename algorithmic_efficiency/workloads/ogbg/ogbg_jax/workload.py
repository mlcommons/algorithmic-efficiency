"""OGB workload implemented in Jax."""
import functools
import itertools
import math
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg.ogbg_jax import input_pipeline
from algorithmic_efficiency.workloads.ogbg.ogbg_jax import metrics
from algorithmic_efficiency.workloads.ogbg.ogbg_jax import models
from algorithmic_efficiency.workloads.ogbg.workload import BaseOgbgWorkload


class OgbgWorkload(BaseOgbgWorkload):

  def __init__(self):
    self._eval_iters = {}
    self._param_shapes = None
    self._init_graphs = None
    self._model = models.GNN()

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    dataset_iter = input_pipeline.get_dataset_iter(split,
                                                   data_rng,
                                                   data_dir,
                                                   global_batch_size)
    if self._init_graphs is None:
      init_graphs, _, _ = next(dataset_iter)
      # Unreplicate the iterator that has the leading dim for pmapping.
      self._init_graphs = jax.tree_map(lambda x: x[0], init_graphs)
    return dataset_iter

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass


  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    if self._init_graphs is None:
      raise ValueError(
          'This should not happen, workload.build_input_queue() should be '
          'called before workload.init_model_fn()!')
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    init_fn = jax.jit(functools.partial(self._model.init, train=False))
    params = init_fn({'params': params_rng, 'dropout': dropout_rng},
                     self._init_graphs)
    params = params['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    return jax_utils.replicate(params), None

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    pass

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Get predicted logits from the network for input graphs."""
    del update_batch_norm  # No BN in the GNN model.
    assert model_state is None
    train = mode == spec.ForwardPassMode.TRAIN
    logits = self._model.apply({'params': params},
                               augmented_and_preprocessed_input_batch,
                               rngs={'dropout': rng},
                               train=train)
    return logits, None

  def _binary_cross_entropy_with_mask(self,
                                      labels: jnp.ndarray,
                                      logits: jnp.ndarray,
                                      mask: jnp.ndarray) -> jnp.ndarray:
    """Binary cross entropy loss for logits, with masked elements."""
    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = jnp.where(mask, labels, -1)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    return relu_logits - (logits * labels) + (jnp.log(1 + jnp.exp(-abs_logits)))

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor]) -> spec.Tensor:  # differentiable
    per_example_losses = self._binary_cross_entropy_with_mask(
        labels=label_batch, logits=logits_batch, mask=mask_batch)
    return per_example_losses

  def _eval_metric(self, labels, logits, masks):
    per_example_losses = self.loss_fn(labels, logits, masks)
    loss = jnp.sum(jnp.where(masks, per_example_losses, 0)) / jnp.sum(masks)
    return metrics.EvalMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=masks)

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_batch(self, params, batch, model_state, rng):
    logits, _ = self.model_fn(
        params,
        batch['inputs'],
        model_state,
        spec.ForwardPassMode.EVAL,
        rng,
        update_batch_norm=False)
    return self._eval_metric(batch['targets'], logits, batch['weights'])

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:
    """Run a full evaluation of the model."""
    data_rng, model_rng = prng.split(rng, 2)
    if split not in self._eval_iters:
      eval_iter = self.build_input_queue(
          data_rng, split, data_dir, global_batch_size=global_batch_size)
      # Note that this stores the entire val dataset in memory.
      self._eval_iters[split] = itertools.cycle(eval_iter)

    total_metrics = None
    num_eval_steps = int(math.ceil(float(num_examples) / global_batch_size))
    # Loop over graph batches in eval dataset.
    for _ in range(num_eval_steps):
      batch = next(self._eval_iters[split])
      batch_metrics = self._eval_batch(params,
                                       batch,
                                       model_state,
                                       model_rng)
      total_metrics = (
          batch_metrics
          if total_metrics is None else total_metrics.merge(batch_metrics))
    if total_metrics is None:
      return {}
    return {k: float(v) for k, v in total_metrics.reduce().compute().items()}
