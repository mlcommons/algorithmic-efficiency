"""OGBG workload implemented in Jax."""
import functools
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp
import jraph
import optax

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg import metrics
from algorithmic_efficiency.workloads.ogbg.ogbg_jax import models
from algorithmic_efficiency.workloads.ogbg.workload import BaseOgbgWorkload


class OgbgWorkload(BaseOgbgWorkload):

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(
          self._param_shapes.unfreeze())
    return self._param_types

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    model = models.GNN(self._num_outputs)
    init_fn = jax.jit(functools.partial(model.init, train=False))
    fake_batch = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, self._num_outputs)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))
    params = init_fn({'params': params_rng, 'dropout': dropout_rng}, fake_batch)
    params = params['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    return jax_utils.replicate(params), None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      dropout_rate: Optional[float],
      aux_dropout_rate: Optional[float],
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Get predicted logits from the network for input graphs.

    aux_dropout_rate is unused.
    """
    del aux_dropout_rate
    del update_batch_norm  # No BN in the GNN model.
    if model_state is not None:
      raise ValueError(
          f'Expected model_state to be None, received {model_state}.')
    train = mode == spec.ForwardPassMode.TRAIN
    model = models.GNN(self._num_outputs, dropout_rate=dropout_rate)
    logits = model.apply({'params': params},
                         augmented_and_preprocessed_input_batch['inputs'],
                         rngs={'dropout': rng},
                         train=train)
    return logits, None

  def _binary_cross_entropy_with_mask(
      self,
      labels: jnp.ndarray,
      logits: jnp.ndarray,
      mask: jnp.ndarray,
      label_smoothing: float = 0.0) -> jnp.ndarray:
    """Binary cross entropy loss for logits, with masked elements."""
    if not (logits.shape == labels.shape == mask.shape):  # pylint: disable=superfluous-parens
      raise ValueError(
          f'Shape mismatch between logits ({logits.shape}), targets '
          f'({labels.shape}), and weights ({mask.shape}).')
    if len(logits.shape) != 2:
      raise ValueError(f'Rank of logits ({logits.shape}) must be 2.')

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = jnp.where(mask, labels, -1)

    # Apply label smoothing.
    smoothed_labels = optax.smooth_labels(labels, label_smoothing)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    return relu_logits - (logits * smoothed_labels) + (
        jnp.log(1 + jnp.exp(-abs_logits)))

  def _eval_metric(self, labels, logits, masks):
    per_example_losses = self.loss_fn(labels, logits, masks)
    loss = jnp.sum(jnp.where(masks, per_example_losses, 0)) / jnp.sum(masks)
    return metrics.EvalMetrics.single_from_model_output(
        loss=loss, logits=logits, labels=labels, mask=masks)

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_batch(self, params, batch, model_state, rng):
    return super()._eval_batch(params, batch, model_state, rng)
