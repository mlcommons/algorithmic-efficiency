"""OGBG workload implemented in Jax."""
import functools
from typing import Any, Dict, Optional, Tuple

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

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is unused."""
    del aux_dropout_rate
    rng, params_rng, dropout_rng = jax.random.split(rng, 3)
    self._model = models.GNN(self._num_outputs, dropout_rate=dropout_rate)
    init_fn = jax.jit(functools.partial(self._model.init, train=False))
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
    self._param_shapes = param_utils.jax_param_shapes(params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    return jax_utils.replicate(params), None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_17'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Get predicted logits from the network for input graphs."""
    del update_batch_norm  # No BN in the GNN model.
    if model_state is not None:
      raise ValueError(
          f'Expected model_state to be None, received {model_state}.')
    train = mode == spec.ForwardPassMode.TRAIN

    logits = self._model.apply({'params': params},
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
    positive_logits = logits >= 0
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    losses = relu_logits - (logits * smoothed_labels) + (
        jnp.log(1 + jnp.exp(-abs_logits)))
    return jnp.where(mask, losses, 0.)

  def _eval_metric(self, labels, logits, masks):
    loss = self.loss_fn(labels, logits, masks)
    return metrics.EvalMetrics.single_from_model_output(
        loss=loss['per_example'], logits=logits, labels=labels, mask=masks)

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_batch(self, params, batch, model_state, rng):
    return super()._eval_batch(params, batch, model_state, rng)

  def _normalize_eval_metrics(
      self, num_examples: int, total_metrics: Dict[str,
                                                   Any]) -> Dict[str, float]:
    """Normalize eval metrics."""
    del num_examples
    total_metrics = total_metrics.reduce()
    return {k: float(v) for k, v in total_metrics.compute().items()}
