import functools
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax import metrics
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax import models
from algorithmic_efficiency.workloads.criteo1tb.workload import \
    BaseCriteo1TbDlrmSmallWorkload


class Criteo1TbDlrmSmallWorkload(BaseCriteo1TbDlrmSmallWorkload):
  """Criteo1TB DLRM-Small Jax workload."""

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.jax_param_types(self._param_shapes)
    return self._param_types

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor,
      label_smoothing: float = 0.0,
      mask_batch: Optional[spec.Tensor] = None) -> spec.Tensor:
    del label_smoothing
    per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      weighted_losses = per_example_losses * mask_batch
      normalization = mask_batch.sum()
    else:
      weighted_losses = per_example_losses
    normalization = label_batch.shape[0]
    return jnp.sum(weighted_losses, axis=-1) / normalization

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    self._model = models.DlrmSmall(
        vocab_sizes=self.vocab_sizes,
        total_vocab_sizes=sum(self.vocab_sizes),
        num_dense_features=self.num_dense_features,
        mlp_bottom_dims=(128, 128),
        mlp_top_dims=(256, 128, 1),
        embed_dim=64)

    rng, init_rng = jax.random.split(rng)
    init_fake_batch_size = 2
    input_size = self.num_dense_features + len(self.vocab_sizes)
    input_shape = (init_fake_batch_size, input_size)
    target_shape = (init_fake_batch_size, input_size)

    initial_variables = jax.jit(self._model.init)(
        init_rng,
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables['params']
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      initial_params)
    return jax_utils.replicate(initial_params), None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del mode
    del rng
    del update_batch_norm
    inputs = augmented_and_preprocessed_input_batch['inputs']
    targets = augmented_and_preprocessed_input_batch['targets']
    logits_batch = self._model.apply({'params': params}, inputs, targets)
    return logits_batch, None

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
  def _eval_batch(self, params, batch, model_state, rng):
    return super()._eval_batch(params, batch, model_state, rng)
