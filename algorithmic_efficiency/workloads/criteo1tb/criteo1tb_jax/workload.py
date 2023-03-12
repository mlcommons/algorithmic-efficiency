"""Criteo1TB workload implemented in Jax."""
import functools
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
import jax.numpy as jnp

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax import models
from algorithmic_efficiency.workloads.criteo1tb.workload import \
    BaseCriteo1TbDlrmSmallWorkload


class Criteo1TbDlrmSmallWorkload(BaseCriteo1TbDlrmSmallWorkload):

  @property
  def eval_batch_size(self) -> int:
    return 524_288

  def _per_example_sigmoid_binary_cross_entropy(
      self, logits: spec.Tensor, targets: spec.Tensor) -> spec.Tensor:
    """Computes the sigmoid binary cross entropy per example.

    Args:
    logits: float array of shape (batch, output_shape).
    targets: float array of shape (batch, output_shape).
    Returns:
      Sigmoid binary cross entropy computed per example, shape (batch,).
    """
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    losses = -1.0 * (targets * log_p + (1 - targets) * log_not_p)
    return losses

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the (masked) loss function at (label_batch, logits_batch).

    Return {'summed': scalar summed loss, 'n_valid_examples': scalar number of
    valid examples in batch, 'per_example': 1-d array of per-example losses}
    (not synced across devices).
    """
    del label_smoothing
    batch_size = label_batch.shape[0]
    label_batch = jnp.reshape(label_batch, (batch_size,))
    logits_batch = jnp.reshape(logits_batch, (batch_size,))
    per_example_losses = self._per_example_sigmoid_binary_cross_entropy(
        logits=logits_batch, targets=label_batch)
    if mask_batch is not None:
      mask_batch = jnp.reshape(mask_batch, (batch_size,))
      per_example_losses *= mask_batch
      n_valid_examples = mask_batch.sum()
    else:
      n_valid_examples = len(per_example_losses)
    summed_loss = per_example_losses.sum()
    return {
        'summed': summed_loss,
        'n_valid_examples': n_valid_examples,
        'per_example': per_example_losses,
    }

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Only dropout is used."""
    del aux_dropout_rate
    self._model = models.DlrmSmall(
        vocab_size=self.vocab_size,
        num_dense_features=self.num_dense_features,
        mlp_bottom_dims=self.mlp_bottom_dims,
        mlp_top_dims=self.mlp_top_dims,
        embed_dim=self.embed_dim,
        dropout_rate=dropout_rate)

    params_rng, dropout_rng = jax.random.split(rng)
    init_fake_batch_size = 2
    num_categorical_features = 26
    input_size = self.num_dense_features + num_categorical_features
    input_shape = (init_fake_batch_size, input_size)

    init_fn = functools.partial(self._model.init, train=False)
    initial_variables = jax.jit(init_fn)(
        {'params': params_rng, 'dropout': dropout_rng},
        jnp.ones(input_shape, jnp.float32))
    initial_params = initial_variables['params']
    self._param_shapes = param_utils.jax_param_shapes(initial_params)
    self._param_types = param_utils.jax_param_types(self._param_shapes)
    return jax_utils.replicate(initial_params), None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key == 'Dense_7'

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del update_batch_norm
    inputs = augmented_and_preprocessed_input_batch['inputs']
    train = mode == spec.ForwardPassMode.TRAIN
    apply_kwargs = {'train': train}
    if train:
      apply_kwargs['rngs'] = {'dropout': rng}
    logits_batch = self._model.apply({'params': params}, inputs, **apply_kwargs)
    return logits_batch, None

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_batch_pmapped(self,
                          params: spec.ParameterContainer,
                          batch: Dict[str, spec.Tensor]) -> spec.Tensor:
    logits, _ = self.model_fn(
        params,
        batch,
        model_state=None,
        mode=spec.ForwardPassMode.EVAL,
        rng=None,
        update_batch_norm=False)
    weights = batch.get('weights')
    if weights is None:
      weights = jnp.ones(len(logits))
    summed_loss = self.loss_fn(
        label_batch=batch['targets'], logits_batch=logits,
        mask_batch=weights)['summed']
    return summed_loss

  def _eval_batch(self,
                  params: spec.ParameterContainer,
                  batch: Dict[str, spec.Tensor]) -> spec.Tensor:
    # We do NOT psum inside of _eval_batch_pmapped, so the returned tensor of
    # shape (local_device_count,) will all be different values.
    return self._eval_batch_pmapped(params, batch).sum()


class Criteo1TbDlrmSmallTestWorkload(Criteo1TbDlrmSmallWorkload):
  vocab_size: int = 32 * 128 * 16
