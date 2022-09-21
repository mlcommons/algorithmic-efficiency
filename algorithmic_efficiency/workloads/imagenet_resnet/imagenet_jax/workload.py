"""ImageNet workload implemented in Jax."""
import functools
import math
from typing import Dict, Optional, Tuple

from flax import jax_utils
import jax
from jax import lax
import jax.numpy as jnp
import optax
import tensorflow_datasets as tfds

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    input_pipeline
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax import \
    models
from algorithmic_efficiency.workloads.imagenet_resnet.workload import \
    BaseImagenetResNetWorkload


class ImagenetResNetWorkload(BaseImagenetResNetWorkload):

  def _build_dataset(self,
                     data_rng: spec.RandomState,
                     split: str,
                     data_dir: str,
                     batch_size: int,
                     cache: Optional[bool] = None,
                     repeat_final_dataset: Optional[bool] = None,
                     num_batches: Optional[int] = None):
    if batch_size % jax.local_device_count() != 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    ds_builder = tfds.builder('imagenet2012:5.*.*', data_dir=data_dir)
    ds_builder.download_and_prepare()
    train = split == 'train'
    if split == 'eval_train':
      split = f'train[:{self.num_eval_train_examples}]'
    ds = input_pipeline.create_input_iter(
        split,
        ds_builder,
        data_rng,
        batch_size,
        self.train_mean,
        self.train_stddev,
        self.center_crop_size,
        self.resize_size,
        self.aspect_ratio_range,
        self.scale_ratio_range,
        train=train,
        cache=not train if cache is None else cache,
        repeat_final_dataset=repeat_final_dataset,
        num_batches=num_batches)
    return ds

  def sync_batch_stats(self, model_state):
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
    avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy(
        {'batch_stats': avg_fn(model_state['batch_stats'])})
    return new_model_state

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

  def initialized(self, key, model):
    input_shape = (1, 224, 224, 3)
    variables = jax.jit(model.init)({'params': key},
                                    jnp.ones(input_shape, model.dtype))
    model_state, params = variables.pop('params')
    return params, model_state

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model_cls = getattr(models, 'ResNet50')
    model = model_cls(num_classes=1000, dtype=jnp.float32)
    self._model = model
    params, model_state = self.initialized(rng, model)
    self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                      params)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0),
      static_broadcasted_argnums=(0,))
  def _eval_model(self, params, batch, state):
    logits, _ = self.model_fn(
        params,
        batch,
        state,
        spec.ForwardPassMode.EVAL,
        rng=None,
        update_batch_norm=False)
    return self._compute_metrics(logits, batch['targets'])

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del mode
    del rng
    variables = {'params': params, **model_state}
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=['batch_stats'])
      return logits, new_model_state
    else:
      logits = self._model.apply(
          variables,
          augmented_and_preprocessed_input_batch['inputs'],
          update_batch_norm=update_batch_norm,
          mutable=False)
      return logits, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self,
              label_batch: spec.Tensor,
              logits_batch: spec.Tensor,
              label_smoothing: float = 0.0) -> spec.Tensor:  # differentiable
    """Cross Entropy Loss"""
    one_hot_labels = jax.nn.one_hot(label_batch, num_classes=1000)
    smoothed_labels = optax.smooth_labels(one_hot_labels, label_smoothing)
    return optax.softmax_cross_entropy(
        logits=logits_batch, labels=smoothed_labels)

  def _compute_metrics(self, logits, labels):
    loss = jnp.sum(self.loss_fn(labels, logits))
    # not accuracy, but nr. of correct predictions
    accuracy = jnp.sum(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = lax.psum(metrics, axis_name='batch')
    return metrics

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0):
    if model_state is not None:
      # Sync batch statistics across replicas before evaluating.
      model_state = self.sync_batch_stats(model_state)
    num_batches = int(math.ceil(num_examples / global_batch_size))
    # We already repeat the dataset indefinitely in tf.data.
    if split not in self._eval_iters:
      self._eval_iters[split] = self.build_input_queue(
          rng,
          split=split,
          global_batch_size=global_batch_size,
          data_dir=data_dir,
          cache=True,
          repeat_final_dataset=True,
          num_batches=num_batches)

    eval_metrics = {}
    for _ in range(num_batches):
      batch = next(self._eval_iters[split])
      # We already average these metrics across devices inside _compute_metrics.
      synced_metrics = self._eval_model(params, batch, model_state)
      for metric_name, metric_value in synced_metrics.items():
        if metric_name not in eval_metrics:
          eval_metrics[metric_name] = 0.0
        eval_metrics[metric_name] += metric_value

    eval_metrics = jax.tree_map(lambda x: float(x[0] / num_examples),
                                eval_metrics)
    return eval_metrics
