"""ImageNet workload implemented in Jax.

python3 submission_runner.py \
    --workload=imagenet_jax \
    --submission_path=workloads/imagenet/imagenet_jax/submission.py \
    --num_tuning_trials=1
"""

from typing import Tuple
from absl import logging
import optax

import tensorflow as tf
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make it
# unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import lax
from flax.training import common_utils
from flax import jax_utils

import spec
from . import input_pipeline
from . import models



class ImagenetWorkload(spec.Workload):
  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self.epoch_metrics = []
    self.model_name = 'ResNet50'
    self.dataset = 'imagenet2012:5.*.*'
    self.num_classes = 1000
    # For faster development testing, uncomment the lines below
    # self.model_name = '_ResNet1'
    # self.dataset = 'imagenette'
    # self.num_classes = 10

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result > 0.69

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 111600 # 31 hours

  @property
  def eval_period_time_sec(self):
    return 6000 # 100 mins

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def _build_dataset(self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size):
    if batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    ds_builder = tfds.builder(self.dataset)
    ds_builder.download_and_prepare()
    ds = input_pipeline.create_input_iter(
      ds_builder,
      batch_size,
      mean_rgb,
      stddev_rgb,
      train=True,
      cache=False)

    self.num_train_examples = ds_builder.info.splits['train'].num_examples
    self.num_eval_examples = ds_builder.info.splits['validation'].num_examples
    self.steps_per_epoch = self.num_train_examples // batch_size
    self.batch_size = batch_size
    return ds

  def build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  def sync_batch_stats(self, model_state):
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = model_state.copy({
      'batch_stats': avg(model_state['batch_stats'])})
    return new_model_state

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError('This should not happen, workload.init_model_fn() '
                       'should be called before workload.param_shapes!')
    return self._param_shapes

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    return (selected_raw_input_batch, selected_label_batch)

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    return (raw_input_batch, raw_label_batch)

  def create_model(self, *, model_cls, **kwargs):
    return model_cls(num_classes=self.num_classes,
                     dtype=jnp.float32,
                     **kwargs)

  def initialized(self, key, model):
    input_shape = (1, 224, 224, 3)
    @jax.jit
    def init(*args):
      return model.init(*args)
    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
    model_state, params = variables.pop('params')
    return params, model_state

  _InitState = Tuple[spec.ParameterTree, spec.ModelAuxillaryState]
  def init_model_fn(
      self,
      rng: spec.RandomState) -> _InitState:
    model_cls = getattr(models, self.model_name)
    model = self.create_model(model_cls=model_cls)
    self._model = model
    params, model_state = self.initialized(rng, model)
    self._param_shapes = jax.tree_map(
      lambda x: spec.ShapeTuple(x.shape),
      params)
    model_state = jax_utils.replicate(model_state)
    params = jax_utils.replicate(params)
    return params, model_state

  def eval_model_fn(self, params, batch, state, rng):
    logits, _ = self.model_fn(
      params,
      batch,
      state,
      spec.ForwardPassMode.EVAL,
      rng,
      update_batch_norm=False)
    return self.compute_metrics(logits, batch['label'])

  def model_fn(
      self,
      params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    variables = {'params': params, **model_state}
    train = mode == spec.ForwardPassMode.TRAIN
    if update_batch_norm:
      logits, new_model_state = self._model.apply(
        variables,
        jax.numpy.squeeze(augmented_and_preprocessed_input_batch['image']),
        train=train,
        mutable=['batch_stats'])
      return logits, new_model_state
    else:
      logits = self._model.apply(
        variables,
        jax.numpy.squeeze(augmented_and_preprocessed_input_batch['image']),
        train=train,
        mutable=False)
      return logits, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    """Cross Entropy Loss"""
    one_hot_labels = common_utils.onehot(label_batch,
                                         num_classes=self.num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits_batch,
                                           labels=one_hot_labels)
    return jnp.mean(xentropy)

  def compute_metrics(self, logits, labels):
    loss = self.loss_fn(labels, logits)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics

  def eval_model(
      self,
      params: spec.ParameterTree,
      model_state: spec.ModelAuxillaryState,
      rng: spec.RandomState,
      data_dir: str):
    """Run a full evaluation of the model."""

    # sync batch statistics across replicas
    model_state = self.sync_batch_stats(model_state)

    epoch_metrics = common_utils.get_metrics(self.epoch_metrics)
    summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
    logging.info('train loss: %.4f, accuracy: %.2f',
                  summary['loss'], summary['accuracy'] * 100)
    self.epoch_metrics = []
    eval_metrics = []

    data_rng, model_rng = jax.random.split(rng, 2)
    eval_batch_size = self.batch_size
    num_batches = self.num_eval_examples // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
        data_rng, split='test', batch_size=eval_batch_size, data_dir=data_dir)
    eval_iter = iter(self._eval_ds)
    total_accuracy = 0.
    for idx in range(num_batches):
      batch = next(eval_iter)
      metrics = jax.pmap(self.eval_model_fn,
                         axis_name='batch',
                         in_axes=(0, 0, 0, None))(
                      params,
                      batch,
                      model_state,
                      rng)
      eval_metrics.append(metrics)
      total_accuracy += jnp.mean(metrics['accuracy'])

    eval_metrics = common_utils.get_metrics(eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    logging.info('eval loss: %.4f, accuracy: %.2f',
                  summary['loss'], summary['accuracy'] * 100)
    return float(total_accuracy / num_batches)


