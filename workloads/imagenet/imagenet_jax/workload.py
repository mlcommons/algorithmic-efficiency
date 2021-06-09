"""ImageNet workload implemented in Jax."""

from typing import Tuple, Any, Callable
import time
import functools
from absl import logging

import tensorflow as tf
# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make it
# unavailable to JAX.
tf.config.experimental.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from jax import lax
import flax
from flax import optim
from flax.training import common_utils

import spec
from . import input_pipeline
from . import models
from . import submission
from . import config as config_lib

config = config_lib.get_config()



# flax.struct.dataclass enables instances of this class to be passed into jax
# transformations like tree_map and pmap.
@flax.struct.dataclass
class TrainState:
  step: int
  optimizer: optim.Optimizer
  model_state: Any


class ImagenetWorkload(spec.Workload):
  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self.epoch_metrics = []

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result > 0.6

  def _build_dataset(self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size):
    if config.batch_size % jax.device_count() > 0:
      raise ValueError('Batch size must be divisible by the number of devices')
    mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    if config.half_precision:
      if platform == 'tpu':
        input_dtype = tf.bfloat16
      else:
        input_dtype = tf.float16
    else:
      input_dtype = tf.float32

    ds_builder = tfds.builder(config.dataset)
    ds = input_pipeline.create_input_iter(
      ds_builder,
      batch_size,
      config.image_size,
      input_dtype,
      mean_rgb,
      stddev_rgb,
      train=split=='train',
      cache=config.cache)

    self.num_train_examples = ds_builder.info.splits['train'].num_examples
    self.num_eval_examples = ds_builder.info.splits['validation'].num_examples
    self.steps_per_epoch = self.num_train_examples // config.batch_size
    return ds

  def build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  def sync_batch_stats(self, state):
    """Sync the batch statistics across replicas."""
    # An axis_name is passed to pmap which can then be used by pmean.
    # In this case each device has its own version of the batch statistics and
    # we average them.
    avg = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
    new_model_state = state.model_state.copy({
      'batch_stats': avg(state.model_state['batch_stats'])})
    return state.replace(model_state=new_model_state)

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError('This should not happen, workload.init_model_fn() '
                       'should be called before workload.param_shapes!')
    return self._param_shapes

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
    return 60

  @property
  def max_allowed_eval_time_sec(self):
    return 20

  @property
  def eval_period_time_sec(self):
    return 30

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

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

  def create_model(self, *, model_cls, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
      if platform == 'tpu':
        model_dtype = jnp.bfloat16
      else:
        model_dtype = jnp.float16
    else:
      model_dtype = jnp.float32
    return model_cls(num_classes=config.num_classes,
                     dtype=model_dtype,
                     **kwargs)

  def initialized(self, key, image_size, model):
    input_shape = (1, image_size, image_size, 3)
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
    model_cls = getattr(models, config.model)
    model = self.create_model(
      model_cls=model_cls, half_precision=config.half_precision)
    params, model_state = self.initialized(rng, config.image_size, model)
    self._param_shapes = jax.tree_map(
      lambda x: spec.ShapeTuple(x.shape),
      params)
    learning_rate_fn = submission.create_learning_rate_fn(
      config,
      self.num_train_examples)

    self.p_train_step = jax.pmap(
      functools.partial(
        submission.train_step,
        model.apply,
        learning_rate_fn=learning_rate_fn,
        loss_fn=self.loss_fn,
        loss_type=self.loss_type,
        model_fn=self.model_fn,
        compute_metrics=self.compute_metrics),
      axis_name='batch')

    self.p_eval_step = jax.pmap(
      functools.partial(
        submission.eval_step,
        model.apply,
        model_fn=self.model_fn,
        compute_metrics=self.compute_metrics),
      axis_name='batch')

    state = TrainState(
      step=0,
      optimizer=None,
      model_state=model_state)
    return params, state

  def model_fn(
      self,
      params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState,
      mode: spec.ForwardPassMode,
      # rng: spec.RandomState, # TODO: Question— I'm not sure how to pass rng in
      # correctly when using pmap
      update_batch_norm: bool,
      mutable: bool, # TODO: Question— Is this redundant to param
      # "update_batch_norm"?
      apply_fn: Callable) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    variables = {'params': params, **model_state.model_state}
    train = mode == spec.ForwardPassMode.TRAIN
    if mutable:
      logits, new_model_state = apply_fn(
        variables,
        jax.numpy.squeeze(augmented_and_preprocessed_input_batch['image']),
        train=train,
        mutable=mutable)
      return logits, new_model_state
    else:
      logits = apply_fn(
        variables,
        jax.numpy.squeeze(augmented_and_preprocessed_input_batch['image']),
        train=train,
        mutable=mutable)
      return logits, None

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:  # differentiable
    one_hot_targets = common_utils.onehot(label_batch,
                                          num_classes=config.num_classes)
    return -jnp.sum(one_hot_targets * logits_batch) / label_batch.size

  def compute_metrics(self, logits, labels):
    loss = self.loss_fn(labels, logits, self.loss_type)
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
    # sync batch statistics across replicas once per epoch
    model_state = self.sync_batch_stats(model_state)

    step = int(model_state.step[0])
    epoch = step // self.steps_per_epoch
    epoch_metrics = common_utils.get_metrics(self.epoch_metrics)
    summary = jax.tree_map(lambda x: x.mean(), epoch_metrics)
    logging.info('train epoch: %d, step: %d, loss: %.4f, accuracy: %.2f',
                  epoch, step, summary['loss'], summary['accuracy'] * 100)
    self.epoch_metrics = []
    eval_metrics = []

    data_rng, model_rng = jax.random.split(rng, 2)
    eval_batch_size = config.batch_size
    num_batches = self.num_eval_examples // eval_batch_size
    if self._eval_ds is None:
      self._eval_ds = self._build_dataset(
        data_rng, split='test', batch_size=eval_batch_size, data_dir=data_dir)
    eval_iter = iter(self._eval_ds)
    total_accuracy = 0.
    eval_step = 1
    start_time = time.time()
    accumulated_eval_time = 0
    for batch in eval_iter:
      metrics = self.p_eval_step(model_state, batch)
      eval_metrics.append(metrics)
      total_accuracy += jnp.mean(metrics['accuracy'])
      eval_step += 1
      # Check if eval time is over limit
      current_time = time.time()
      accumulated_eval_time += current_time - start_time
      is_time_remaining = (
        accumulated_eval_time < self.max_allowed_eval_time_sec)
      if not is_time_remaining:
        num_batches = eval_step
        break

    eval_metrics = common_utils.get_metrics(eval_metrics)
    summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                  epoch, summary['loss'], summary['accuracy'] * 100)
    return float(total_accuracy / num_batches)
