import copy
import functools
import importlib
import os
import pickle

from absl import app
from absl import flags
from absl import logging
import flax
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict
import jax
import jax.numpy as jnp
from jraph import GraphsTuple
import numpy as np
import optax
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.workloads.ogbg import \
    input_pipeline as ogbg_input_pipeline
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.workload import \
    _graph_map
import submission_runner
from tests.modeldiffs import diff as diff_utils

flags.DEFINE_integer(
    'global_batch_size',
    16,
    ('Global Batch size to use when running an individual workload. Otherwise '
     'a per-device batch size of 2 is used.'))
flags.DEFINE_boolean('use_fake_input_queue', True, 'Use fake data examples.')
flags.DEFINE_string('log_file', '/tmp/log.pkl', 'The log file')
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, PYTORCH_DEVICE, N_GPUS = pytorch_utils.pytorch_setup()
N_GPUS = max(N_GPUS, jax.local_device_count())
tf.config.set_visible_devices([], 'GPU')


def _make_fake_image_batch(batch_shape, data_shape, num_classes):
  examples = np.random.normal(size=(*batch_shape,
                                    *data_shape)).astype(np.float32)
  labels = np.random.randint(0, num_classes, size=batch_shape)
  masks = np.ones(batch_shape, dtype=np.float32)
  return {'inputs': examples, 'targets': labels, 'weights': masks}


def _pytorch_map(inputs):
  if USE_PYTORCH_DDP:
    return jax.tree_map(
        lambda a: torch.as_tensor(a[RANK], device=PYTORCH_DEVICE), inputs)
  return jax.tree_map(lambda a: torch.as_tensor(a, device=PYTORCH_DEVICE),
                      inputs)


class _FakeTokenizer:

  def detokenize(self, *args):
    del args
    return tf.constant('this is a fake sequence?')


@flax.struct.dataclass
class _FakeMetricsCollection:

  def merge(self, *args):
    del args
    return self

  def compute(self):
    return {
        'wer': 0.0,
        'ctc_loss': 0.0,
    }

  def unreplicate(self):
    return self


class _FakeMetricsBundle:

  def gather_from_model_output(self, *args, **kwargs):
    del args
    del kwargs
    return _FakeMetricsCollection()


class _FakeMetricsLogger:

  def __init__(self):
    self.filename = FLAGS.log_file
    self.scalars = []
    self.eval_results = []

  def append_scalar_metrics(self, scalars, step):
    if USE_PYTORCH_DDP:
      for k in sorted(scalars):
        scalars[k] = torch.FloatTensor([scalars[k]]).to(PYTORCH_DEVICE)
        dist.all_reduce(scalars[k])
        scalars[k] = scalars[k].item() / N_GPUS
    if RANK == 0:
      self.scalars.append(scalars)
      self.save()

  def append_eval_metrics(self, result):
    if RANK == 0:
      self.eval_results.append(result)
      self.save()

  def save(self):
    with open(self.filename, 'wb') as f:
      pickle.dump({'scalars': self.scalars, 'eval_results': self.eval_results},
                  f)


def jax_init_optimizer(workload: spec.Workload,
                       model_params: spec.ParameterContainer,
                       model_state: spec.ModelAuxiliaryState,
                       hyperparameters: spec.Hyperparameters,
                       rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)

  opt_init_fn, opt_update_fn = optax.chain(
      optax.add_decayed_weights(0),
      optax.sgd(learning_rate=0.001))
  optimizer_state = opt_init_fn(params_zeros_like)

  return jax_utils.replicate(optimizer_state), opt_update_fn


def pytorch_init_optimizer(workload: spec.Workload,
                           model_params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           hyperparameters: spec.Hyperparameters,
                           rng: spec.RandomState) -> spec.OptimizerState:
  """Creates an AdamW optimizer and a learning rate schedule."""
  del model_state
  del rng

  optimizer_state = {
      'optimizer':
          torch.optim.SGD(model_params.parameters(), lr=0.001, weight_decay=0)
  }

  return optimizer_state


def _make_one_batch_workload(workload_class,
                             workload_name,
                             framework,
                             global_batch_size,
                             use_fake_input_queue):

  class _OneEvalBatchWorkload(workload_class):

    def __init__(self):
      kwargs = {}
      if 'librispeech' in workload_name:
        kwargs['use_specaug'] = False
      self.init_kwargs = kwargs
      super().__init__(**kwargs)
      self.summary_writer = None
      self.metrics_logger = _FakeMetricsLogger()
      if 'librispeech' in workload_name:
        self.tokenizer = _FakeTokenizer()

    def init_model_fn(self, rng, dropout_rate=None, aux_dropout_rate=None):
      # pylint: disable=line-too-long
      if framework == 'jax':
        compare_module = importlib.import_module(
            f"tests.modeldiffs.{workload_name}.compare")
        jax_params, model_state, _ = diff_utils.torch2jax(
          jax_workload=super(),
          pytorch_workload=compare_module.PytWorkload(**self.init_kwargs),
          key_transform=compare_module.key_transform,
          sd_transform=compare_module.sd_transform)
        return FrozenDict(**jax_utils.replicate(jax_params)), (FrozenDict(**jax_utils.replicate(model_state)) if model_state is not None else model_state)
      else:
        return super().init_model_fn([0],
                                     dropout_rate=0.0,
                                     aux_dropout_rate=0.0)

    @property
    def num_eval_train_examples(self):
      return global_batch_size

    @property
    def num_validation_examples(self):
      return global_batch_size

    @property
    def num_test_examples(self):
      super_num_test = super().num_test_examples
      if super_num_test is not None:
        return global_batch_size
      return None

    def _build_input_queue(self, *args, **kwargs):
      if not use_fake_input_queue:
        return super()._build_input_queue(*args, **kwargs)
      del args
      del kwargs

      np.random.seed(42)
      if framework == 'jax' or USE_PYTORCH_DDP:
        batch_shape = (N_GPUS, global_batch_size // N_GPUS)
      else:
        batch_shape = (global_batch_size,)

      if workload_name == 'criteo1tb':
        targets = np.ones(batch_shape)
        targets[0] = 0
        fake_batch = {
            'inputs': np.ones((*batch_shape, 13 + 26)),
            'targets': targets,
            'weights': np.ones(batch_shape),
        }
      elif workload_name in ['imagenet_resnet', 'imagenet_vit']:
        data_shape = (224, 224, 3)
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=data_shape, num_classes=1000)
        if framework == 'pytorch':
          fake_batch['inputs'] = fake_batch['inputs'].transpose((0, 1, 4, 2, 3))
      elif 'librispeech' in workload_name:
        inputs = np.random.normal(size=(*batch_shape, 320000))
        targets = np.random.randint(low=1, high=1024, size=(*batch_shape, 256))
        fake_batch = {
            'inputs': (inputs, np.zeros_like(inputs)),
            'targets': (targets, np.zeros_like(targets)),
        }
        if RANK == 0:
          np.save(f"{framework}.npz", fake_batch)
      elif workload_name == 'ogbg':
        tf.random.set_seed(5)

        def _fake_iter():
          while True:
            fake_batch = dict(
                num_nodes=tf.ones((1,), dtype=tf.int64),
                edge_index=tf.ones((1, 2), dtype=tf.int64),
                node_feat=tf.random.normal((1, 9)),
                edge_feat=tf.random.normal((1, 3)),
                labels=tf.ones((self._num_outputs,)))
            yield fake_batch

        fake_batch_iter = ogbg_input_pipeline._get_batch_iterator(
            _fake_iter(), global_batch_size)
        fake_batch = next(fake_batch_iter)  # pylint: disable=stop-iteration-return
        if framework == 'pytorch':
          fake_batch['inputs'] = _graph_map(_pytorch_map, fake_batch['inputs'])
      elif workload_name == 'wmt':
        max_len = 256
        fake_batch = {
            'inputs':
                np.random.randint(
                    low=0, high=32000, size=(*batch_shape, max_len)),
            'targets':
                np.random.randint(
                    low=0, high=32000, size=(*batch_shape, max_len)),
        }
        self._tokenizer = _FakeTokenizer()
      elif workload_name == 'fastmri':
        data_shape = (320, 320)
        fake_batch = {
            'inputs':
                _make_fake_image_batch(
                    batch_shape, data_shape=data_shape, num_classes=1000)
                ['inputs'],
            'targets':
                _make_fake_image_batch(
                    batch_shape, data_shape=data_shape, num_classes=1000)
                ['inputs'],
            'mean':
                np.zeros(batch_shape),
            'std':
                np.ones(batch_shape),
            'volume_max':
                np.zeros(batch_shape),
            'weights':
                np.ones(batch_shape),
        }
        if RANK == 0:
          np.save(f"{framework}.npz", fake_batch)
      else:
        raise ValueError(
            'Workload {} does not have a fake batch defined, you '
            'can add it or use --use_fake_input_queue=false.'.format(
                workload_name))

      if framework == 'pytorch':

        def to_device(k, v):
          dtype = (
              torch.long if k == 'targets' else
              torch.bool if k == 'weights' else torch.float)
          if USE_PYTORCH_DDP:
            v = v[RANK]
          return torch.as_tensor(v, device=PYTORCH_DEVICE, dtype=dtype)

        new_fake_batch = {}
        for k, v in fake_batch.items():
          if isinstance(v, np.ndarray):
            new_fake_batch[k] = to_device(k, v)
          elif isinstance(v, tuple) and not isinstance(v, GraphsTuple):
            new_fake_batch[k] = tuple(map(functools.partial(to_device, k), v))
          else:
            new_fake_batch[k] = v
        fake_batch = new_fake_batch

      while True:
        yield fake_batch

    def eval_model(self, *args, **kwargs):
      eval_result = super().eval_model(*args, **kwargs)
      self.metrics_logger.append_eval_metrics(eval_result)
      return eval_result

  return _OneEvalBatchWorkload()


def _test_submission(
    workload_name,
    framework,
    submission_path,  #  search_space_path,
    data_dir,
    use_fake_input_queue):
  logging.info(f'========= Testing {workload_name} in {framework}.')
  FLAGS.framework = framework
  workload_metadata = copy.deepcopy(submission_runner.WORKLOADS[workload_name])
  workload_metadata['workload_path'] = os.path.join(
      submission_runner.BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + '_' + framework,
      'workload.py')
  workload_class = submission_runner.import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'],
      return_class=True)

  submission_module_path = submission_runner.convert_filepath_to_module(
      submission_path)
  submission_module = importlib.import_module(submission_module_path)
  if framework == 'pytorch':
    init_optimizer_state = pytorch_init_optimizer
  else:
    init_optimizer_state = jax_init_optimizer
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  global_batch_size = FLAGS.global_batch_size
  workload = _make_one_batch_workload(workload_class,
                                      workload_name,
                                      framework,
                                      global_batch_size,
                                      use_fake_input_queue)

  hyperparameters = {}

  rng = prng.PRNGKey(200)
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)
  input_queue = workload._build_input_queue(
      data_rng, 'train', data_dir=data_dir, global_batch_size=global_batch_size)
  model_params, model_state = workload.init_model_fn(model_init_rng)
  optimizer_state = init_optimizer_state(workload,
                                         model_params,
                                         model_state,
                                         hyperparameters,
                                         opt_init_rng)

  global_step = 0
  data_select_rng, update_rng, eval_rng = prng.split(rng, 3)
  batch = data_selection(workload,
                         input_queue,
                         optimizer_state,
                         model_params,
                         model_state,
                         hyperparameters,
                         global_step,
                         data_select_rng)
  if USE_PYTORCH_DDP:
    torch.cuda.empty_cache()
    dist.barrier()
  for _ in range(10):
    optimizer_state, model_params, model_state = update_params(
        workload=workload,
        current_param_container=model_params,
        current_params_types=workload.model_params_types,
        model_state=model_state,
        hyperparameters=hyperparameters,
        batch=batch,
        loss_type=workload.loss_type,
        optimizer_state=optimizer_state,
        eval_results=[],
        global_step=global_step,
        rng=update_rng)

    eval_result = workload.eval_model(
        global_batch_size,
        model_params,
        model_state,
        eval_rng,
        data_dir,
        imagenet_v2_data_dir=None,
        global_step=0)

  return eval_result


def main(_):
  profiler = PassThroughProfiler()
  framework = FLAGS.framework
  if framework == 'pytorch':
    pytorch_utils.pytorch_init(USE_PYTORCH_DDP, RANK, profiler)
  workload_name = FLAGS.workload

  _test_submission(
      workload_name,
      framework,
      submission_path='reference_algorithms' +
      f'/target_setting_algorithms/{framework}_adamw.py',
      data_dir=FLAGS.data_dir,
      use_fake_input_queue=FLAGS.use_fake_input_queue)

  if USE_PYTORCH_DDP:
    # cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)
