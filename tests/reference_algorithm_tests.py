"""Test that each reference submission can run a train and eval step.

This is a brief test that runs the for the workload and reference submission
code for one train and one eval step for all workloads, without the real data
iterator because it is not realistic to have all datasets available at testing
time. For end-to-end tests of submission_runner.py see
submission_runner_test.py.

Assumes that each reference submission is using the external tuning ruleset and
that it is defined in:
# pylint: disable=line-too-long
"reference_algorithms/target_setting_algorithms/{workload}/{workload}_{framework}/submission.py"
"reference_algorithms/target_setting_algorithms/{workload}/tuning_search_space.json".

python3 tests/reference_algorithm_tests.py \
    --workload=criteo1tb \
    --framework=jax \
    --global_batch_size=16 \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_adamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/criteo1tb/tuning_search_space.json
"""

import copy
import functools
import importlib
import json
import os
import pickle

from absl import flags
from absl import logging
from absl.testing import absltest
import flax
from flax import jax_utils
from flax.core.frozen_dict import FrozenDict
import jax
from jraph import GraphsTuple
import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency import halton
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.workloads import workloads
from algorithmic_efficiency.workloads.ogbg import \
    input_pipeline as ogbg_input_pipeline
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.workload import \
    _graph_map
import submission_runner
from tests.modeldiffs import diff as diff_utils

flags.DEFINE_integer(
    'global_batch_size',
    -1,
    ('Global Batch size to use when running an individual workload. Otherwise '
     'a per-device batch size of 2 is used.'))
flags.DEFINE_integer('num_train_steps', 1, 'Number of steps to train.')
flags.DEFINE_boolean('use_fake_input_queue', True, 'Use fake data examples.')
flags.DEFINE_string('log_file', '/tmp/log.pkl', 'The log file')
flags.DEFINE_boolean(
    'all',
    False,
    'Run all workloads instead of using --workload and --framework.')
flags.DEFINE_boolean('identical',
                     False,
                     'Run jax and pytorch with identical weights.')
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, PYTORCH_DEVICE, N_GPUS = pytorch_utils.pytorch_setup()
tf.config.set_visible_devices([], 'GPU')
_EXPECTED_METRIC_NAMES = {
    'cifar': ['train/loss', 'validation/loss', 'test/accuracy'],
    'criteo1tb': ['train/loss', 'validation/loss'],
    'criteo1tb_test': ['train/loss', 'validation/loss'],
    'fastmri': ['train/ssim', 'validation/ssim'],
    'imagenet_resnet': ['train/accuracy', 'validation/accuracy'],
    'imagenet_vit': ['train/accuracy', 'validation/accuracy'],
    'librispeech_conformer': ['train/wer', 'validation/wer', 'train/ctc_loss'],
    'librispeech_deepspeech': ['train/wer', 'validation/wer', 'train/ctc_loss'],
    'mnist': ['train/loss', 'validation/accuracy', 'test/accuracy'],
    'ogbg': [
        'train/accuracy', 'validation/loss', 'test/mean_average_precision'
    ],
    'wmt': ['train/bleu', 'validation/loss', 'validation/accuracy'],
}


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
  return jax.tree_map(
      lambda a: torch.as_tensor(a, device=PYTORCH_DEVICE).view(-1, a.shape[-1])
      if len(a.shape) == 3 else torch.as_tensor(a, device=PYTORCH_DEVICE).view(
          -1),
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


class _FakeMetricsLogger:

  def __init__(self):
    self.filename = FLAGS.log_file
    self.scalars = []
    self.eval_results = []

  def append_scalar_metrics(self, scalars, step):
    if USE_PYTORCH_DDP:
      for k in sorted(scalars):
        scalars[k] = torch.as_tensor([scalars[k]], device=PYTORCH_DEVICE)
        dist.all_reduce(scalars[k], op=dist.ReduceOp.AVG)
        scalars[k] = scalars[k].item()
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


class _FakeMetricsBundle:

  def gather_from_model_output(self, *args, **kwargs):
    del args
    del kwargs
    return _FakeMetricsCollection()


def _make_one_batch_workload(workload_class,
                             workload_name,
                             framework,
                             global_batch_size,
                             use_fake_input_queue,
                             n_gpus):

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
      if not (FLAGS.identical and
              os.path.exists(f'tests/modeldiffs/{workload_name}/compare.py')):
        return super().init_model_fn(
            rng, dropout_rate=dropout_rate, aux_dropout_rate=aux_dropout_rate)
      if framework == 'jax':
        compare_module = importlib.import_module(
            f'tests.modeldiffs.{workload_name}.compare')
        jax_params, model_state, _ = diff_utils.torch2jax(
          jax_workload=super(),
          pytorch_workload=compare_module.PyTorchWorkload(**self.init_kwargs),
          key_transform=compare_module.key_transform,
          sd_transform=compare_module.sd_transform)
        return (FrozenDict(**jax_utils.replicate(jax_params)),
                FrozenDict(**jax_utils.replicate(model_state))
                if model_state is not None else model_state)
      return super().init_model_fn([0], dropout_rate=0.0, aux_dropout_rate=0.0)

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
        batch_shape = (n_gpus, global_batch_size // n_gpus)
      else:
        batch_shape = (global_batch_size,)

      if workload_name == 'cifar':
        if framework == 'jax':
          data_shape = (32, 32, 3)
        else:
          data_shape = (3, 32, 32)
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=data_shape, num_classes=10)
      elif workload_name == 'criteo1tb' or workload_name == 'criteo1tb_test':
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
          num_dims = len(fake_batch['inputs'].shape)
          fake_batch['inputs'] = fake_batch['inputs'].transpose(
              (*range(num_dims - 3), num_dims - 1, num_dims - 3, num_dims - 2))
      elif 'librispeech' in workload_name:
        rate = 16000
        l = None
        while l is None or l.shape[-1] < 320000:
          duration = 0.5
          freq = 2**(np.random.rand(*batch_shape, 1) * 13)
          wav = np.sin(2 * np.pi * freq * np.arange(rate * duration) / rate)
          if l is None:
            l = wav
          else:
            l = np.concatenate([l, wav], axis=-1)
        inputs = l
        targets = np.random.randint(low=1, high=1024, size=(*batch_shape, 256))
        tgt_pad = np.arange(0, 256)[tuple([None] * len(batch_shape))]
        tgt_lengths = np.random.randint(
            low=100, high=256, size=(*batch_shape, 1))
        tgt_pad = 1 * (tgt_pad > tgt_lengths)
        fake_batch = {
            'inputs': (inputs, np.zeros_like(inputs)),
            'targets': (targets, tgt_pad),
        }
      elif workload_name == 'mnist':
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=(28, 28, 1), num_classes=10)
      elif workload_name == 'ogbg':
        tf.random.set_seed(5)

        def _fake_iter():
          while True:
            fake_batch = {
                'num_nodes':
                    tf.ones((1,), dtype=tf.int64),
                'edge_index':
                    tf.ones((1, 2), dtype=tf.int64),
                'node_feat':
                    tf.random.normal((1, 9)),
                'edge_feat':
                    tf.random.normal((1, 3)),
                'labels':
                    tf.cast(
                        tf.random.uniform((self._num_outputs,),
                                          minval=0,
                                          maxval=2,
                                          dtype=tf.int32),
                        tf.float32),
            }
            yield fake_batch

        fake_batch_iter = ogbg_input_pipeline._get_batch_iterator(
            _fake_iter(), global_batch_size)
        fake_batch = next(fake_batch_iter)  # pylint: disable=stop-iteration-return
        if framework == 'pytorch':
          fake_batch['inputs'] = _graph_map(_pytorch_map, fake_batch['inputs'])
          fake_batch['targets'] = _pytorch_map(fake_batch['targets'])
          fake_batch['weights'] = _pytorch_map(fake_batch['weights'])
      elif workload_name == 'wmt':
        max_len = 256
        fake_batch = {
            'inputs':
                np.random.randint(
                    low=0, high=32000, size=(*batch_shape, max_len)),
            'targets':
                np.random.randint(
                    low=0, high=32000, size=(*batch_shape, max_len)),
            'weights':
                np.random.randint(low=0, high=2, size=(*batch_shape, max_len)),
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
      else:
        raise ValueError(
            'Workload {} does not have a fake batch defined, you '
            'can add it or use --use_fake_input_queue=false.'.format(
                workload_name))

      if framework == 'pytorch':

        def to_device(k, v):
          dtype = (
              torch.long if (k == 'targets' and workload_name != 'fastmri') else
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
      # We set the number of examples to the batch size for all splits, so only
      # yield two batches, one for each call to eval_model().
      num_batches = 2
      # For WMT we also iterate through the eval iters a second time to complute
      # the BLEU score.
      if workload_name == 'wmt':
        num_batches *= 2

      def _data_gen():
        for _ in range(num_batches * FLAGS.num_train_steps):
          yield fake_batch

      return _data_gen()

    def eval_model(self, *args, **kwargs):
      eval_result = super().eval_model(*args, **kwargs)
      self.metrics_logger.append_eval_metrics(eval_result)
      return eval_result

  return _OneEvalBatchWorkload()


def _test_submission(workload_name,
                     framework,
                     submission_path,
                     search_space_path,
                     data_dir,
                     use_fake_input_queue,
                     n_gpus):
  logging.info(f'========= Testing {workload_name} in {framework}.')
  FLAGS.framework = framework
  workload_metadata = copy.deepcopy(submission_runner.WORKLOADS[workload_name])
  workload_metadata['workload_path'] = os.path.join(
      submission_runner.BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + '_' + framework,
      'workload.py')
  workload_class = workloads.import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'],
      return_class=True)
  print(f'Workload class for {workload_name} is {workload_class}')

  submission_module_path = workloads.convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  if FLAGS.all:
    if FLAGS.global_batch_size > 0:
      raise ValueError('Cannot set --global_batch_size and --all.')
    global_batch_size = 2 * n_gpus
  else:
    global_batch_size = FLAGS.global_batch_size
    if FLAGS.global_batch_size < 0:
      raise ValueError('Must set --global_batch_size.')
  workload = _make_one_batch_workload(workload_class,
                                      workload_name,
                                      framework,
                                      global_batch_size,
                                      use_fake_input_queue,
                                      n_gpus)

  # Get a sample hyperparameter setting.
  hyperparameters = {}
  if search_space_path != 'None':
    with open(search_space_path, 'r', encoding='UTF-8') as search_space_file:
      hyperparameters = halton.generate_search(
          json.load(search_space_file), num_trials=1)[0]

  rng = prng.PRNGKey(0)
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)
  input_queue = workload._build_input_queue(
      data_rng, 'train', data_dir=data_dir, global_batch_size=global_batch_size)
  model_params, model_state = workload.init_model_fn(model_init_rng)
  optimizer_state = init_optimizer_state(workload,
                                         model_params,
                                         model_state,
                                         hyperparameters,
                                         opt_init_rng)

  if USE_PYTORCH_DDP:
    torch.cuda.empty_cache()
    dist.barrier()
  for global_step in range(FLAGS.num_train_steps):
    step_rng = prng.fold_in(rng, global_step)
    data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)
    batch = data_selection(workload,
                           input_queue,
                           optimizer_state,
                           model_params,
                           model_state,
                           hyperparameters,
                           global_step,
                           data_select_rng)
    optimizer_state, model_params, model_state = update_params(
        workload=workload,
        current_param_container=model_params,
        current_params_types=workload.model_params_types,
        model_state=model_state,
        hyperparameters=hyperparameters,
        batch=batch,
        loss_type=workload.loss_type,
        optimizer_state=optimizer_state,
        train_state={},
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
        global_step=global_step)
  _ = workload.eval_model(
      global_batch_size,
      model_params,
      model_state,
      eval_rng,
      data_dir,
      imagenet_v2_data_dir=None,
      global_step=global_step)
  return eval_result


def _make_paths(repo_location, framework, workload_name):
  if '_' in workload_name:
    dataset_name = workload_name.split('_')[0]
  else:
    dataset_name = workload_name
  workload_dir = (
      f'{repo_location}/reference_algorithms/target_setting_algorithms/'
      f'{workload_name}')
  search_space_path = f'{workload_dir}/tuning_search_space.json'
  submission_path = (f'reference_algorithms/target_setting_algorithms/'
                     f'{workload_name}/{dataset_name}_{framework}/'
                     'submission.py')
  full_submission_path = f'{repo_location}/{submission_path}'
  if not os.path.exists(full_submission_path):
    return None, None
  return search_space_path, submission_path


class ReferenceSubmissionTest(absltest.TestCase):
  """Tests for reference submissions."""

  def _assert_eval_result(self, workload_name, eval_result):
    expected_names = _EXPECTED_METRIC_NAMES[workload_name]
    actual_names = list(eval_result.keys())
    for expected_name in expected_names:
      self.assertIn(expected_name, actual_names)

  def test_submission(self):
    profiler = PassThroughProfiler()
    # Example: /home/znado/algorithmic-efficiency/tests
    self_location = os.path.dirname(os.path.realpath(__file__))
    # Example: /home/znado/algorithmic-efficiency
    repo_location = '/'.join(self_location.split('/')[:-1])
    if FLAGS.tuning_ruleset != 'external':
      raise ValueError('--tuning_ruleset must be set to "external".')
    if FLAGS.all:
      if FLAGS.submission_path:
        raise ValueError('Cannot set --submission_path and --all.')
      if FLAGS.tuning_search_space:
        raise ValueError('Cannot set --tuning_search_space and --all.')
      references_dir = (
          f'{repo_location}/reference_algorithms/target_setting_algorithms')
      for workload_name in os.listdir(references_dir):
        for framework in ['jax', 'pytorch']:
          if framework == 'pytorch':
            pytorch_utils.pytorch_init(USE_PYTORCH_DDP, RANK, profiler)
          # First jax operation has to be called after pytorch_init.
          n_gpus = max(N_GPUS, jax.local_device_count())
          search_space_path, submission_path = _make_paths(
              repo_location, framework, workload_name)
          if search_space_path is None:
            continue
          eval_result = _test_submission(
              workload_name,
              framework,
              submission_path,
              search_space_path,
              data_dir=FLAGS.data_dir,
              use_fake_input_queue=FLAGS.use_fake_input_queue,
              n_gpus=n_gpus)
          self._assert_eval_result(workload_name, eval_result)
    else:
      framework = FLAGS.framework
      if framework == 'pytorch':
        pytorch_utils.pytorch_init(USE_PYTORCH_DDP, RANK, profiler)
      # First jax operation has to be called after pytorch_init.
      n_gpus = max(N_GPUS, jax.local_device_count())
      workload_name = FLAGS.workload
      if FLAGS.submission_path and FLAGS.tuning_search_space:
        search_space_path = FLAGS.tuning_search_space
        submission_path = FLAGS.submission_path
      else:
        search_space_path, submission_path = _make_paths(
            repo_location, framework, workload_name)
      eval_result = _test_submission(
          workload_name,
          framework,
          submission_path,
          search_space_path,
          data_dir=FLAGS.data_dir,
          use_fake_input_queue=FLAGS.use_fake_input_queue,
          n_gpus=n_gpus)
      self._assert_eval_result(workload_name, eval_result)

    if USE_PYTORCH_DDP:
      # cleanup
      dist.destroy_process_group()


if __name__ == '__main__':
  absltest.main()
