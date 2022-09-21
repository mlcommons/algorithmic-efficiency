"""Test that each reference submission can run a train and eval step.

This is a brief test that runs the for the workload and reference submission
code for one train and one eval step for all workloads, without the real data
iterator because it is not realistic to have all datasets available at testing
time. For end-to-end tests of submission_runner.py see
submission_runner_test.py.

Assumes that each reference submission is using the external tuning ruleset and
that it is defined in:
"reference_submissions/{workload}/{workload}_{framework}/submission.py"
"reference_submissions/{workload}/tuning_search_space.json".
"""
import copy
import importlib
import json
import os

from absl import flags
from absl import logging
from absl.testing import absltest
import flax
import jax
import jraph
import numpy as np
import tensorflow as tf
import torch

from algorithmic_efficiency import halton
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency.workloads.ogbg import \
    input_pipeline as ogbg_input_pipeline
import submission_runner

flags.DEFINE_integer(
    'global_batch_size',
    -1,
    ('Global Batch size to use when running an individual workload. Otherwise '
     'a per-device batch size of 2 is used.'))
flags.DEFINE_boolean('use_fake_input_queue', True, 'Use fake data examples.')
flags.DEFINE_boolean(
    'run_all',
    False,
    'Run all workloads instead of using --workload and --framework.')
FLAGS = flags.FLAGS
PYTORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

_EXPECTED_METRIC_NAMES = {
    'cifar': ['train/loss', 'validation/loss', 'test/accuracy'],
    'criteo1tb': ['train/loss', 'validation/loss'],
    'fastmri': ['train/ssim', 'validation/ssim'],
    'imagenet_resnet': ['train/accuracy', 'validation/accuracy'],
    'imagenet_vit': ['train/accuracy', 'validation/accuracy'],
    'librispeech_conformer': [
        'train/word_error_rate',
        'validation/word_error_rate',
        'train/ctc_loss',
    ],
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
  masks = np.ones((*batch_shape, *data_shape), dtype=np.float32)
  return {'inputs': examples, 'targets': labels, 'weights': masks}


def _tile(x):
  # Go from n_node.shape = (1,) -> (local_device_count, 1).
  return np.tile(
      np.expand_dims(x, axis=0),
      (jax.local_device_count(), *[1] * len(x.shape)))


def _graph_tuple_to_device(graph_tuple):
  return jraph.GraphsTuple(
      n_node=torch.from_numpy(graph_tuple.n_node).to(
          PYTORCH_DEVICE, dtype=torch.long),
      n_edge=torch.from_numpy(graph_tuple.n_edge).to(
          PYTORCH_DEVICE, dtype=torch.long),
      nodes=torch.from_numpy(graph_tuple.nodes).to(
          PYTORCH_DEVICE, dtype=torch.float),
      edges=torch.from_numpy(graph_tuple.edges).to(
          PYTORCH_DEVICE, dtype=torch.float),
      globals=graph_tuple.globals,
      senders=torch.from_numpy(graph_tuple.senders).to(
          PYTORCH_DEVICE, dtype=torch.long),
      receivers=torch.from_numpy(graph_tuple.receivers).to(
          PYTORCH_DEVICE, dtype=torch.long))


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
        'word_error_rate': 0.0,
        'ctc_loss': 0.0,
    }

  def unreplicate(self):
    return self


class _FakeMetricsBundle:

  def gather_from_model_output(self, *args, **kwargs):
    del args
    del kwargs
    return _FakeMetricsCollection()


def _make_one_batch_workload(workload_class,
                             workload_name,
                             framework,
                             global_batch_size,
                             use_fake_input_queue):

  class _OneEvalBatchWorkload(workload_class):

    def __init__(self):
      super().__init__()
      self.summary_writer = None
      if 'librispeech' in workload_name:
        self.metrics_bundle = _FakeMetricsBundle()

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

    def build_input_queue(self, *args, **kwargs):
      if not use_fake_input_queue:
        return super().build_input_queue(*args, **kwargs)
      del args
      del kwargs

      np.random.seed(42)
      if framework == 'jax':
        num_devices = jax.local_device_count()
        batch_shape = (num_devices, global_batch_size // num_devices)
      else:
        batch_shape = (global_batch_size,)

      if workload_name == 'cifar':
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=(32, 32, 3), num_classes=10)
      elif workload_name == 'criteo1tb':
        targets = np.ones(batch_shape)
        targets[0] = 0
        fake_batch = {
            'inputs': np.ones((*batch_shape, 13 + 26)),
            'targets': targets,
            'weights': np.ones(batch_shape),
        }
      elif workload_name in ['imagenet_resnet', 'imagenet_vit']:
        if framework == 'jax':
          data_shape = (224, 224, 3)
        else:
          data_shape = (3, 224, 224)
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=data_shape, num_classes=1000)
      elif 'librispeech' in workload_name:
        inputs = np.random.normal((*batch_shape, 320000))
        targets = np.random.normal((*batch_shape, 256))
        fake_batch = {
            'inputs': (inputs, inputs),
            'targets': (targets, targets),
        }
      elif workload_name == 'mnist':
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=(28, 28, 1), num_classes=10)
      elif workload_name == 'ogbg':
        # TODO(znado): fix the memory usage of this for pytorch.
        fake_batch = {
            'edge_feat': tf.ones((1,)),
            'node_feat': tf.ones((1,)),
            'edge_index': tf.ones((1, 2)),
            'labels': tf.ones((1,)),
            'num_nodes': tf.ones((1,)),
        }

        def _fake_iter():
          while True:
            yield fake_batch

        fake_batch_iter = ogbg_input_pipeline._get_batch_iterator(
            _fake_iter(), global_batch_size)
        fake_batch = next(fake_batch_iter)  # pylint: disable=stop-iteration-return
        if framework == 'pytorch':
          fake_batch['inputs'] = _graph_tuple_to_device(fake_batch['inputs'])
      elif workload_name == 'wmt':
        max_len = 256
        fake_batch = {
            'inputs': np.ones((*batch_shape, max_len)),
            'targets': np.ones((*batch_shape, max_len), dtype=np.int64),
        }
        self._tokenizer = _FakeTokenizer()
      elif workload_name == 'fastmri':
        fake_batch = {
            'inputs': np.zeros((*batch_shape, 320, 320)),
            'targets': np.zeros((*batch_shape, 320, 320)),
            'mean': np.zeros(batch_shape),
            'std': np.ones(batch_shape),
            'volume_max': np.zeros(batch_shape),
            'weights': np.ones(batch_shape),
        }
      else:
        raise ValueError(
            'Workload {} does not have a fake batch defined, you '
            'can add it or use --use_fake_input_queue=false.'.format(
                workload_name))

      if framework == 'pytorch':

        def to_device(k, v):
          dtype = torch.long if k in ['targets', 'weights'] else torch.float
          return torch.from_numpy(v).to(PYTORCH_DEVICE, dtype=dtype)

        new_fake_batch = {}
        for k, v in fake_batch.items():
          if isinstance(v, np.ndarray):
            new_fake_batch[k] = to_device(k, v)
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
      for _ in range(num_batches):
        yield fake_batch

  return _OneEvalBatchWorkload()


def _test_submission(workload_name,
                     framework,
                     submission_path,
                     search_space_path,
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

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  get_batch_size = submission_module.get_batch_size
  global_batch_size = get_batch_size(workload_name)
  if FLAGS.run_all:
    if FLAGS.global_batch_size < 0:
      raise ValueError('Cannot set --batch_size and --run_all.')
    global_batch_size = 2 * jax.local_device_count()
  else:
    global_batch_size = FLAGS.global_batch_size
  workload = _make_one_batch_workload(workload_class,
                                      workload_name,
                                      framework,
                                      global_batch_size,
                                      use_fake_input_queue)

  # Get a sample hyperparameter setting.
  with open(search_space_path, 'r', encoding='UTF-8') as search_space_file:
    hyperparameters = halton.generate_search(
        json.load(search_space_file), num_trials=1)[0]

  rng = prng.PRNGKey(0)
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)
  input_queue = workload.build_input_queue(
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
                         hyperparameters,
                         global_step,
                         data_select_rng)
  _, model_params, model_state = update_params(
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
      global_step=0)
  _ = workload.eval_model(
      global_batch_size,
      model_params,
      model_state,
      eval_rng,
      data_dir,
      global_step=0)
  return eval_result


def _make_paths(repo_location, framework, workload_name):
  if '_' in workload_name:
    dataset_name = workload_name.split('_')[0]
  else:
    dataset_name = workload_name
  workload_dir = f'{repo_location}/reference_submissions/{workload_name}'
  search_space_path = f'{workload_dir}/tuning_search_space.json'
  submission_path = (f'reference_submissions/{workload_name}/'
                     f'{dataset_name}_{framework}/submission.py')
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
    # Example: /home/znado/algorithmic-efficiency/tests
    self_location = os.path.dirname(os.path.realpath(__file__))
    # Example: /home/znado/algorithmic-efficiency
    repo_location = '/'.join(self_location.split('/')[:-1])
    if FLAGS.run_all:
      references_dir = f'{repo_location}/reference_submissions'
      for workload_name in os.listdir(references_dir):
        for framework in ['jax', 'pytorch']:
          search_space_path, submission_path = _make_paths(
              repo_location, framework, workload_name)
          if search_space_path is None:
            continue
          eval_result = _test_submission(
              workload_name,
              framework,
              submission_path,
              search_space_path,
              data_dir=None,
              use_fake_input_queue=FLAGS.use_fake_input_queue)
          self._assert_eval_result(workload_name, eval_result)
    else:
      framework = FLAGS.framework
      workload_name = FLAGS.workload
      search_space_path, submission_path = _make_paths(
          repo_location, framework, workload_name)
      eval_result = _test_submission(
          workload_name,
          framework,
          submission_path,
          search_space_path,
          data_dir=None,
          use_fake_input_queue=FLAGS.use_fake_input_queue)
      self._assert_eval_result(workload_name, eval_result)


if __name__ == '__main__':
  absltest.main()
