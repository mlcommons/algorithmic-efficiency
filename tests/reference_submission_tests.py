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
import jax
import jraph
import numpy as np
import tensorflow as tf
import torch

from algorithmic_efficiency import halton
from algorithmic_efficiency import random_utils as prng
import submission_runner

flags.DEFINE_boolean('use_fake_input_queue', True, 'Use fake data examples.')
FLAGS = flags.FLAGS
PYTORCH_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

_EXPECTED_METRIC_NAMES = {
    'cifar': ['train/loss', 'validation/loss', 'test/accuracy'],
    'criteo1tb': [
        'train/loss', 'train/average_precision', 'validation/auc_roc'
    ],
    'fastmri': ['train/ssim', 'validation/ssim'],
    'imagenet_resnet': ['train/accuracy', 'validation/accuracy'],
    'imagenet_vit': ['train/accuracy', 'validation/accuracy'],
    'librispeech': [
        'train/word_error_rate',
        'validation/word_error_rate',
        'train/word_error_rate',
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


class _FakeTokenizer:

  def detokenize(self, *args):
    del args
    return tf.constant('this is a fake sequence?')


def _make_one_batch_workload(workload_class,
                             workload_name,
                             framework,
                             global_batch_size,
                             use_fake_input_queue):

  class _OneEvalBatchWorkload(workload_class):

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
        batch_shape = (1, global_batch_size)
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
      elif workload_name == 'librispeech':
        fake_batch = {
            'indices': np.ones((8,)),
            'features': np.ones((8, 1593, 161)),
            'transcripts': np.ones((8, 246)),
            'input_lengths': np.ones((8,)),
        }
      elif workload_name == 'mnist':
        fake_batch = _make_fake_image_batch(
            batch_shape, data_shape=(28, 28, 1), num_classes=10)
      elif workload_name == 'ogbg':
        num_classes = 128
        fake_graph = jraph.GraphsTuple(
            n_node=np.asarray([1]),
            n_edge=np.asarray([1]),
            nodes=np.random.normal(size=(1, 9)),
            edges=np.random.normal(size=(1, 3)),
            globals=np.zeros((1, num_classes)),
            senders=np.asarray([0]),
            receivers=np.asarray([0]))
        if framework == 'jax':
          fake_graph = jax.tree_map(lambda x: np.expand_dims(x, axis=0),
                                    fake_graph)
        labels = fake_graph.globals
        fake_graph = fake_graph._replace(globals={})
        fake_batch = {
            'inputs': fake_graph,
            'targets': labels,
            'weights': np.random.normal(size=labels.shape),
        }
      elif workload_name == 'wmt':
        max_len = 256
        fake_batch = {
            'inputs': np.ones((*batch_shape, max_len)),
            'targets': np.ones((*batch_shape, max_len), dtype=np.int64),
        }
        self._tokenizer = _FakeTokenizer()
      else:
        raise ValueError(
            f'Workload {workload_name} does not have a fake batch defined, you '
            'can add it or use --use_fake_input_queue=false.')

      if framework == 'pytorch':
        fake_batch = {
            k: torch.from_numpy(v).to(PYTORCH_DEVICE) for k,
            v in fake_batch.items()
        }
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
  global_batch_size = 2
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
  eval_result = workload.eval_model(global_batch_size,
                                    model_params,
                                    model_state,
                                    eval_rng,
                                    data_dir)
  _ = workload.eval_model(global_batch_size,
                          model_params,
                          model_state,
                          eval_rng,
                          data_dir)
  return eval_result


class ReferenceSubmissionTest(absltest.TestCase):
  """Tests for reference submissions."""

  def test_submission(self):
    # Example: /home/znado/algorithmic-efficiency/tests
    self_location = os.path.dirname(os.path.realpath(__file__))
    # Example: /home/znado/algorithmic-efficiency
    repo_location = '/'.join(self_location.split('/')[:-1])
    references_dir = f'{repo_location}/reference_submissions'
    for workload_name in os.listdir(references_dir):
      if '_' in workload_name:
        dataset_name = workload_name.split('_')[0]
      else:
        dataset_name = workload_name
      workload_dir = f'{repo_location}/reference_submissions/{workload_name}'
      search_space_path = f'{workload_dir}/tuning_search_space.json'
      for framework in ['jax', 'pytorch']:
        submission_dir = f'{workload_dir}/{dataset_name}_{framework}'
        if not os.path.exists(submission_dir):
          continue
        if 'fastmri' not in workload_dir: #DO NOT SUBMIT
          continue
        submission_path = (f'reference_submissions/{workload_name}/'
                           f'{dataset_name}_{framework}/submission.py')
        logging.info(f'========= Testing {workload_name} in {framework}.')
        eval_result = _test_submission(
            workload_name,
            framework,
            submission_path,
            search_space_path,
            data_dir=None,
            use_fake_input_queue=FLAGS.use_fake_input_queue)
        expected_names = _EXPECTED_METRIC_NAMES[workload_name]
        actual_names = list(eval_result.keys())
        for expected_name in expected_names:
          self.assertIn(expected_name, actual_names)


if __name__ == '__main__':
  absltest.main()
