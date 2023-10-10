"""Tests for submission_runner.py.

This is an end-to-end test for MNIST in PyTorch and Jax that requires the
dataset to be available. For testing the workload and reference submission code
for all workloads, see reference_algorithm_tests.py.
"""
import copy
import os
import sys

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from algorithmic_efficiency.profiler import PassThroughProfiler
import submission_runner

FLAGS = flags.FLAGS
# Needed to avoid UnparsedFlagAccessError
# (see https://github.com/google/model_search/pull/8).
FLAGS(sys.argv)

_MNIST_DEV_ALGO_DIR = 'reference_algorithms/target_setting_algorithms/mnist'


class SubmissionRunnerTest(parameterized.TestCase):
  """Tests for reference submissions."""

  @parameterized.named_parameters(
      dict(
          testcase_name='mnist_jax',
          workload='mnist',
          framework='jax',
          submission_path=(f'{_MNIST_DEV_ALGO_DIR}/mnist_jax/submission.py'),
          tuning_search_space=(
              f'{_MNIST_DEV_ALGO_DIR}/tuning_search_space.json')),
      dict(
          testcase_name='mnist_pytorch',
          workload='mnist',
          framework='pytorch',
          submission_path=(
              f'{_MNIST_DEV_ALGO_DIR}/mnist_pytorch/submission.py'),
          tuning_search_space=(
              f'{_MNIST_DEV_ALGO_DIR}/tuning_search_space.json')),
  )
  def test_submission(self,
                      workload,
                      framework,
                      submission_path,
                      tuning_search_space):
    FLAGS.framework = framework
    workload_metadata = copy.deepcopy(submission_runner.WORKLOADS[workload])
    workload_metadata['workload_path'] = os.path.join(
        submission_runner.BASE_WORKLOADS_DIR,
        workload_metadata['workload_path'] + '_' + framework,
        'workload.py')
    workload_obj = submission_runner.import_workload(
        workload_path=workload_metadata['workload_path'],
        workload_class_name=workload_metadata['workload_class_name'],
        workload_init_kwargs={})
    score = submission_runner.score_submission_on_workload(
        workload_obj,
        workload,
        submission_path,
        data_dir='~/tensorflow_datasets',  # The default in TFDS.
        tuning_ruleset='external',
        tuning_search_space=tuning_search_space,
        num_tuning_trials=1,
        profiler=PassThroughProfiler(),
        max_global_steps=500,
    )
    logging.info(score)

  def test_convert_filepath_to_module(self):
    """Sample test for the `convert_filepath_to_module` function."""
    test_path = os.path.abspath(__file__)
    module_path = submission_runner.convert_filepath_to_module(test_path)
    self.assertNotIn('.py', module_path)
    self.assertNotIn('/', module_path)
    self.assertIsInstance(module_path, str)


if __name__ == '__main__':
  absltest.main()
