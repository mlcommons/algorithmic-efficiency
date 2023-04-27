"""Tests for submission.py for baselines.

This is an end-to-end test for all baselines on MNIST in PyTorch and Jax that 
requires the dataset to be available.
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

MAX_GLOBAL_STEPS = 5

baselines = [
    'adafactor',
    'adamw',
    'lamb',
    'momentum',
    'nadamw',
    'nesterov',
    'sam',
    'shampoo',
]
frameworks = [
    # 'pytorch', # will enable this once all pytorch baselines are ready
    'jax',
]

named_parameters = []
for f in frameworks:
  for b in baselines:
    named_parameters.append(
        dict(
            testcase_name=f'{b}_{f}',
            workload='mnist',
            framework=f'{f}',
            submission_path=(f'baselines/{b}/{f}/submission.py'),
            tuning_search_space=(f'baselines/{b}/tuning_search_space.json')))


class BaselineTest(parameterized.TestCase):
  """Tests for reference submissions."""

  @parameterized.named_parameters(*named_parameters)
  def test_baseline_submission(self,
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


if __name__ == '__main__':
  absltest.main()
