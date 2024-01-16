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
from algorithmic_efficiency.workloads import workloads
import submission_runner

FLAGS = flags.FLAGS
# Needed to avoid UnparsedFlagAccessError
# (see https://github.com/google/model_search/pull/8).
FLAGS(sys.argv)

MAX_GLOBAL_STEPS = 5

baselines = {
    'jax': [
        'adafactor',
        'adamw',
        'lamb',
        'momentum',
        'nadamw',
        'nesterov',
        'sam',
        'shampoo',
    ],
    'pytorch': [
        'adamw',
        'momentum',
        'nadamw',
        'nesterov',
    ],
}

frameworks = [
    'pytorch',
    'jax',
]

baseline_path = "reference_algorithms/paper_baselines"

named_parameters = []
for f in frameworks:
  for b in baselines[f]:
    named_parameters.append(
        dict(
            testcase_name=f'{b}_{f}',
            workload='mnist',
            framework=f'{f}',
            submission_path=f'{baseline_path}/{b}/{f}/submission.py',
            tuning_search_space=f'{baseline_path}/{b}/tuning_search_space.json')
    )


class BaselineTest(parameterized.TestCase):
  """Tests for reference submissions."""

  @parameterized.named_parameters(*named_parameters)
  def test_baseline_submission(self,
                               workload,
                               framework,
                               submission_path,
                               tuning_search_space):
    FLAGS.framework = framework
    workload_metadata = copy.deepcopy(workloads.WORKLOADS[workload])
    workload_metadata['workload_path'] = os.path.join(
        workloads.BASE_WORKLOADS_DIR,
        workload_metadata['workload_path'] + '_' + framework,
        'workload.py')
    workload_obj = workloads.import_workload(
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
        max_global_steps=MAX_GLOBAL_STEPS,
    )
    logging.info(score)


if __name__ == '__main__':
  absltest.main()
