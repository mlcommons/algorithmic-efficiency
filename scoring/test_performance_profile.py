import os

from absl.testing import absltest

from scoring import performance_profile
from scoring import scoring_utils

TEST_LOGFILE = 'test_data/adamw_fastmri_jax_04-18-2023-13-10-58.log'
TEST_DIR = 'test_data/experiment_dir'
NUM_EVALS = 18


class Test(absltest.TestCase):

  def test_get_workloads_time_to_target(self):
    pass

  def test_get_best_trial_index(self):
    pass

  def test_compute_performance_profiles(self):
    pass


if __name__ == '__main__':
  absltest.main()
