from absl.testing import absltest

from scoring import scoring_utils
from scoring.scoring import NUM_TRIALS
from scoring.scoring import NUM_WORKLOADS

TEST_LOGFILE = 'scoring/test_data/adamw_fastmri_jax_04-18-2023-13-10-58.log'
TEST_DIR = 'scoring/test_data/experiment_dir'
NUM_EVALS = 18


class Test(absltest.TestCase):

  def test_get_trials_dict(self):
    trials_dict = scoring_utils.get_trials_dict(TEST_LOGFILE)
    self.assertEqual(len(trials_dict['1']['global_step']), NUM_EVALS)

  def test_get_trials_df_dict(self):
    trials_dict = scoring_utils.get_trials_df_dict(TEST_LOGFILE)
    for df in trials_dict.values():
      self.assertEqual(len(df.index), NUM_EVALS)

  def test_get_trials_df(self):
    df = scoring_utils.get_trials_df(TEST_LOGFILE)
    for column in df.columns:
      self.assertEqual(len(df.at['1', column]), NUM_EVALS)

  def test_get_experiment_df(self):
    df = scoring_utils.get_experiment_df(TEST_DIR)
    assert len(df) == 2

  def test_scores(self):
    df = scoring_utils.get_experiment_df(TEST_DIR)
    performance_profile_df = scoring.compute_performance_profiles(
        {'my.submission': df},
        time_col='score',
        min_tau=1.0,
        max_tau=None,
        reference_submission_tag=None,
        num_points=100,
        scale='linear',
        verbosity=0)


if __name__ == '__main__':
  absltest.main()
