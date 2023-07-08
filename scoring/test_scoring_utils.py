from absl.testing import absltest
import scoring_utils

TEST_LOGFILE = 'test_data/trial_0/adamw_fastmri_jax_04-18-2023-13-10-58.log'
NUM_EVALS = 18


class Test(absltest.TestCase):

  def test_get_trials_dict(self):
    trials_dict = scoring_utils.get_trials_dict(TEST_LOGFILE)
    self.assertEqual(len(trials_dict['1']['global_step']), NUM_EVALS)

  def test_get_trials_df_dict(self):
    trials_dict = scoring_utils.get_trials_df_dict(TEST_LOGFILE)
    for trial in trials_dict:
      df = trials_dict[trial]
      self.assertEqual(len(df.index), NUM_EVALS)

  def test_get_trials_df(self):
    df = scoring_utils.get_trials_df(TEST_LOGFILE)
    for column in df.columns:
      self.assertEqual(len(df.at['1', column]), NUM_EVALS)

  def test_get_dr(self):
    

if __name__ == '__main__':
  absltest.main()
