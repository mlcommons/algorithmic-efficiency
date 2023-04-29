from absl.testing import absltest
import scoring_utils 

TEST_LOGFILE = 'test_data/trial_0/adamw_fastmri_jax_04-18-2023-13-10-58.log'
EXPERIMENT_DIR = 'test_data/'

class Test(absltest.TestCase):

    def test_convert_metrics_line_to_dict(self):
        line = scoring_utils.get_metrics_line(TEST_LOGFILE)
        results_dict = scoring_utils.convert_metrics_line_to_dict(line)

    def test_get_tuning_run_df(self):
        df = scoring_utils.get_tuning_run_df(TEST_LOGFILE)

    def test_get_trials_df(self):
        df = scoring_utils.get_trials_df(EXPERIMENT_DIR)

if __name__ == '__main__':
    absltest.main()