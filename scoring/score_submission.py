import os

from absl import app
from absl import flags
from absl import logging
import scoring_utils
from tabulate import tabulate
from scoring import score_profile

flags.DEFINE_string(
    'experiment_path',
    None,
    'Path to experiment directory containing workload directories.')
flags.DEFINE_string('submission_tag', 'my.submission', 'Submission tag.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
flags.DEFINE_boolean('compute_performance_profiles',
                    True, 
                    'Whether or not to compute the performance profiles.')
FLAGS = flags.FLAGS


def main(_):
  df = scoring_utils.get_experiment_df(FLAGS.experiment_path)
  results = {
      FLAGS.submission_tag: df,
  }
  table = tabulate(df, headers='keys', tablefmt='psql')
  logging.info(df)
  for workload, group in df.groupby('workload'):
    target_metric_name, target_metric_value = scoring_utils.get_workload_validation_target(workload)
    print(target_metric_name)
    print(target_metric_value)
    # print(workload)
    # print(group)

  if FLAGS.compute_performance_profiles:
    performance_profile_df = score_profile.compute_performance_profiles(
        results,
        time_col='score',
        min_tau=1.0,
        max_tau=None,
        reference_submission_tag=None,
        num_points=100,
        scale='linear',
        verbosity=0)
    if not os.path.exists(FLAGS.output_dir):
      os.mkdir(FLAGS.output_dir)
    score_profile.plot_performance_profiles(
        performance_profile_df, 'score', save_dir=FLAGS.output_dir)
    perf_df = tabulate(performance_profile_df.T, headers='keys', tablefmt='psql')
    logging.info(f'Performance profile:\n {perf_df}')


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_path')
  app.run(main)
