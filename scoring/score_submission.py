import operator
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd
import scoring_utils
from tabulate import tabulate

from scoring import performance_profile
from scoring.performance_profile import check_if_minimized

flags.DEFINE_string(
    'experiment_path',
    None,
    'Path to experiment directory containing workload directories.')
flags.DEFINE_string('submission_tag', 'my.submission', 'Submission tag.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
flags.DEFINE_boolean('compute_performance_profiles',
                     False,
                     'Whether or not to compute the performance profiles.')
FLAGS = flags.FLAGS


def get_summary_df(workload, workload_df):
  validation_metric, validation_target = scoring_utils.get_workload_validation_target(workload)
  is_minimized = check_if_minimized(validation_metric)
  target_op = operator.le if is_minimized else operator.ge
  best_op = min if is_minimized else max
  idx_op = np.argmin if is_minimized else np.argmax

  summary_df = pd.DataFrame()
  summary_df['workload'] = workload_df['workload']
  summary_df['trial'] = workload_df['trial']
  summary_df['target metric name'] = validation_metric
  summary_df['target metric value'] = validation_target

  summary_df['target reached'] = workload_df[validation_metric].apply(
      lambda x: target_op(x, validation_target)).apply(np.any)
  summary_df['best target'] = workload_df[validation_metric].apply(
      lambda x: best_op(x))
  workload_df['index best eval'] = workload_df[validation_metric].apply(
      lambda x: idx_op(x))
  summary_df['submission time'] = workload_df.apply(
      lambda x: x['accumulated_submission_time'][x['index best eval']], axis=1)
  summary_df['score'] = summary_df.apply(
      lambda x: x['submission time'] if x['target reached'] else np.inf, axis=1)

  return summary_df


def main(_):
  df = scoring_utils.get_experiment_df(FLAGS.experiment_path)
  results = {
      FLAGS.submission_tag: df,
  }

  dfs = []
  for workload, group in df.groupby('workload'):
    summary_df = get_summary_df(workload, group)
    dfs.append(summary_df)

  df = pd.concat(dfs)
  print(tabulate(df, headers='keys', tablefmt='psql'))

  if FLAGS.compute_performance_profiles:
    performance_profile_df = performance_profile.compute_performance_profiles(
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
    performance_profile.plot_performance_profiles(
        performance_profile_df, 'score', save_dir=FLAGS.output_dir)
    perf_df = tabulate(
        performance_profile_df.T, headers='keys', tablefmt='psql')
    logging.info(f'Performance profile:\n {perf_df}')


if __name__ == '__main__':
  flags.mark_flag_as_required('experiment_path')
  app.run(main)
