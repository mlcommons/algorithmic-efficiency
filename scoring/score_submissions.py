"""This script can
1. Summarize the raw submission times for each workload run in a set of studies and trials.
2. Produce the performance profiles and scores of a group of submissions. 
Note that for performance profiles and final scores are computed w.r.t. a group of submissions.
If you only have logs for one submission you may group it with some reference submission
to compare the performance.

Example usage:
python3 score_submissions.py \
  --submission_directory $HOME/algorithmic-efficiency/prize_qualification_baselines/logs \
  --strict True
  --compute_performance_profiles
"""

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

flags.DEFINE_string(
    'submission_directory',
    None,
    'Path to submission directory containing experiment directories.')
flags.DEFINE_string('output_dir',
                    'scoring_results',
                    'Path to save performance profile table and plot.')
flags.DEFINE_boolean('compute_performance_profiles',
                     False,
                     'Whether or not to compute the performance profiles.')
flags.DEFINE_boolean(
    'strict',
    False,
    'Whether to enforce scoring criteria on variant performance and on'
    '5-trial median performance. Note that during official scoring this '
    'flag will be set to True.')
flags.DEFINE_boolean(
    'self_tuning_ruleset',
    False,
    'Whether to score on self-tuning ruleset or externally tuned ruleset')
FLAGS = flags.FLAGS


def get_summary_df(workload, workload_df):
  validation_metric, validation_target = scoring_utils.get_workload_validation_target(workload)
  is_minimized = performance_profile.check_if_minimized(validation_metric)
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
  summary_df['submission time (s)'] = workload_df.apply(
      lambda x: x['accumulated_submission_time'][x['index best eval']], axis=1)
  summary_df['submission time to target (s)'] = summary_df.apply(
      lambda x: x['submission time (s)'] if x['target reached'] else np.inf,
      axis=1)

  return summary_df


def print_submission_summary(df):
  dfs = []
  for workload, group in df.groupby('workload'):
    summary_df = get_summary_df(workload, group)
    dfs.append(summary_df)

  df = pd.concat(dfs)
  logging.info('\n' + tabulate(df, headers='keys', tablefmt='psql'))


def main(_):
  results = {}

  for submission in os.listdir(FLAGS.submission_directory):
    experiment_path = os.path.join(FLAGS.submission_directory, submission)
    df = scoring_utils.get_experiment_df(experiment_path)
    results[submission] = df
    print_submission_summary(df)

  if not FLAGS.strict:
    logging.warning(
        'You are running with strict=False. This will relax '
        'scoring criteria on the held-out workloads, number of trials and number '
        'of studies. Your score may not be an accurate representation '
        'under competition scoring rules. To enforce the criteria set strict=True.'
    )
  if FLAGS.compute_performance_profiles:
    performance_profile_df = performance_profile.compute_performance_profiles(
        results,
        time_col='score',
        min_tau=1.0,
        max_tau=None,
        reference_submission_tag=None,
        num_points=100,
        scale='linear',
        verbosity=0,
        self_tuning_ruleset=FLAGS.self_tuning_ruleset,
        strict=FLAGS.strict)
    if not os.path.exists(FLAGS.output_dir):
      os.mkdir(FLAGS.output_dir)
    performance_profile.plot_performance_profiles(
        performance_profile_df, 'score', save_dir=FLAGS.output_dir)
    perf_df = tabulate(
        performance_profile_df.T, headers='keys', tablefmt='psql')
    logging.info(f'Performance profile:\n {perf_df}')


if __name__ == '__main__':
  # flags.mark_flag_as_required('submission_directory')
  app.run(main)
