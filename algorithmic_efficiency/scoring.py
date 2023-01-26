"""Performance and scoring code.

The three primary methods exposed by the `scoring` module are:
- `compute_performance_profiles`: generates performance profiles for a set of
  submissions over all workloads as defined in the scoring rules:
  https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md
- `compute_leaderboard_score`: computes final scores from performance profiles.
- `plot_performance_profiles`: plot performance profiles for a set of
  submissions.

The two primary inputs to `compute_performance_profiles` are
1. A dictionary of pandas DataFrames, where each key is a globally unique
  identifier for a submission and each value is a DataFrame containing one row
  per trial per workload in that submission. At minimum, this DataFrame should
  include a column of np.arrays indicating time (e.g., 'global_step'), a column
  of np.arrays indicating performance (e.g., 'valid/accuracy') for each
  workload and a column 'workload' that indicates the workload identifier.
2. A dictionary of workload metadata describing each workload in the form:
  {
    'workload_identifier': {
      'target': VALUE,
      'metric': 'valid/error_rate'
    }
  }
  The keys in this dictionary should match the workload identifiers used in
  the dictionary of submissions.
"""

import itertools
import operator
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MIN_EVAL_METRICS = [
    'ce_loss',
    'error_rate',
    'ctc_loss',
    'wer',
    'l1_loss',
]
MAX_EVAL_METRICS = ['average_precision', 'ssim', 'bleu_score']


def generate_eval_cols(metrics):
  splits = ['train', 'valid', 'test']
  return [f'{split}/{col}' for split, col in itertools.product(splits, metrics)]


MINIMIZE_REGISTRY = {k: True for k in generate_eval_cols(MIN_EVAL_METRICS)}
MINIMIZE_REGISTRY.update(
    {k: False for k in generate_eval_cols(MAX_EVAL_METRICS)})
MINIMIZE_REGISTRY['train_cost'] = True


def check_if_minimized(col_name):
  """Guess if the eval metric column name should be minimized or not."""
  for prefix in ['best_', 'final_']:
    col_name = col_name.replace(prefix, '')

  for col in MINIMIZE_REGISTRY:
    if col in col_name:
      return MINIMIZE_REGISTRY[col]

  raise ValueError(f'Column {col_name} not found in `MINIMIZE_REGISTRY` as '
                   'either a column name or a substring of a column name.')


def get_index_that_reaches_best(workload_df, metric_col):
  """Get the eval index in which a workload reaches the best on metric_col.

  Args:
    workload_df: A subset of a submission's trials DataFrame that
      includes only the trials in a single workload.
    metric_col: Name of array column in workload_df (e.g., `valid/l1_loss`).

  Returns:
    Tuple of trial index, time index, and best value where the workload
      reached the best metric_col. Return (-1, -1, -1) if no undiverged trials.
  """
  is_minimized = check_if_minimized(metric_col)
  series = workload_df[metric_col]

  series = series[series != np.nan]

  op = np.min if is_minimized else np.max
  best = series.apply(op)

  op_idx = np.argmin if is_minimized else np.argmax
  best_idx = series.apply(op_idx)

  if best.empty:
    return -1, -1, -1
  else:
    trial = best.idxmin() if is_minimized else best.idxmax()
    return trial, best_idx[trial], best[trial]


def get_index_that_reaches_target(workload_df, metric_col, target):
  """Get the eval index in which a workload reaches the target metric_col.

  Args:
    workload_df: A subset of a submission's trials DataFrame that
      includes only the trials in a single workload.
    metric_col: Name of array column in workload_df (e.g., `valid/l1_loss`).
    target: Target value for metric_col.

  Returns:
    Tuple of trial index and time index where the workload reached the target
      metric_col. Return (-1, -1) if not reached.
  """
  is_minimized = check_if_minimized(metric_col)
  series = workload_df[metric_col]

  series = series[series != np.nan]

  op = operator.le if is_minimized else operator.ge
  target_reached = series.apply(lambda x: op(x, target))

  # Remove trials that never reach the target
  target_reached = target_reached[target_reached.apply(np.any)]

  # If we have no trials that have reached the target, return -1. Else, return
  # the eval index of the earliest point the target is reached.
  if target_reached.empty:
    return -1, -1
  else:
    index_reached = target_reached.apply(np.argmax)
    trial = index_reached.idxmin()
    return trial, index_reached[trial]


def get_times_for_submission(submission,
                             submission_tag,
                             workload_metadata,
                             time_col='global_step',
                             verbosity=1):
  """Get times to target for each workload in a submission.

  Args:
    submission: A DataFrame containing one row for each trial in each workload
      for a given submission.
    submission_tag: Globally unique identified for a submission.
    workload_metadata: Dictionary keyed by workload names with value of
      dictionary with `target` and `metric` as keys.
    time_col: A string indicating which column to use for time.
    verbosity: Debug level of information; choice of (1, 2, 3).

  Returns:
    DataFrame with columns `submission`, `workload`, and time_col.
  """
  workloads = []
  submission_name = submission_tag.split('.')[1]

  for workload, group in submission.groupby('workload'):
    metric = workload_metadata[workload]['metric']
    target = workload_metadata[workload]['target']
    trial_idx, time_idx = get_index_that_reaches_target(group, metric, target)
    if time_idx > -1:
      time_val = group[time_col].loc[trial_idx][time_idx]
    else:
      time_val = float('inf')

    workloads.append({
        'submission': submission_name,
        'workload': workload,
        time_col: time_val,
    })

    if verbosity > 0:
      print('  hparams:')
      if time_idx > -1:
        hparams = group.loc[trial_idx, 'hparams']
        for key, val in hparams.items():
          print(f'  - {key}: {val}')
      else:
        print('Submission did not reach target')
  df = pd.DataFrame.from_records(workloads)
  df = df.pivot(index='submission', columns='workload', values=time_col)

  return df


def compute_performance_profiles(results,
                                 workload_metadata,
                                 time_col='global_step',
                                 min_tau=1.0,
                                 max_tau=None,
                                 reference_submission_tag=None,
                                 num_points=100,
                                 scale='linear',
                                 verbosity=0):
  """Compute performance profiles for a set of submission by some time column.

  Args:
    results: Dict where keys are submission names and values are a DataFrame of
      trials where each row is a trial and each column is a field for a given
      trial. Results should contain keys for each workload's metric, time_col,
      'workload'. See file header comment for more details.
    workload_metadata: Dictionary keyed by workload names with value of
      dictionary with `target` and `metric` as keys.
    time_col: A string indicating which column to use for time.
    min_tau: Minimum tau to use for plotting.
    max_tau: Maximum tau to use for plotting.
    reference_submission_tag: If specified, must be an element of
      `submission_tags`. Used as the denominator for computing tau. Otherwise,
      the minimum time to target is computed per-workload and used as the
      denominator for tau.
    num_points: Number of points to use for plotting.
    scale: Linear or log scale for the x-axis.
    verbosity: Debug level of information; choice of (1, 2, 3).

  Returns:
    A DataFrame of performance profiles for the set of submissions given in
      `results` based on `time_col`. Each row represents a submission and each
      column represents rho(tau) for some value of tau (df.volumns are the
      different values of tau).
  """
  dfs = []

  for submission_tag, result in results.items():
    print(f'\nComputing performance profile with respect to `{time_col}` for '
          f'{submission_tag}')
    dfs.append(
        get_times_for_submission(result,
                                 submission_tag,
                                 workload_metadata,
                                 time_col,
                                 verbosity))
  df = pd.concat(dfs)

  if verbosity > 0:
    print(f'\n`{time_col}` to reach target:')
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None,
                           'display.width',
                           1000):
      print(df)

  # Divide by the fastest.
  if reference_submission_tag is None:
    df.update(df.div(df.min(axis=0), axis=1))
  else:
    df.update(df.div(df.loc[reference_submission_tag, :], axis=1))

  if verbosity > 0:
    print(f'\n`{time_col}` to reach target normalized to best:')
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None,
                           'display.width',
                           1000):
      print(df)

  # If no max_tau is supplied, choose the value of tau that would plot all non
  # inf or nan data.
  if max_tau is None:
    max_tau = df.replace(float('inf'), -1).replace(np.nan, -1).values.max()

  if scale == 'linear':
    points = np.linspace(min_tau, max_tau, num=num_points)
  elif scale == 'log':
    points = np.logspace(
        np.log10(min_tau), np.log10(max_tau), num=num_points, base=10.0)

  def rho(r, tau):
    return (r <= tau).sum(axis=1) / len(r.columns)

  perf_df = pd.concat([rho(df, tau) for tau in points], axis=1)

  cols = points
  if scale == 'log':
    cols = np.log10(points)
  perf_df.columns = cols

  return perf_df


def compute_leaderboard_score(df, normalize=False):
  """Compute leaderboard score by taking integral of performance profile.

  Args:
    df: pd.DataFrame returned from `compute_performance_profiles`.
    normalize: divide by the range of the performance profile's tau.

  Returns:
    pd.DataFrame with one column of scores indexed by submission.
  """
  scores = np.trapz(df, x=df.columns)
  if normalize:
    scores /= df.columns.max() - df.columns.min()
  return pd.DataFrame(scores, columns=['score'], index=df.index)


def maybe_save_figure(save_dir, name, ext='pdf'):
  """Maybe save the current matplotlib.pyplot figure."""
  if save_dir:
    path = os.path.join(save_dir, f'{name}.{ext}')
    with open(path, 'wb') as fout:
      plt.savefig(fout, format=ext)


def maybe_save_df_to_csv(save_dir, df, path, **to_csv_kwargs):
  if save_dir:
    path = os.path.join(save_dir, path)
    with open(path, 'w') as fout:
      df.to_csv(fout, **to_csv_kwargs)


def plot_performance_profiles(perf_df,
                              df_col,
                              scale='linear',
                              save_dir=None,
                              figsize=(30, 10),
                              font_size=18):
  """Plot performance profiles.

  Args:
    perf_df: A DataFrame of performance profiles where each row represents a
      submission and each column represents rho(tau) for some value of tau
      (df.volumns are the different values of tau).
    df_col: The column in the original submission results DataFrame used to
      compute the performance profile. This argument is only used for axis
      and file naming.
    scale: Whether or not the data in perf_df is on a linear or log scale. This
      argument is only used for axis and file naming.
    save_dir: If a valid directory is provided, save both the plot and perf_df
      to the provided directory.
    figsize: The size of the plot.
    font_size: The font size to use for the legend.

  Returns:
    None. If a valid save_dir is provided, save both the plot and perf_df.
  """
  fig = perf_df.T.plot(figsize=figsize)
  df_col_display = f'log10({df_col})' if scale == 'log' else df_col
  fig.set_xlabel(
      f'Ratio of `{df_col_display}` to best submission', size=font_size)
  fig.set_ylabel('Proportion of workloads', size=font_size)
  fig.legend(prop={'size': font_size}, bbox_to_anchor=(1.0, 1.0))
  maybe_save_figure(save_dir, f'performance_profile_by_{df_col_display}')
  maybe_save_df_to_csv(save_dir,
                       perf_df,
                       f'performance_profile_{df_col_display}.csv')
