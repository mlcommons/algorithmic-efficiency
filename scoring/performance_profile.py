"""Performance and scoring code.

The three primary methods exposed by the `scoring` module are:
- `compute_performance_profiles`: generates performance profiles for a set of
  submissions over all workloads as defined in the scoring section:
  https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md
- `compute_leaderboard_score`: computes final scores from performance profiles.
- `plot_performance_profiles`: plot performance profiles for a set of
  submissions.

The two primary inputs to `compute_performance_profiles` are
1. A dictionary of pandas DataFrames, where each key is a globally unique
  identifier for a submission and each value is a DataFrame containing one row
  per trial per workload in that submission. At minimum, this DataFrame should
  include a column of np.arrays indicating time (e.g., 'global_step'), a column
  of np.arrays indicating performance (e.g., 'validation/accuracy') for each
  workload and a column 'workload' that indicates the workload identifier.
2. A dictionary of workload metadata describing each workload in the form:
  {
    'workload_identifier': {
      'target': VALUE,
      'metric': 'validation/error_rate',
    }
  }
  The keys in this dictionary should match the workload identifiers used in
  the dictionary of submissions.
"""
import itertools
import operator
import os
import re

from absl import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import re

import logging

from algorithmic_efficiency.workloads.workloads import get_base_workload_name
import algorithmic_efficiency.workloads.workloads as workloads_registry
from scoring import scoring_utils

WORKLOADS = workloads_registry.WORKLOADS
BASE_WORKLOADS = workloads_registry.BASE_WORKLOADS
WORKLOAD_NAME_PATTERN = '(.*)(_jax|_pytorch)'
BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'
# These global variables have to be set according to the current set of
# workloads and rules for the scoring to be correct.
# We do not use the workload registry since it contains test and development
# workloads as well.
NUM_BASE_WORKLOADS = 8
NUM_VARIANT_WORKLOADS = 6
NUM_TRIALS = 5
NUM_STUDIES = 5

MIN_EVAL_METRICS = [
    'ce_loss',
    'error_rate',
    'ctc_loss',
    'wer',
    'l1_loss',
    'loss',
]

MAX_EVAL_METRICS = ['mean_average_precision', 'ssim', 'accuracy', 'bleu']

#MPL params
mpl.rcParams['figure.figsize'] = (16, 10)  # Width, height in inches
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = [
    'Times New Roman'
] + mpl.rcParams['font.serif']  # Add Times New Roman as first choice
mpl.rcParams['font.size'] = 22
mpl.rcParams['savefig.dpi'] = 300  # Set resolution for saved figures

# Plot Elements
mpl.rcParams['lines.linewidth'] = 3  # Adjust line thickness if needed
mpl.rcParams['lines.markersize'] = 6  # Adjust marker size if needed
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(
    color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd"])  # Example color cycle (consider ColorBrewer or viridis)
mpl.rcParams['axes.labelsize'] = 22  # Axis label font size
mpl.rcParams['xtick.labelsize'] = 20  # Tick label font size
mpl.rcParams['ytick.labelsize'] = 20

# Legends and Gridlines
mpl.rcParams['legend.fontsize'] = 20  # Legend font size
mpl.rcParams[
    'legend.loc'] = 'best'  # Let matplotlib decide the best legend location
mpl.rcParams['axes.grid'] = True  # Enable grid
mpl.rcParams['grid.alpha'] = 0.4  # Gridline transparency


def print_dataframe(df):
  tabulated_df = tabulate(df.T, headers='keys', tablefmt='psql')
  logging.info(tabulated_df)


def generate_eval_cols(metrics):
  splits = ['train', 'validation']
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


def get_best_trial_index(workload_df,
                         validation_metric,
                         validation_target=None):
  """Get the eval index in which a workload reaches the target metric_col.

  Args:
    workload_df: A subset of a submission's trials DataFrame that
      includes only the trials in a single workload.
    metric_col: Name of array column in workload_df (e.g. `validation/l1_loss`).
    target: Target value for metric_col.

  Returns:
    Tuple of trial index and time index where the workload reached the target
      metric_col. Return (-1, -1) if not reached.
  """
  is_minimized = check_if_minimized(validation_metric)
  validation_series = workload_df[validation_metric]
  validation_series = validation_series[validation_series != np.nan]

  op = operator.le if is_minimized else operator.ge
  validation_target_reached = validation_series.apply(
      lambda x: op(x, validation_target))
  target_reached = pd.Series(validation_target_reached)

  # Remove trials that never reach the target
  target_reached = target_reached[target_reached.apply(np.any)]

  # If no trials reach the target return -1. Else, return the eval index
  # of the earliest point the target is reached.
  if len(target_reached) == 0:
    return -1, -1
  else:
    index_reached = target_reached.apply(np.argmax)
    trial = index_reached.idxmin()
    return trial, index_reached[trial]


def get_workloads_time_to_target(submission,
                                 submission_name,
                                 time_col='global_step',
                                 verbosity=1,
                                 self_tuning_ruleset=False,
                                 strict=False):
  """Get times to target for each workload in a submission.

  Args:
    submission: A DataFrame containing one row for each trial in each workload
      for a given submission.
    submission_name: Globally unique identifier for a submission.
    time_col: A string indicating which column to use for time.
    verbosity: Debug level of information; choice of (1, 2, 3).

  Returns:
    DataFrame with columns `submission`, `workload`, and time_col.
  """
  workloads = []

  # Check number of workloads in submission
  num_workloads = len(submission.groupby('workload'))
  if num_workloads != NUM_BASE_WORKLOADS + NUM_VARIANT_WORKLOADS:
    if strict:
      raise ValueError(
          f'Expecting {NUM_BASE_WORKLOADS + NUM_VARIANT_WORKLOADS} workloads '
          f'but found {num_workloads} workloads.')
    logging.warning(
        f'Expecting {NUM_BASE_WORKLOADS + NUM_VARIANT_WORKLOADS} workloads '
        f'but found {num_workloads} workloads.')

  # For each workload get submission time get the submission times to target.
  for workload, group in submission.groupby('workload'):
    validation_metric, validation_target = scoring_utils.get_workload_metrics_and_targets(workload)

    # Check number of studies
    time_vals_per_study = []
    num_studies = len(group.groupby('study'))
    if num_studies != NUM_STUDIES:
      if strict:
        raise ValueError(f'Expecting {NUM_STUDIES} trials for workload '
                         f'{workload} but found {num_studies} trials.')
      else:
        logging.warning(f'Expecting {NUM_STUDIES} trials for workload '
                        f'{workload} but found {num_studies} trials.')

    # For each study check trials
    for study, group in group.groupby('study'):

      # Check number of trials per study
      num_trials = len(group)
      if num_trials != NUM_TRIALS and not self_tuning_ruleset:
        if strict:
          raise ValueError(
              f'In Study {study}: Expecting {NUM_TRIALS} trials for workload '
              f'{workload} but found {num_trials} trials.')
        else:
          logging.warning(
              f'In Study {study}: Expecting {NUM_TRIALS} trials for workload '
              f'{workload} but found {num_trials} trials.')

      # Get trial and time index that reaches target
      trial_idx, time_idx = get_best_trial_index(
          group, validation_metric, validation_target)
      if time_idx > -1:
        time_val = group[time_col].loc[trial_idx][time_idx]
      else:
        time_val = float('inf')
      time_vals_per_study.append(time_val)

    workloads.append({
        'submission': submission_name,
        'workload': re.sub(r'_(jax|pytorch)$', '', workload),
        time_col: np.median(time_vals_per_study),
    })

  df = pd.DataFrame.from_records(workloads)
  df = df.pivot(index='submission', columns='workload', values=time_col)
  logging.info("HELLOOOOOOOOO")
  print_dataframe(df)
  return df


def variant_criteria_filter(base_workload, variant_workload):

  def filter(x):
    try:
      if x[variant_workload] == np.inf:
        return np.inf
      else:
        return x[base_workload]
    except KeyError as e:
      print(x.keys())
      raise e

  return filter


def compute_performance_profiles(submissions,
                                 time_col='global_step',
                                 min_tau=1.0,
                                 max_tau=None,
                                 reference_submission_tag=None,
                                 num_points=100,
                                 scale='linear',
                                 verbosity=0,
                                 strict=False,
                                 self_tuning_ruleset=False):
  """Compute performance profiles for a set of submission by some time column.

  Args:
    results: Dict where keys are submission names and values are a DataFrame of
      trials where each row is a trial and each column is a field for a given
      trial. Results should contain keys for each workload's metric, time_col,
      'workload'. See file header comment for more details.
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

  for submission_tag, submission in submissions.items():
    logging.info(
        f'\nComputing performance profile with respect to `{time_col}` for '
        f'{submission_tag}')
    # Get time to targets for each submission across studies and trials
    dfs.append(
        get_workloads_time_to_target(submission,
                                     submission_tag,
                                     time_col,
                                     verbosity,
                                     self_tuning_ruleset,
                                     strict))
  df = pd.concat(dfs)

  logging.info("TIME TO TARGET")
  print_dataframe(df)

  # Set score to inf if not within 4x of fastest submission
  best_scores = df.min(axis=0)
  df[df.apply(lambda x: x > 4 * best_scores, axis=1)] = np.inf

  logging.info("4X of budget")
  print_dataframe(df)

  # For each held-out workload if variant target was not hit set submission to inf
  framework = None
  for workload in df.keys():
    if workload not in BASE_WORKLOADS:
      # If variants do not have finite score set base_workload score to inf
      base_workload = get_base_workload_name(workload)
      df[base_workload] = df.apply(
          variant_criteria_filter(base_workload, workload), axis=1)

  logging.info("HELDOUT_WORKLOAD FILTER")
  print_dataframe(df)

  df = df[BASE_WORKLOADS]

  if verbosity > 0:
    logging.info('\n`{time_col}` to reach target:')
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None,
                           'display.width',
                           1000):
      logging.info(df)

  # Divide by the fastest.
  if reference_submission_tag is None:
    df.update(df.div(df.min(axis=0), axis=1))
  else:
    df.update(df.div(df.loc[reference_submission_tag, :], axis=1))

  if verbosity > 0:
    logging.info('\n`{time_col}` to reach target normalized to best:')
    with pd.option_context('display.max_rows',
                           None,
                           'display.max_columns',
                           None,
                           'display.width',
                           1000):
      logging.info(df)

  logging.info('DIVIDE BY FASTEST')
  print_dataframe(df)

  # If no max_tau is supplied, choose the value of tau that would plot all non
  # inf or nan data.
  if max_tau is None:
    max_tau = df.replace(float('inf'), -1).replace(np.nan, -1).values.max()

  logging.info('AFTER MAYBE SETTING MAX TAU')
  print_dataframe(df)

  if scale == 'linear':
    points = np.linspace(min_tau, max_tau, num=num_points)
  elif scale == 'log':
    points = np.logspace(
        np.log10(min_tau), np.log10(max_tau), num=num_points, base=10.0)

  def rho(r, tau):
    return (r <= tau).sum(axis=1) / NUM_BASE_WORKLOADS

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
                              figsize=(30, 10)):
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
  fig = perf_df.T.plot(figsize=figsize, alpha=0.7)
  df_col_display = f'log10({df_col})' if scale == 'log' else df_col
  fig.set_xlabel(f'Ratio of `{df_col_display}` to best submission')
  fig.set_ylabel('Proportion of workloads')
  fig.legend(bbox_to_anchor=(1.0, 1.0))
  plt.tight_layout()
  maybe_save_figure(save_dir, f'performance_profile_by_{df_col_display}')
  maybe_save_df_to_csv(save_dir,
                       perf_df,
                       f'performance_profile_{df_col_display}.csv')
