import json
import os
import re

import pandas as pd
from absl import logging 

TRIAL_LINE_REGEX = '(.*) --- Tuning run (\d+)/(\d+) ---'
METRICS_LINE_REGEX = '(.*) Metrics: ({.*})'
TRIAL_DIR_REGEX = 'trial_(\d+)'
MEASUREMENTS_FILENAME = 'eval_measurements.csv'


#### File IO helper functions ###
def get_logfile_paths(logdir):
  """Gets all files ending in .log in logdir
    """
  filenames = os.listdir(logdir)
  logfile_paths = []
  for f in filenames:
    if f.endswith(".log"):
      f = os.path.join(logdir, f)
      logfile_paths.append(f)
  return logfile_paths


### Logfile reading helper functions ###
def decode_metrics_line(line):
  """Convert metrics line to dict.
    Args:
        line: str 

    Returns:
        dict_of_lists: dict  where keys are metric names and vals 
        are lists of values.
            e.g. {'loss':[5.1, 3.2, 1.0],
                  'step':[100, 200, 300]}
    """
  eval_results = []
  dict_str = re.match(METRICS_LINE_REGEX, line).group(2)
  dict_str = dict_str.replace("'", "\"")
  dict_str = dict_str.replace("(", "")
  dict_str = dict_str.replace(")", "")
  dict_str = dict_str.replace("DeviceArray", "")
  dict_str = dict_str.replace(", dtype=float32", "")
  dict_str = dict_str.replace("nan", "0")
  metrics_dict = json.loads(dict_str)
  for item in metrics_dict['eval_results']:
    if isinstance(item, dict):
      eval_results.append(item)

  keys = eval_results[0].keys()

  dict_of_lists = {}
  for key in keys:
    dict_of_lists[key] = []

  for eval_results_dict in eval_results:
    for key in eval_results_dict.keys():
      val = eval_results_dict[key]
      dict_of_lists[key].append(val)

  return dict_of_lists


def get_trials_dict(logfile):
  """Get a dict of dicts with metrics for each 
    tuning run.

    Returns:
        trials_dict: Dict of dicts where outer dict keys
            are trial indices and inner dict key-value pairs
            are metrics and list of values.
            e.g. {'trial_0': {'loss':[5.1, 3.2, 1.0],
                            'step':[100, 200, 300]},
                  'trial_1': {'loss':[5.1, 3.2, 1.0],
                            'step':[100, 200, 300]}}
    """
  trial = 0
  metrics_lines = {}
  with open(logfile, 'r') as f:
    for line in f:
      if re.match(TRIAL_LINE_REGEX, line):
        trial = re.match(TRIAL_LINE_REGEX, line).group(2)
      if re.match(METRICS_LINE_REGEX, line):
        metrics_lines[trial] = decode_metrics_line(line)
  if len(metrics_lines) == 0:
    raise ValueError(f"Log file does not have a metrics line {logfile}")
  return metrics_lines


### Results formatting helper functions ###
def get_trials_df_dict(logfile):
  """Get a dict with dataframes with metrics for each 
    tuning run. 
    Preferable format for saving dataframes for tables.
    Args:
        logfile: str path to logfile.

    Returns:
        DataFrame where indices are index of eval and 
        columns are metric names.
    """
  trials_dict = get_trials_dict(logfile)
  trials_df_dict = {}
  for trial in trials_dict:
    metrics = trials_dict[trial]
    trials_df_dict[trial] = pd.DataFrame(metrics)
  return trials_df_dict


def get_trials_df(logfile):
  """Gets a df of per trial results from a logfile.
    Args:
        experiment_dir: str

    Returns:
        df: DataFrame where indices are trials, columns are 
            metric names and values are lists.
            e.g 
            +---------+-----------------+-----------------+
            |         | loss            | step            |
            |---------+-----------------+-----------------|
            | trial_0 | [5.1, 3.2, 1.0] | [100, 200, 300] |
            | trial_1 | [5.1, 3.2, 1.0] | [100, 200, 300] |
            +---------+-----------------+-----------------+
    """
  trials_dict = get_trials_dict(logfile)
  df = pd.DataFrame(trials_dict).transpose()
  return df


## Get scoring code
def get_experiment_df(experiment_dir):
  """Gets a df of per trial results from an experiment dir.
  The output df can be provided as input to 
  scoring.compute_performance_profiles. 
  Args:
      experiment_dir: path to experiment directory containing 
        results for workloads.
        The directory structure is assumed to be:
        + experiment_dir
          + <workload>
            + <trial>
              - eval_measurements.csv

  Returns:
      df: DataFrame where indices are trials, columns are 
          metric names and values are lists.
          e.g 
          +----+------------+---------+--------------------+--------------------+
          |    | workload   | trial   | validation/accuracy| score              |
          |----+------------+---------+--------------------+--------------------|
          |  0 | mnist_jax  | trial_1 | [0.0911, 0.0949]   | [10.6396, 10.6464] |
          +----+------------+---------+--------------------+--------------------+
  """
  df = pd.DataFrame()
  workload_dirs = os.listdir(experiment_dir)
  for workload in workload_dirs:
    data = {
        'workload': workload,
    }
    trial_dirs = [
        t for t in os.listdir(os.path.join(experiment_dir, workload))
        if re.match(TRIAL_DIR_REGEX, t)
    ]
    for trial in trial_dirs:
      eval_measurements_filepath = os.path.join(
          experiment_dir,
          workload,
          trial,
          MEASUREMENTS_FILENAME,
      )
      try:
        trial_df = pd.read_csv(eval_measurements_filepath)
      except FileNotFoundError as e:
        logging.info(f"Could not read {eval_measurements_filepath}")
        continue
      columns = df.columns.tolist()
      data['trial'] = trial
      for column in trial_df.columns:
        values = trial_df[column].to_numpy()
        data[column] = values
      trial_df = pd.DataFrame([data])
      df = pd.concat([df, trial_df], ignore_index=True)
  return df
