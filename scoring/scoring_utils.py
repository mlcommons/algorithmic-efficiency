import os
import pandas as pd
import json

metrics_prefix = 'Metrics: '


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
def get_metrics_line(logfile):
  """Extracts the metrics line from a logfile.
    """
  with open(logfile, 'r') as f:
    for line in f:
      if metrics_prefix in line:
        return line
  raise ValueError(f"Log file does not have a metrics line {logfile}")


def convert_metrics_line_to_dict(line):
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
  dict_str = line.split(metrics_prefix)[1]
  dict_str = dict_str.replace("'", "\"")
  dict_str = dict_str.replace("(", "")
  dict_str = dict_str.replace(")", "")
  dict_str = dict_str.replace("DeviceArray", "")
  dict_str = dict_str.replace(", dtype=float32", "")
  dict_str = dict_str.replace("nan", "0")
  try:
    metrics_dict = json.loads(dict_str)
  except Exception as e:
    print('Error loading metrics line')
    raise (e)
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


### Results formatting helper functions ###
def get_results_dict(logfile):
  """
    Returns results as dict.

    Returns:
        Dict where keys are metric names and values
        are list of measurements, e.g.:
        {'loss':[5.1, 3.2, 1.0],
         'step':[100, 200, 300]}
    """
  results_line = get_metrics_line(logfile)
  results_dict = convert_metrics_line_to_dict(results_line)
  return results_dict


def get_tuning_run_df(logfile):
  """Get dataframe for single tuning run from logfile.
    Args:
        logfile: str path to logfile.

    Returns:
        DataFrame where indices are index of eval and 
        columns are metric names.
    """
  line = get_metrics_line(logfile)
  tuning_run_dict = convert_metrics_line_to_dict(line)
  return pd.DataFrame(tuning_run_dict)


def get_trials_df(experiment_dir):
  """Gets a df of per trial results from an experiment_dir.
    This df can be provided as input to 
    scoring.compute_performance_profiles. 
    Args:
        experiment_dir: str
        trials_dict: Dict(Dict) where outer dict keys
            are trial indices and inner dict key-value pairs
            are metrics and list of values.
            e.g. {'trial_0': {'loss':[5.1, 3.2, 1.0],
                            'step':[100, 200, 300]},
                'trial_1': {'loss':[5.1, 3.2, 1.0],
                            'step':[100, 200, 300]}}

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
  trial_dirs = os.listdir(experiment_dir)
  trials_dict = {}
  for trial_dir in trial_dirs:
    logfile_paths = get_logfile_paths(os.path.join(experiment_dir, trial_dir))
    if len(logfile_paths) > 1:
      logfile_path = logfile_paths[-1]
    else:
      logfile_path = logfile_paths[0]
    results_dict = get_results_dict(logfile_path)
    trials_dict[trial_dir] = results_dict
  df = pd.DataFrame(trials_dict).transpose()
  return df
