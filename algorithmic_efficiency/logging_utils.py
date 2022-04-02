"""Save information about the training progress of a workload to disk.

Use the `--logging_dir` CLI argument with submission_runner.py to enable this
functionality.

Four files are written to the `logging_dir` folder:
  1. `metadata.json` is created at the start of a workload and it includes the
    datetime, workload name, and system configuration.
  2. `packages.txt` is created at the start of a workload and it includes a list
    of the currently installed OS and python packages.
  3. `trial_[n]/measurements.csv` is created for each hyperparameter tuning
    trial and a row is appended for every model evaluation. The information
    included is loss, accuracy, training step, time elapsed, hparams, workload
    properties, and hardware utilization.
  4. `trial_[n]/metadata.json` is created at the end of each hyperparameter
    tuning trial and includes the result of the trial. The last row of
    measurements.csv and this file are very similar but can differ in the number
    of steps and runtime in one situation: when the tuning trial ran out of time
    but completed one or more steps after the last model evaluation.
"""
from datetime import datetime
import glob
import json
import os
import platform
import re
import subprocess
from typing import Any, Optional

from absl import flags
from absl import logging
import GPUtil
import pandas as pd
import psutil

from algorithmic_efficiency import spec

FLAGS = flags.FLAGS


def concatenate_csvs(path: str):
  """Join all files named "measurements.csv" in a given folder recursively.

  In this logging module, one "measurements.csv" is produced at the granularity
  of each hyperparameter tuning run. This function is provided as a convienence
  to users to join their CSV data. We leave it to users to do this because we do
  not want to create data duplication if there is no user need."""
  search_path = os.path.join(path, '**/measurements.csv')
  input_csvs = list(glob.iglob(search_path, recursive=True))
  if input_csvs:
    df = pd.read_csv(input_csvs.pop())
    for file in input_csvs:
      df = df.append(pd.read_csv(file))
    output_filepath = os.path.join(path, 'all_measurements.csv')
    df.to_csv(output_filepath, index=False)


def _get_utilization() -> dict:
  """Collect system-wide hardware performance measurements.

  High-level utilization measurements for the GPU (if available), CPU,
  temperature, memory, disk, and network.

  The performance measurements are all system-wide because we can't guarentee
  how many processes Jax or PyTorch will start and not all measurements are
  available on a per-process basis (eg. network).
  """
  measurements = {}

  # CPU
  measurements['cpu.util.avg_percent_since_last'] = psutil.cpu_percent(
      interval=None)  # non-blocking (cpu util percentage since last call)
  measurements['cpu.freq.current'] = psutil.cpu_freq().current

  # Temp
  sensor_temps = psutil.sensors_temperatures()
  for key in sensor_temps.keys():
    # Take the first temp reading for each kind of device (CPU, GPU, Disk, etc.)
    value = sensor_temps[key][0].current
    measurements[f'temp.{key}.current'] = value

  # Memory
  memory_util = psutil.virtual_memory()
  measurements['mem.total'] = memory_util.total
  measurements['mem.available'] = memory_util.available
  measurements['mem.used'] = memory_util.used
  measurements['mem.percent_used'] = memory_util.percent

  # Disk
  disk_io_counters = psutil.disk_io_counters()
  measurements['mem.read_bytes_since_boot'] = disk_io_counters.read_bytes
  measurements['mem.write_bytes_since_boot'] = disk_io_counters.write_bytes

  # Network
  net_io_counters = psutil.net_io_counters()
  measurements['net.bytes_sent_since_boot'] = net_io_counters.bytes_sent
  measurements['net.bytes_recv_since_boot'] = net_io_counters.bytes_recv

  # GPU
  gpus = GPUtil.getGPUs()
  if gpus:
    gpu_count = len(gpus)
    measurements['gpu.count'] = gpu_count
    avg_gpu_load = 0
    avg_gpu_memory_util = 0
    avg_gpu_memory_total = 0
    avg_gpu_memory_used = 0
    avg_gpu_memory_free = 0
    avg_gpu_temperature = 0
    for gpu in gpus:
      idx = gpu.id
      measurements[f'gpu.{idx}.compute.util'] = gpu.load
      measurements[f'gpu.{idx}.mem.util'] = gpu.memoryUtil
      measurements[f'gpu.{idx}.mem.total'] = gpu.memoryTotal
      measurements[f'gpu.{idx}.mem.used'] = gpu.memoryUsed
      measurements[f'gpu.{idx}.mem.free'] = gpu.memoryFree
      measurements[f'gpu.{idx}.temp.current'] = gpu.temperature
      # Note: GPU wattage was not available from gputil as of writing
      avg_gpu_load += gpu.load
      avg_gpu_memory_util += gpu.memoryUtil
      avg_gpu_memory_total += gpu.memoryTotal
      avg_gpu_memory_used += gpu.memoryUsed
      avg_gpu_memory_free += gpu.memoryFree
      avg_gpu_temperature += gpu.temperature
    measurements['gpu.avg.compute.util'] = avg_gpu_load / gpu_count
    measurements['gpu.avg.mem.util'] = avg_gpu_memory_util / gpu_count
    measurements['gpu.avg.mem.total'] = avg_gpu_memory_total / gpu_count
    measurements['gpu.avg.mem.used'] = avg_gpu_memory_used / gpu_count
    measurements['gpu.avg.mem.free'] = avg_gpu_memory_free / gpu_count
    measurements['gpu.avg.temp.current'] = avg_gpu_temperature / gpu_count

  return measurements


def _get_git_commit_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse',
                                  'HEAD']).decode('ascii').strip()


def _get_git_branch() -> str:
  return subprocess.check_output(['git', 'branch',
                                  '--show-current']).decode('ascii').strip()


def _get_cpu_model_name() -> str:
  output = subprocess.check_output(['lscpu']).decode('ascii').strip()
  return re.findall(r"(?=Model name:\s{1,}).*",
                    output)[0].split('Model name:')[1].strip()


def _get_os_package_list() -> str:
  return subprocess.check_output(['dpkg', '-l']).decode('ascii').strip()


def _get_pip_package_list() -> str:
  return subprocess.check_output(['pip', 'freeze']).decode('ascii').strip()


def _is_primitive_type(item: Any) -> bool:
  primitive = (float, int, str, bool)
  return isinstance(item, primitive)


def _get_extra_metadata_as_dict(extra_metadata_string_list: list) -> dict:
  """Parse the extra_metadata CLI argument from string into dict.

  For example this program was executed with --record_extra_metadata="key=value"
  then {'extra.key':'value'} is returned.
  """
  metadata = {}
  if not extra_metadata_string_list:
    return metadata
  try:
    for item in extra_metadata_string_list:
      key, value = item.split("=")
      metadata['extra.' + key] = value
  except:
    logging.error(
        'Failed to parse extra_metadata CLI arguments. Please check your '
        'command.')
    raise
  return metadata


def _get_workload_properties(workload: spec.Workload) -> dict:
  """Parse the workload class to extract its basic properties.

  Each workload has properties such as target_value, num_train_examples,
  train_stddev, etc. We want to record these to enable follow-up analysis.
  Instead of hardcoding a list of properties to be extracted, we automatically
  extract any float, int, str, or bool because each workload is different and
  may change."""
  workload_properties = {}
  skip_list = ['param_shapes']
  keys = [
      key for key in dir(workload)
      if not key.startswith('_') and key not in skip_list
  ]
  for key in keys:
    try:
      attr = getattr(workload, key)
    except:
      logging.warn(
          f'Unable to record workload.{key} information. Continuing without it.'
      )
    if _is_primitive_type(attr):
      workload_properties[f'workload.{key}'] = attr
  return workload_properties


class NoOpRecorder(object):
  """ This dummy class returns None for all possible function calls.

  This class makes it easy to turn off all functionality by swapping in this
  class.
   """

  def no_op(self, *args, **kw):
    pass

  def __getattr__(self, _):
    return self.no_op


class Recorder:
  """Save information about the training progress of a workload to disk.

  This class should be instantiated once per workload. Logging files are written
  to seperate workload specific folders.

  Four files are written to the `logging_dir` folder:
    1. `metadata.json` is created at the start of a workload and it includes the
      datetime, workload name, and system configuration.
    2. `packages.txt` is created at the start of a workload and it includes a
      list of the currently installed OS and python packages.
    3. `trial_[n]/measurements.csv` is created for each hyperparameter tuning
      trial and a row is appended for every model evaluation. The information
      included is loss, accuracy, training step, time elapsed, hparams, workload
      properties, and hardware utilization.
    4. `trial_[n]/metadata.json` is created at the end of each hyperparameter
      tuning trial and includes the result of the trial. The last row of
      measurements.csv and this file are very similar but can differ in the
      number of steps and runtime in one situation: when the tuning trial ran
      out of time but completed one or more steps after the last model
      evaluation.

  Joining measurement CSVs across workloads or hyperparameter tuning trials is
  left to users, although a convienence function called "concatenate_csvs()" is
  provided. The data format of "measurements.csv" is designed to be safe to
  arbitrarily join CSVs without attribute name conflicts across both workloads
  and across hyperparameter tuning trials.
  """

  def __init__(self,
               workload: spec.Workload,
               workload_name: str,
               logging_dir: str,
               submission_path: str,
               tuning_ruleset: str,
               tuning_search_space_path: Optional[str] = None,
               num_tuning_trials: Optional[int] = None):
    self.workload_name = workload_name
    self.workload = workload
    self.logging_dir = logging_dir
    self.submission_path = submission_path
    self.tuning_ruleset = tuning_ruleset
    self.tuning_search_space_path = tuning_search_space_path
    self.num_tuning_trials = num_tuning_trials
    self.status = 'INCOMPLETE'
    self.last_epoch_evaluated = None
    self.workload_log_dir = os.path.join(self.logging_dir, self.workload_name)
    if os.path.isdir(self.workload_log_dir):
      logging.warn(
          'Warning: You may overwrite data because recording output path '
          f'already exists: {self.workload_log_dir}')
    # Record initial information about workload at startup
    self._write_workload_metadata_file()
    self._write_package_list_file()

  def _write_workload_metadata_file(self, score: float = None):
    """Write "metadata.json" to disk.

    It is is created at the start of a workload and includes the datetime,
    workload name, and system configuration."""
    metadata = {}

    # Workload Information
    metadata['workload'] = self.workload_name
    metadata['datetime'] = datetime.now().isoformat()
    metadata['status'] = self.status
    if score:
      metadata['score'] = score
    metadata['logging_dir'] = self.logging_dir
    metadata['submission_path'] = self.submission_path
    metadata['tuning_ruleset'] = self.tuning_ruleset
    metadata['num_tuning_trials'] = self.num_tuning_trials

    if self.tuning_search_space_path:
      metadata['tuning_search_space_path'] = self.tuning_search_space_path
      with open(self.tuning_search_space_path, 'r') as search_space_file:
        tuning_search_space = json.load(search_space_file)
        metadata['tuning_search_space'] = tuning_search_space

    workload_properties = _get_workload_properties(self.workload)
    metadata.update(workload_properties)

    if FLAGS.extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      metadata.update(extra_metadata)

    # System Information
    metadata['os_platform'] = \
        platform.platform()  # Ex. 'Linux-5.4.48-x86_64-with-glibc2.29'
    metadata['python_version'] = platform.python_version()  # Ex. '3.8.10'
    metadata['python_compiler'] = platform.python_compiler()  # Ex. 'GCC 9.3.0'
    # Note: do not store hostname as that may be sensitive

    try:
      metadata['git_branch'] = _get_git_branch()
      metadata['git_commit_hash'] = _get_git_commit_hash()
      # Note: do not store git repo url as it may be sensitive or contain a
      # secret.
    except:
      logging.warn('Unable to record git information. Continuing without it.')

    try:
      metadata['cpu_model_name'] = _get_cpu_model_name()
      metadata['cpu_count'] = psutil.cpu_count()
    except:
      logging.warn('Unable to record cpu information. Continuing without it.')

    gpus = GPUtil.getGPUs()
    if gpus:
      try:
        metadata['gpu_model_name'] = gpus[0].name
        metadata['gpu_count'] = len(gpus)
        metadata['gpu_driver'] = gpus[0].driver
      except:
        logging.warn('Unable to record gpu information. Continuing without it.')

    # Save workload metadata.json
    os.makedirs(self.workload_log_dir, exist_ok=True)
    metadata_filepath = os.path.join(self.workload_log_dir, 'metadata.json')
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
      json.dump(metadata, f, ensure_ascii=False, indent=4)

  def _write_package_list_file(self):
    """Write "packages.txt" to disk.

    It is created at the start of a workload and includes a list of the
    currently installed OS and python packages."""
    # Get package lists
    try:
      os_package_list = _get_os_package_list()
      pip_package_list = _get_pip_package_list()
    except:
      logging.warn(
          'Unable to record package information. Continuing without it.')
      return

    # Save to disk
    packages_filepath = os.path.join(self.workload_log_dir, 'packages.txt')
    with open(packages_filepath, 'w', encoding='utf-8') as f:
      f.write('Python Packages:\n')
      f.write(pip_package_list)
      f.write('\n\nOS Packages:\n')
      f.write(os_package_list)

  def _write_trial_metadata_file(
      self, workload: spec.Workload,
      hyperparameters: Optional[spec.Hyperparamters], trial_idx: int,
      global_step: int, batch_size: int, latest_eval_result: dict,
      global_start_time: float, accumulated_submission_time: float,
      goal_reached: bool, is_time_remaining: bool, training_complete: bool):
    metadata = self._get_eval_measurements(
        workload, hyperparameters, trial_idx, global_step, batch_size,
        latest_eval_result, global_start_time, accumulated_submission_time,
        goal_reached, is_time_remaining, training_complete)
    metadata['status'] = 'COMPLETE'

    # Save trial metadata.json
    metadata_filepath = os.path.join(self.workload_log_dir,
                                     'trial_' + str(trial_idx), 'metadata.json')
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
      json.dump(metadata, f, ensure_ascii=False, indent=4)

  def trial_complete(self, workload: spec.Workload,
                     hyperparameters: Optional[spec.Hyperparamters],
                     trial_idx: int, global_step: int, batch_size: int,
                     latest_eval_result: dict, global_start_time: float,
                     accumulated_submission_time: float, goal_reached: bool,
                     is_time_remaining: bool, training_complete: bool):
    self._write_trial_metadata_file(workload, hyperparameters, trial_idx,
                                    global_step, batch_size, latest_eval_result,
                                    global_start_time,
                                    accumulated_submission_time, goal_reached,
                                    is_time_remaining, training_complete)

  def workload_complete(self, score: float):
    """At the end of the workload write COMPLETE to the metadata file."""
    self.status = 'COMPLETE'
    self._write_workload_metadata_file(score)

  def _get_eval_measurements(self, workload: spec.Workload,
                             hyperparameters: Optional[spec.Hyperparamters],
                             trial_idx: int, global_step: int, batch_size: int,
                             latest_eval_result: dict, global_start_time: float,
                             accumulated_submission_time: float,
                             goal_reached: bool, is_time_remaining: bool,
                             training_complete: bool) -> dict:
    """Collect all evaluation measurements and metadata in one dict."""
    measurements = {}
    measurements['workload'] = self.workload_name
    measurements['trial_idx'] = trial_idx

    # Record training measurements
    measurements['datetime'] = datetime.now().isoformat()
    measurements['accumulated_submission_time'] = accumulated_submission_time
    if latest_eval_result:
      for key, value in latest_eval_result.items():
        measurements[key] = value
    measurements['global_step'] = global_step
    steps_per_epoch = workload.num_train_examples // batch_size
    measurements['epoch'] = global_step / steps_per_epoch
    measurements['steps_per_epoch'] = steps_per_epoch
    measurements['global_start_time'] = global_start_time
    measurements['goal_reached'] = goal_reached
    measurements['is_time_remaining'] = is_time_remaining
    measurements['training_complete'] = training_complete

    # Record hyperparameters
    measurements['batch_size'] = batch_size
    if hyperparameters:
      hparams_dict = hyperparameters._asdict()
      # prefix every key with "hparam." to make more human-readable and to
      # avoid overlap with other possible kys in the measurements dict.
      hparams_dict = {f'hparam.{k}': v for k, v in hparams_dict.items()}
      measurements.update(hparams_dict)

    # Record workload properties
    workload_properties = _get_workload_properties(workload)
    measurements.update(workload_properties)

    if FLAGS.extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      measurements.update(extra_metadata)

    # Record utilization
    utilization_measurements = _get_utilization()
    measurements.update(utilization_measurements)

    return measurements

  def save_eval(self, workload: spec.Workload,
                hyperparameters: Optional[spec.Hyperparamters], trial_idx: int,
                global_step: int, batch_size: int, latest_eval_result: dict,
                global_start_time: float, accumulated_submission_time: float,
                goal_reached: bool, is_time_remaining: bool,
                training_complete: bool):
    """"Write or append to "measurements.csv".

    A "measurements.csv" is created for each hyperparameter tuning trial and a
    row is appended for every model evaluation. The information included is
    loss, accuracy, training step, time elapsed, hparams, workload properties,
    and hardware utilization."""
    measurements = self._get_eval_measurements(
        workload, hyperparameters, trial_idx, global_step, batch_size,
        latest_eval_result, global_start_time, accumulated_submission_time,
        goal_reached, is_time_remaining, training_complete)

    # Save to CSV file
    trial_output_path = os.path.join(self.workload_log_dir,
                                     'trial_' + str(trial_idx))
    os.makedirs(trial_output_path, exist_ok=True)
    csv_path = os.path.join(trial_output_path, 'measurements.csv')
    logging.info(f'Recording measurements to: {csv_path}')
    self._append_to_csv(measurements, csv_path)

  def check_eval_frequency_override(self, workload: spec.Workload,
                                    global_step: int, batch_size: int):
    """Parse the eval_frequency_override cli argument and return whether or not
    the user wants to eval this step."""
    if not FLAGS.eval_frequency_override:
      return False

    try:
      freq, unit = FLAGS.eval_frequency_override.split(' ')
      freq = int(freq)
      assert (unit in ['epoch', 'step'])
    except:
      logging.error(
          'Failed to parse eval_frequency_override CLI arguments. Please check '
          'your command.')
      raise

    if unit == 'step':
      if global_step % freq == 0:
        return True

    elif unit == 'epoch':
      steps_per_epoch = workload.num_train_examples // batch_size
      epoch = global_step // steps_per_epoch
      if epoch != self.last_epoch_evaluated:
        self.last_epoch_evaluated = epoch
        return True

  def _append_to_csv(self, data: dict, csv_path: str):
    """Open a CSV, append more data, and save back to disk."""
    if os.path.isfile(csv_path):
      df = pd.read_csv(csv_path)
    else:
      df = pd.DataFrame()  # Initialize empty dataframe if no data is saved yet
    df = df.append(data, ignore_index=True)
    df.to_csv(csv_path, index=False)
