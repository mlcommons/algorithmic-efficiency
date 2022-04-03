"""Save information about the training progress of a workload to disk.

Use the `--logging_dir` CLI argument with submission_runner.py to enable this
functionality.
"""
from datetime import datetime
import glob
import json
import os
import platform
import re
import subprocess
from typing import Any, Optional

from absl import logging
import GPUtil
import pandas as pd
import psutil

from algorithmic_efficiency import spec


def concatenate_csvs(path: str, output_name='all.csv'):
  """Join all CSV files in a given folder recursively.

   This function is provided as a convienence to users to join their CSV data.
   We leave it to users to do this because we do not want to create data
   duplication if there is no user need."""
  search_path = os.path.join(path, '**/*.csv')
  input_csvs = list(glob.iglob(search_path, recursive=True))
  if input_csvs:
    df = pd.read_csv(input_csvs.pop())
    for file in input_csvs:
      df = df.append(pd.read_csv(file))
    output_filepath = os.path.join(path, output_name)
    df.to_csv(output_filepath, index=False)


def _get_utilization() -> dict:
  """Collect system-wide hardware performance measurements.

  High-level utilization measurements for the GPU (if available), CPU,
  temperature, memory, disk, and network.

  The performance measurements are all system-wide because we can't guarentee
  how many processes Jax or PyTorch will start and not all measurements are
  available on a per-process basis (eg. network).
  """
  util_data = {}

  # CPU
  util_data['cpu.util.avg_percent_since_last'] = psutil.cpu_percent(
      interval=None)  # non-blocking (cpu util percentage since last call)
  util_data['cpu.freq.current'] = psutil.cpu_freq().current

  # Temp
  sensor_temps = psutil.sensors_temperatures()
  for key in sensor_temps.keys():
    # Take the first temp reading for each kind of device (CPU, GPU, Disk, etc.)
    value = sensor_temps[key][0].current
    util_data[f'temp.{key}.current'] = value

  # Memory
  memory_util = psutil.virtual_memory()
  util_data['mem.total'] = memory_util.total
  util_data['mem.available'] = memory_util.available
  util_data['mem.used'] = memory_util.used
  util_data['mem.percent_used'] = memory_util.percent

  # Disk
  disk_io_counters = psutil.disk_io_counters()
  util_data['mem.read_bytes_since_boot'] = disk_io_counters.read_bytes
  util_data['mem.write_bytes_since_boot'] = disk_io_counters.write_bytes

  # Network
  net_io_counters = psutil.net_io_counters()
  util_data['net.bytes_sent_since_boot'] = net_io_counters.bytes_sent
  util_data['net.bytes_recv_since_boot'] = net_io_counters.bytes_recv

  # GPU
  gpus = GPUtil.getGPUs()
  if gpus:
    gpu_count = len(gpus)
    util_data['gpu.count'] = gpu_count
    avg_gpu_load = 0
    avg_gpu_memory_util = 0
    avg_gpu_memory_total = 0
    avg_gpu_memory_used = 0
    avg_gpu_memory_free = 0
    avg_gpu_temperature = 0
    for gpu in gpus:
      idx = gpu.id
      util_data[f'gpu.{idx}.compute.util'] = gpu.load
      util_data[f'gpu.{idx}.mem.util'] = gpu.memoryUtil
      util_data[f'gpu.{idx}.mem.total'] = gpu.memoryTotal
      util_data[f'gpu.{idx}.mem.used'] = gpu.memoryUsed
      util_data[f'gpu.{idx}.mem.free'] = gpu.memoryFree
      util_data[f'gpu.{idx}.temp.current'] = gpu.temperature
      # Note: GPU wattage was not available from gputil as of writing
      avg_gpu_load += gpu.load
      avg_gpu_memory_util += gpu.memoryUtil
      avg_gpu_memory_total += gpu.memoryTotal
      avg_gpu_memory_used += gpu.memoryUsed
      avg_gpu_memory_free += gpu.memoryFree
      avg_gpu_temperature += gpu.temperature
    util_data['gpu.avg.compute.util'] = avg_gpu_load / gpu_count
    util_data['gpu.avg.mem.util'] = avg_gpu_memory_util / gpu_count
    util_data['gpu.avg.mem.total'] = avg_gpu_memory_total / gpu_count
    util_data['gpu.avg.mem.used'] = avg_gpu_memory_used / gpu_count
    util_data['gpu.avg.mem.free'] = avg_gpu_memory_free / gpu_count
    util_data['gpu.avg.temp.current'] = avg_gpu_temperature / gpu_count

  return util_data


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


def _get_extra_metadata_as_dict(extra_metadata: list) -> dict:
  """Parse the extra_metadata CLI argument from string into dict.

  For example this program was executed with --extra_metadata="key=value"
  then {'extra.key':'value'} is returned.
  """
  metadata = {}
  if not extra_metadata:
    return metadata
  for item in extra_metadata:
    try:
      key, value = item.split("=")
      metadata['extra.' + key] = value
    except:
      raise ValueError(
          f'Failed to parse this extra_metadata CLI argument: {item}. ' +
          'Please check your command.')
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

  def _no_op(self, *args, **kw):
    pass

  def __getattr__(self, _):
    return self._no_op


class Recorder:
  """Save information about the training progress of a workload to disk.

  This class should be instantiated once per workload. Logging files are written
  to seperate workload specific folders.

  Four files are written to the `logging_dir` folder:
    1. `workload_results.json` includes the datetime, workload name, score,
      and the system configuration used to generate the result.
    2. `trial_[n]/trial_results.json` is created at the end of each
      hyperparameter tuning trial and includes the result of the trial. The
      last row of eval_results.csv and this file are very similar but can
      differ in the number of steps and runtime in one situation: when the
      tuning trial ran out of time but completed one or more steps after the
      last model evaluation.
    3. `trial_[n]/eval_results.csv` is created for each hyperparameter tuning
      trial. A row is appended for every model evaluation. The information
      included is loss, accuracy, training step, time elapsed, hparams, workload
      properties, and hardware utilization.
    4. `packages.txt` is created at the start of a workload and it includes a
      list of the currently installed OS and python packages.
  """

  def __init__(
      self,
      workload: spec.Workload,
      workload_name: str,
      logging_dir: str,
      submission_path: str,
      tuning_ruleset: str,
      tuning_search_space_path: Optional[str] = None,
      num_tuning_trials: Optional[int] = None,
      extra_metadata: Optional[str] = None,
  ):
    self._workload_name = workload_name
    self._workload = workload
    self._logging_dir = logging_dir
    self._submission_path = submission_path
    self._tuning_ruleset = tuning_ruleset
    self._tuning_search_space_path = tuning_search_space_path
    self._num_tuning_trials = num_tuning_trials
    self._extra_metadata = extra_metadata
    self._last_epoch_evaluated = None
    self._workload_log_dir = os.path.join(self._logging_dir,
                                          self._workload_name)
    if os.path.isdir(self._workload_log_dir):
      logging.warn(
          'Warning: You may overwrite data because recording output path '
          f'already exists: {self._workload_log_dir}')
    # Record initial information about workload at startup
    self._write_workload_results_file(status='INCOMPLETE')
    self._write_package_list_file()

  def _write_workload_results_file(
      self,
      score: float = None,
      status: str = None,
  ):
    """Write "workload_results.json" to disk.

    It is is created at the start of a workload and includes the datetime,
    workload name, and system configuration.

    Args:
      score: (optional) final score of the training run
      status: (optional) status the training run ex. "INCOMPLETE" if training is
        in progress
    """
    workload_data = {}

    # Workload Information
    workload_data['workload'] = self._workload_name
    workload_data['datetime'] = datetime.now().isoformat()
    if status:
      workload_data['status'] = status
    if score:
      workload_data['score'] = score
    workload_data['logging_dir'] = self._logging_dir
    workload_data['submission_path'] = self._submission_path
    workload_data['tuning_ruleset'] = self._tuning_ruleset
    workload_data['num_tuning_trials'] = self._num_tuning_trials

    if self._tuning_search_space_path:
      workload_data['tuning_search_space_path'] = self._tuning_search_space_path
      with open(self._tuning_search_space_path, 'r') as search_space_file:
        tuning_search_space = json.load(search_space_file)
        workload_data['tuning_search_space'] = tuning_search_space

    workload_properties = _get_workload_properties(self._workload)
    workload_data.update(workload_properties)

    if self._extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(self._extra_metadata)
      workload_data.update(extra_metadata)

    # System Information
    workload_data['os_platform'] = \
        platform.platform()  # Ex. 'Linux-5.4.48-x86_64-with-glibc2.29'
    workload_data['python_version'] = platform.python_version()  # Ex. '3.8.10'
    workload_data['python_compiler'] = platform.python_compiler(
    )  # Ex. 'GCC 9.3.0'
    # Note: do not store hostname as that may be sensitive

    try:
      workload_data['git_branch'] = _get_git_branch()
      workload_data['git_commit_hash'] = _get_git_commit_hash()
      # Note: do not store git repo url as it may be sensitive or contain a
      # secret.
    except:
      logging.warn('Unable to record git information. Continuing without it.')

    try:
      workload_data['cpu_model_name'] = _get_cpu_model_name()
      workload_data['cpu_count'] = psutil.cpu_count()
    except:
      logging.warn('Unable to record cpu information. Continuing without it.')

    gpus = GPUtil.getGPUs()
    if gpus:
      try:
        workload_data['gpu_model_name'] = gpus[0].name
        workload_data['gpu_count'] = len(gpus)
        workload_data['gpu_driver'] = gpus[0].driver
      except:
        logging.warn('Unable to record gpu information. Continuing without it.')

    # Save workload_results.json
    os.makedirs(self._workload_log_dir, exist_ok=True)
    results_filepath = os.path.join(self._workload_log_dir,
                                    'workload_results.json')
    with open(results_filepath, 'w', encoding='utf-8') as f:
      json.dump(workload_data, f, ensure_ascii=False, indent=4)

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
    packages_filepath = os.path.join(self._workload_log_dir, 'packages.txt')
    with open(packages_filepath, 'w', encoding='utf-8') as f:
      f.write('Python Packages:\n')
      f.write(pip_package_list)
      f.write('\n\nOS Packages:\n')
      f.write(os_package_list)

  def trial_complete(
      self,
      workload: spec.Workload,
      hyperparameters: Optional[spec.Hyperparamters],
      trial_idx: int,
      global_step: int,
      batch_size: int,
      latest_eval_result: dict,
      global_start_time: float,
      accumulated_submission_time: float,
      goal_reached: bool,
      is_time_remaining: bool,
      training_complete: bool,
  ):
    trial_data = self._get_eval_measurements(workload,
                                             hyperparameters,
                                             trial_idx,
                                             global_step,
                                             batch_size,
                                             latest_eval_result,
                                             global_start_time,
                                             accumulated_submission_time,
                                             goal_reached,
                                             is_time_remaining,
                                             training_complete)
    trial_data['status'] = 'COMPLETE'

    # Save trial_results.json
    results_filepath = os.path.join(self._workload_log_dir,
                                    'trial_' + str(trial_idx),
                                    'trial_results.json')
    with open(results_filepath, 'w', encoding='utf-8') as f:
      json.dump(trial_data, f, ensure_ascii=False, indent=4)

  def workload_complete(self, score: float):
    """Set status to 'COMPLETE' in the workload_results.json file."""
    self._write_workload_results_file(score=score, status='COMPLETE')

  def _get_eval_measurements(
      self,
      workload: spec.Workload,
      hyperparameters: Optional[spec.Hyperparamters],
      trial_idx: int,
      global_step: int,
      batch_size: int,
      latest_eval_result: dict,
      global_start_time: float,
      accumulated_submission_time: float,
      goal_reached: bool,
      is_time_remaining: bool,
      training_complete: bool,
  ) -> dict:
    """Collect all evaluation measurements and metadata in one dict."""
    eval_data = {}
    eval_data['workload'] = self._workload_name
    eval_data['trial_idx'] = trial_idx

    # Record training measurements
    eval_data['datetime'] = datetime.now().isoformat()
    eval_data['accumulated_submission_time'] = accumulated_submission_time
    if latest_eval_result:
      for key, value in latest_eval_result.items():
        eval_data[key] = value
    eval_data['global_step'] = global_step
    steps_per_epoch = workload.num_train_examples // batch_size
    eval_data['epoch'] = global_step / steps_per_epoch
    eval_data['steps_per_epoch'] = steps_per_epoch
    eval_data['global_start_time'] = global_start_time
    eval_data['goal_reached'] = goal_reached
    eval_data['is_time_remaining'] = is_time_remaining
    eval_data['training_complete'] = training_complete

    # Record hyperparameters
    eval_data['batch_size'] = batch_size
    if hyperparameters:
      hparams_dict = hyperparameters._asdict()
      # prefix every key with "hparam." to make more human-readable and to
      # avoid overlap with other possible keys in the measurements dict.
      hparams_dict = {f'hparam.{k}': v for k, v in hparams_dict.items()}
      eval_data.update(hparams_dict)

    # Record workload properties
    workload_properties = _get_workload_properties(workload)
    eval_data.update(workload_properties)

    if self._extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(self._extra_metadata)
      eval_data.update(extra_metadata)

    # Record utilization
    utilization_measurements = _get_utilization()
    eval_data.update(utilization_measurements)

    return eval_data

  def save_eval(
      self,
      workload: spec.Workload,
      hyperparameters: Optional[spec.Hyperparamters],
      trial_idx: int,
      global_step: int,
      batch_size: int,
      latest_eval_result: dict,
      global_start_time: float,
      accumulated_submission_time: float,
      goal_reached: bool,
      is_time_remaining: bool,
      training_complete: bool,
  ):
    """"Write or append to "eval_results.csv".

    A "eval_results.csv" is created for each hyperparameter tuning trial and a
    row is appended for every model evaluation. The information included is
    loss, accuracy, training step, time elapsed, hparams, workload properties,
    and hardware utilization."""
    eval_data = self._get_eval_measurements(workload,
                                            hyperparameters,
                                            trial_idx,
                                            global_step,
                                            batch_size,
                                            latest_eval_result,
                                            global_start_time,
                                            accumulated_submission_time,
                                            goal_reached,
                                            is_time_remaining,
                                            training_complete)

    # Save to CSV file
    results_filepath = os.path.join(self._workload_log_dir,
                                    'trial_' + str(trial_idx))
    os.makedirs(results_filepath, exist_ok=True)
    csv_path = os.path.join(results_filepath, 'eval_results.csv')
    logging.info(f'Recording measurements to: {csv_path}')
    self._append_to_csv(eval_data, csv_path)

  def check_eval_frequency_override(
      self,
      eval_frequency_override: str,
      workload: spec.Workload,
      global_step: int,
      batch_size: int,
  ):
    """Parse the eval_frequency_override CLI argument and return whether or not
    the user wants to eval this step."""
    if not eval_frequency_override:
      return False

    try:
      freq, unit = eval_frequency_override.split(' ')
      freq = int(freq)
      assert freq > 0
      assert (unit in ['epoch', 'step'])
    except:
      raise ValueError(
          'Failed to parse eval_frequency_override CLI argument: ' +
          f'{eval_frequency_override}. Please check your command.')

    if unit == 'step':
      if global_step % freq == 0:
        return True

    elif unit == 'epoch':
      steps_per_epoch = workload.num_train_examples // batch_size
      epoch = global_step // steps_per_epoch
      if epoch != self._last_epoch_evaluated:
        self._last_epoch_evaluated = epoch
        return True

  def _append_to_csv(
      self,
      data: dict,
      csv_path: str,
  ):
    """Open a CSV, append more data, and save back to disk."""
    if os.path.isfile(csv_path):
      df = pd.read_csv(csv_path)
    else:
      df = pd.DataFrame()  # Initialize empty dataframe if no data is saved yet
    df = df.append(data, ignore_index=True)
    df.to_csv(csv_path, index=False)
