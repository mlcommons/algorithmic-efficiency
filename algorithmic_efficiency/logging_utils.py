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


def concatenate_csvs(path: str) -> None:
  """Join all files named "metrics.csv" in a given folder recursively.

  In this logging module, one "metrics.csv" is produced at the granularity of
  each hyperparameter tuning run. This function is provided as a convienence to
  users to join their CSV data. We leave it to users to do this because we do
  not want to create data duplication if there is no user need."""
  search_path = os.path.join(path, '**/metrics.csv')
  input_csvs = list(glob.iglob(search_path, recursive=True))
  if input_csvs:
    df = pd.read_csv(input_csvs.pop())
    for file in input_csvs:
      df = df.append(pd.read_csv(file))
    output_filepath = os.path.join(path, 'all_metrics.csv')
    df.to_csv(output_filepath, index=False)


def _get_utilization() -> dict:
  """Collect system-wide hardware performance metrics.

  High-level utilization metrics for the GPU (if available), CPU, temperature,
  memory, disk, and network.

  The performance metrics are all system-wide because we can't guarentee how
  many processes Jax or PyTorch will start and not all metrics are available on
  a per-process basis (eg. network).
  """
  metrics = {}

  # CPU
  metrics['cpu.util.avg_percent_since_last'] = psutil.cpu_percent(
      interval=None)  # non-blocking (cpu util percentage since last call)
  metrics['cpu.freq.current'] = psutil.cpu_freq().current

  # Temp
  sensor_temps = psutil.sensors_temperatures()
  for key in sensor_temps.keys():
    # Take the first temp reading for each kind of device (CPU, GPU, Disk, etc.)
    value = sensor_temps[key][0].current
    metrics[f'temp.{key}.current'] = value

  # Memory
  memory_util = psutil.virtual_memory()
  metrics['mem.total'] = memory_util.total
  metrics['mem.available'] = memory_util.available
  metrics['mem.used'] = memory_util.used
  metrics['mem.percent_used'] = memory_util.percent

  # Disk
  disk_io_counters = psutil.disk_io_counters()
  metrics['mem.read_bytes_since_boot'] = disk_io_counters.read_bytes
  metrics['mem.write_bytes_since_boot'] = disk_io_counters.write_bytes

  # Network
  net_io_counters = psutil.net_io_counters()
  metrics['net.bytes_sent_since_boot'] = net_io_counters.bytes_sent
  metrics['net.bytes_recv_since_boot'] = net_io_counters.bytes_recv

  # GPU
  gpus = GPUtil.getGPUs()
  if gpus:
    gpu_count = len(gpus)
    metrics[f'gpu.count'] = gpu_count
    avg_gpu_load = 0
    avg_gpu_memoryUtil = 0
    avg_gpu_memoryTotal = 0
    avg_gpu_memoryUsed = 0
    avg_gpu_memoryFree = 0
    avg_gpu_temperature = 0
    for gpu in gpus:
      id = gpu.id
      metrics[f'gpu.{id}.compute.util'] = gpu.load
      metrics[f'gpu.{id}.mem.util'] = gpu.memoryUtil
      metrics[f'gpu.{id}.mem.total'] = gpu.memoryTotal
      metrics[f'gpu.{id}.mem.used'] = gpu.memoryUsed
      metrics[f'gpu.{id}.mem.free'] = gpu.memoryFree
      metrics[f'gpu.{id}.temp.current'] = gpu.temperature
      # Note: GPU wattage was not available from gputil as of writing
      avg_gpu_load += gpu.load
      avg_gpu_memoryUtil += gpu.memoryUtil
      avg_gpu_memoryTotal += gpu.memoryTotal
      avg_gpu_memoryUsed += gpu.memoryUsed
      avg_gpu_memoryFree += gpu.memoryFree
      avg_gpu_temperature += gpu.temperature
    metrics[f'gpu.avg.compute.util'] = avg_gpu_load / gpu_count
    metrics[f'gpu.avg.mem.util'] = avg_gpu_memoryUtil / gpu_count
    metrics[f'gpu.avg.mem.total'] = avg_gpu_memoryTotal / gpu_count
    metrics[f'gpu.avg.mem.used'] = avg_gpu_memoryUsed / gpu_count
    metrics[f'gpu.avg.mem.free'] = avg_gpu_memoryFree / gpu_count
    metrics[f'gpu.avg.temp.current'] = avg_gpu_temperature / gpu_count

  return metrics


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
  then {'key':'value'} is returned.
  """
  metadata = {}
  try:
    for item in extra_metadata_string_list:
      key, value = item.split("=")
      metadata[key] = value
  except Exception as e:
    logging.error(
        'Failed to parse extra_metadata CLI arguments. Please check your command.'
    )
    raise e
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
    except Exception as e:
      logging.warn(
          f'Unable to record workload.{key} information. Continuing without it.'
      )
    if _is_primitive_type(attr):
      workload_properties[f'workload.{key}'] = attr
  return workload_properties


class No_Op_Recorder(object):
  """ This dummy class returns None for all possible function calls.

  This class makes it easy to turn off all functionality by swapping in this
  class.
   """

  def no_op(*args, **kw):
    pass

  def __getattr__(self, _):
    return self.no_op


class Recorder:
  """Save information about the training progress of a workload to disk.

  This class should be instantiated once per workload. Logging files are written
  to seperate workload specific folders.

  Three files are written to the given "log_dir" folder:
  1. "metadata.json" is created at the start of a workload and it includes the
     datetime, workload name, and system configuration.
  2. "metrics.csv" is created for each hyperparameter tuning trail and a row is
     appended for every model evaluation. The information included is loss,
     accuracy, training step, time elapsed, hparams, workload properties,
     and hardware utilization.
  3. "packages.txt" is created at the start of a workload and it includes a
     list of the currently installed OS and python packages.

  Joining metric CSVs across workloads or across hyperparameter tuning trials is
  left to users, although a convienence function called "concatenate_csvs()" is
  provided. The data format of "metrics.csv" is designed to be safe to
  arbitrarily join CSVs without attribute name conflicts across both workloads
  and across hyperparameter tuning trials.
  """

  def __init__(self, workload: spec.Workload, workload_name: str,
               log_dir: str) -> None:
    self.workload_name = workload_name
    self.log_dir = log_dir
    self.workload_log_dir = os.path.join(self.log_dir, self.workload_name)
    if os.path.isdir(self.workload_log_dir):
      logging.warn(
          'Warning: You may overwrite data because recording output path '
          f'already exists: {self.workload_log_dir}')
    # Record initial information about workload at startup
    self._write_metadata_file(workload)
    self._write_package_list_file()

  def _write_metadata_file(self, workload: spec.Workload) -> None:
    """Write "metadata.json" to disk.

    It is is created at the start of a workload and includes the datetime,
    workload name, and system configuration."""
    metadata = {}
    metadata['workload'] = self.workload_name
    metadata['log_dir'] = self.log_dir
    metadata['datetime'] = datetime.now().isoformat()
    metadata['python_version'] = platform.python_version()  # Ex. '3.8.10'
    metadata['python_compiler'] = platform.python_compiler()  # Ex. 'GCC 9.3.0'
    metadata['os_platform'] = \
        platform.platform()  # Ex. 'Linux-5.4.48-x86_64-with-glibc2.29'
    # Note: do not store hostname as that may be sensitive

    try:
      metadata['git_branch'] = _get_git_branch()
      metadata['git_commit_hash'] = _get_git_commit_hash()
      # Note: do not store git repo url as it may be sensitive or contain a
      # secret.
    except Exception as e:
      logging.warn('Unable to record git information. Continuing without it.')

    try:
      metadata['cpu_model_name'] = _get_cpu_model_name()
      metadata['cpu_count'] = psutil.cpu_count()
    except Exception as e:
      logging.warn('Unable to record cpu information. Continuing without it.')

    gpus = GPUtil.getGPUs()
    if gpus:
      try:
        metadata['gpu_model_name'] = gpus[0].name
        metadata['gpu_count'] = len(gpus)
        metadata['gpu_driver'] = gpus[0].driver
      except Exception as e:
        logging.warn('Unable to record gpu information. Continuing without it.')

    # Record workload properties
    workload_properties = _get_workload_properties(workload)
    metadata.update(workload_properties)

    if 'extra_metadata' in FLAGS and FLAGS.extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      metadata.update(extra_metadata)

    # Save metadata.json
    os.makedirs(self.workload_log_dir, exist_ok=True)
    metadata_filepath = os.path.join(self.workload_log_dir, 'metadata.json')
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
      json.dump(metadata, f, ensure_ascii=False, indent=4)

  def _write_package_list_file(self) -> None:
    """Write "packages.txt" to disk.

    It is created at the start of a workload and includes a list of the
    currently installed OS and python packages."""
    # Get package lists
    try:
      os_package_list = _get_os_package_list()
      pip_package_list = _get_pip_package_list()
    except Exception as e:
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

  def eval(self, workload: spec.Workload,
           hyperparameters: Optional[spec.Hyperparamters], run_idx: int,
           global_step: int, batch_size: int, latest_eval_result: dict,
           global_start_time: float, accumulated_submission_time: float,
           goal_reached: bool) -> None:
    """"Write or append to "metrics.csv".

    A "metrics.csv" is created for each hyperparameter tuning trail and a row is
    appended for every model evaluation. The information included is loss,
    accuracy, training step, time elapsed, hparams, workload properties,
    and hardware utilization."""
    metrics = {}
    metrics['workload'] = self.workload_name
    metrics['framework'] = FLAGS.framework
    metrics['run_idx'] = run_idx

    # Record training metrics
    metrics['accumulated_submission_time'] = accumulated_submission_time
    metrics['global_step'] = global_step
    steps_per_epoch = workload.num_train_examples // batch_size
    metrics['epoch'] = global_step / steps_per_epoch
    for key, value in latest_eval_result.items():
      metrics[key] = value

    # Record hyperparameters
    metrics['batch_size'] = batch_size
    if hyperparameters:
      hparams_dict = hyperparameters._asdict()
      # prefix every key with "hparam." to make more human-readable and to
      # avoid overlap with other possible kys in the metrics dict.
      hparams_dict = {f'hparam.{k}': v for k, v in hparams_dict.items()}
      metrics.update(hparams_dict)

    # Record workload properties
    workload_properties = _get_workload_properties(workload)
    metrics.update(workload_properties)

    # Record miscellaneous metadata
    metrics['steps_per_epoch'] = steps_per_epoch
    metrics['global_start_time'] = global_start_time
    metrics['goal_reached'] = goal_reached
    if 'extra_metadata' in FLAGS and FLAGS.extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      metrics.update(extra_metadata)

    # Record utilization
    utilization_metrics = _get_utilization()
    metrics.update(utilization_metrics)

    # Save to CSV file
    run_output_path = os.path.join(self.workload_log_dir, 'run_' + str(run_idx))
    os.makedirs(run_output_path, exist_ok=True)
    csv_path = os.path.join(run_output_path, 'metrics.csv')
    logging.info(f'Recording metrics to: {csv_path}')
    self._append_to_csv(metrics, csv_path)

  def _append_to_csv(self, metrics: dict, csv_path: str) -> None:
    # Open most recent data and save data to filesystem immediately to minimize
    # data loss.
    if os.path.isfile(csv_path):
      df = pd.read_csv(csv_path)
    else:
      df = pd.DataFrame()  # Initialize empty dataframe if no data is saved yet
    df = df.append(metrics, ignore_index=True)
    df.to_csv(csv_path, index=False)
