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
  then {'key':'value'} is returned.
  """
  metadata = {}
  try:
    for item in extra_metadata_string_list:
      key, value = item.split("=")
      metadata[key] = value
  except:
    logging.error(
        'Failed to parse extra_metadata CLI arguments. Please check your'
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

  Three files are written to the given "log_dir" folder:
  1. "metadata.json" is created at the start of a workload and it includes the
     datetime, workload name, and system configuration.
  2. "measurements.csv" is created for each hyperparameter tuning trial and a
     row is appended for every model evaluation. The information included is
     loss, accuracy, training step, time elapsed, hparams, workload properties,
     and hardware utilization.
  3. "packages.txt" is created at the start of a workload and it includes a
     list of the currently installed OS and python packages.

  Joining measurement CSVs across workloads or hyperparameter tuning trials is
  left to users, although a convienence function called "concatenate_csvs()" is
  provided. The data format of "measurements.csv" is designed to be safe to
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

  def eval(self, workload: spec.Workload,
           hyperparameters: Optional[spec.Hyperparamters], run_idx: int,
           global_step: int, batch_size: int, latest_eval_result: dict,
           global_start_time: float, accumulated_submission_time: float,
           goal_reached: bool) -> None:
    """"Write or append to "measurements.csv".

    A "measurements.csv" is created for each hyperparameter tuning trial and a
    row is appended for every model evaluation. The information included is
    loss, accuracy, training step, time elapsed, hparams, workload properties,
    and hardware utilization."""
    measurements = {}
    measurements['workload'] = self.workload_name
    measurements['framework'] = FLAGS.framework
    measurements['run_idx'] = run_idx

    # Record training measurements
    measurements['accumulated_submission_time'] = accumulated_submission_time
    measurements['global_step'] = global_step
    steps_per_epoch = workload.num_train_examples // batch_size
    measurements['epoch'] = global_step / steps_per_epoch
    for key, value in latest_eval_result.items():
      measurements[key] = value

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

    # Record miscellaneous metadata
    measurements['steps_per_epoch'] = steps_per_epoch
    measurements['global_start_time'] = global_start_time
    measurements['goal_reached'] = goal_reached
    if 'extra_metadata' in FLAGS and FLAGS.extra_metadata:
      extra_metadata = _get_extra_metadata_as_dict(FLAGS.extra_metadata)
      measurements.update(extra_metadata)

    # Record utilization
    utilization_measurements = _get_utilization()
    measurements.update(utilization_measurements)

    # Save to CSV file
    run_output_path = os.path.join(self.workload_log_dir, 'run_' + str(run_idx))
    os.makedirs(run_output_path, exist_ok=True)
    csv_path = os.path.join(run_output_path, 'measurements.csv')
    logging.info(f'Recording measurements to: {csv_path}')
    self._append_to_csv(measurements, csv_path)

  def _append_to_csv(self, measurements: dict, csv_path: str) -> None:
    # Open most recent data and save data to filesystem immediately to minimize
    # data loss.
    if os.path.isfile(csv_path):
      df = pd.read_csv(csv_path)
    else:
      df = pd.DataFrame()  # Initialize empty dataframe if no data is saved yet
    df = df.append(measurements, ignore_index=True)
    df.to_csv(csv_path, index=False)
