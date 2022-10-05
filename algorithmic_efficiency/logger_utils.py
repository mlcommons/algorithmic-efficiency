import concurrent.futures
import functools
import json
import logging
import operator
import os.path
import time

from absl import logging as absl_logging
from clu import metric_writers
from flax.training import checkpoints as flax_checkpoints
# from init2winit import checkpoint
import jax
import jax.numpy as jnp
import numpy as np
import GPUtil
import pandas as pd
import psutil
import platform
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
import subprocess
from typing import Any, Optional

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  logging.exception('Unable to import wandb.')
  wandb = None

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


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
  # sensor_temps = psutil.sensors_temperatures()
  # for key in sensor_temps.keys():  # pylint: disable=consider-using-dict-items
  #   # Take the first temp reading for each kind of device (CPU, GPU, Disk, etc.)
  #   value = sensor_temps[key][0].current
  #   util_data[f'temp.{key}.current'] = value

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


def _get_system_hardware_info() -> dict:
  system_hardware_info = {}
  try:
    system_hardware_info['cpu_model_name'] = _get_cpu_model_name()
    system_hardware_info['cpu_count'] = psutil.cpu_count()
  except:  # pylint: disable=bare-except
    logging.warn('Unable to record cpu information. Continuing without it.')

  gpus = GPUtil.getGPUs()
  if gpus:
    try:
      system_hardware_info['gpu_model_name'] = gpus[0].name
      system_hardware_info['gpu_count'] = len(gpus)
      system_hardware_info['gpu_driver'] = gpus[0].driver
    except:  # pylint: disable=bare-except
      logging.warn('Unable to record gpu information. Continuing without it.')

  return system_hardware_info


def _get_system_software_info() -> dict:
  system_software_info = {}

  system_software_info['os_platform'] = \
      platform.platform()  # Ex. 'Linux-5.4.48-x86_64-with-glibc2.29'
  system_software_info['python_version'] = platform.python_version(
  )  # Ex. '3.8.10'
  system_software_info['python_compiler'] = platform.python_compiler(
  )  # Ex. 'GCC 9.3.0'
  # Note: do not store hostname as that may be sensitive

  try:
    system_software_info['git_branch'] = _get_git_branch()
    system_software_info['git_commit_hash'] = _get_git_commit_hash()
    # Note: do not store git repo url as it may be sensitive or contain a
    # secret.
  except:  # pylint: disable=bare-except
    logging.warn('Unable to record git information. Continuing without it.')

  return system_software_info

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
    except:  # pylint: disable=bare-except
      raise ValueError(  # pylint: disable=raise-missing-from
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
    except:  # pylint: disable=bare-except
      logging.warn(
          f'Unable to record workload.{key} information. Continuing without it.'
      )
    if _is_primitive_type(attr):
      workload_properties[f'workload.{key}'] = attr
  return workload_properties

def get_meta_data(workload):
  meta_data = {}
  workload_properties = _get_workload_properties(workload)
  meta_data.update(workload_properties)
  utilization_measurements = _get_utilization()
  meta_data.update(utilization_measurements)
  system_software_info = _get_system_software_info()
  meta_data.update(system_software_info)
  system_hardware_info = _get_system_hardware_info()
  meta_data.update(system_hardware_info)
  return meta_data

# def get_meta_data(workload):
#   meta_data = {}
#   workload_properties = _get_workload_properties(workload)
#   meta_data.update(workload_properties)
#   utilization_measurements = _get_utilization()
#   meta_data.update(utilization_measurements)
#   system_software_info = _get_system_software_info()
#   meta_data.update(system_software_info)
#   system_hardware_info = _get_system_hardware_info()
#   meta_data.update(system_hardware_info)
#   return meta_data

def set_up_loggers(train_dir, flags):
  """Creates a logger for eval metrics."""
  csv_path = os.path.join(train_dir, 'measurements.csv')
  checkpoint_path = os.path.join(train_dir, 'checkpoints')
  os.makedirs(checkpoint_path, exist_ok=True)
  metrics_logger = MetricLogger(
      csv_path=csv_path,
      checkpoint_path=checkpoint_path,
      events_dir=train_dir,
      flags=flags
  )
  return metrics_logger


class MetricLogger(object):
  """Used to log all measurements during training.

  Note: Writes are not atomic, so files may become corrupted if preempted at
  the wrong time.
  """

  def __init__(self,
               csv_path='',
               checkpoint_path='',
               events_dir=None,
               flags=None):
    """Create a recorder for metrics, as CSV or JSON.
    Args:
      csv_path: A filepath to a CSV file to append to.
      checkpoint_path: Where to save checkpoints.
      events_dir: Optional. If specified, save tfevents summaries to this
        directory.

    """
    self._measurements = {}
    self._csv_path = csv_path
    self._checkpoint_path = checkpoint_path

    if events_dir:
      self._tb_metric_writer = metric_writers.create_default_writer(events_dir)
      if wandb is not None:
        wandb.init(dir=events_dir)
        wandb.config.update(flags)

  def append_scalar_metrics(self, metrics, global_step):
    """Record a dictionary of scalar metrics at a given step.
    Args:
      metrics: a Dict of metric names to scalar values. 'global_step' is the
        only required key.
    """
    metrics['global_step'] = global_step

    try:
      with open(self._csv_path, "r") as csv_file:
        measurements = pd.read_csv(csv_file)
        measurements = measurements.append([metrics])
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
      measurements = pd.DataFrame([metrics], columns=sorted(metrics.keys()))
      if isinstance(e, pd.errors.EmptyDataError):
        logging.info('Measurements file is empty. Create a new one, starting '
                     'with metrics from this step.')

    with open(self._csv_path, "w") as csv_file:
      measurements.to_csv(csv_file, index=False)

    if self._tb_metric_writer:
      self._tb_metric_writer.write_scalars(
          step=int(metrics['global_step']), scalars=metrics)
      self._tb_metric_writer.flush()

    if wandb is not None:
      wandb.log(metrics)


  # def write_pytree(self, pytree, prefix='training_metrics'):
  #   """Record a serializable pytree to disk, overwriting any previous state.
  #   Args:
  #     pytree: Any serializable pytree
  #     prefix: The prefix for the checkpoint.  Save path is
  #       self._pytree_path/prefix
  #   """
  #   state = dict(pytree=pytree)
  #   checkpoint.save_checkpoint(
  #       self._pytree_path,
  #       step='',
  #       state=state,
  #       prefix=prefix,
  #       max_to_keep=None)
  #
  # def append_pytree(self, pytree, prefix='training_metrics'):
  #   """Append and record a serializable pytree to disk.
  #   The pytree will be saved to disk as a list of pytree objects. Everytime
  #   this function is called, it will load the previous saved state, append the
  #   next pytree to the list, then save the appended list.
  #   Args:
  #     pytree: Any serializable pytree.
  #     prefix: The prefix for the checkpoint.
  #   """
  #   # Read the latest (and only) checkpoint, then append the new state to it
  #   # before saving back to disk.
  #   old_state = flax_checkpoints.restore_checkpoint(
  #       self._pytree_path, target=None, prefix=prefix)
  #   # Because we pass target=None, checkpointing will return the raw state
  #   # dict, where 'pytree' is a dict with keys ['0', '1', ...] instead of a
  #   # list.
  #   if old_state:
  #     state_list = old_state['pytree']
  #     state_list = [state_list[str(i)] for i in range(len(state_list))]
  #   else:
  #     state_list = []
  #   state_list.append(pytree)
  #
  #   self.write_pytree(state_list)
  #
  # def append_json_object(self, json_obj):
  #   """Append a json serializable object to the json file."""
  #
  #   if not self._json_path:
  #     raise ValueError('Attempting to write to a null json path')
  #   if exists(self._json_path):
  #     with gfile.GFile(self._json_path) as json_file:
  #       json_objs = json.loads(json_file.read())
  #     json_objs.append(json_obj)
  #   else:
  #     json_objs = [json_obj]
  #   # TODO(gdahl,gilmer): Should this be an atomic file?
  #   with gfile.GFile(self._json_path, 'w') as json_file:
  #     json_file.write(json.dumps(json_objs))
