import collections
import json
import logging
import os.path
import platform
import re
import subprocess
from typing import Any, Optional

from absl import flags
from clu import metric_writers
import GPUtil
import pandas as pd
import psutil

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, RANK, DEVICE, _ = pytorch_setup()

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  logging.exception('Unable to import wandb.')
  wandb = None


def makedir(dir_name: str, exist_ok: bool = True) -> None:
  if RANK == 0:
    # only one worker should create the required dir
    os.makedirs(name=dir_name, exist_ok=exist_ok)


def _get_last_run_dir_index(runs):
  # Run names have format run_{index}
  indices = [int(run.split('_')[1]) for run in runs]
  return max(indices)


def get_log_dir(experiment_dir,
                workload,
                framework,
                experiment_name,
                interactive,
                resume_last_run):
  if RANK != 0:
    return

  # Construct path to experiment workload directory
  experiment_dir = os.path.expanduser(experiment_dir)
  workload_dir_name = f'{workload}_{framework}'
  if experiment_name is None:
    experiment_path = os.path.join(experiment_dir, workload_dir_name)
  else:
    experiment_path = os.path.join(experiment_dir,
                                   experiment_name,
                                   workload_dir_name)

  # Get either a new run dir or previous run dir
  if os.path.exists(experiment_path):
    runs = os.listdir(experiment_path)

    if len(runs) != 0:
      if resume_last_run:
        run_dir = f'run_{_get_last_run_dir_index(runs)}'
      elif interactive:
        while True:
          run_dir = input('Found existing runs: {}.\n'
                          'Enter run name to resume '
                          'or press ENTER to start new run:'.format(runs))
          if run_dir in runs or run_dir == '':
            break
          print('Please enter valid run. Try again.')
        if run_dir == '':
          run_dir = f'run_{_get_last_run_dir_index(runs) + 1}'
      else:
        raise ValueError(
            'The flag --interactive=False and --resume_last_run={}.'.format(resume_last_run)
            'Please use --resume_last_run flag if --interactive=False.')
    else:
      run_dir = 'run_0'
  else:
    run_dir = 'run_0'

  # Setup log dir
  logging_dir_path = os.path.join(experiment_path, run_dir)
  logging.info(f'Creating experiment run directory at {logging_dir_path}')
  makedir(logging_dir_path)
  return logging_dir_path


def write_hparams(hparams: spec.Hyperparameters,
                  tuning_dir: str) -> spec.Hyperparameters:
  hparams_file_name = os.path.join(tuning_dir, 'hparams.json')
  if os.path.exists(hparams_file_name):
    # If hparams.json already exist, use the previously saved hyperparameters.
    logging.info('Loading hparams from %s', hparams_file_name)
    with open(hparams_file_name, 'r') as f:
      hparams_dict = json.load(f)
    hparams = collections.namedtuple('Hyperparameters',
                                     hparams_dict)(**hparams_dict)
    return hparams
  else:
    logging.info('Saving hparams to %s', hparams_file_name)
    if RANK == 0:
      with open(hparams_file_name, 'w') as f:
        f.write(json.dumps(hparams._asdict(), indent=2))
    return hparams


def write_json(name: str, log_dict: dict, indent: int = 2):
  if RANK == 0:
    with open(name, 'w') as f:
      f.write(json.dumps(log_dict, indent=indent))


def _get_utilization() -> dict:
  util_data = {}

  # CPU
  util_data['cpu.util.avg_percent_since_last'] = psutil.cpu_percent(
      interval=None)  # non-blocking (cpu util percentage since last call)
  util_data['cpu.freq.current'] = psutil.cpu_freq().current

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
    logging.info('Unable to record cpu information. Continuing without it.')

  gpus = GPUtil.getGPUs()
  if gpus:
    try:
      system_hardware_info['gpu_model_name'] = gpus[0].name
      system_hardware_info['gpu_count'] = len(gpus)
      system_hardware_info['gpu_driver'] = gpus[0].driver
    except:  # pylint: disable=bare-except
      logging.info('Unable to record gpu information. Continuing without it.')

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
    logging.info('Unable to record git information. Continuing without it.')

  return system_software_info


def _get_git_commit_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse',
                                  'HEAD']).decode('ascii').strip()


def _get_git_branch() -> str:
  return subprocess.check_output(['git', 'rev-parse', '--abbrev-ref',
                                  'HEAD']).decode('ascii').strip()


def _get_cpu_model_name() -> str:
  output = subprocess.check_output(['lscpu']).decode('ascii').strip()
  return re.findall(r'(?=Model name:\s{1,}).*',
                    output)[0].split('Model name:')[1].strip()


def _is_primitive_type(item: Any) -> bool:
  primitive = (float, int, str, bool)
  return isinstance(item, primitive)


def _get_workload_properties(workload: spec.Workload) -> dict:
  workload_properties = {}
  skip_list = ['param_shapes', 'model_params_types']
  keys = [
      key for key in dir(workload)
      if not key.startswith('_') and key not in skip_list
  ]
  for key in keys:
    try:
      attr = getattr(workload, key)
    except:  # pylint: disable=bare-except
      logging.info(
          f'Unable to record workload.{key} information. Continuing without it.'
      )
    if _is_primitive_type(attr):
      workload_properties[f'workload.{key}'] = attr
  return workload_properties


def get_meta_data(workload: spec.Workload) -> dict:
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


class MetricLogger(object):
  """Used to log all measurements during training.

  Note: Writes are not atomic, so files may become corrupted if preempted at
  the wrong time.
  """

  def __init__(self,
               csv_path: str = '',
               events_dir: Optional[str] = None,
               configs: Optional[flags.FLAGS] = None) -> None:
    self._measurements = {}
    self._csv_path = csv_path
    self.use_wandb = configs.use_wandb

    if events_dir:
      self._tb_metric_writer = metric_writers.create_default_writer(events_dir)
      if wandb is not None and self.use_wandb:
        wandb.init(
            dir=events_dir, tags=[flags.FLAGS.workload, flags.FLAGS.framework])
        wandb.config.update(configs)

  def append_scalar_metrics(self,
                            metrics: dict,
                            global_step: int,
                            preemption_count: Optional[int] = None) -> None:
    metrics['global_step'] = global_step
    if preemption_count is not None:
      metrics['preemption_count'] = preemption_count

    try:
      with open(self._csv_path, 'r') as csv_file:
        measurements = pd.read_csv(csv_file)
        measurements = measurements.append([metrics])
    except (pd.errors.EmptyDataError, FileNotFoundError) as e:
      measurements = pd.DataFrame([metrics], columns=sorted(metrics.keys()))
      if isinstance(e, pd.errors.EmptyDataError):
        logging.info('Measurements file is empty. Create a new one, starting '
                     'with metrics from this step.')

    with open(self._csv_path, 'w') as csv_file:
      measurements.to_csv(csv_file, index=False)

    if self._tb_metric_writer:
      self._tb_metric_writer.write_scalars(
          step=int(metrics['global_step']), scalars=metrics)
      self._tb_metric_writer.flush()

    if wandb is not None and self.use_wandb:
      wandb.log(metrics)

  def finish(self) -> None:
    if wandb is not None and self.use_wandb:
      wandb.finish()


class PassThroughMetricLogger(object):

  def __init__(self,
               csv_path: str = '',
               events_dir: Optional[str] = None,
               configs: Optional[flags.FLAGS] = None) -> None:
    pass

  def append_scalar_metrics(self,
                            metrics: dict,
                            global_step: int,
                            preemption_count: Optional[int] = None) -> None:
    pass

  def finish(self) -> None:
    pass


def set_up_loggers(train_dir: str,
                   configs: flags.FLAGS) -> Optional[MetricLogger]:
  csv_path = os.path.join(train_dir, 'measurements.csv')
  if RANK == 0:
    metrics_logger = MetricLogger(
        csv_path=csv_path, events_dir=train_dir, configs=configs)
  else:
    metrics_logger = PassThroughMetricLogger(
        csv_path=csv_path, events_dir=train_dir, configs=configs)
  return metrics_logger
