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
import pandas as pd
from algorithmic_efficiency.pytorch_utils import pytorch_setup

try:
  import wandb  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError:
  logging.exception('Unable to import wandb.')
  wandb = None


USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def set_up_loggers(train_dir, xm_work_unit=None):
  """Creates a logger for eval metrics."""
  csv_path = os.path.join(train_dir, 'measurements.csv')
  pytree_path = os.path.join(train_dir, 'training_metrics')
  metrics_logger = MetricLogger(
      csv_path=csv_path,
      pytree_path=pytree_path,
      xm_work_unit=xm_work_unit,
      events_dir=train_dir)
  return metrics_logger


class MetricLogger(object):
  """Used to log all measurements during training.

  Note: Writes are not atomic, so files may become corrupted if preempted at
  the wrong time.
  """
  def __init__(self,
               csv_path='',
               json_path='',
               pytree_path='',
               events_dir=None,
               **logger_kwargs):
    """Create a recorder for metrics, as CSV or JSON.
    Args:
      csv_path: A filepath to a CSV file to append to.
      json_path: An optional filepath to a JSON file to append to.
      pytree_path: Where to save trees of numeric arrays.
      events_dir: Optional. If specified, save tfevents summaries to this
        directory.
      **logger_kwargs: Optional keyword arguments, whose only valid parameter
        name is an optional XM WorkUnit used to also record metrics to XM as
        MeasurementSeries.
    """
    self._measurements = {}
    self._csv_path = csv_path
    self._json_path = json_path
    self._pytree_path = pytree_path
    # if logger_kwargs:
    #   if len(logger_kwargs.keys()) > 1 or 'xm_work_unit' not in logger_kwargs:
    #     raise ValueError(
    #         'The only logger_kwarg that should be passed to MetricLogger is '
    #         'xm_work_unit.')
    #   self._xm_work_unit = logger_kwargs['xm_work_unit']
    # else:
    #   self._xm_work_unit = None

    # self._tb_metric_writer = None
    if events_dir:
      self._tb_metric_writer = metric_writers.create_default_writer(events_dir)

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
