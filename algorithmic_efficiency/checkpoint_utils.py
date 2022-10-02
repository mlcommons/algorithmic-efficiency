# coding=utf-8
# Copyright 2022 The init2winit Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements checkpointing of nested python containers of numpy arrays.
This is useful for training neural networks with stax, where model parameters
are nested numpy arrays.
"""
import copy
import os
import sys
import threading
from typing import Sequence

from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import checkpoints as flax_checkpoints
import jax
import jax.numpy as jnp


FLAGS = flags.FLAGS


def load_pytree(pytree_file):
  """Loads the checkpointed pytree."""
  latest = load_latest_checkpoint(pytree_file, target=None)
  if latest:
    # Because we pass target=None, flax checkpointing will return the raw
    # state dict, where 'state' will be a dict with keys ['0', '1', ...]
    # instead of a list.
    return latest['pytree']
  return None


def replicate_checkpoint(
    latest,
    pytree_keys: Sequence[str],
    replicate=True):
  """Restores from the provided checkpoint.
  Args:
    latest: A dict representing the state of the
      checkpoint we want to restore.
    pytree_keys: A sequence of keys into `latest` that are pytrees, which will
      be replicated if replicate=True.
    replicate: If set, replicate the state across devices.
  Returns:
    Tuple of (pytree, extra_dict) where pytree is a JAX pytree holding the
    arrays that need to be replicated/unreplicated and extra_dict holds any
    additional python state. We expect extra_dict to have the keys of
    'global_step', 'preemption_count', 'sum_train_cost', but old checkpoints
    might be missing something.
  """
  logging.info('Loaded model parameters from latest checkpoint.')
  # Old checkpoints without 'sum_train_cost' can still be restored, but the
  # train() function will break. Evals and curvature stuff should be fine,
  # however.
  expected = ['global_step', 'preemption_count', 'sum_train_cost']
  if any(k not in latest for k in expected):
    logging.warn('Checkpoint state missing keys, obtained %s expected %s',
                 list(latest.keys()), expected)

  pytree = {k: latest[k] for k in pytree_keys}
  if replicate:
    pytree = jax_utils.replicate(pytree)
  extra_dict = {k: latest[k] for k in latest.keys() if k not in pytree_keys}
  return pytree, extra_dict


def replicate_and_maybe_restore_checkpoint(
    unreplicated_optimizer_state,
    unreplicated_params,
    unreplicated_batch_stats,
    unreplicated_training_metrics_state,
    train_dir,
    external_checkpoint_path=None):
  """Replicates everything, and optionally restores from a checkpoint.
  The checkpoint logic is as follows: if there is a checkpoint in `train_dir`,
  restore it.  Else, if `external_checkpoint_path` is set, restore the
  checkpoint found there.  Else, don't restore any checkpoint, and just
  return the passed-in optimizer_state, params, batch_stats, and
  metrics_grabber.
  This function is also responsible for replicating the optimizer_state, params,
  batch_stats, and training_metrics_grabber across multiple devices.
  Args:
    unreplicated_optimizer_state: unreplicated optimizer state
    unreplicated_params: unreplicated params
    unreplicated_batch_stats: unreplicated batch stats
    unreplicated_training_metrics_state: unreplicated metrics state
    train_dir: (str) The training directory where we will look for a checkpoint.
    external_checkpoint_path: (str) If this argument is set, then we will load
    the external checkpoint stored there.
  Returns:
    replicated_optimizer_state
    replicated_params
    replicated_batch_stats
    replicated_training_metrics_state
    global_step (int)
    sum_train_cost (float)
    preemption_count (int)
    is_restored (bool): True if we've restored the latest checkpoint
                        in train_dir.
  """
  uninitialized_global_step = -1
  unreplicated_checkpoint_state = dict(
      params=unreplicated_params,
      optimizer_state=unreplicated_optimizer_state,
      batch_stats=unreplicated_batch_stats,
      training_metrics_grabber=unreplicated_training_metrics_state,
      global_step=uninitialized_global_step,
      preemption_count=0,
      sum_train_cost=0.0)
  latest_ckpt = load_latest_checkpoint(train_dir,
                                       target=unreplicated_checkpoint_state)
  # Load_latest_checkpoint() will return unreplicated_checkpoint_state if
  # train_dir does not exist or if it exists and contains no checkpoints.
  # Note that we could likely change the below line to:
  # found_checkpoint = latest_ckpt != unreplicated_checkpoint_state
  found_checkpoint = (latest_ckpt['global_step'] != uninitialized_global_step)

  # If there's a latest checkpoint in the train_dir, restore from that.
  if found_checkpoint:
    ckpt_to_return = latest_ckpt
    is_restored = True  # We do want trainer to increment preemption_count.
  # Else, if external_checkpoint_path is non-null, restore from that checkpoint.
  elif external_checkpoint_path is not None:
    # TODO(jeremycohen) This code will crash if we try to load an external
    # checkpoint which was trained with a different num_train_steps.  The issue
    # is that some of the fields in the training metrics state are arrays of
    # shape [num_train_steps].  In the future we may want to handle these
    # arrays explicitly, in order to avoid this crash.
    ckpt_to_return = load_checkpoint(external_checkpoint_path,
                                     target=unreplicated_checkpoint_state)
    is_restored = False  # We don't want trainer to increment preemption_count.
  else:  # Else, don't restore from any checkpoint.
    return (
        jax_utils.replicate(unreplicated_optimizer_state),
        jax_utils.replicate(unreplicated_params),
        jax_utils.replicate(unreplicated_batch_stats),
        jax_utils.replicate(unreplicated_training_metrics_state),
        0,  # global_step
        jnp.zeros(jax.local_device_count()),  # sum_train_cost
        0,  # preemption_count
        False)  # is_restored

  pytree_dict, extra_state = replicate_checkpoint(
      ckpt_to_return,
      pytree_keys=[
          'optimizer_state',
          'params',
          'batch_stats',
          'training_metrics_grabber',
      ])
  return (
      pytree_dict['optimizer_state'],
      pytree_dict['params'],
      pytree_dict['batch_stats'],
      pytree_dict['training_metrics_grabber'],
      extra_state['global_step'],
      extra_state['sum_train_cost'],
      extra_state['preemption_count'],
      is_restored)


def save_unreplicated_checkpoint_background(
    train_dir,
    optimizer_state,
    params,
    batch_stats,
    training_metrics_state,
    global_step,
    preemption_count,
    sum_train_cost,
    max_to_keep=1):
  """Saves pytree, step, preemption_count, and sum_train_cost to train_dir."""
  logging.info('Saving checkpoint to ckpt_%d', global_step)
  unreplicated_optimizer_state = jax.device_get(
      jax_utils.unreplicate(optimizer_state))
  unreplicated_params = jax.device_get(jax_utils.unreplicate(params))
  unreplicated_batch_stats = jax.device_get(jax_utils.unreplicate(batch_stats))
  unreplicated_training_metrics_state = jax.device_get(
      jax_utils.unreplicate(training_metrics_state))
  state = dict(global_step=global_step,
               preemption_count=preemption_count,
               sum_train_cost=sum_train_cost,
               optimizer_state=unreplicated_optimizer_state,
               params=unreplicated_params,
               batch_stats=unreplicated_batch_stats,
               training_metrics_grabber=unreplicated_training_metrics_state)
  save_checkpoint_background(
      train_dir,
      global_step,
      state,
      max_to_keep=max_to_keep)
  logging.info('Done saving checkpoint.')


_save_checkpoint_background_thread = None
_save_checkpoint_background_error = None
_save_checkpoint_background_lock = threading.RLock()


def _save_checkpoint_background_catch_error(*args, **kwargs):
  """Call save_checkpoint with provided args, store exception if any."""
  global _save_checkpoint_background_error
  try:
    save_checkpoint(*args, **kwargs)
    _save_checkpoint_background_error = None
  except BaseException as err:  # pylint: disable=broad-except
    logging.exception('Error while saving checkpoint in background.')
    _save_checkpoint_background_error = err


def wait_for_checkpoint_save():
  """Wait until last checkpoint save (if any) to finish."""
  with _save_checkpoint_background_lock:
    if _save_checkpoint_background_thread:
      _save_checkpoint_background_thread.join()
    if _save_checkpoint_background_error:
      raise _save_checkpoint_background_error


def save_checkpoint_background(*args, **kwargs):
  """Saves checkpoint to train_dir/checkpoint_name in a background thread.
  Args:
    *args:
    **kwargs: See save_checkpoint for a descrition of the arguments.
  The process is prevented from exiting until the last checkpoint as been saved.
  At most one checkpoint can be saved simultaneously. If the function is called
  while a previous checkpoint is being saved, the function will block until that
  previous checkpoint saving finishes.
  The provided state can be mutated after this function returns.
  Raises error raised by save_checkpoint during the previous call, if any.
  """
  with _save_checkpoint_background_lock:
    wait_for_checkpoint_save()
    global _save_checkpoint_background_thread
    # Copy everything for state, rest is negligeable, do it to keep it simple.
    args = copy.deepcopy(args)
    kwargs = copy.deepcopy(kwargs)
    _save_checkpoint_background_thread = threading.Thread(
        target=_save_checkpoint_background_catch_error,
        args=args, kwargs=kwargs, daemon=False)
    _save_checkpoint_background_thread.start()


def save_checkpoint(train_dir,
                    step,
                    state,
                    prefix='ckpt_',
                    max_to_keep=None):
  """Saves checkpoint to train_dir/{prefix}{step}.
  A list of checkpoints will be stored in train_dir. The user
  is responsible for using unique checkpoint names when calling save_checkpoint
  repeatedly. If the same train_dir and checkpoint name are used more than once,
  the latest file will become corrupt. This may become an issue if max_to_keep
  is not None.
  Args:
    train_dir: (str) Directory to create the checkpoint directory in.
    step: (int) Step of the checkpoint.
    state: (dict) The state to save.
    prefix: (str) Prefix of the checkpoint name.
    max_to_keep: (int) Checkpoints older than the max_to_keep'th will be
      deleted. Defaults to never deleting.
  Returns:
    The path of the checkpoint directory.
  """
  if max_to_keep is None:
    max_to_keep = sys.maxsize
  flax_checkpoints.save_checkpoint(
      train_dir,
      target=state,
      step=step,
      prefix=prefix,
      keep=max_to_keep,
      overwrite=True)
  save_dir = os.path.join(train_dir, prefix + str(step))
  return save_dir


def load_checkpoint(
    checkpoint_path,
    target=None,
    prefix='ckpt_'):
  """Loads the specified checkpoint."""
  restored = flax_checkpoints.restore_checkpoint(
      checkpoint_path, target=target, prefix=prefix)
  return restored


def load_latest_jax_checkpoint(train_dir, target=None, prefix='ckpt_'):
  """Loads the most recent checkpoint listed in train_dir.
  Args:
    train_dir: the directory to read checkpoints from.
    target: used for Flax checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    prefix: the prefix of the names of checkpoint files.
  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67.
  """
  restored = flax_checkpoints.restore_checkpoint(
      train_dir, target=target, prefix=prefix)
  return restored


def load_latest_pytorch_checkpoint(train_dir, target=None, prefix='ckpt_'):
  """Loads the most recent checkpoint listed in train_dir.
  Args:
    train_dir: the directory to read checkpoints from.
    target: used for Flax checkpointing, a pytree whose structure will be used
      to structure the restored checkpoint data.
    prefix: the prefix of the names of checkpoint files.
  Returns:
    The state restored from the checkpoint. If using Flax checkpointing and
    target=None, this will return a unstructured dictionary containing the
    checkpoint data, as returned by to_state_dict in serialization.py:
    https://github.com/google/flax/blob/master/flax/serialization.py#L67.
  """
  restored = flax_checkpoints.restore_checkpoint(
      train_dir, target=target, prefix=prefix)
  return restored
