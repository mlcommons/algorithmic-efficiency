"""Utilities for checkpointing.

Note: Code adapted from
https://github.com/google/init2winit/blob/master/init2winit/checkpoint.py.
"""

import os
from typing import Sequence, Tuple

from absl import logging
from flax import jax_utils
from flax.training import checkpoints as flax_checkpoints
from flax.training.checkpoints import latest_checkpoint
import jax
import numpy as np
from tensorflow.io import gfile  # pytype: disable=import-error
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

_, _, DEVICE, _ = pytorch_setup()
CheckpointReturn = Tuple[spec.OptimizerState,
                         spec.ParameterContainer,
                         spec.ModelAuxiliaryState,
                         dict,
                         list,
                         int,
                         int]


def maybe_restore_checkpoint(framework: str,
                             optimizer_state: spec.OptimizerState,
                             model_params: spec.ParameterContainer,
                             model_state: spec.ModelAuxiliaryState,
                             train_state: dict,
                             eval_results: list,
                             global_step: int,
                             preemption_count: int,
                             checkpoint_dir: str) -> CheckpointReturn:
  """Optionally restores from a checkpoint.

  The checkpoint logic is as follows: if there is a checkpoint in
  `checkpoint_dir`, restore it. Else, don't restore any checkpoint, and
  just return the passed-in optimizer_state, model_params,
  model_state, and train_state.

  Args:
    framework: Current framework (e.g., `jax` or `pytorch`).
    optimizer_state: Optimizer state.
    model_params: Model parameters.
    model_state: Model state such as batch statistics when batch
      normalization is used.
    train_state: Training state such as `last_eval_time`.
    eval_results: Previous evaluation results.
    global_step: Global step.
    preemption_count: Number of preemptions.
    checkpoint_dir: The training directory where we will look for a checkpoint.

  Returns:
    A tuple of (optimizer_state, model_params, model_state,
    train_state, eval_results, global_step, preemption_count).
  """
  if framework == 'jax':
    opt_state, opt_update_fn = optimizer_state
  else:
    opt_state, opt_update_fn = optimizer_state, None

  uninitialized_global_step = -1
  uninitialized_preemption_count = -1
  checkpoint_state = {
      'model_params': model_params,
      'optimizer_state': opt_state,
      'model_state': model_state,
      'train_state': train_state,
      'eval_results': None,
      'global_step': uninitialized_global_step,
      'preemption_count': uninitialized_preemption_count,
  }

  if framework == 'jax':
    latest_ckpt = flax_checkpoints.restore_checkpoint(
        checkpoint_dir, target=checkpoint_state)
    save_path = os.path.join(checkpoint_dir,
                             'checkpoint_' + str(latest_ckpt['global_step']))
  else:
    latest_ckpt = checkpoint_state
    save_path = latest_checkpoint(checkpoint_dir)
    if save_path is not None:
      latest_ckpt = torch.load(save_path, map_location=DEVICE)

  # Load_latest_checkpoint() will return checkpoint_state if
  # checkpoint_dir does not exist or if it exists and contains no checkpoints.
  found_checkpoint = latest_ckpt['global_step'] != uninitialized_global_step

  if not found_checkpoint:
    return (optimizer_state,
            model_params,
            model_state,
            train_state,
            eval_results,
            global_step,
            preemption_count)

  # If there's the latest checkpoint in the checkpoint_dir, restore from that.
  if framework == 'jax':
    checkpoint_state = replicate_checkpoint(
        latest_ckpt,
        pytree_keys=[
            'optimizer_state',
            'model_params',
            'model_state',
        ])
    checkpoint_state['optimizer_state'] = (checkpoint_state['optimizer_state'],
                                           opt_update_fn)
    checkpoint_state['eval_results'] = [
        (value, key) for key, value in latest_ckpt['eval_results'].items()
    ]

  else:
    checkpoint_state = latest_ckpt
    if isinstance(
        model_params,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
      model_params = model_params.module
    model_params.load_state_dict(checkpoint_state['model_params'])
    checkpoint_state['model_params'] = model_params
    for key in optimizer_state.keys():
      optimizer_state[key].load_state_dict(
          checkpoint_state['optimizer_state'][key])
      checkpoint_state['optimizer_state'][key] = optimizer_state[key]

    logging.info(f'Loaded checkpoint from {save_path}.')
  return (checkpoint_state['optimizer_state'],
          checkpoint_state['model_params'],
          checkpoint_state['model_state'],
          checkpoint_state['train_state'],
          list(checkpoint_state['eval_results']),
          checkpoint_state['global_step'],
          checkpoint_state['preemption_count'] + 1)


def replicate_checkpoint(latest: dict,
                         pytree_keys: Sequence[str],
                         replicate: bool = True) -> dict:
  """Restores from the provided checkpoint.

  Args:
    latest: A dict representing the state of the
      checkpoint we want to restore.
    pytree_keys: A sequence of keys into `latest` that are pytrees, which will
      be replicated if replicate=True.
    replicate: If set, replicate the state across devices.

  Returns:
    A JAX pytree holding the arrays that need to be replicated/unreplicated.
  """
  pytree = {k: latest[k] for k in pytree_keys}
  if replicate:
    pytree = jax_utils.replicate(pytree)
  extra_dict = {k: latest[k] for k in latest.keys() if k not in pytree_keys}
  pytree.update(extra_dict)
  return pytree


def save_checkpoint(framework: str,
                    optimizer_state: spec.OptimizerState,
                    model_params: spec.ParameterContainer,
                    model_state: spec.ModelAuxiliaryState,
                    train_state: dict,
                    eval_results: list,
                    global_step: int,
                    preemption_count: int,
                    checkpoint_dir: str,
                    save_intermediate_checkpoints: bool) -> None:
  """Save the checkpoint in `checkpoint_dir`.

  Args:
    framework: Current framework (e.g., `jax` or `pytorch`).
    optimizer_state: Optimizer state.
    model_params: Model parameters.
    model_state: Model state such as batch statistics when batch
      normalization is used.
    train_state: Training state such as `last_eval_time`.
    eval_results: Previous evaluation results.
    global_step: Global step.
    preemption_count: Number of preemptions.
    checkpoint_dir: The training directory where we will look for a checkpoint.
    save_intermediate_checkpoints: Whether to save intermediate checkpoints.

  Returns:
    A tuple of (optimizer_state, model_params, model_state,
    train_state, eval_results, global_step, preemption_count).
  """
  if framework == 'jax':
    model_params = jax.device_get(jax_utils.unreplicate(model_params))
    opt_state, _ = optimizer_state
    opt_state = jax.device_get(jax_utils.unreplicate(opt_state))
    model_state = jax.device_get(jax_utils.unreplicate(model_state))
  else:
    if isinstance(
        model_params,
        (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
      model_params = model_params.module
    model_params = model_params.state_dict()
    optimizer_state_dict = {}
    for key in optimizer_state.keys():
      if hasattr(optimizer_state[key], 'state_dict'):
        optimizer_state_dict[key] = optimizer_state[key].state_dict()
      else:
        logging.warning(
            f'The optimizer state for key {key} is not saved, because '
            f'{type(optimizer_state[key])} has not implemented a state_dict() '
            'method.')
    opt_state = optimizer_state_dict

  checkpoint_state = {
      'model_params': model_params,
      'optimizer_state': opt_state,
      'model_state': model_state,
      'train_state': train_state,
      'eval_results': tuple(eval_results),
      'global_step': global_step,
      'preemption_count': preemption_count,
  }

  save_path = os.path.join(checkpoint_dir, f'checkpoint_{global_step}')
  if framework == 'jax':
    flax_checkpoints.save_checkpoint(
        checkpoint_dir,
        target=checkpoint_state,
        step=global_step,
        overwrite=True,
        keep=np.Inf if save_intermediate_checkpoints else 1)
  else:
    if not save_intermediate_checkpoints:
      checkpoint_files = gfile.glob(
          os.path.join(checkpoint_dir, 'checkpoint_*'))
      for path in checkpoint_files:
        logging.info('Removing checkpoint at %s', path)
        gfile.rmtree(path)
    torch.save(checkpoint_state, save_path)

  logging.info(f'Saved checkpoint to {save_path}.')
