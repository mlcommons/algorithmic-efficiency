import os
from typing import Sequence, Tuple

from absl import flags
from absl import logging
from flax import jax_utils
from flax.training import checkpoints as flax_checkpoints
import jax
import torch
import torch.distributed as dist

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup

FLAGS = flags.FLAGS

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
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
  if framework == 'jax':
    opt_state, opt_update_fn = optimizer_state
  else:
    opt_state, opt_update_fn = optimizer_state, None

  uninitialized_global_step = -1
  checkpoint_state = dict(
      model_params=model_params,
      optimizer_state=opt_state,
      model_state=model_state,
      train_state=train_state,
      eval_results=None,
      global_step=uninitialized_global_step,
      preemption_count=preemption_count)

  if framework == 'jax':
    latest_ckpt = flax_checkpoints.restore_checkpoint(
        checkpoint_dir, target=checkpoint_state)
    save_path = os.path.join(checkpoint_dir,
                             'checkpoint_' + str(latest_ckpt['global_step']))
  else:
    latest_ckpt = checkpoint_state
    save_path = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.exists(save_path):
      latest_ckpt = torch.load(save_path, map_location=DEVICE)

  # Load_latest_checkpoint() will return checkpoint_state if
  # checkpoint_dir does not exist or if it exists and contains no checkpoints.
  found_checkpoint = (latest_ckpt['global_step'] != uninitialized_global_step)

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
    model_params.load_state_dict(checkpoint_state['model_params'])
    checkpoint_state['model_params'] = model_params
    for key in optimizer_state.keys():
      optimizer_state[key].load_state_dict(
          checkpoint_state['optimizer_state'][key])
      checkpoint_state['optimizer_state'][key] = optimizer_state[key]

    logging.info('Loaded checkpoint from %s', save_path)
  return (checkpoint_state['optimizer_state'],
          checkpoint_state['model_params'],
          checkpoint_state['model_state'],
          checkpoint_state['train_state'],
          list(checkpoint_state['eval_results']),
          checkpoint_state['global_step'],
          preemption_count + 1)


def replicate_checkpoint(latest: dict,
                         pytree_keys: Sequence[str],
                         replicate: bool = True) -> dict:
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
                    checkpoint_dir: str) -> None:
  if RANK == 0:
    if framework == 'jax':
      model_params = jax.device_get(jax_utils.unreplicate(model_params))
      opt_state, _ = optimizer_state
      opt_state = jax.device_get(jax_utils.unreplicate(opt_state))
      model_state = jax.device_get(jax_utils.unreplicate(model_state))
    else:
      model_params = model_params.state_dict()
      optimizer_state_dict = {}
      for key in optimizer_state.keys():
        optimizer_state_dict[key] = optimizer_state[key].state_dict()
      opt_state = optimizer_state_dict

    checkpoint_state = dict(
        model_params=model_params,
        optimizer_state=opt_state,
        model_state=model_state,
        train_state=train_state,
        eval_results=tuple(eval_results),
        global_step=global_step,
        preemption_count=preemption_count)

    if framework == 'jax':
      save_path = os.path.join(checkpoint_dir, 'checkpoint_' + str(global_step))
      flax_checkpoints.save_checkpoint(
          checkpoint_dir,
          target=checkpoint_state,
          step=global_step,
          overwrite=True)
    else:
      save_path = os.path.join(checkpoint_dir, 'checkpoint')
      torch.save(checkpoint_state, save_path)
    logging.info('Saved checkpoint to %s', save_path)

  if USE_PYTORCH_DDP:
    dist.barrier()
