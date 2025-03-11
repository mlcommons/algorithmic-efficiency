"""LM workload parent class."""

import abc
import math
import os
from typing import Any, Dict, Optional, Tuple

import jax
import numpy as np
import torch

from algoperf import spec
from algoperf.workloads.lm import input_pipeline

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ


class BaseLmWorkload(spec.Workload):
  """A LM workload."""

  _vocab_size: int = 32000

  def __init__(self) -> None:
    super().__init__()
    self._tokenizer = None

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int,
                         num_batches: Optional[int] = None,
                         repeat_final_dataset: bool = False):
    is_training = split == 'train'
    ds, self._tokenizer = input_pipeline.get_lm_dataset(
        data_rng,
        split,
        data_dir,
        is_training=is_training,
        vocab_size=self._vocab_size,
        global_batch_size=global_batch_size,
        num_batches=num_batches,
        repeat_final_dataset=repeat_final_dataset)
    
    for batch in iter(ds):
      yield batch

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str,
                           global_step: int = 0) -> Dict[str, float]:
    """Run a full evaluation of the model."""

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense or one-hot labels.
      logits_batch: spec.Tensor,
      mask_batch: Optional[spec.Tensor] = None,
      label_smoothing: float = 0.0) -> Dict[str, spec.Tensor]:  # differentiable
    """Evaluate the loss function at (label_batch, logits_batch)."""
    pass