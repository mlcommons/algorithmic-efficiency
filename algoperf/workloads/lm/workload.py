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
    ds = input_pipeline.get_lm_dataset(
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

  def _eval_model_on_split():
    pass
  
  def eval_period_time_sec():
    pass
  
  def has_reached_test_target():
    pass
  
  def has_reached_validation_target():
    pass
  
  def init_model_fn():
    pass
  
  def is_output_params():
    pass
  
  def loss_fn():
    pass
  
  def loss_type():
    pass
  
  def max_allowed_runtime_sec():
    pass
  
  def model_fn():
    pass
  
  def num_eval_train_examples():
    pass
  
  def num_test_examples():
    pass
  
  def num_train_examples():
    pass
  
  def num_validation_examples():
    pass
  
  def step_hint():
    pass
  
  def test_target_value():
    pass
  
  def train_mean():
    pass
  
  def train_stddev():
    pass
  
  def validation_target_value():
    pass
  
  def target_metric_name():
    pass