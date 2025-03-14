
import os
import numpy as np
import tensorflow as tf
import torch

from datasets import load_from_disk

from absl import app
from absl import flags
from absl import logging

from algoperf.profiler import PassThroughProfiler
from algoperf import random_utils as prng
from algoperf.pytorch_utils import pytorch_init
from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.lm.input_pipeline import get_lm_dataset


tf.config.set_visible_devices([], 'GPU')

# Environment variables
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disables tensorRT, cuda warnings.
# disable only for deepspeech if it works fine for other workloads
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
# (nico)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags.DEFINE_enum(
    'framework',
    None,
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')

FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


DATASET_PATH = "/fast/najroldi/data/lm/slim_pajama/new_sp_15B_tokens"
BATCH_SIZE = 2
RNG_SEED = 1996  # Fixed random seed for reproducibility


def main(_):
  profiler = PassThroughProfiler()
  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  rng = prng.PRNGKey(RNG_SEED)
  data_rng, _, _, _ = prng.split(rng, 4)
  
  print(f"data_rng = {data_rng}")

  # Load the dataset
  ds = get_lm_dataset(
      data_rng=data_rng,
      split="train",
      data_dir=DATASET_PATH,
      is_training=True,
      vocab_size=0,  # Not needed but kept for function signature
      global_batch_size=BATCH_SIZE,
  )
  # Check if `ds` acts as a generator
  if hasattr(ds, '__iter__'):
      print("Dataset is an iterable/generator.")

  # Fetch first batch
  try:
      first_batch = next(iter(ds))
      print(f"Successfully retrieved first batch.")
  except Exception as e:
      print(f"Error retrieving first batch: {e}")
      return

  # Print structure of a batch
  print(f"First batch keys: {first_batch.keys()}")
  print(f"First batch shapes:")
  for key, value in first_batch.items():
      print(f"  - {key}: {value.shape} (dtype: {value.dtype})")

  # Validate batch dimensions
  assert "inputs" in first_batch and "targets" in first_batch, "Missing expected keys!"
  assert first_batch["inputs"].shape[0] == BATCH_SIZE, "Batch size mismatch!"
  assert first_batch["inputs"].shape == first_batch["targets"].shape, "Inputs and targets should have the same shape!"

  print(f"Dataset is correctly batched and structured.")
  print(f"Test completed successfully.")

if __name__ == '__main__':
  flags.mark_flag_as_required('framework')
  app.run(main)
