import os
import tensorflow as tf
import torch
from datasets import load_from_disk

from algoperf.workloads.lm.input_pipeline import get_lm_dataset

DATASET_PATH = "/fast/najroldi/data/lm/slim_pajama/new_sp_15B_tokens"
BATCH_SIZE = 2
SEED = 42  # Fixed random seed for reproducibility

tf_seed = SEED

# Load the dataset
ds = get_lm_dataset(
    data_rng=[tf_seed],  # Ensure correct seed type
    split="train",
    data_dir=DATASET_PATH,
    is_training=True,
    vocab_size=0,  # Not needed but kept for function signature
    global_batch_size=BATCH_SIZE,
)
