import os
import tensorflow as tf
import torch
from datasets import load_from_disk

from algoperf.workloads.lm.input_pipeline import get_lm_dataset

DATASET_PATH = "/fast/najroldi/data/lm/slim_pajama/new_sp_15B_tokens"
BATCH_SIZE = 2
SEED = 42  # Fixed random seed for reproducibility


def test_tf_dataset():
    """Tests if get_lm_dataset correctly loads the HF dataset as a TensorFlow dataset."""
    
    print(f"Loading dataset from: {DATASET_PATH}")

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

    print("Testing TensorFlow Dataset Output...")
    for batch in ds.take(2):  # Take two batches to test
        print("Inputs:", batch["inputs"].numpy())  # Convert to NumPy for inspection
        print("Targets:", batch["targets"].numpy())

def test_pytorch_dataloader():
    """Tests if the TensorFlow dataset can be converted to PyTorch format correctly."""
    
    # Use the same TensorFlow-compatible seed
    tf_seed = tf.constant(SEED, dtype=tf.int64)

    # Load the dataset
    ds = get_lm_dataset(
        data_rng=[tf_seed],  # Ensure correct seed type
        split="train",
        data_dir=DATASET_PATH,
        is_training=True,
        vocab_size=0,
        global_batch_size=BATCH_SIZE,
    )

    def _input_queue_generator():
        """Generator that converts TF dataset batches to PyTorch tensors."""
        for batch in iter(ds):
            batch = {k: torch.tensor(v.numpy()) for k, v in batch.items()}  # Convert to PyTorch tensors
            yield batch

    dataloader = _input_queue_generator()

    print("\nTesting PyTorch DataLoader Output...")
    for _ in range(2):  # Take two batches
        batch = next(dataloader)
        print("Inputs:", batch["inputs"])
        print("Targets:", batch["targets"])

# Run tests
if __name__ == "__main__":
    test_tf_dataset()
    test_pytorch_dataloader()