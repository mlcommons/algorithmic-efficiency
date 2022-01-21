"""Data pipeline for LibriSpeech dataset modified from https://github.com/lsari/librispeech_100/blob/main/dataset.py."""

import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
import torch


class LibriSpeechDataset(torch.utils.data.Dataset):

  def __init__(self, feat_csv):
    self.df = pd.read_csv(feat_csv)
    self.sample_size = len(self.df)
    self.df["id"] = list(range(self.sample_size))

  def __getitem__(self, idx):
    sample = self.df.iloc[idx]
    trn_ids = json.loads(sample["trans_ids"])

    feature = np.load(sample["features"])
    index = sample["id"]
    return index, feature, trn_ids

  def __len__(self):
    return self.sample_size

  def pad_collate(self, batch):
    max_input_len = 2453
    max_target_len = 405

    for i, elem in enumerate(batch):
      index, f, trn = elem
      input_length = np.array(f.shape[0])
      input_dim = f.shape[1]
      feature = np.zeros((max_input_len, input_dim), dtype=np.float)
      feature[:f.shape[0], :f.shape[1]] = f
      target_len = len(trn)
      target_padding = np.zeros(max_target_len)
      target_padding[target_len:] = 1 
      trn = np.pad(
          trn, (0, max_target_len - len(trn)), "constant", constant_values=0)

      batch[i] = (int(index), feature, trn, input_length, target_padding)

    batch.sort(key=lambda x: x[3], reverse=True)

    index = np.array([x[0] for x in batch], dtype=jnp.int32)
    feature = np.array([x[1] for x in batch], dtype=jnp.float32)
    trn = np.array([x[2] for x in batch], dtype=jnp.int32)
    input_length = np.array([x[3] for x in batch], dtype=jnp.int32)
    target_padding = np.array([x[4] for x in batch], dtype=jnp.int32)

    return index, feature, trn, input_length, target_padding
