"""Data pipeline for LibriSpeech dataset.

Modified from https://github.com/lsari/librispeech_100/blob/main/dataset.py.
"""

import json

import numpy as np
import pandas as pd
import torch


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, feat_csv, aligned_on: int = 1):
        self.df = pd.read_csv(feat_csv)
        self.aligned_on = aligned_on
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
        max_input_len = 0
        max_target_len = 0

        # Get maximum length of entire batch
        for elem in batch:
            index, feature, trn = elem
            max_input_len = max(max_input_len, feature.shape[0])
            max_target_len = max(max_target_len, len(trn))

        # Pad to alignment value
        max_input_len = max_input_len + (-max_input_len) % self.aligned_on
        max_target_len = max_target_len + (-max_target_len) % self.aligned_on

        # Pad samples to new maximum
        for i, elem in enumerate(batch):
            index, f, trn = elem
            input_length = np.array(f.shape[0])
            input_dim = f.shape[1]
            feature = np.zeros((max_input_len, input_dim), dtype=np.float)
            feature[: f.shape[0], : f.shape[1]] = f
            trn = np.pad(
                trn, (0, max_target_len - len(trn)), "constant", constant_values=0
            )

            batch[i] = {
                "indices": int(index),
                "features": feature,
                "transcripts": trn,
                "input_lengths": input_length,
            }

        batch.sort(key=lambda x: x["input_lengths"], reverse=True)

        return torch.utils.data.dataloader.default_collate(batch)
