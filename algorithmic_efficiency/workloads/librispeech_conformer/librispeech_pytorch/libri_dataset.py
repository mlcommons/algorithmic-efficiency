import csv

from absl import logging
import numpy as np
import torch


class LibriDataset(torch.utils.data.Dataset):

  def __init__(self, split, data_dir):
    super().__init__()
    self.data_dir = data_dir
    splits = split.split("+")
    ids = []
    for split in splits:
      logging.info('loading split = %s', split)
      feat_csv = '{}/{}.csv'.format(data_dir, split)

      with open(feat_csv, newline='') as csvfile:
        data = list(csv.reader(csvfile))

      for example in data[1:]:
        ids.append('{}/{}'.format(split, example[1]))
    self.ids = ids

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, index):
    example_id = self.ids[index]
    data_dir = self.data_dir
    audio = np.load('{}/{}_audio.npy'.format(data_dir, example_id))
    targets = np.load('{}/{}_targets.npy'.format(data_dir, example_id))

    audio_paddings = np.zeros_like(audio, dtype=np.float32)
    audio_paddings = np.pad(
        audio_paddings, (0, 320000 - audio.shape[0]), constant_values=1.0)
    audio = np.pad(audio, (0, 320000 - audio.shape[0]), constant_values=0.0)

    target_paddings = np.zeros_like(targets, dtype=np.float32)
    target_paddings = np.pad(
        target_paddings, (0, 256 - target_paddings.shape[0]),
        constant_values=1.0)
    targets = np.pad(targets, (0, 256 - targets.shape[0]), constant_values=0)
    audio = audio.astype(np.float32)
    audio_paddings = audio_paddings.astype(np.float32)
    targets = targets.astype(np.float32)
    target_paddings = target_paddings.astype(np.float32)
    return (audio, audio_paddings), (targets, target_paddings)
