"""Data loader for pre-processed Criteo data.

Similar to how the NVIDIA example works, we split data from the last day into a
validation and test split (taking the first half for test and second half for
validation). See here for the NVIDIA example:
https://github.com/NVIDIA/DeepLearningExamples/blob/4e764dcd78732ebfe105fc05ea3dc359a54f6d5e/PyTorch/Recommendation/DLRM/preproc/run_spark_cpu.sh#L119.
"""
import json
import math
import os

from absl import app
import numpy as np

# Hide any GPUs from TF.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from typing import Optional, Sequence

import jax
import tensorflow as tf
import torch

from algorithmic_efficiency import data_utils
from algorithmic_efficiency.workloads.criteo1tb import input_pipeline

gold_example = {
    'inputs':
        np.array([
            4.3820267e+00,
            6.5206213e+00,
            2.3978953e+00,
            5.1532917e+00,
            6.9314718e-01,
            0.0000000e+00,
            0.0000000e+00,
            4.4998097e+00,
            1.9459102e+00,
            0.0000000e+00,
            6.9314718e-01,
            9.2467690e+00,
            2.4849067e+00,
            2.7102520e+07,
            1.8506000e+04,
            9.5080000e+03,
            3.9020000e+03,
            2.6690000e+03,
            1.0000000e+00,
            3.6620000e+03,
            8.2900000e+02,
            4.1000000e+01,
            1.4658379e+07,
            2.5537880e+06,
            3.0605000e+04,
            5.0000000e+00,
            4.4800000e+02,
            4.6330000e+03,
            5.6000000e+01,
            0.0000000e+00,
            3.3100000e+02,
            1.3000000e+01,
            2.6837320e+07,
            2.1765420e+07,
            2.3281244e+07,
            3.5562700e+05,
            8.7050000e+03,
            1.0200000e+02,
            7.0000000e+00
        ],
                 dtype=np.float32),
    'targets':
        0.0,
    'weights':
        1.0
}


def main(_):
  ds = input_pipeline.get_criteo1tb_dataset(
      split='validation',
      shuffle_rng=jax.random.PRNGKey(0),
      data_dir='/home/znado/criteo',
      num_dense_features=13,
      global_batch_size=524_288)
  gold_features = gold_example['inputs'][:13]
  for bi, batch in enumerate(iter(ds)):
    if bi < 90:
      continue
    # print(jax.tree_map(lambda x: x[0], batch))
    # assert False
    # print(bi)
    # print(batch['inputs'][:, :13].shape)
    # Matches elementwise on each dense dim.
    np_features = batch['inputs'][0, :, :13]
    diffs = np.sum(np.abs(gold_features - np_features), axis=1)
    print(np.min(diffs))
    matches = np.sum(gold_features == np_features, axis=1) == 13
    found = np.sum(matches)
    if found > 0:
      print('exact match')
      match_index = np.argmax(matches)
      print(bi, match_index, jax.tree_map(lambda x: x[0, match_index], batch))
      break
    if np.min(diffs) < 1e-6:
      print('approx match')
      match_index = np.argmin(diffs)
      print(bi, match_index, jax.tree_map(lambda x: x[0, match_index], batch))
      break


if __name__ == '__main__':
  app.run(main)
