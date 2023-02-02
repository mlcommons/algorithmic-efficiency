r"""MLCommons dataset setup script for development workloads.

If you already have a copy of a dataset(s), you can skip download it and provide
the path when running your algorithm with submission_runner.py via --data_dir.

Example command:

python3 datasets/development_dataset_setup.py \
  --data_dir=~/data \
  --framework=jax
"""

import os

from absl import app
from absl import flags
from absl import logging
import tensorflow_datasets as tfds
from torchvision.datasets import CIFAR10

flags.DEFINE_boolean(
    'all',
    False,
    'Whether or not to download all datasets. If false, can download some '
    'combination of datasets by setting the individual dataset flags below.')
flags.DEFINE_boolean('mnist',
                     False,
                     'If --all=false, whether or not to download MNIST.')
flags.DEFINE_boolean('cifar',
                     False,
                     'If --all=false, whether or not to download CIFAR-10.')

flags.DEFINE_string(
    'data_dir',
    '.',
    'The path to the folder where datasets should be downloaded.')

flags.DEFINE_string('framework', None, 'Can be either jax or pytorch.')
FLAGS = flags.FLAGS


def _maybe_mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)


def download_mnist(data_dir):
  tfds.builder('mnist', data_dir=data_dir).download_and_prepare()


def download_cifar(data_dir, framework):
  if framework == 'jax':
    tfds.builder('cifar10:3.0.2', data_dir=data_dir).download_and_prepare()
  elif framework == 'pytorch':
    CIFAR10(root=data_dir, train=True, download=True)
    CIFAR10(root=data_dir, train=False, download=True)
  else:
    raise ValueError('Invalid value for framework: {}'.format(framework))


def main(_):
  data_dir = FLAGS.data_dir
  bad_chars = [';', ' ', '&', '"']
  if any(s in data_dir for s in bad_chars):
    raise ValueError(f'Invalid data_dir: {data_dir}.')
  data_dir = os.path.abspath(os.path.expanduser(data_dir))
  logging.info('Downloading data to %s...', data_dir)

  if FLAGS.all or FLAGS.mnist:
    logging.info('Downloading MNIST...')
    download_mnist(data_dir)

  if FLAGS.all or FLAGS.cifar:
    logging.info('Downloading CIFAR...')
    download_cifar(data_dir, FLAGS.framework)


if __name__ == '__main__':
  app.run(main)
