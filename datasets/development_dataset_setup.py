r"""MLCommons dataset setup script for development workloads.

If you already have a copy of a dataset(s), you can skip download it and provide
the path when running your algorithm with submission_runner.py via --data_dir.

Note that some functions use subprocess.Popen(..., shell=True), which can be
dangerous if the user injects code into the --data_dir or --temp_dir flags. We
do some basic sanitization in main(), but submitters should not let untrusted
users run this script on their systems.

If mounting a GCS bucket with gcsfuse, --temp_dir should NOT be a path to the
GCS bucket, as this can result in *orders of magnitude* slower download speeds
due to write speed issues (--data_dir can include the GCS bucket though).

Note that some of the disk usage number below may be underestimates if the temp
and final data dir locations are on the same drive.

Criteo download size: ~350GB
Criteo final disk size: ~1TB
FastMRI download size:
FastMRI final disk size:
LibriSpeech download size:
LibriSpeech final disk size:
OGBG download size:
OGBG final disk size:
WMT download size: (1.58 GiB + ) =
WMT final disk size:
_______________________
Total download size:
Total disk size:

Example command:

python3 datasets/development_dataset_setup.py \
  --data_dir=~/data \
  --temp_dir=/tmp/mlcommons_data
  --framework=jax
"""
# pylint: disable=logging-format-interpolation
# pylint: disable=consider-using-with
import os
import subprocess

from absl import app
from absl import flags
from absl import logging
import tensorflow_datasets as tfds

FRAMEWORKS = ['pytorch', 'jax']

flags.DEFINE_boolean(
    'interactive_deletion',
    True,
    'If true, user will be prompted before any files are deleted. If false, no '
    'files will be deleted.')
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
flags.DEFINE_string(
    'temp_dir',
    '/tmp',
    'A local path to a folder where temp files can be downloaded.')
flags.DEFINE_integer(
    'num_decompression_threads',
    8,
    'The number of threads to use in parallel when decompressing.')

flags.DEFINE_string('framework', None, 'Can be either jax or pytorch.')
FLAGS = flags.FLAGS


def _maybe_mkdir(d):
  if not os.path.exists(d):
    os.makedirs(d)


def _maybe_prompt_for_deletion(paths, interactive_deletion):
  if not interactive_deletion:
    return
  files_for_deletion = '\n'.join(paths)
  logging.info('\n\n\nWARNING: the following temp files will be DELETED:'
               f'\n{files_for_deletion}')
  delete_str = input('Confirm deletion? [y/N]: ')
  if delete_str.lower() == 'y':
    del_cmd = 'rm ' + ' '.join(f'"{s}"' for s in paths)
    logging.info(f'Running deletion command:\n{del_cmd}')
    subprocess.Popen(del_cmd, shell=True).communicate()
  else:
    logging.info('Skipping deletion.')


def download_mnist(data_dir):
  tfds.builder('mnist:3.0.1', data_dir=data_dir).download_and_prepare()


def main(_):
  data_dir = FLAGS.data_dir
  tmp_dir = FLAGS.temp_dir
  bad_chars = [';', ' ', '&', '"']
  if any(s in data_dir for s in bad_chars):
    raise ValueError(f'Invalid data_dir: {data_dir}.')
  if any(s in tmp_dir for s in bad_chars):
    raise ValueError(f'Invalid temp_dir: {tmp_dir}.')
  data_dir = os.path.abspath(os.path.expanduser(data_dir))
  logging.info('Downloading data to %s...', data_dir)

  if FLAGS.all or FLAGS.mnist:
    logging.info('Downloading MNIST...')
    download_mnist(data_dir)


# pylint: enable=logging-format-interpolation
# pylint: enable=consider-using-with

if __name__ == '__main__':
  app.run(main)
