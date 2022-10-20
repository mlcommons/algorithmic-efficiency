r"""MLCommons dataset setup script.

If you already have a copy of a dataset(s), you can skip download it and provide
the path when running your algorithm with submission_runner.py via --data_dir.

Note that in order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives). All of these
files will be downloaded to the provided --temp_dir, and the user can manually
delete these after downloading has finished.

If mounting a GCS bucket with gcsfuse, --temp_dir should NOT be a path to the
GCS bucket, as this can result in *orders of magnitude* slower download speeds
due to write speed issues (--dataset_dir can include the GCS bucket though).

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

Some datasets require signing a form before downloading:

FastMRI:
Fill out form on https://fastmri.med.nyu.edu/ and run this script with the
links that are emailed to you for "knee_singlecoil_train" and
"knee_singlecoil_val".

ImageNet:
Register on https://image-net.org/ and run this script with the links to the
ILSVRC2012 train and validation images.

Note for tfds ImageNet, you may have to increase the max number of files allowed
open at once using `ulimit -n 8192`.

Example command:

python3 dataset_setup.py --dataset_dir=/data --temp_dir=/tmp/mlcommons_data
"""
import os

from absl import app
from absl import flags
from absl import logging
import subprocess
import tensorflow_datasets as tfds

from datasets import librispeech_preprocess
from datasets import librispeech_tokenizer


flags.DEFINE_boolean(
    'all',
    True,
    'Whether or not to download all datasets. If false, can download some '
    'combination of datasets by setting the individual dataset flags below.')
flags.DEFINE_boolean(
    'criteo',
    False,
    'If --all=false, whether or not to download Criteo.')
flags.DEFINE_boolean(
    'fastmri',
    False,
    'If --all=false, whether or not to download FastMRI.')
flags.DEFINE_boolean(
    'librispeech',
    False,
    'If --all=false, whether or not to download LibriSpeech.')
flags.DEFINE_boolean(
    'ogbg',
    False,
    'If --all=false, whether or not to download OGBG.')
flags.DEFINE_boolean(
    'wmt',
    False,
    'If --all=false, whether or not to download WMT.')

flags.DEFINE_string(
    'dataset_dir',
    None,
    'The path to the folder where datasets should be downloaded.')
flags.DEFINE_string(
    'temp_dir',
    '/tmp',
    'A local path to a folder where temp files can be downloaded.')

flags.DEFINE_string(
    'imagenet_train_url',
    None,
    'Only necessary if you want this script to `wget` the ImageNet train '
    'split. If not, you can supply the path to --data_dir in '
    'submission_runner.py.')
flags.DEFINE_string(
    'imagenet_val_url',
    None,
    'Only necessary if you want this script to `wget` the ImageNet validation '
    'split. If not, you can supply the path to --data_dir in '
    'submission_runner.py.')
flags.DEFINE_string(
    'fastmri_knee_singlecoil_train_url',
    None,
    'Only necessary if you want this script to `wget` the FastMRI train '
    'split. If not, you can supply the path to --data_dir in '
    'submission_runner.py.')
flags.DEFINE_string(
    'fastmri_knee_singlecoil_val_url',
    None,
    'Only necessary if you want this script to `wget` the FastMRI validation '
    'split. If not, you can supply the path to --data_dir in '
    'submission_runner.py.')

flags.DEFINE_integer(
    'num_decompression_threads',
    8,
    'The number of threads to use in parallel when decompressing.')

flags.DEFINE_boolean('train_tokenizer', True, 'Train Librispeech tokenizer.')
FLAGS = flags.FLAGS


def download_criteo(dataset_dir, tmp_dir, num_decompression_threads):
  criteo_dir = os.path.join(dataset_dir, 'criteo')
  tmp_criteo_dir = os.path.join(tmp_dir, 'criteo')
  processes = []
  for day in range(24):
    if day in [0, 1, 2, 3, 23]:  # DO NOT SUBMIT
      continue
    logging.info(f'Downloading Criteo day {day}...')
    #
    # DOWNLOADING STRAIGHT TO gcsfuse mounted GCS IS 4000x SLOWER!!!!
    #
    wget_cmd = (
        f'wget --directory-prefix={tmp_criteo_dir} '
        f'https://storage.googleapis.com/criteo-cail-datasets/day_{day}.gz')
    input_path = os.path.join(tmp_criteo_dir, f'day_{day}.gz')
    output_path = os.path.join(criteo_dir, f'day_{day}.csv')
    unzip_cmd = (
        f'pigz -d -c -p{num_decompression_threads} {input_path} > '
        f'{output_path}')
    command_str = f'{wget_cmd} && {unzip_cmd}'
    logging.info(f'Running Criteo download command:\n{command_str}')
    # Note that shell=True can be dangerous if the user injects code into the
    # --dataset_dir flag. We do some basic sanitization in main(), but
    # submitters should not let untrusted users run this script on their
    # systems.
    processes.append(subprocess.Popen(command_str, shell=True))
  for p in processes:
    p.communicate()


def download_fastmri(
    dataset_dir,
    tmp_dir,
    knee_singlecoil_train_url,
    knee_singlecoil_val_url):
  pass


def download_imagenet(
    dataset_dir, tmp_dir, imagenet_train_url, imagenet_val_url):
  download_imagenet_v2(dataset_dir)


def download_imagenet_v2(dataset_dir):
  tfds.builder(
      'imagenet_v2/matched-frequency:3.0.0',
      data_dir=dataset_dir).download_and_prepare()


def download_librispeech(dataset_dir, tmp_dir, train_tokenizer):
  # After extraction the result is a folder named Librispeech containing audio
  # files in .flac format along with transcripts containing name of audio file
  # and corresponding transcription.
  tmp_librispeech_dir = os.path.join(tmp_dir, 'librispeech')

  for split in ['dev', 'test']:
    for version in ['clean', 'other']:
      wget_cmd = (
          f'wget --directory-prefix={tmp_librispeech_dir} '
          f'http://www.openslr.org/resources/12/{split}-{version}.tar.gz')
      subprocess.Popen(wget_cmd, shell=True)
      subprocess.Popen(f'tar xzvf {split}-{version}.tar.gz', shell=True)

  tars = [
    'raw-metadata.tar.gz',
    'train-clean-100.tar.gz',
    'train-clean-360.tar.gz',
    'train-other-500.tar.gz',
  ]
  for tar_filename in tars:
    wget_cmd = (
        f'wget --directory-prefix={tmp_librispeech_dir} '
        f'http://www.openslr.org/resources/12/{tar_filename}')
    subprocess.Popen(wget_cmd, shell=True)
    tar_path = os.path.join(tmp_librispeech_dir, tar_filename)
    subprocess.Popen(f'tar xzvf {tar_path}', shell=True)

  if train_tokenizer:
    librispeech_tokenizer.run(train=True, data_dir=tmp_librispeech_dir)

    # Preprocess data.
    tokenizer_vocab_path = os.path.join(tmp_librispeech_dir, 'spm_model.vocab')
    librispeech_dir = os.path.join(dataset_dir, 'criteo')
    librispeech_preprocess.run(
        input_dir=tmp_librispeech_dir,
        output_dir=librispeech_dir,
        tokenizer_vocab_path=tokenizer_vocab_path)


def download_ogbg(dataset_dir, tmp_dir):
  pass


def download_wmt(dataset_dir):
  """WMT14 and WMT17 de-en."""
  for ds_name in ['wmt14_translate/de-en:1.0.0', 'wmt17_translate/de-en:1.0.0']:
    tfds.builder(ds_name, data_dir=dataset_dir).download_and_prepare()


def main(_):
  dataset_dir = FLAGS.dataset_dir
  tmp_dir = FLAGS.temp_dir
  num_decompression_threads = FLAGS.num_decompression_threads
  if ';' in dataset_dir or ' ' in dataset_dir or '&' in dataset_dir:
    raise ValueError(f'Invalid dataset_dir: {dataset_dir}.')
  dataset_dir = os.path.abspath(dataset_dir)
  logging.info('Downloading data to %s...', dataset_dir)
  if FLAGS.all or FLAGS.criteo:
    logging.info('Downloading criteo...')
    download_criteo(dataset_dir, tmp_dir, num_decompression_threads)
  if FLAGS.all or FLAGS.fastmri:
    logging.info('Downloading FastMRI...')
    knee_singlecoil_train_url = FLAGS.fastmri_knee_singlecoil_train_url
    knee_singlecoil_val_url = FLAGS.fastmri_knee_singlecoil_val_url
    if knee_singlecoil_train_url is None or knee_singlecoil_val_url is None:
      raise ValueError(
          'Must provide both --fastmri_knee_singlecoil_{train,val}_url to '
          'download the FastMRI dataset. Sign up for the URLs at '
          'https://fastmri.med.nyu.edu/.')
    # download_fastmri(
    #     dataset_dir,
    #     tmp_dir,
    #     knee_singlecoil_train_url,
    #     knee_singlecoil_val_url)
  if FLAGS.all or FLAGS.fastmri:
    logging.info('Downloading ImageNet...')
    imagenet_train_url = FLAGS.imagenet_train_url
    imagenet_val_url = FLAGS.imagenet_val_url
    if imagenet_train_url is None or imagenet_val_url is None:
      raise ValueError(
          'Must provide both --imagenet_{train,val}_url to download the '
          'ImageNet dataset. Sign up for the URLs at https://image-net.org/.')
    # download_imagenet(
    #     dataset_dir, tmp_dir, imagenet_train_url, imagenet_val_url)
  if FLAGS.all or FLAGS.librispeech:
    logging.info('Downloading Librispeech...')
    download_librispeech(dataset_dir, tmp_dir, FLAGS.train_tokenizer)
  if FLAGS.all or FLAGS.ogbg:
    logging.info('Downloading OGBG...')
    # download_ogbg(dataset_dir, tmp_dir)
  if FLAGS.all or FLAGS.wmt:
    logging.info('Downloading WMT...')
    # download_wmt(dataset_dir)


if __name__ == '__main__':
  app.run(main)
