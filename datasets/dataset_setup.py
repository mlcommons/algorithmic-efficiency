r"""MLCommons dataset setup script.

If you already have a copy of a dataset(s), you can skip download it and provide
the path when running your algorithm with submission_runner.py via --data_dir.

Note that in order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives) without a user
confirmation. Deleting temp files is particularly important for Criteo 1TB, as
there can be multiple copies of the dataset on disk during preprocessing if
files are not cleaned up. If you do not want any temp files to be deleted, you
can pass --interactive_deletion=false and then all files will be downloaded to
the provided --temp_dir, and the user can manually delete these after
downloading has finished.

Note that some functions use subprocess.Popen(..., shell=True), which can be
dangerous if the user injects code into the --data_dir or --temp_dir flags. We
do some basic sanitization in main(), but submitters should not let untrusted
users run this script on their systems.

If mounting a GCS bucket with gcsfuse, --temp_dir should NOT be a path to the
GCS bucket, as this can result in *orders of magnitude* slower download speeds
due to write speed issues (--data_dir can include the GCS bucket though).

Note that some of the disk usage number below may be underestimates if the temp
and final data dir locations are on the same drive.

Criteo 1TB download size: ~350GB
Criteo 1TB final disk size: ~1TB
FastMRI download size: ~90GB
FastMRI final disk size: ~110GB
ImageNet download size: ~150GB
ImageNet final disk size: ~150GB
LibriSpeech download size: ~60GB
LibriSpeech final disk size: ~350GB
OGBG download size: ~37MB
OGBG final disk size: ~800MB
WMT download size: ~3GB
WMT final disk size: ~3GB
_______________________
Total download size: ~650GB
Total disk size: ~1.1TB

Some datasets require signing a form before downloading:

FastMRI:
Fill out form on https://fastmri.med.nyu.edu/ and run this script with the
links that are emailed to you for "knee_singlecoil_train" and
"knee_singlecoil_val".

ImageNet:
Register on https://image-net.org/ and run this script with the links to the
ILSVRC2012 train and validation images.

Note for tfds ImageNet, you may have to increase the max number of files
allowed open at once using `ulimit -n 8192`.

Example command:

python3 datasets/dataset_setup.py \
  --data_dir=~/data \
  --temp_dir=/tmp/mlcommons_data
  --imagenet \
  --imagenet_train_url=<train_url> \
  --imagenet_val_url=<val_url>\
  --framework=jax
"""
# pylint: disable=logging-format-interpolation
# pylint: disable=consider-using-with

# isort: off
import tensorflow_datasets as tfds
from torchvision.datasets import CIFAR10

from algoperf.workloads.wmt import tokenizer
from algoperf.workloads.wmt.input_pipeline import \
    normalize_feature_names
from datasets import librispeech_preprocess
from datasets import librispeech_tokenizer

import datasets as hf_datasets
from transformers import AutoTokenizer

import math
import functools
import itertools
import os
import shutil
import subprocess
import tarfile

from typing import Dict, List, Any
from absl import app
from absl import flags
from absl import logging
import re
import requests
import tqdm
import urllib.parse

import tensorflow as tf

IMAGENET_TRAIN_TAR_FILENAME = 'ILSVRC2012_img_train.tar'
IMAGENET_VAL_TAR_FILENAME = 'ILSVRC2012_img_val.tar'

FASTMRI_TRAIN_TAR_FILENAME = 'knee_singlecoil_train.tar.xz'
FASTMRI_VAL_TAR_FILENAME = 'knee_singlecoil_val.tar.xz'
FASTMRI_TEST_TAR_FILENAME = 'knee_singlecoil_test.tar.xz'

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

flags.DEFINE_boolean('criteo1tb',
                     False,
                     'If --all=false, whether or not to download Criteo 1TB.')
flags.DEFINE_boolean('cifar',
                     False,
                     'If --all=false, whether or not to download CIFAR-10.')
flags.DEFINE_boolean('fastmri',
                     False,
                     'If --all=false, whether or not to download FastMRI.')
flags.DEFINE_boolean('imagenet',
                     False,
                     'If --all=false, whether or not to download Imagenet.')
flags.DEFINE_boolean('librispeech',
                     False,
                     'If --all=false, whether or not to download LibriSpeech.')
flags.DEFINE_boolean('finewebedu',
                     False,
                     'If --all=false, whether or not to download FineWebEdu.')
flags.DEFINE_boolean('mnist',
                     False,
                     'If --all=false, whether or not to download MNIST.')
flags.DEFINE_boolean('ogbg',
                     False,
                     'If --all=false, whether or not to download OGBG.')
flags.DEFINE_boolean('wmt',
                     False,
                     'If --all=false, whether or not to download WMT.')

flags.DEFINE_string(
    'data_dir',
    '~/data',
    'The path to the folder where datasets should be downloaded.')
flags.DEFINE_string(
    'temp_dir',
    '/tmp/mlcommons',
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
flags.DEFINE_string(
    'fastmri_knee_singlecoil_test_url',
    None,
    'Only necessary if you want this script to `wget` the FastMRI test '
    'split. If not, you can supply the path to --data_dir in '
    'submission_runner.py.')

flags.DEFINE_integer(
    'num_decompression_threads',
    8,
    'The number of threads to use in parallel when decompressing.')

flags.DEFINE_string('framework', None, 'Can be either jax or pytorch.')

flags.DEFINE_boolean('skip_download', False, 'Skips data download.')

FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


def _download_url(url, data_dir, name=None):
  if not name:
    file_path = os.path.join(data_dir, url.split('/')[-1])
  else:
    file_path = os.path.join(data_dir, name)
  logging.info(f'Downloading URL {url} to {file_path}')

  response = requests.get(url, stream=True, timeout=600)
  total_size_in_bytes = int(response.headers.get('Content-length', 0))
  total_size_in_mib = total_size_in_bytes / (2**20)
  progress_bar = tqdm.tqdm(total=total_size_in_mib, unit='MiB', unit_scale=True)
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if os.path.exists(file_path):
    while True:
      overwrite = input('File already exists {}.\n Overwrite? (Y/n)'.format(
          file_path)).lower()
      if overwrite in ['y', 'n']:
        break
      logging.info('Invalid response. Try again.')
    if overwrite == 'n':
      logging.info(f'Skipping download URL {url} to {file_path}')
      return

  with open(file_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=2**10):
      chunk_size_in_mib = len(chunk) / (2**20)
      progress_bar.update(chunk_size_in_mib)
      f.write(chunk)
  progress_bar.close()
  if (progress_bar.total != 0 and progress_bar.n != progress_bar.total):
    raise RuntimeError(
        ('Download corrupted, size {n} MiB from {url} does not match '
         'expected size {size} MiB').format(
             url=url, n=progress_bar.n, size=progress_bar.total))


def download_criteo1tb(data_dir,
                       tmp_dir,
                       num_decompression_threads,
                       interactive_deletion):
  criteo_dir = os.path.join(data_dir, 'criteo1tb')
  tmp_criteo_dir = os.path.join(tmp_dir, 'criteo1tb')
  _maybe_mkdir(criteo_dir)
  _maybe_mkdir(tmp_criteo_dir)

  # Forked from
  # https://github.com/iamleot/transferwee/blob/master/transferwee.py.
  user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) '
                'Gecko/20100101 Firefox/102.0')
  criteo_wetransfer_url = (
      'https://criteo.wetransfer.com/downloads/'
      '4bbea9b4a54baddea549d71271a38e2c20230428071257/d4f0d2')
  _, _, transfer_id, security_hash = urllib.parse.urlparse(
      criteo_wetransfer_url).path.split('/')

  session = requests.Session()
  session.headers.update({
      'User-Agent': user_agent,
      'x-requested-with': 'XMLHttpRequest',
  })
  r = session.get('https://wetransfer.com/')
  m = re.search('name="csrf-token" content="([^"]+)"', r.text)
  if m:
    session.headers.update({'x-csrf-token': m.group(1)})

  get_url_request = session.post(
      f'https://wetransfer.com/api/v4/transfers/{transfer_id}/download',
      json={
          'intent': 'entire_transfer',
          'security_hash': security_hash,
      })
  session.close()

  download_url = get_url_request.json().get('direct_link')

  logging.info(f'Downloading ~342GB Criteo 1TB data .zip file:\n{download_url}')
  download_request = requests.get(  # pylint: disable=missing-timeout
      download_url,
      headers={'User-Agent': user_agent},
      stream=True)

  all_days_zip_filepath = os.path.join(tmp_criteo_dir, 'all_days.zip')
  if not FLAGS.skip_download:
    download = True
    if os.path.exists(all_days_zip_filepath):
      while True:
        overwrite = input('File already exists {}.\n Overwrite? (Y/n)'.format(
            all_days_zip_filepath)).lower()
        if overwrite in ['y', 'n']:
          break
        logging.info('Invalid response. Try again.')
      if overwrite == 'n':
        logging.info(f'Skipping download to {all_days_zip_filepath}')
        download = False

    if download:
      with open(all_days_zip_filepath, 'wb') as f:
        for chunk in download_request.iter_content(chunk_size=1024):
          f.write(chunk)

  unzip_cmd = f'unzip {all_days_zip_filepath} -d {tmp_criteo_dir}'
  logging.info(f'Running Criteo 1TB unzip command:\n{unzip_cmd}')
  p = subprocess.Popen(unzip_cmd, shell=True)
  p.communicate()
  _maybe_prompt_for_deletion([all_days_zip_filepath], interactive_deletion)

  # Unzip the individual days.
  processes = []
  gz_paths = []
  for day in range(24):
    input_path = os.path.join(tmp_criteo_dir, f'day_{day}.gz')
    gz_paths.append(input_path)
    unzipped_path = os.path.join(criteo_dir, f'day_{day}.csv')
    unzip_cmd = (f'pigz -d -c -p{num_decompression_threads} "{input_path}" > '
                 f'"{unzipped_path}"')
    logging.info(f'Running Criteo unzip command for day {day}:\n{unzip_cmd}')
    processes.append(subprocess.Popen(unzip_cmd, shell=True))
  for p in processes:
    p.communicate()
  _maybe_prompt_for_deletion(gz_paths, interactive_deletion)

  # Split into files with 5M lines each: day_1.csv -> day_1_[0-39].csv.
  unzipped_paths = []
  for batch in range(6):
    batch_processes = []
    for day_offset in range(4):
      day = batch * 4 + day_offset
      unzipped_path = os.path.join(criteo_dir, f'day_{day}.csv')
      unzipped_paths.append(unzipped_path)
      split_path = os.path.join(criteo_dir, f'day_{day}_')
      split_cmd = ('split -a 2 -d -l 5000000 '
                   f'"{unzipped_path}" "{split_path}"')
      logging.info(f'Running Criteo 1TB split command:\n{split_cmd}')
      batch_processes.append(subprocess.Popen(split_cmd, shell=True))
    for p in batch_processes:
      p.communicate()
  _maybe_prompt_for_deletion(unzipped_paths, interactive_deletion)


def download_cifar(data_dir, framework):
  data_dir = os.path.join(data_dir, 'cifar10')
  if framework == 'jax':
    tfds.builder('cifar10:3.0.2', data_dir=data_dir).download_and_prepare()
  elif framework == 'pytorch':
    CIFAR10(root=data_dir, train=True, download=True)
    CIFAR10(root=data_dir, train=False, download=True)
  else:
    raise ValueError('Invalid value for framework: {}'.format(framework))


def extract_filename_from_url(url, start_str='knee', end_str='.xz'):
  """ The url filenames are sometimes couched within a urldefense+aws access id
  etc. string. Unfortunately querying the content disposition in requests fails
  (not provided)... so fast search is done here within the url.
   """
  failure = -1
  start = url.find(start_str)
  end = url.find(end_str)
  if failure in (start, end):
    raise ValueError(
        f'Unable to locate filename wrapped in {start_str}--{end_str} in {url}')
  end += len(end_str)  # make it inclusive
  return url[start:end]


def download_fastmri(data_dir,
                     fastmri_train_url,
                     fastmri_val_url,
                     fastmri_test_url):
  data_dir = os.path.join(data_dir, 'fastmri')
  # Download fastmri train dataset
  knee_train_filename = extract_filename_from_url(fastmri_train_url)
  logging.info(
      'Downloading fastmri train dataset from {}'.format(fastmri_train_url))
  _download_url(
      url=fastmri_train_url, data_dir=data_dir, name=knee_train_filename)

  # Download fastmri val dataset
  knee_val_filename = extract_filename_from_url(fastmri_val_url)
  logging.info(
      'Downloading fastmri val dataset from {}'.format(fastmri_val_url))
  _download_url(url=fastmri_val_url, data_dir=data_dir, name=knee_val_filename)

  # Download fastmri test dataset
  knee_test_filename = extract_filename_from_url(fastmri_test_url)

  logging.info(
      'Downloading fastmri test dataset from {}'.format(fastmri_test_url))
  _download_url(
      url=fastmri_test_url, data_dir=data_dir, name=knee_test_filename)
  return data_dir


def extract(source, dest, mode='r:xz'):
  if not os.path.exists(dest):
    os.makedirs(dest)
  logging.info(f'Extracting {source} to {dest}')
  tar = tarfile.open(source, mode)
  logging.info('Opened tar')

  tar.extractall(dest)
  tar.close()


def setup_fastmri(data_dir):
  data_dir = os.path.join(data_dir, 'fastmri')
  train_tar_file_path = os.path.join(data_dir, FASTMRI_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, FASTMRI_VAL_TAR_FILENAME)
  test_tar_file_path = os.path.join(data_dir, FASTMRI_TEST_TAR_FILENAME)

  # Unzip tar file into subdirectories
  logging.info('Unzipping {} to {}'.format(train_tar_file_path, data_dir))
  extract(train_tar_file_path, data_dir)
  logging.info('Unzipping {} to {}'.format(val_tar_file_path, data_dir))
  extract(val_tar_file_path, data_dir)
  logging.info('Unzipping {} to {}'.format(test_tar_file_path, data_dir))
  extract(test_tar_file_path, data_dir)
  logging.info('Extraction completed!')

  # Rename folders to match what the workload expects
  os.rename(
      os.path.join(data_dir, "singlecoil_train"),
      os.path.join(data_dir, "knee_singlecoil_train"),
  )
  os.rename(
      os.path.join(data_dir, "singlecoil_val"),
      os.path.join(data_dir, "knee_singlecoil_val"),
  )
  os.rename(
      os.path.join(data_dir, "singlecoil_test"),
      os.path.join(data_dir, "knee_singlecoil_test"),
  )
  logging.info("Set up fastMRI dataset complete")


def download_imagenet(data_dir, imagenet_train_url, imagenet_val_url):
  """Downloads imagenet tar files to $DATA_DIR/imagenet/."""
  data_dir = os.path.join(data_dir, 'imagenet')
  imagenet_train_filepath = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  imagenet_val_filepath = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)

  # If the data was already downloaded for JAX it will have
  # been moved to the manual_download_dir.
  # Get paths in manual_download_dir.
  imagenet_jax_data_dir = os.path.join(data_dir, 'jax')
  manual_download_dir = os.path.join(imagenet_jax_data_dir,
                                     'downloads',
                                     'manual')
  imagenet_train_download_filepath = os.path.join(manual_download_dir,
                                                  IMAGENET_TRAIN_TAR_FILENAME)
  imagenet_val_download_filepath = os.path.join(manual_download_dir,
                                                IMAGENET_VAL_TAR_FILENAME)

  # Download imagenet train dataset
  if not os.path.exists(imagenet_train_filepath) and not os.path.exists(
      imagenet_train_download_filepath):
    logging.info(
        'Downloading imagenet train dataset from {}'.format(imagenet_train_url))
    _download_url(url=imagenet_train_url, data_dir=data_dir)

  # Download imagenet val dataset
  if not os.path.exists(imagenet_val_filepath) and not os.path.exists(
      imagenet_val_download_filepath):
    logging.info('Downloading imagenet validation dataset from {}'.format(
        imagenet_val_url))
    _download_url(url=imagenet_val_url, data_dir=data_dir)

  # Download imagenet test set
  download_imagenet_v2(data_dir)


def setup_imagenet(data_dir, framework=None):
  data_dir = os.path.join(data_dir, 'imagenet')
  if framework == 'jax':
    setup_imagenet_jax(data_dir)

  elif framework == 'pytorch':
    setup_imagenet_pytorch(data_dir)

  else:
    raise ValueError('Invalid value for framework: {}'.format(framework))


def setup_imagenet_jax(data_dir):
  train_tar_file_path = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)
  test_dir_path = os.path.join(data_dir, 'imagenet_v2')

  # Setup jax dataset dir
  imagenet_jax_data_dir = os.path.join(data_dir, 'jax')
  manual_download_dir = os.path.join(imagenet_jax_data_dir,
                                     'downloads',
                                     'manual')
  os.makedirs(manual_download_dir, exist_ok=True)

  # Copy tar file into jax/downloads/manual
  logging.info('Checking if tar files already exists in jax/downloads/manual.')
  if not os.path.exists(
      os.path.join(manual_download_dir, IMAGENET_TRAIN_TAR_FILENAME)):
    logging.info('Moving {} to {}'.format(train_tar_file_path,
                                          manual_download_dir))
    shutil.move(train_tar_file_path, manual_download_dir)
  if not os.path.exists(
      os.path.join(manual_download_dir, IMAGENET_VAL_TAR_FILENAME)):
    logging.info('Moving {} to {}'.format(val_tar_file_path,
                                          manual_download_dir))
    shutil.move(val_tar_file_path, manual_download_dir)
  if not os.path.exists(os.path.join(imagenet_jax_data_dir, 'imagenet_v2')):
    logging.info('Moving imagenet_v2 to {}'.format(
        os.path.join(imagenet_jax_data_dir, 'imagenet_v2')))
    shutil.move(test_dir_path,
                os.path.join(imagenet_jax_data_dir, 'imagenet_v2'))
  logging.info('Preparing imagenet data.')
  ds_builder = tfds.builder(
      'imagenet2012:5.1.0', data_dir=os.path.join(imagenet_jax_data_dir))
  ds_builder.download_and_prepare()
  logging.info('Set up imagenet dataset for jax framework complete')


def setup_imagenet_pytorch(data_dir):
  train_tar_file_path = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)
  test_dir_path = os.path.join(data_dir, 'imagenet_v2')

  # Check if downloaded data has been moved
  manual_download_dir = os.path.join(data_dir, 'jax', 'downloads', 'manual')
  if not os.path.exists(train_tar_file_path):
    if os.path.exists(
        os.path.join(manual_download_dir, IMAGENET_TRAIN_TAR_FILENAME)):
      train_tar_file_path = os.path.join(manual_download_dir,
                                         IMAGENET_TRAIN_TAR_FILENAME)
  if not os.path.exists(val_tar_file_path):
    if os.path.exists(
        os.path.join(manual_download_dir, IMAGENET_VAL_TAR_FILENAME)):
      val_tar_file_path = os.path.join(manual_download_dir,
                                       IMAGENET_VAL_TAR_FILENAME)

  # Setup pytorch dataset dir
  imagenet_pytorch_data_dir = os.path.join(data_dir, 'pytorch')
  if not os.path.exists(os.path.join(imagenet_pytorch_data_dir, 'train')):
    os.makedirs(os.path.join(imagenet_pytorch_data_dir, 'train'))
  if not os.path.exists(os.path.join(imagenet_pytorch_data_dir, 'val')):
    os.makedirs(os.path.join(imagenet_pytorch_data_dir, 'val'))

  # Move tar files and imagenet_v2 into pytorch directory
  if not os.path.exists(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_TRAIN_TAR_FILENAME)):
    logging.info('Moving {} to {}'.format(train_tar_file_path,
                                          imagenet_pytorch_data_dir))
    shutil.move(train_tar_file_path, imagenet_pytorch_data_dir)
  if not os.path.exists(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_VAL_TAR_FILENAME)):
    logging.info('Moving {} to {}'.format(val_tar_file_path,
                                          imagenet_pytorch_data_dir))
    shutil.move(val_tar_file_path, imagenet_pytorch_data_dir)
  if not os.path.exists(os.path.join(imagenet_pytorch_data_dir, 'imagenet_v2')):
    logging.info('Moving imagenet_v2 to {}'.format(
        os.path.join(imagenet_pytorch_data_dir, 'imagenet_v2')))
    shutil.move(test_dir_path,
                os.path.join(imagenet_pytorch_data_dir, 'imagenet_v2'))

  # Extract train data\
  logging.info('Extracting imagenet train data')
  extract(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_TRAIN_TAR_FILENAME),
      os.path.join(imagenet_pytorch_data_dir, 'train'),
      mode='r:')

  train_tar_filenames = os.listdir(
      os.path.join(imagenet_pytorch_data_dir, 'train'))
  for tar_filename in train_tar_filenames:
    if tar_filename.endswith('.tar'):
      dir_name = tar_filename[:-4]
      extract(
          os.path.join(imagenet_pytorch_data_dir, 'train', tar_filename),
          os.path.join(imagenet_pytorch_data_dir, 'train', dir_name),
          mode='r:')

  # Extract val data
  logging.info('Extracting imagenet val data')
  extract(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_VAL_TAR_FILENAME),
      os.path.join(imagenet_pytorch_data_dir, 'val'),
      mode='r:')

  valprep_command = [
      'wget',
      '-qO-',
      'https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh'
  ]
  valprep_download = subprocess.Popen(valprep_command, stdout=subprocess.PIPE)
  valprep_process = subprocess.Popen(['bash'],
                                     stdin=valprep_download.stdout,
                                     cwd=os.path.expanduser(
                                         os.path.join(imagenet_pytorch_data_dir,
                                                      'val')))
  valprep_download.stdout.close()
  valprep_process.communicate()
  logging.info('Set up imagenet dataset for pytorch framework complete')


def download_imagenet_v2(data_dir):
  tfds.builder(
      'imagenet_v2/matched-frequency:3.0.0',
      data_dir=data_dir).download_and_prepare()


def download_librispeech(data_dir, tmp_dir):
  # After extraction the result is a folder named Librispeech containing audio
  # files in .flac format along with transcripts containing name of audio file
  # and corresponding transcription.
  tmp_librispeech_dir = os.path.join(tmp_dir, 'librispeech')
  extracted_data_dir = os.path.join(tmp_librispeech_dir, 'LibriSpeech')
  final_data_dir = os.path.join(data_dir, 'librispeech')

  _maybe_mkdir(tmp_librispeech_dir)
  _maybe_mkdir(final_data_dir)

  for split in ['dev', 'test']:
    for version in ['clean', 'other']:
      if split == 'test' and version == 'other':
        continue
      wget_cmd = (
          f'wget --directory-prefix={tmp_librispeech_dir} '
          f'http://www.openslr.org/resources/12/{split}-{version}.tar.gz')
      subprocess.Popen(wget_cmd, shell=True).communicate()
      tar_path = os.path.join(tmp_librispeech_dir, f'{split}-{version}.tar.gz')
      subprocess.Popen(
          f'tar xzvf {tar_path} --directory {tmp_librispeech_dir}',
          shell=True).communicate()

  tars = [
      'raw-metadata.tar.gz',
      'train-clean-100.tar.gz',
      'train-clean-360.tar.gz',
      'train-other-500.tar.gz',
  ]
  for tar_filename in tars:
    wget_cmd = (f'wget --directory-prefix={tmp_librispeech_dir} '
                f'http://www.openslr.org/resources/12/{tar_filename}')
    subprocess.Popen(wget_cmd, shell=True).communicate()
    tar_path = os.path.join(tmp_librispeech_dir, tar_filename)
    subprocess.Popen(
        f'tar xzvf {tar_path} --directory {tmp_librispeech_dir}',
        shell=True).communicate()

  tokenizer_vocab_path = os.path.join(final_data_dir, 'spm_model.vocab')

  if not os.path.exists(tokenizer_vocab_path):
    librispeech_tokenizer.run(
        train=True,
        input_dir=extracted_data_dir,
        tokenizer_vocab_path=tokenizer_vocab_path)

  librispeech_preprocess.run(
      input_dir=extracted_data_dir,
      output_dir=final_data_dir,
      tokenizer_vocab_path=tokenizer_vocab_path)


def download_mnist(data_dir):
  data_dir = os.path.join(data_dir, 'MNIST')  # Capitalization to match PyTorch
  tfds.builder('mnist', data_dir=data_dir).download_and_prepare()


def download_ogbg(data_dir):
  data_dir = os.path.join(data_dir, 'ogbg')
  tfds.builder('ogbg_molpcba:0.1.3', data_dir=data_dir).download_and_prepare()


def download_wmt(data_dir):
  """WMT14 and WMT17 de-en."""
  data_dir = os.path.join(data_dir, 'wmt')
  for ds_name in ['wmt14_translate/de-en:1.0.0', 'wmt17_translate/de-en:1.0.0']:
    dataset_builder = tfds.builder(ds_name, data_dir=data_dir)
    dataset_builder.download_and_prepare()

    if ds_name == 'wmt17_translate/de-en:1.0.0':
      ds = dataset_builder.as_dataset(split='train', shuffle_files=False)
      ds = ds.map(
          functools.partial(normalize_feature_names, dataset_builder.info),
          num_parallel_calls=tf.data.AUTOTUNE)
      # Tokenize data.
      vocab_path = os.path.join(data_dir, 'wmt_sentencepiece_model')
      tokenizer.train_tokenizer(
          ds, vocab_path=vocab_path, vocab_size=32000, max_corpus_chars=10**7)


def download_finewebedu(data_dir, tmp_dir):
  """Download FineWebEdu-10B."""

  data_dir = os.path.join(data_dir, 'finewebedu')
  tmp_dir = os.path.join(tmp_dir, 'lm') if tmp_dir is not None \
      else os.path.expanduser("~/.cache/huggingface/datasets")
  _maybe_mkdir(data_dir)
  _maybe_mkdir(tmp_dir)

  # Use local disk instead of NFS for temp storage
  os.environ["TMPDIR"] = tmp_dir

  ds = hf_datasets.load_dataset(
    'HuggingFaceFW/fineweb-edu', 
    name='sample-10BT', 
    split='train',
    cache_dir=tmp_dir
  )

  ds = ds.shuffle(seed=1996)  # shuffle so that multiproc has shards of similar size

  seq_len = 2048
  max_seq_length = seq_len+1
  map_setup = dict(batched=True, batch_size=1024, num_proc=8)

  # Tokenize
  tokenizer = AutoTokenizer.from_pretrained('gpt2')
  logging.info(f"Vocab size of tokenizer = {len(tokenizer)}")
  def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
    add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
    return tokenizer(
      add_eos_batched(examples["text"]),
      return_special_tokens_mask=False, 
      return_attention_mask=False
    )
  tokenizer.model_max_length = 1e30  # prevent truncation during tokenization
  tokenized_dataset = ds.map(
    tokenize, 
    remove_columns=['text', 'id', 'dump', 'url', 'file_path', 'language', 
                    'language_score', 'token_count', 'score', 'int_score'],  
    **map_setup
  )
  tokenizer.model_max_length = seq_len
  
  tokenized_dataset.save_to_disk(os.path.join(data_dir, f"fwedu_10B_tokenized"))

  # Concat in chunks of max_seq_len
  # TODO (nico): this might take to much memory
  # TODO (nico): bug fix: Python's shutil.rmtree tried to delete .nfs file, but it was still in use (OSError: [Errno 16] Device or resource busy
  # TODO (nico): make it sequential or increase batch_size in the map_setup
  def concat_chunck(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """Concatenate text and generate chunks of max_seq_length"""
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    result = {
      k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] 
      for k, t in concatenated_examples.items()
    }
    return result
  lm_dataset = tokenized_dataset.map(concat_chunck, **map_setup)
  n_tokens = len(lm_dataset) * max_seq_length
  logging.info(f"Number of tokens in dataset: {n_tokens:_}")

  # Split dataset into training and validation sets
  # TODO (nico): avoid (single doc) contamination, by splitting before concatenation
  VAL_TOKENS = 10_000_000
  val_samples = VAL_TOKENS // max_seq_length + 1
  val_dataset = lm_dataset.select(range(val_samples))
  train_dataset = lm_dataset.select(range(val_samples, len(lm_dataset)))
  logging.info(f"Number of tokens in val_dataset: {len(val_dataset) * max_seq_length :_}")
  logging.info(f"Number of tokens in train_dataset: {len(train_dataset) * max_seq_length :_}")

  # Save datasets
  train_dataset.save_to_disk(os.path.join(data_dir, f"train"))
  val_dataset.save_to_disk(os.path.join(data_dir, f"val"))


def main(_):
  data_dir = FLAGS.data_dir
  tmp_dir = FLAGS.temp_dir
  num_decompression_threads = FLAGS.num_decompression_threads
  bad_chars = [';', ' ', '&', '"']
  if any(s in data_dir for s in bad_chars):
    raise ValueError(f'Invalid data_dir: {data_dir}.')
  if any(s in tmp_dir for s in bad_chars):
    raise ValueError(f'Invalid temp_dir: {tmp_dir}.')
  data_dir = os.path.abspath(os.path.expanduser(data_dir))
  tmp_dir = os.path.abspath(os.path.expanduser(tmp_dir))
  if not FLAGS.skip_download:
    logging.info('Downloading data to %s...', data_dir)

  if FLAGS.all or FLAGS.criteo1tb:
    logging.info('Downloading criteo1tb...')
    download_criteo1tb(data_dir,
                       tmp_dir,
                       num_decompression_threads,
                       FLAGS.interactive_deletion)

  if FLAGS.all or FLAGS.mnist:
    logging.info('Downloading MNIST...')
    download_mnist(data_dir)

  if FLAGS.all or FLAGS.fastmri:
    logging.info('Starting fastMRI download...\n')
    logging.info('Downloading FastMRI...')
    knee_singlecoil_train_url = FLAGS.fastmri_knee_singlecoil_train_url
    knee_singlecoil_val_url = FLAGS.fastmri_knee_singlecoil_val_url
    knee_singlecoil_test_url = FLAGS.fastmri_knee_singlecoil_test_url
    if None in (knee_singlecoil_train_url,
                knee_singlecoil_val_url,
                knee_singlecoil_test_url):
      raise ValueError(
          'Must provide three --fastmri_knee_singlecoil_[train,val,test]_url '
          'to download the FastMRI dataset.\nSign up for the URLs at '
          'https://fastmri.med.nyu.edu/.')

    if not FLAGS.skip_download:
      download_fastmri(data_dir,
                       knee_singlecoil_train_url,
                       knee_singlecoil_val_url,
                       knee_singlecoil_test_url)

    logging.info('fastMRI download completed. Extracting...')
    setup_fastmri(data_dir)

  if FLAGS.all or FLAGS.imagenet:
    flags.mark_flag_as_required('imagenet_train_url')
    flags.mark_flag_as_required('imagenet_val_url')
    imagenet_train_url = FLAGS.imagenet_train_url
    imagenet_val_url = FLAGS.imagenet_val_url
    if imagenet_train_url is None or imagenet_val_url is None:
      raise ValueError(
          'Must provide both --imagenet_{train,val}_url to download the '
          'ImageNet dataset. Sign up for the URLs at https://image-net.org/.')
    if FLAGS.framework is None:
      raise ValueError(
          'Please specify either jax or pytorch framework through framework '
          'flag.')
    if not FLAGS.skip_download:
      logging.info('Downloading ImageNet...')
      download_imagenet(data_dir, imagenet_train_url, imagenet_val_url)
    setup_imagenet(data_dir, framework=FLAGS.framework)

  if FLAGS.all or FLAGS.librispeech:
    logging.info('Downloading Librispeech...')
    download_librispeech(data_dir, tmp_dir)

  if FLAGS.all or FLAGS.cifar:
    logging.info('Downloading CIFAR...')
    download_cifar(data_dir, FLAGS.framework)

  if FLAGS.all or FLAGS.ogbg:
    logging.info('Downloading OGBG...')
    download_ogbg(data_dir)

  if FLAGS.all or FLAGS.wmt:
    logging.info('Downloading WMT...')
    download_wmt(data_dir)

  if FLAGS.all or FLAGS.finewebedu:
    if not FLAGS.skip_download:
      logging.info('Downloading FineWebEdu-10B...')
      download_finewebedu(data_dir)


# pylint: enable=logging-format-interpolation
# pylint: enable=consider-using-with

if __name__ == '__main__':
  app.run(main)
