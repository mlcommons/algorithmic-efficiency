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

import os
import shutil
import subprocess
import tarfile

from absl import app
from absl import flags
from absl import logging
import requests
import tensorflow_datasets as tfds
from torchvision.datasets import CIFAR10
import tqdm

IMAGENET_TRAIN_TAR_FILENAME = 'ILSVRC2012_img_train.tar'
IMAGENET_VAL_TAR_FILENAME = 'ILSVRC2012_img_val.tar'

FASTMRI_TRAIN_TAR_FILENAME = 'knee_singlecoil_train.tar'
FASTMRI_VAL_TAR_FILENAME = 'knee_singlecoil_val.tar'
FASTMRI_TEST_TAR_FILENAME = 'knee_singlecoil_test.tar'

from datasets import librispeech_preprocess
from datasets import librispeech_tokenizer

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

flags.DEFINE_boolean('criteo',
                     False,
                     'If --all=false, whether or not to download Criteo.')
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

flags.DEFINE_string('framework', None, 'Can be either jax or pytorch.')
flags.DEFINE_boolean('train_tokenizer', True, 'Train Librispeech tokenizer.')
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


def _download_url(url, data_dir):
  data_dir = os.path.expanduser(data_dir)
  file_path = os.path.join(data_dir, url.split('/')[-1])
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
      logging.info('Skipping download to {}'.format(file_path))
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


def download_criteo(data_dir,
                    tmp_dir,
                    num_decompression_threads,
                    interactive_deletion):
  criteo_dir = os.path.join(data_dir, 'criteo')
  tmp_criteo_dir = os.path.join(tmp_dir, 'criteo')
  _maybe_mkdir(criteo_dir)
  _maybe_mkdir(tmp_criteo_dir)
  processes = []
  gz_paths = []
  # Download and unzip.
  for day in range(24):
    logging.info(f'Downloading Criteo day {day}...')
    wget_cmd = (
        f'wget --no-clobber --directory-prefix="{tmp_criteo_dir}" '
        f'https://sacriteopcail01.z16.web.core.windows.net/day_{day}.gz')
    input_path = os.path.join(tmp_criteo_dir, f'day_{day}.gz')
    gz_paths.append(input_path)
    unzipped_path = os.path.join(criteo_dir, f'day_{day}.csv')
    unzip_cmd = (f'pigz -d -c -p{num_decompression_threads} "{input_path}" > '
                 f'"{unzipped_path}"')
    command_str = f'{wget_cmd} && {unzip_cmd}'
    logging.info(f'Running Criteo download command:\n{command_str}')
    processes.append(subprocess.Popen(command_str, shell=True))
  for p in processes:
    p.communicate()
  _maybe_prompt_for_deletion(gz_paths, interactive_deletion)
  # Split into files with 1M lines each: day_1.csv -> day_1_[0-40].csv.
  for batch in range(6):
    batch_processes = []
    unzipped_paths = []
    for day_offset in range(4):
      day = batch * 4 + day_offset
      unzipped_path = os.path.join(criteo_dir, f'day_{day}.csv')
      unzipped_paths.append(unzipped_path)
      split_path = os.path.join(criteo_dir, f'day_{day}_')
      split_cmd = ('split -a 3 -d -l 1000000 --additional-suffix=.csv '
                   f'"{unzipped_path}" "{split_path}"')
      logging.info(f'Running Criteo split command:\n{split_cmd}')
      batch_processes.append(subprocess.Popen(split_cmd, shell=True))
    for p in batch_processes:
      p.communicate()
    _maybe_prompt_for_deletion(unzipped_paths, interactive_deletion)


def download_cifar(data_dir, framework):
  if framework == 'jax':
    tfds.builder('cifar10:3.0.2', data_dir=data_dir).download_and_prepare()
  elif framework == 'pytorch':
    CIFAR10(root=data_dir, train=True, download=True)
    CIFAR10(root=data_dir, train=False, download=True)
  else:
    raise ValueError('Invalid value for framework: {}'.format(framework))


def download_fastmri(data_dir,
                     fastmri_train_url,
                     fastmri_val_url,
                     fastmri_test_url):

  data_dir = os.path.join(data_dir, 'fastmri')

  # Download fastmri train dataset
  logging.info(
      'Downloading fastmri train dataset from {}'.format(fastmri_train_url))
  _download_url(url=fastmri_train_url, data_dir=data_dir).download()

  # Download fastmri val dataset
  logging.info(
      'Downloading fastmri val dataset from {}'.format(fastmri_val_url))
  _download_url(url=fastmri_val_url, data_dir=data_dir).download()

  # Download fastmri test dataset
  logging.info(
      'Downloading fastmri test dataset from {}'.format(fastmri_test_url))
  _download_url(url=fastmri_test_url, data_dir=data_dir).download()


def extract(source, dest):
  if not os.path.exists(dest):
    os.path.makedirs(dest)

  tar = tarfile.open(source)
  tar.extractall(dest)
  tar.close()


def setup_fastmri(data_dir):
  train_tar_file_path = os.path.join(data_dir, FASTMRI_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, FASTMRI_VAL_TAR_FILENAME)
  test_tar_file_path = os.path.join(data_dir, FASTMRI_TEST_TAR_FILENAME)

  # Make train, val and test subdirectories
  fastmri_data_dir = os.path.join(data_dir, 'fastmri')
  train_data_dir = os.path.join(fastmri_data_dir, 'train')
  os.makedirs(train_data_dir)
  val_data_dir = os.path.join(fastmri_data_dir, 'val')
  os.makedirsval_data_dir()
  test_data_dir = os.path.join(fastmri_data_dir, 'test')
  os.makedirs(test_data_dir)

  # Unzip tar file into subdirectories
  logging.info('Unzipping {} to {}'.format(train_tar_file_path,
                                           fastmri_data_dir))
  extract(train_tar_file_path, train_data_dir)
  logging.info('Unzipping {} to {}'.format(val_tar_file_path, fastmri_data_dir))
  extract(val_tar_file_path, val_data_dir)
  logging.info('Unzipping {} to {}'.format(val_tar_file_path, fastmri_data_dir))
  extract(test_tar_file_path, test_data_dir)
  logging.info('Set up imagenet dataset for jax framework complete')


def download_imagenet(data_dir, imagenet_train_url, imagenet_val_url):
  data_dir = os.path.join(data_dir, 'imagenet')

  # Download imagnet train dataset
  logging.info(
      'Downloading imagenet train dataset from {}'.format(imagenet_train_url))
  _download_url(url=imagenet_train_url, data_dir=data_dir).download()

  # Download imagenet val dataset
  logging.info('Donwloading imagenet validation dataset from {}'.format(
      imagenet_val_url))
  _download_url(url=imagenet_val_url, data_dir=data_dir).download()

  # Download imagenet test set
  download_imagenet_v2(data_dir)


def setup_imagenet(data_dir, framework=None):
  if framework == 'jax':
    setup_imagenet_jax(data_dir)

  elif framework == 'pytorch':
    setup_imagenet_pytorch(data_dir)

  else:
    raise ValueError('Invalid value for framework: {}'.format(framework))


def setup_imagenet_jax(data_dir):
  train_tar_file_path = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)

  # Setup jax dataset dir
  imagenet_jax_data_dir = os.path.join(data_dir, 'jax')
  os.makedirs(imagenet_jax_data_dir)

  # Copy tar file into jax
  logging.info('Copying {} to {}'.format(train_tar_file_path,
                                         imagenet_jax_data_dir))
  shutil.copy(train_tar_file_path, imagenet_jax_data_dir)
  logging.info('Copying {} to {}'.format(val_tar_file_path,
                                         imagenet_jax_data_dir))
  shutil.copy(val_tar_file_path, imagenet_jax_data_dir)
  logging.info('Set up imagenet dataset for jax framework complete')


def setup_imagenet_pytorch(data_dir):
  train_tar_file_path = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)

  # Setup jax dataset dir
  imagenet_pytorch_data_dir = os.path.join(data_dir, 'pytorch')
  os.makedirs(imagenet_pytorch_data_dir)
  os.makedirs(os.path.join(imagenet_pytorch_data_dir, 'train'))
  os.makedirs(os.path.join(imagenet_pytorch_data_dir, 'val'))

  # Copy tar file into pytorch directory
  logging.info('Copying {} to {}'.format(train_tar_file_path,
                                         imagenet_pytorch_data_dir))
  shutil.copy(train_tar_file_path, imagenet_pytorch_data_dir)
  logging.info('Copying {} to {}'.format(val_tar_file_path,
                                         imagenet_pytorch_data_dir))
  shutil.copy(val_tar_file_path, imagenet_pytorch_data_dir)

  # Extract train data\
  logging.info('Extracting imagenet train data')
  extract(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_TRAIN_TAR_FILENAME),
      os.path.join(imagenet_pytorch_data_dir, 'train'))

  train_tar_filenames = os.listdir(
      os.path.join(imagenet_pytorch_data_dir, 'train'))
  for tar_filename in train_tar_filenames:
    if tar_filename.endswith('.tar'):
      dir_name = tar_filename[:-4]
      extract(
          os.path.join(imagenet_pytorch_data_dir, IMAGENET_TRAIN_TAR_FILENAME),
          os.path.join(imagenet_pytorch_data_dir, 'train', dir_name))

  # Extract val data
  logging.info('Extracting imagenet val data')
  extract(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_VAL_TAR_FILENAME),
      os.path.join(imagenet_pytorch_data_dir, 'val'))

  valprep_command = [
      'wget',
      '-qO-',
      ('https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/'
       'valprep.sh'),
  ]
  valprep_process = subprocess.Popen(valprep_command, shell=True)
  valprep_process.communicate()
  logging.info('Set up imagenet dataset for pytorch framework complete')


def download_imagenet_v2(data_dir):
  tfds.builder(
      'imagenet_v2/matched-frequency:3.0.0',
      data_dir=data_dir).download_and_prepare()


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
      subprocess.Popen(wget_cmd, shell=True).communicate()
      subprocess.Popen(
          f'tar xzvf {split}-{version}.tar.gz', shell=True).communicate()

  tars = [
      'raw-metadata.tar.gz',
      'train-clean-100.tar.gz',
      'train-clean-360.tar.gz',
      'train-other-500.tar.gz',
  ]
  for tar_filename in tars:
    wget_cmd = (f'wget --directory-prefix={tmp_librispeech_dir} '
                f'http://www.openslr.org/resources/12/{tar_filename}')
    subprocess.Popen(wget_cmd, shell=True)
    tar_path = os.path.join(tmp_librispeech_dir, tar_filename)
    subprocess.Popen(f'tar xzvf {tar_path}', shell=True)

  if train_tokenizer:
    librispeech_tokenizer.run(train=True, data_dir=tmp_librispeech_dir)

    # Preprocess data.
    tokenizer_vocab_path = os.path.join(tmp_librispeech_dir, 'spm_model.vocab')
    librispeech_dir = os.path.join(dataset_dir, 'librispeech')
    librispeech_preprocess.run(
        input_dir=tmp_librispeech_dir,
        output_dir=librispeech_dir,
        tokenizer_vocab_path=tokenizer_vocab_path)


def download_mnist(data_dir):
  tfds.builder('mnist', data_dir=data_dir).download_and_prepare()


def download_ogbg(data_dir):
  tfds.builder('ogbg_molpcba:0.1.3', data_dir=data_dir).download_and_prepare()


def download_wmt(data_dir):
  """WMT14 and WMT17 de-en."""
  for ds_name in ['wmt14_translate/de-en:1.0.0', 'wmt17_translate/de-en:1.0.0']:
    tfds.builder(ds_name, data_dir=data_dir).download_and_prepare()


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
  logging.info('Downloading data to %s...', data_dir)

  if FLAGS.all or FLAGS.criteo:
    logging.info('Downloading criteo...')
    download_criteo(data_dir,
                    tmp_dir,
                    num_decompression_threads,
                    FLAGS.interactive_deletion)

  if FLAGS.all or FLAGS.mnist:
    logging.info('Downloading MNIST...')
    download_mnist(data_dir)

  if FLAGS.all or FLAGS.fastmri:
    logging.info('Downloading FastMRI...')
    knee_singlecoil_train_url = FLAGS.fastmri_knee_singlecoil_train_url
    knee_singlecoil_val_url = FLAGS.fastmri_knee_singlecoil_val_url
    knee_singlecoil_test_url = FLAGS.fastmri_knee_singlecoil_test_url
    if (knee_singlecoil_train_url is None or knee_singlecoil_val_url is None or
        knee_singlecoil_val_url is None):
      raise ValueError(
          'Must provide both --fastmri_knee_singlecoil_{train,val}_url to '
          'download the FastMRI dataset. Sign up for the URLs at '
          'https://fastmri.med.nyu.edu/.')
    download_fastmri(data_dir,
                     tmp_dir,
                     knee_singlecoil_train_url,
                     knee_singlecoil_val_url,
                     knee_singlecoil_test_url)

  if FLAGS.all or FLAGS.imagenet:
    flags.mark_flag_as_required('imagenet_train_url')
    flags.mark_flag_as_required('imagenet_val_url')
    logging.info('Downloading ImageNet...')
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
    download_imagenet(data_dir, imagenet_train_url, imagenet_val_url)
    setup_imagenet(data_dir, framework=FLAGS.framework)

  if FLAGS.all or FLAGS.librispeech:
    logging.info('Downloading Librispeech...')
    download_librispeech(data_dir, tmp_dir, train_tokenizer=True)

  if FLAGS.all or FLAGS.cifar:
    logging.info('Downloading CIFAR...')
    download_cifar(data_dir, FLAGS.framework)

  if FLAGS.all or FLAGS.ogbg:
    logging.info('Downloading OGBG...')
    download_ogbg(data_dir)

  if FLAGS.all or FLAGS.wmt:
    logging.info('Downloading WMT...')
    download_wmt(data_dir)


# pylint: enable=logging-format-interpolation
# pylint: enable=consider-using-with

if __name__ == '__main__':
  app.run(main)
