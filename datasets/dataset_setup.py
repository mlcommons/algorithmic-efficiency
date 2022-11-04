r"""MLCommons dataset setup script.

If you already have a copy of a dataset(s), you can skip download it and provide
the path when running your algorithm with submission_runner.py via --data_dir.

Note that in order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives). All of these
files will be downloaded to the provided --temp_dir, and the user can manually
delete these after downloading has finished.

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

python3 datasets/dataset_setup.py --data_dir=/data --temp_dir=/tmp/mlcommons_data
python3 datasets/dataset_setup.py --data_dir=~/data --all=False \
--imagenet=True \
--imagenet_train_url=<train_url> \
--imagenet_val_url=<val_url>\
--framework=jax
"""
import os
import shutil
import subprocess
import tarfile

from absl import app
from absl import flags
from absl import logging
import requests
import tensorflow_datasets as tfds
import tqdm

FRAMEWORKS = ['pytorch', 'jax']

KiB = 2**10
MiB = 2**20
GiB = 2**30
TiB = 2**40
PiB = 2**50

IMAGENET_TRAIN_TAR_FILENAME = 'ILSVRC2012_img_train.tar'
IMAGENET_VAL_TAR_FILENAME = 'ILSVRC2012_img_val.tar'

FASTMRI_TRAIN_TAR_FILENAME = 'knee_singlecoil_train.tar'
FASTMRI_VAL_TAR_FILENAME = 'knee_singlecoil_val.tar'
FASTMRI_TEST_TAR_FILENAME = 'knee_singlecoil_test.tar'

from datasets import librispeech_preprocess
from datasets import librispeech_tokenizer

flags.DEFINE_boolean(
    'all',
    True,
    'Whether or not to download all datasets. If false, can download some '
    'combination of datasets by setting the individual dataset flags below.')
flags.DEFINE_boolean('criteo',
                     False,
                     'If --all=false, whether or not to download Criteo.')
flags.DEFINE_boolean('fastmri',
                     False,
                     'If --all=false, whether or not to download FastMRI.')
flags.DEFINE_boolean('imagenet',
                     False,
                     'If --all=false, whether or not to download Imagenet.')
flags.DEFINE_boolean('librispeech',
                     False,
                     'If --all=false, whether or not to download LibriSpeech.')
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


class _Downloader:

  def __init__(self, url, data_dir):
    self.url = url
    self.data_dir = os.path.expanduser(data_dir)
    self.file_path = os.path.join(data_dir, self.url.split('/')[-1])
    self.response = requests.get(self.url, stream=True)
    self.progress_bar = self.setup_progress_bar()

  def setup_data_dirs(self):
    if not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)

  def setup_progress_bar(self):
    total_size_in_bytes = int(self.response.headers.get('Content-length', 0))
    total_size_in_MiB = total_size_in_bytes / MiB
    progress_bar = tqdm.tqdm(
        total=total_size_in_MiB, unit='MiB', unit_scale=True)
    return progress_bar

  def download(self):
    self.setup_data_dirs()

    if os.path.exists(self.file_path):
      while True:
        overwrite = input("File already exists {}.\n Overwrite? (Y/n)".format(
            self.file_path)).lower()
        if overwrite in ['y', 'n']:
          break
        else:
          print("Invalid response. Try again.")
      if overwrite == 'n':
        logging.info("Skipping download to {}".format(self.file_path))
        return

    with open(self.file_path, "wb") as f:
      for chunk in self.response.iter_content(chunk_size=KiB):
        chunk_size_in_MiB = len(chunk) / MiB
        self.progress_bar.update(chunk_size_in_MiB)
        f.write(chunk)
    self.progress_bar.close()
    if self.progress_bar.total != 0 and self.progress_bar.n != self.progress_bar.total:
      raise Exception(
          "Download corrupted, size {n} MiB from {url} does not match expected size {size} MiB"
          .format(
              url=self.url, n=self.progress_bar.n,
              size=self.progress_bar.total))


def download_criteo(data_dir, tmp_dir, num_decompression_threads):
  criteo_dir = os.path.join(data_dir, 'criteo')
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
    unzip_cmd = (f'pigz -d -c -p{num_decompression_threads} {input_path} > '
                 f'{output_path}')
    command_str = f'{wget_cmd} && {unzip_cmd}'
    logging.info(f'Running Criteo download command:\n{command_str}')
    # Note that shell=True can be dangerous if the user injects code into the
    # --data_dir flag. We do some basic sanitization in main(), but
    # submitters should not let untrusted users run this script on their
    # systems.
    processes.append(subprocess.Popen(command_str, shell=True))
  for p in processes:
    p.communicate()


def download_fastmri(
    data_dir,
    fastmri_train_url,
    fastmri_val_url,
    fastmri_test_url,
):

  data_dir = os.path.join(data_dir, 'fastmri')

  # Download fastmri train dataset
  logging.info(
      'Downloading fastmri train dataset from {}'.format(fastmri_train_url))
  _Downloader(url=fastmri_train_url, data_dir=data_dir).download()

  # Download fastmri val dataset
  logging.info(
      "Downloading fastmri val dataset from {}".format(fastmri_val_url))
  _Downloader(url=fastmri_val_url, data_dir=data_dir).download()

  # Download fastmri test dataset
  logging.info(
      "Downloading fastmri test dataset from {}".format(fastmri_test_url))
  _Downloader(url=fastmri_test_url, data_dir=data_dir).download()


def setup_fastmri(data_dir):
  train_tar_file_path = os.path.join(data_dir, FASTMRI_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, FASTMRI_VAL_TAR_FILENAME)
  test_tar_file_path = os.path.join(data_dir, FASTMRI_TEST_TAR_FILENAME)

  # Make train, val and test subdirectories
  fastmri_data_dir = os.path.join(data_dir, 'fastmri')
  train_data_dir = os.makedirs(os.path.join(fastmri_data_dir, 'train'))
  val_data_dir = os.makedirs(os.path.join(fastmri_data_dir, 'val'))
  test_data_dir = os.makedirs(os.path.join(fastmri_data_dir, 'test'))

  # Unzip tar file into subdirectories
  logging.info("Unzipping {} to {}".format(train_tar_file_path,
                                           fastmri_data_dir))
  extract(train_tar_file_path, train_data_dir)
  logging.info("Unzipping {} to {}".format(val_tar_file_path, fastmri_data_dir))
  extract(val_tar_file_path, val_data_dir)
  logging.info("Unzipping {} to {}".format(val_tar_file_path, fastmri_data_dir))
  extract(test_tar_file_path, test_data_dir)
  logging.info("Set up imagenet dataset for jax framework complete")


def download_imagenet(data_dir, tmp_dir, imagenet_train_url, imagenet_val_url):
  data_dir = os.path.join(data_dir, 'imagenet')

  # Download imagnet train dataset
  logging.info(
      'Downloading imagenet train dataset from {}'.format(imagenet_train_url))
  _Downloader(url=imagenet_train_url, data_dir=data_dir).download()

  # Download imagenet val dataset
  logging.info('Donwloading imagenet validation dataset from {}'.format(
      imagenet_val_url))
  _Downloader(url=imagenet_val_url, data_dir=data_dir).download()

  # Download imagenet test set
  download_imagenet_v2(data_dir)


def setup_imagenet(data_dir, framework=None):
  if framework == 'jax':
    setup_imagenet_jax(data_dir)

  elif framework == 'pytorch':
    setup_imagenet_pytorch(data_dir)

  else:
    raise ValueError("Invalid value for framework: {}".format(framework))


def setup_imagenet_jax(data_dir):
  train_tar_file_path = os.path.join(data_dir, IMAGENET_TRAIN_TAR_FILENAME)
  val_tar_file_path = os.path.join(data_dir, IMAGENET_VAL_TAR_FILENAME)

  # Setup jax dataset dir
  imagenet_jax_data_dir = os.path.join(data_dir, 'jax')
  os.makedirs(imagenet_jax_data_dir)

  # Copy tar file into jax
  logging.info("Copying {} to {}".format(train_tar_file_path,
                                         imagenet_jax_data_dir))
  shutil.copy(train_tar_file_path, imagenet_jax_data_dir)
  logging.info("Copying {} to {}".format(val_tar_file_path,
                                         imagenet_jax_data_dir))
  shutil.copy(val_tar_file_path, imagenet_jax_data_dir)
  logging.info("Set up imagenet dataset for jax framework complete")


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
  logging.info("Copying {} to {}".format(val_tar_file_path,
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
          os.path.join(imagenet_pytorch_data_dir, 'train', dirname))

  # Extract val data
  logging.info('Extracting imagenet val data')
  extract(
      os.path.join(imagenet_pytorch_data_dir, IMAGENET_VAL_TAR_FILENAME),
      os.path.join(imagenet_pytorch_data_dir, 'val'))

  cwd = os.path.join(imagenet_pytorch_data_dir, 'train')
  valprep_command = [
      'wget',
      '-qO-',
      'https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh'
  ]
  valprep_process = subprocess.Popen(
      valprep_command, cwd=cwd, stdout=subprocess.PIPE)
  out = subprocess.check_output(['bash'], cwd=cwd, stdin=valprep_process.stdout)
  logging.info("Set up imagenet dataset for pytorch framework complete")


def extract(source, dest):
  if not os.path.exists(dest):
    os.path.makedirs(dest)

  tar = tarfile.open(source)
  tar.extractall(dest)
  tar.close()


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
      subprocess.Popen(wget_cmd, shell=True)
      subprocess.Popen(f'tar xzvf {split}-{version}.tar.gz', shell=True)

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
    librispeech_dir = os.path.join(dataset_dir, 'criteo')
    librispeech_preprocess.run(
        input_dir=tmp_librispeech_dir,
        output_dir=librispeech_dir,
        tokenizer_vocab_path=tokenizer_vocab_path)


def download_ogbg(data_dir, tmp_dir):
  pass


def download_wmt(data_dir):
  """WMT14 and WMT17 de-en."""
  for ds_name in ['wmt14_translate/de-en:1.0.0', 'wmt17_translate/de-en:1.0.0']:
    tfds.builder(ds_name, data_dir=data_dir).download_and_prepare()


def main(_):
  data_dir = FLAGS.data_dir
  tmp_dir = FLAGS.temp_dir
  num_decompression_threads = FLAGS.num_decompression_threads
  if ';' in data_dir or ' ' in data_dir or '&' in data_dir:
    raise ValueError(f'Invalid data_dir: {data_dir}.')
  data_dir = os.path.abspath(os.path.expanduser(data_dir))
  logging.info('Downloading data to %s...', data_dir)

  if FLAGS.all or FLAGS.criteo:
    logging.info('Downloading criteo...')
    download_criteo(data_dir, tmp_dir, num_decompression_threads)
  if FLAGS.all or FLAGS.fastmri:
    logging.info('Downloading FastMRI...')
    knee_singlecoil_train_url = FLAGS.fastmri_knee_singlecoil_train_url
    knee_singlecoil_val_url = FLAGS.fastmri_knee_singlecoil_val_url
    knee_singlecoil_test_url = FLAGS.fastmri_knee_singlecoil_test_url
    if knee_singlecoil_train_url is None or knee_singlecoil_val_url is None or knee_singlecoil_val_url is None:
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
          'Please specify either jax or pytorch framework through framework flag.'
      )
    download_imagenet(data_dir, tmp_dir, imagenet_train_url, imagenet_val_url)
    setup_imagenet(data_dir, framework=framework)
  if FLAGS.all or FLAGS.librispeech:
    logging.info('Downloading Librispeech...')
    # download_librispeech(data_dir, tmp_dir)
  if FLAGS.all or FLAGS.ogbg:
    logging.info('Downloading OGBG...')
    # download_ogbg(data_dir, tmp_dir)
  if FLAGS.all or FLAGS.wmt:
    logging.info('Downloading WMT...')
    download_wmt(data_dir)


if __name__ == '__main__':
  app.run(main)
