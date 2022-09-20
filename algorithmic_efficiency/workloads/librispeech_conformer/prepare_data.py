"""Data preprocessing for LibriSpeech.
Modified from https://github.com/lsari/librispeech_100.
"""

import argparse
import multiprocessing.dummy
import os
from os.path import exists
import sys
import threading
import time

import numpy as np
import pandas as pd
from pydub import AudioSegment
import tensorflow as tf
import tensorflow_text as tftxt

gfile = tf.io.gfile
copy = tf.io.gfile.copy
exists = tf.io.gfile.exists
rename = tf.io.gfile.rename
parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', help='path to training data directory', type=str)

parser.add_argument(
    '--tokenizer_vocab_path',
    help='path to sentence piece tokenizer vocab file',
    type=str)

tf.config.set_visible_devices([], 'GPU')

TRANSCRIPTION_MAX_LENGTH = 256
AUDIO_MAX_LENGTH = 320000

# taken from TFDS page for librispeech dataset :
# https://www.tensorflow.org/datasets/catalog/librispeech
librispeech_example_counts = {
    'train-clean-100': 28539,
    'train-clean-360': 104014,
    'train-other-500': 148688,
    'test-clean': 2620,
    'dev-clean': 2703,
    'dev-other': 2864
}


class Counter:
  """A threadsafe counter."""
  lock = threading.Lock()
  value = 0

  def inc(self):
    with self.lock:
      self.value += 1

  def val(self):
    with self.lock:
      return self.value


def report_progress(count, total, start_time):
  """Print a progress bar to stdout."""
  now = time.time()
  size = 50
  filled = int(round(size * count / float(total)))
  percent = round(100. * count / float(total), 1)
  bar = "-" * filled + "." * (size - filled)
  sys.stdout.write("[%s] %d%% (%d of %d) %.2f sample/sec\r" %
                   (bar, percent, count, total, count / (now - start_time)))
  sys.stdout.flush()


def preprocess_data(data_folder, tokenizer, split):
  finished = Counter()
  skipped = Counter()
  start_time = time.time()

  def process(index):
    data_folder, speaker_folder, chapter_folder = index
    utterance_ids = []

    trans_file = (f'{data_folder}/{speaker_folder}/{chapter_folder}/'
                  f'{speaker_folder}-{chapter_folder}.trans.txt')
    if not exists(trans_file):
      skipped.inc()
      return utterance_ids

    with open(trans_file, 'r', encoding='UTF-8') as f:
      for l in f:
        utt, trans = l.strip().split(' ', maxsplit=1)
        audio_path = (
            f'{data_folder}/{speaker_folder}/{chapter_folder}/{utt}.flac')

        if not os.path.isfile(audio_path):
          skipped.inc()
          continue

        if len(trans) > TRANSCRIPTION_MAX_LENGTH:
          skipped.inc()
          continue

        sound = load_audio(audio_path)
        sound = np.array(sound, dtype=np.int64)

        if sound.shape[0] > AUDIO_MAX_LENGTH:
          skipped.inc()
          continue

        targets = tokenizer.tokenize(trans).numpy().astype(np.int32)

        np.save('data/{}/{}_audio.npy'.format(split, utt), sound)
        np.save('data/{}/{}_targets.npy'.format(split, utt), targets)

        finished.inc()
        report_progress(finished.val() + skipped.val(),
                        librispeech_example_counts[split],
                        start_time)

        utterance_ids.append(utt)
    return utterance_ids

  paths = []
  for _, speaker_folder in enumerate(os.listdir(data_folder)):
    for chapter_folder in os.listdir(f'{data_folder}/{speaker_folder}'):
      paths.append((data_folder, speaker_folder, chapter_folder))

  sys.stdout.write('\r')
  pool = multiprocessing.dummy.Pool(32)
  file_trans = pool.map(process, paths)

  file_trans = list(np.concatenate(file_trans).flat)

  end_time = time.time()
  elapsed_time = end_time - start_time

  print(' \n time taken to preprocess split : ',
        split,
        ' = ',
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

  final_count = finished.val() + skipped.val()
  return pd.DataFrame(file_trans, columns=['id']), final_count


def load_audio(audio_path):
  audio_segment = AudioSegment.from_file(audio_path, 'flac')
  audio = np.array(audio_segment.get_array_of_samples(), dtype=np.int64)

  return audio


def load_tokenizer(model_path: str = 'spm_model.vocab',
                   add_bos: bool = False,
                   add_eos: bool = True,
                   reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  if not exists(model_path):
    print('tokenizer not found, please train one ....')

  with gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def main():
  args = parser.parse_args()
  data_dir = args.data_dir
  tokenizer = load_tokenizer(args.tokenizer_vocab_path)

  save_dir = 'data/'
  os.makedirs(save_dir, exist_ok=True)

  # put whatever splits required in this list below
  subset_list = [
      # 'train-clean-100',
      # 'dev-clean',
      # 'dev-other',
      # 'test-clean',
      'train-clean-360',
      'train-other-500',
  ]
  for subset in subset_list:
    print('processing split = ', subset)
    os.makedirs(save_dir + '/' + subset, exist_ok=True)
    example_ids, num_entries = preprocess_data(f'{data_dir}/{subset}', tokenizer, subset)  # pylint: disable=line-too-long

    if num_entries != librispeech_example_counts[subset]:
      raise ValueError('preprocessed dataframe final count not equal to '
                       'expected count: {} vs expected {}'.format(
                           num_entries, librispeech_example_counts[subset]))
    example_ids.to_csv('data/{}.csv'.format(subset))


if __name__ == '__main__':
  main()
