"""Data preprocessing for LibriSpeech.
Modified from https://github.com/lsari/librispeech_100.
"""

import multiprocessing.dummy
import os
from os.path import exists
import sys
import threading
import time

from absl import logging
import numpy as np
import pandas as pd
from pydub import AudioSegment
import tensorflow as tf

from datasets import librispeech_tokenizer

gfile = tf.io.gfile
copy = tf.io.gfile.copy
exists = tf.io.gfile.exists
rename = tf.io.gfile.rename

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
    'dev-other': 2864,
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


def preprocess_data(in_folder, out_folder, tokenizer, split):
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

        np.save('{}/{}/{}_audio.npy'.format(out_folder, split, utt), sound)
        np.save('{}/{}/{}_targets.npy'.format(out_folder, split, utt), targets)

        finished.inc()
        report_progress(finished.val() + skipped.val(),
                        librispeech_example_counts[split],
                        start_time)

        utterance_ids.append(utt)
    return utterance_ids

  paths = []
  for _, speaker_folder in enumerate(os.listdir(in_folder)):
    for chapter_folder in os.listdir(f'{in_folder}/{speaker_folder}'):
      paths.append((in_folder, speaker_folder, chapter_folder))

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


def run(input_dir, output_dir, tokenizer_vocab_path):
  tokenizer = librispeech_tokenizer.load_tokenizer(tokenizer_vocab_path)
  os.makedirs(output_dir, exist_ok=True)

  subset_list = [
      'train-clean-100',
      'train-clean-360',
      'train-other-500',
      'dev-clean',
      'dev-other',
      'test-clean',
      'test-other',
  ]
  for subset in subset_list:
    logging.info('Processing split = %s...', subset)
    in_dir = os.path.join(input_dir, subset)
    out_dir = os.path.join(output_dir, subset)
    os.makedirs(out_dir, exist_ok=True)
    example_ids, num_entries = preprocess_data(
      in_dir, output_dir, tokenizer, subset)

    if num_entries != librispeech_example_counts[subset]:
      raise ValueError('Preprocessed dataframe final count not equal to '
                       'expected count: {} vs expected {}'.format(
                           num_entries, librispeech_example_counts[subset]))
    example_ids.to_csv(os.path.join(output_dir, f'{subset}.csv'))
