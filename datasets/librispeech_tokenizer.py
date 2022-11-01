"""Sentence Piece Tokenizer and ops for tokenizing / de-tokenizing a dataset.

Forked from:
https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/examples/lm1b/tokenizer.py
"""

from absl import flags
from absl import logging
import os
import tempfile
from typing import Dict

import sentencepiece as spm
import tensorflow as tf
import tensorflow_text as tftxt

gfile = tf.io.gfile
copy = tf.io.gfile.copy
exists = tf.io.gfile.exists
rename = tf.io.gfile.rename

Features = Dict[str, tf.Tensor]

flags.DEFINE_string('data_dir', '', 'Path to training data directory.')
flags.DEFINE_boolean(
    'train',
    False,
    'Whether to train a new tokenizer or load existing one to test.')
FLAGS = flags.FLAGS


def dump_chars_for_training(data_folder, splits, maxchars: int = int(1e7)):
  char_count = 0
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/ds_chars') as outfp:
    for split in splits:
      data_folder = data_folder + '/' + split
      for _, speaker_folder in enumerate(os.listdir(data_folder)):
        if char_count > maxchars:
          break

        for chapter_folder in os.listdir(f'{data_folder}/{speaker_folder}'):
          trans_file = (f'{data_folder}/{speaker_folder}/{chapter_folder}/'
                        f'{speaker_folder}-{chapter_folder}.trans.txt')
          if not exists(trans_file):
            logging.info('path does not exist -> %s', trans_file)
            continue
          with open(trans_file, 'r', encoding='UTF-8') as f:
            for l in f:
              _, line = l.strip().split(' ', maxsplit=1)
              line = line + '\n'
              char_count += len(line)
              if char_count > maxchars:
                break

              logging.info(line)
              outfp.write(str.encode(line))
  return outfp


def train_tokenizer(data_dir: str,
                    splits,
                    vocab_size: int = 1024,
                    model_path: str = 'spm_model.vocab',
                    maxchars: int = int(1e7),
                    model_type: str = 'unigram',
                    character_coverage: float = 1.0):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    data_dir: string path to data
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    data_keys: Tuple[str]: keys of dataset to use for training.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  charfile = dump_chars_for_training(data_dir, splits, maxchars=maxchars)

  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/sp_tmp') as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = ' '.join([
      f'--input={charfile.name}',
      f'--vocab_size={vocab_size}',
      f'--character_coverage={character_coverage}',
      f'--model_prefix={model_fp.name}',
      f'--model_type={model_type}'
  ])
  spm.SentencePieceTrainer.Train(argstr)

  copy_rename_path = abs_model_path + '.rntmp'
  copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
  rename(copy_rename_path, abs_model_path, overwrite=True)
  logging.info('Copied %s to %s', model_fp.name + '.model', abs_model_path)

  return abs_model_path


def load_tokenizer(model_filepath):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  if not exists(model_filepath):
    logging.info('Tokenizer not found.')

  with gfile.GFile(model_filepath, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=False, add_eos=True, reverse=False)
  return sp_tokenizer


def run(train, data_dir):
  logging.info('Data dir: %s', data_dir)

  if train:
    logging.info('Training...')
    splits = ['train-clean-100']
    train_tokenizer(data_dir, splits)
  else:
    tokenizer = load_tokenizer(os.path.join(data_dir, 'spm_model.vocab'))
    test_input = 'OPEN SOURCE ROCKS'
    tokens = tokenizer.tokenize(test_input)
    detokenized = tokenizer.detokenize(tokens).numpy().decode('utf-8')

    logging.info('Original input = %s', test_input)
    logging.info('Output after after tokenizing and detokenizing = %s',
                 detokenized)

    if detokenized == test_input:
      logging.info('Tokenizer working correctly!')


def main():
  run(FLAGS.train, FLAGS.data_dir)


if __name__ == '__main__':
  main()
