"""Sentence Piece Tokenizer and ops for tokenizing / de-tokenizing a dataset.

forked from
https://github.com/google/flax/blob/b60f7f45b90f8fc42a88b1639c9cc88a40b298d3/examples/lm1b/tokenizer.py
"""

import argparse
import os
import sys
import tempfile
from typing import Dict, Iterable, Tuple

import sentencepiece as spm
import tensorflow as tf
import tensorflow_text as tftxt

gfile = tf.io.gfile
copy = tf.io.gfile.copy
exists = tf.io.gfile.exists
rename = tf.io.gfile.rename

Features = Dict[str, tf.Tensor]
parser = argparse.ArgumentParser()


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


parser.add_argument(
    '--data_dir', help='path to training data directory', type=str)
parser.add_argument(
    '--train',
    help='whether to train a new tokenizer or load existing one to test',
    type=str2bool,
    default=False)


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
            print('path does not exist -> {}'.format(trans_file))
            continue
          with open(trans_file, 'r', encoding='UTF-8') as f:
            for l in f:
              _, line = l.strip().split(' ', maxsplit=1)
              line = line + '\n'
              char_count += len(line)
              if char_count > maxchars:
                break

              print(line)
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
  print('copied %s to %s', model_fp.name + '.model', abs_model_path)

  return abs_model_path


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
  print(args)

  print(args.train)
  print(args.data_dir)

  if args.train:
    print('train mode invoked')
    print('passed in data dir = ', args.data_dir)
    splits = ['train-clean-100']
    train_tokenizer(args.data_dir, splits)
  else:
    tokenizer = load_tokenizer()
    test_input = 'OPEN SOURCE ROCKS'
    tokens = tokenizer.tokenize(test_input)
    detokenized = tokenizer.detokenize(tokens).numpy().decode('utf-8')

    print('original input = ', test_input)
    print('output after after tokenizing and detokenizing = ', detokenized)

    if detokenized == test_input:
      print('tokenizer loaded and tested, works correctly!')


if __name__ == '__main__':
  main()
