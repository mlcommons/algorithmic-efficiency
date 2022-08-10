"""Data preprocessing for LibriSpeech.

Modified from https://github.com/lsari/librispeech_100.
"""

import argparse
import os
from os.path import exists

import librosa
import numpy as np
import pandas as pd
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

TRANSCRIPTION_MAX_LENGTH = 256
AUDIO_MAX_LENGTH = 320000


def preprocess_data(data_folder, tokenizer, split):
  file_trans = []
  num_entries = 0
  for j, speaker_folder in enumerate(os.listdir(data_folder)):
    if not speaker_folder.isdigit():
      continue
    for chapter_folder in os.listdir(f'{data_folder}/{speaker_folder}'):
      trans_file = (f'{data_folder}/{speaker_folder}/{chapter_folder}/'
                    f'{speaker_folder}-{chapter_folder}.trans.txt')
      if not exists(trans_file):
        print('path does not exist -> {}'.format(trans_file))
        continue

      with open(trans_file, 'r', encoding='UTF-8') as f:
        for l in f:
          utt, trans = l.strip().split(' ', maxsplit=1)
          audio_path = (
              f'{data_folder}/{speaker_folder}/{chapter_folder}/{utt}.flac')
          assert os.path.isfile(audio_path)
          if len(trans) > TRANSCRIPTION_MAX_LENGTH:
            continue

          sound, _ = librosa.load(audio_path, sr=16000)
          sound = np.array(sound, dtype=np.float32)
          if sound.shape[0] > AUDIO_MAX_LENGTH:
            continue

          targets = tokenizer.tokenize(trans).numpy().astype(np.int32)
          num_entries = num_entries + 1

          print('{}) transcription = {}, audio = {}'.format(utt, len(trans), len(sound)))

          np.save('data/{}/{}_audio.npy'.format(split, utt), sound)
          np.save('data/{}/{}_targets.npy'.format(split, utt), targets)


          if num_entries > 10:
            return pd.DataFrame(file_trans, columns=['id'])
        
          file_trans.append([utt])

    return pd.DataFrame(file_trans, columns=['id'])


def load_audio(audio_path):
  sound, _ = librosa.load(audio_path, sr=16000)
  audio_duration = librosa.get_duration(filename=f'{audio_path}')

  if len(sound.shape) > 1:
    if sound.shape[1] == 1:
      sound = sound.squeeze()
    else:
      sound = sound.mean(axis=1)  # multiple channels, average

  return sound, audio_duration


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

  subset_list = ['train-clean-100',] #'test-clean']
  for subset in subset_list:
    print(subset)
    os.makedirs(save_dir + '/' + subset, exist_ok=True)
    df = preprocess_data(f'{data_dir}/{subset}', tokenizer, subset)
    print(df)
    
    df.to_csv('data/{}3.csv'.format(subset))


if __name__ == '__main__':
  main()
