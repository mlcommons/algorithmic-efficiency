"""Data preprocessing for LibriSpeech modified from https://github.com/lsari/librispeech_100."""

import json
import os
import sys

import librosa
import numpy as np
import pandas as pd

WINDOW_SIZE = 0.02
WINDOW_STRIDE = 0.01
WINDOW = 'hamming'


def check_characters(string, labels):
  for c in string:
    if c not in labels:
      return False
  return True


def insert_underscore_between_pairs(a_string):
  if len(a_string) <= 1:
    return a_string
  if a_string[0] == a_string[1]:
    return a_string[0] + '_' + insert_underscore_between_pairs(a_string[1:])
  return a_string[0] + insert_underscore_between_pairs(a_string[1:])


def analyze_transcripts(train_data_dir, ignore_space=False):
  char_labels = set()
  for i, speaker_folder in enumerate(os.listdir(train_data_dir)):
    if i % 10 == 0:
      print(i)
    for chapter_folder in os.listdir(f'{train_data_dir}/{speaker_folder}'):
      trans_file = f'{train_data_dir}/{speaker_folder}/{chapter_folder}/{speaker_folder}-{chapter_folder}.trans.txt'
      with open(trans_file, 'r') as f:
        for line in f:
          _, trans = line.strip().split(' ', maxsplit=1)
          if ignore_space:
            char_labels = char_labels.union(set(trans.replace(' ', '')))
          else:
            char_labels = char_labels.union(set(trans))

  output_labels = ['_'] + sorted(list(char_labels))
  output_label_dict = {c: i for i, c in enumerate(output_labels)}
  os.makedirs('data', exist_ok=True)
  with open('data/labels.json', 'w') as f:
    json.dump(output_label_dict, f, indent=4)

  return output_label_dict


def get_txt(data_dir, labels_dict, ignore_space=False, add_additional_blank=False):
  file_trans = []
  for i, speaker_folder in enumerate(os.listdir(data_dir)):
    if i % 20 == 0:
      print(f'{i}th speaker')
    if not speaker_folder.isdigit():
      continue
    for chapter_folder in os.listdir(f'{data_dir}/{speaker_folder}'):
      trans_file = f'{data_dir}/{speaker_folder}/{chapter_folder}/{speaker_folder}-{chapter_folder}.trans.txt'
      with open(trans_file, 'r') as f:
        for l in f:
          utt, trans = l.strip().split(' ', maxsplit=1)
          audio_path = f'{data_dir}/{speaker_folder}/{chapter_folder}/{utt}.flac'
          assert os.path.isfile(audio_path)
          if check_characters(trans, labels_dict) and len(trans) > 10:
            # insert a underscore between pair of identical characters
            if add_additional_blank:
              trans = insert_underscore_between_pairs(trans)
            if ignore_space:
              trans = trans.replace(' ', '')
            trans_ids = [labels_dict[c] for c in trans]
            file_trans.append(
                [audio_path, trans, trans_ids, f'speaker-{speaker_folder}'])

  df = pd.DataFrame(
      file_trans, columns=['file', 'trans', 'trans_ids', 'speaker'])
  return df


def load_audio(audio_path):
  sound, sample_rate = librosa.load(audio_path, sr=16000)
  duration = librosa.get_duration(filename=f'{audio_path}')

  if len(sound.shape) > 1:
    if sound.shape[1] == 1:
      sound = sound.squeeze()
    else:
      sound = sound.mean(axis=1)  # multiple channels, average
  return sound, sample_rate, duration


def extract_spect_mvn(audio_path):
  y, sample_rate, duration = load_audio(audio_path)

  n_fft = int(sample_rate * WINDOW_SIZE)
  win_length = n_fft
  hop_length = int(sample_rate * WINDOW_STRIDE)
  # STFT
  D = librosa.stft(
      y,
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window=WINDOW)
  spect, _ = librosa.magphase(D)
  # S = log(S+1)
  spect = np.log1p(spect)
  # spect = torch.FloatTensor(spect)
  # if self.normalize:
  print(spect.shape)
  mean = np.mean(spect)
  std = np.std(spect)
  spect -= mean
  spect /= std
  return spect.T, duration


if __name__ == '__main__':
  data_dir = sys.argv[1]
  pytorch_or_jax = sys.argv[2]

  if pytorch_or_jax == 'pytorch':
    add_additional_blank = False
  else:
    add_additional_blank = True

  trans_dir = os.getcwd() + '/data'
  save_dir = os.getcwd() + '/data/stft/'
  os.makedirs(trans_dir, exist_ok=True)
  os.makedirs(save_dir, exist_ok=True)

  output_label_dict = analyze_transcripts('{}/train-clean-100'.format(data_dir))

  subset_list = [
      'dev-clean', 'test-clean', 'dev-other', 'test-other', 'train-clean-100'
  ]
  for subset in subset_list:
    print(subset)
    df = get_txt('{}/{}'.format(data_dir, subset), output_label_dict, add_additional_blank=add_additional_blank)
    df.to_csv('data/trans_{}.csv'.format(subset))

  for subset in subset_list:
    df = pd.read_csv('data/trans_{}.csv'.format(subset))
    dataset = []
    for i, row in df.iterrows():
      S, duration = extract_spect_mvn(row['file'])
      wave, extension = os.path.splitext(os.path.basename(row['file']))
      feat_name = '{}_{:.3f}_{:.3f}.npy'.format(wave, 0, duration)
      save_path = os.path.join(save_dir, feat_name)
      np.save(save_path, S)

      row['features'] = save_path
      row['duration'] = duration
      row.pop('Unnamed: 0')
      dataset.append(row)

    features_df = pd.DataFrame(dataset)
    features_df.to_csv('data/features_{}.csv'.format(subset))

