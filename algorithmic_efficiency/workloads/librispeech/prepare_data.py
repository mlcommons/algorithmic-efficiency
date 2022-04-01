"""Data preprocessing for LibriSpeech.

Modified from https://github.com/lsari/librispeech_100.
"""

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


def analyze_transcripts(train_data_dir, ignore_space=False):
  char_labels = set()
  for j, speaker_folder in enumerate(os.listdir(train_data_dir)):
    if j % 10 == 0:
      print(j)
    for chapter_folder in os.listdir(f'{train_data_dir}/{speaker_folder}'):
      trans_file = (f'{train_data_dir}/{speaker_folder}/{chapter_folder}/'
                    f'{speaker_folder}-{chapter_folder}.trans.txt')
      with open(trans_file, 'r', encoding='UTF-8') as f:
        for line in f:
          _, trans = line.strip().split(' ', maxsplit=1)
          if ignore_space:
            char_labels = char_labels.union(set(trans.replace(' ', '')))
          else:
            char_labels = char_labels.union(set(trans))

  output_labels = ['_'] + sorted(list(char_labels))
  output_label_dict_inner = {c: i for i, c in enumerate(output_labels)}
  os.makedirs('data', exist_ok=True)
  with open('data/labels.json', 'w', encoding='UTF-8') as f:
    json.dump(output_label_dict_inner, f, indent=4)

  return output_label_dict_inner


def get_txt(data_folder, labels_dict, ignore_space=False):
  file_trans = []
  for j, speaker_folder in enumerate(os.listdir(data_folder)):
    if j % 20 == 0:
      print(f'{j}th speaker')
    if not speaker_folder.isdigit():
      continue
    for chapter_folder in os.listdir(f'{data_folder}/{speaker_folder}'):
      trans_file = (f'{data_folder}/{speaker_folder}/{chapter_folder}/'
                    f'{speaker_folder}-{chapter_folder}.trans.txt')
      with open(trans_file, 'r', encoding='UTF-8') as f:
        for l in f:
          utt, trans = l.strip().split(' ', maxsplit=1)
          audio_path = (
              f'{data_folder}/{speaker_folder}/{chapter_folder}/{utt}.flac')
          assert os.path.isfile(audio_path)
          if check_characters(trans, labels_dict) and len(trans) > 10:
            if ignore_space:
              trans = trans.replace(' ', '')
            trans_ids = [labels_dict[c] for c in trans]
            file_trans.append(
                [audio_path, trans, trans_ids, f'speaker-{speaker_folder}'])

  return pd.DataFrame(
      file_trans, columns=['file', 'trans', 'trans_ids', 'speaker'])


def load_audio(audio_path):
  sound, sample_rate = librosa.load(audio_path, sr=16000)
  audio_duration = librosa.get_duration(filename=f'{audio_path}')

  if len(sound.shape) > 1:
    if sound.shape[1] == 1:
      sound = sound.squeeze()
    else:
      sound = sound.mean(axis=1)  # multiple channels, average
  return sound, sample_rate, audio_duration


def extract_spect_mvn(audio_path):
  y, sample_rate, audio_duration = load_audio(audio_path)

  n_fft = int(sample_rate * WINDOW_SIZE)
  win_length = n_fft
  hop_length = int(sample_rate * WINDOW_STRIDE)
  # STFT
  d = librosa.stft(
      y,
      n_fft=n_fft,
      hop_length=hop_length,
      win_length=win_length,
      window=WINDOW)
  spect, _ = librosa.magphase(d)
  # S = log(S+1)
  spect = np.log1p(spect)
  # spect = torch.FloatTensor(spect)
  # if self.normalize:
  print(spect.shape)
  mean = np.mean(spect)
  std = np.std(spect)
  spect -= mean
  spect /= std
  return spect.T, audio_duration


def main(data_dir):
  trans_dir = os.getcwd() + 'data'
  save_dir = os.getcwd() + '/data/stft/'
  os.makedirs(trans_dir, exist_ok=True)
  os.makedirs(save_dir, exist_ok=True)

  output_label_dict = analyze_transcripts(f'{data_dir}/train-clean-100')

  subset_list = [
      'dev-clean', 'test-clean', 'dev-other', 'test-other', 'train-clean-100'
  ]
  for subset in subset_list:
    print(subset)
    df = get_txt(f'{data_dir}/{subset}', output_label_dict)
    df.to_csv(f'data/trans_{subset}.csv')

  for subset in subset_list:
    df = pd.read_csv(f'data/trans_{subset}.csv')
    dataset = []
    for _, row in df.iterrows():
      s, duration = extract_spect_mvn(row['file'])
      wave, _ = os.path.splitext(os.path.basename(row['file']))
      feat_name = '{}_{:.3f}_{:.3f}.npy'.format(wave, 0, duration)
      save_path = os.path.join(save_dir, feat_name)
      np.save(save_path, s)

      row['features'] = save_path
      row['duration'] = duration
      row.pop('Unnamed: 0')
      dataset.append(row)

    features_df = pd.DataFrame(dataset)
    features_df.to_csv('data/features_{}.csv'.format(subset))


if __name__ == '__main__':
  main(data_dir=sys.argv[1])
