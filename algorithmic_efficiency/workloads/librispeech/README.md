## 1. Overview

Speech recognition accepts raw audio samples and produces a corresponding text
transcription. We train a CNN LSTM model on LibriSpeech clean 100 hour dataset
and evaluate it on Librispeech test clean dataset. We get Character Error Rate
(CER) 0.0995 after 52760 steps or 52825.76 seconds on a 8 V-100 GPU machine
(1.00 step/second).

## 2. Download and preprocess dataset

```bash
mkdir data_dir
# Make script executable
chmod +x download_data.sh
cd data_dir

# Download and unzip LibriSpeech dataset
../download_data.sh

# [Optional] Delete the tar.gz files to save disk space
# rm *.tar.gz

cd ..
mkdir work_dir
cd work_dir

# Run preprocessing script
python ../prepare_data.py ../data_dir/LibriSpeech/
```

The raw dataset is under `data_dir` and the preprocessed dataset is under
`work_dir`.


## Running workload
Run from root of repository:

### PyTorch
```bash
python algorithmic_efficiency/submission_runner.py --framework=pytorch --workload=librispeech_pytorch --submission_path=baselines/librispeech/librispeech_pytorch/submission.py --tuning_search_space=baselines/librispeech/tuning_search_space.json --data_dir algorithmic_efficiency/workloads/librispeech/work_dir/data/
```

## 3. Reference

https://github.com/lsari/librispeech_100

