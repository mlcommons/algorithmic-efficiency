To run an experiment on TPU, you need to install the dependencies and execute the submission runner

```BASH
sudo apt update
sudo apt install libsndfile-dev -y
python3 -m pip install --upgrade pip
sudo python3 -m pip uninstall -y tf-nightly tb-nightly
python3 -m pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install --upgrade flax optax git+https://github.com/parlance/ctcdecode Levenshtein pandas librosa sndfile tensorflow tensorboard

git clone https://github.com/ClashLuke/algorithmic-efficiency/
cd algorithmic-efficiency
git checkout librispeech-jax
cd workloads/librispeech/librispeech_jax
mkdir data
cd data
bash ../../download_data.sh
cd LibriSpeech
python3 ../../../prepare_data.py . jax

cd ../../../../..
python3 submission_runner.py --workload=librispeech_jax --submission_path=workloads/librispeech/librispeech_jax/submission.py tuning_search_space=baselines/librispeech/tuning_search_space.json --data_dir=`pwd`/workloads/librispeech/librispeech_jax/data/LibriSpeech/data
```
