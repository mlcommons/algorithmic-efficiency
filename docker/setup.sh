#!/bin/sh

GIT_URL=https://github.com/sourabh2k15/algorithmic-efficiency.git

echo "Setting up machine"
apt-get update
apt-get install -y curl tar
apt-get install -y git python3 pip

echo "Setting up gsutil"
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-413.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-413.0.0-linux-x86_64.tar.gz
yes | ./google-cloud-sdk/install.sh

echo "Setting up directories"
mkdir -p data/criteo
mkdir -p experiment_runs/

echo "Setting up algorithmic_efficiency repo"
git clone $GIT_URL
cd algorithmic-efficiency/

pip install -e '.[pytorch_cpu]'
pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
pip install -e '.[full]'

echo "Setting up data"
cd ..

# ./google-cloud-sdk/bin/gsutil -m cp -r gs://mlcommons-data/criteo/* data/criteo/

# python3 dataset_setup.py --data_dir=~/data --all=False --criteo
python3 dataset_setup.py --data_dir=~/data --all=False --ogbg
# python3 dataset_setup.py --data_dir=~/data --all=False --ogbg
