#!/bin/sh

while getopts u:d:f: flag
do
    case "${flag}" in
        u) GIT_URL=${OPTARG};;
        d) DATASET=${OPTARG};;
        f) FRAMEWORK=${OPTARG};;
    esac
done

echo "git url : $GIT_URL";
echo "dataset: $DATASET";
echo "framework: $FRAMEWORK";

echo "Setting up machine"
apt-get update
apt-get install -y curl tar
apt-get install -y git python3 pip wget

echo "Setting up gsutil"
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-413.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-413.0.0-linux-x86_64.tar.gz
yes | ./google-cloud-sdk/install.sh

echo "Setting up directories"
mkdir -p data/
mkdir -p experiment_runs/

echo "Setting up algorithmic_efficiency repo"
git clone $GIT_URL
cd algorithmic-efficiency/

if [ "$FRAMEWORK" = "jax" ]; then
    echo "framework = jax, installing jax GPU torch CPU"
    pip install -e '.[pytorch_cpu]'
    pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
else
    echo "framework = torch, installing torch GPU jax CPU"
    pip install -e '.[jax_cpu]'
    pip install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
fi

pip install -e '.[full]'

echo "Setting up data"
cd ..

yes | python3 dataset_setup.py --data_dir=~/data --temp_dir=~/data --all=False --$DATASET