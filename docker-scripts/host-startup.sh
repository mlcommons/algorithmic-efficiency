#! /bin/bash

LOGIN_USER=smedapati
STARTUP_SUCCESS_FILE=/home/$LOGIN_USER/.ran-startup-script

# add user
sudo useradd -m $LOGIN_USER

# no more 'sudo docker' after this
sudo groupadd docker
sudo usermod -aG docker $LOGIN_USER
newgrp docker
echo "Setup Users"

# host machine requires nvidia drivers.
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-7
echo "Installed cuda 11.7 and GPU drivers"

# install docker
sudo apt-get update && sudo apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io
echo "Installed Docker"

# install nvidia docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "Installed nvidia docker"

# make sure docker-credential-gcloud is in PATH
# https://stackoverflow.com/questions/54494386/gcloud-auth-configure-docker-on-gcp-vm-instance-with-ubuntu-not-setup-properly
sudo ln -s /snap/google-cloud-cli/current/bin/docker-credential-gcloud /usr/local/bin
# make gcloud docker's credential helper
gcloud auth configure-docker --quiet
yes | gcloud auth configure-docker us-central1-docker.pkg.dev
echo "Setup gcloud docker artifact repo connection"

# create file which will be checked on next reboot
touch /home/$LOGIN_USER/.ran-startup-script

echo "Pulling and running docker image"
docker pull us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/mlcommons:ogbg
docker run -it --gpus all us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/mlcommons:ogbg

