# Startup script for Google Click to Deploy Deep Learning VM 
# This script will install the NVIDIA drivers, NVIDIA Container Toolkit and 
# pull docker image base_image:latest from mlcommons-docker-repo (internal access only).

# Overwrite NVIDIA driver versions and reinstall NVIDIA drivers for CUDA 11.7 
sudo chmod 777 /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_VERSION=535.104.05" > /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_DEB="nvidia-driver-local-repo-ubuntu1804-535.104.05_1.0-1_amd64.deb"" >> /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_CUDA_VERSION="11.7.1"" >> /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_PKG=nvidia-driver-535" >> /opt/deeplearning/driver-version.sh
sudo /opt/deeplearning/install-driver.sh

# Install NVIDIA Container Toolkit 
echo "Installing NVIDIA Container Toolkit"
sudo rm /etc/apt/sources.list.d/nvidia-docker.list

if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]
then 
echo "Setting up package repository and GPG key"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi
echo "Installing nvidia-container-toolkit"
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
echo "Configuring Docker daemon to recognize and install NVIDIA Container Runtime"
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Override libcmalloc
sudo apt-get install libtcmalloc-minimal4
sudo export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Run gcloud credential helper for Google Container Repository
echo "Running gcloud credential helper"
yes | gcloud auth configure-docker us-central1-docker.pkg.dev
sleep 30

# Pull latest algorithmic efficiency image
echo "Pulling docker image"
docker pull us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/base_image:latest
