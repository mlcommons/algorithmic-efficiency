# Overwrite NVIDIA driver versions and reinstall NVIDIA drivers for CUDA 11.7 
sudo chmod 777 /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_VERSION=515.65.01" > /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_DEB="nvidia-driver-local-repo-ubuntu1804-515.65.01_1.0-1_amd64.deb"" >> /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_CUDA_VERSION="11.7.1"" >> /opt/deeplearning/driver-version.sh
sudo echo "export DRIVER_UBUNTU_PKG=nvidia-driver-515" >> /opt/deeplearning/driver-version.sh
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

# Run gcloud credential helper for Google Container Repository
echo "Running gcloud credential helper"
yes | gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull latest algorithmic efficiency image
echo "Pulling docker image"
docker pull us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/base_image:latest