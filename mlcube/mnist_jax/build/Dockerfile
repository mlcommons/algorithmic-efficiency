
ARG DOCKER_CUDA=11.2.0
ARG UBUNTU_VERSION=20.04

FROM nvidia/cuda:${DOCKER_CUDA}-base-ubuntu${UBUNTU_VERSION}

ARG CUDA=11.2.0
ARG CUDA_APT=11-2
ARG CUDA_MAJOR=11
ARG CUDA_MINOR=2
ARG CUDA_PYPI=112
ARG CUDNN=8
ARG CUDNN_MAJOR=8
ARG CUDNN_MINOR=None

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
 locales
RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'
# Ubuntu 20.04
# Fix timezone issue: https://blog.programster.org/docker-ubuntu-20-04-automate-setting-timezone
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        python3-dev \
        curl
RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install CUDA libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcublas-dev-${CUDA_APT} \
        libcublas-${CUDA_APT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-command-line-tools-${CUDA_APT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libcufft-dev-${CUDA_APT} \
        libcufft-${CUDA_APT} \
        libcurand-dev-${CUDA_APT} \
        libcurand-${CUDA_APT} \
        libcusolver-dev-${CUDA_APT} \
        libcusolver-${CUDA_APT} \
        libcusparse-dev-${CUDA_APT} \
        libcusparse-${CUDA_APT}

RUN apt-get update && apt-get install -y \
    libcudnn${CUDNN_MAJOR}=*cuda${CUDA_MAJOR}.${CUDA_MINOR} \
    libcudnn${CUDNN_MAJOR}-dev=*cuda${CUDA_MAJOR}.${CUDA_MINOR}

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    ssh

RUN git clone https://github.com/mlcommons/algorithmic-efficiency.git
WORKDIR "algorithmic-efficiency"

# Install pip dependencies
ENV PIP_NO_CACHE_DIR=true
RUN pip3 install -e .

# NOTE: JAX don't yet have a "cuda112" version of jaxlib 0.1.69.
ARG JAX_CUDA_PYPI=111
RUN pip3 install -e .[jax-gpu] -f 'https://storage.googleapis.com/jax-releases/jax_releases.html'
RUN pip3 install jaxlib==0.1.69+cuda${JAX_CUDA_PYPI} -f 'https://storage.googleapis.com/jax-releases/jax_releases.html'

# # NOTE: PyTorch don't yet have a "cu112" version of torch 1.8.1.
ARG PYTORCH_CUDA_PYPI=111
RUN pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu${PYTORCH_CUDA_PYPI}
RUN pip install -f https://download.pytorch.org/whl/torch_stable.html torchvision==0.9.1+cu${PYTORCH_CUDA_PYPI}
RUN pip install -f https://download.pytorch.org/whl/torch_stable.html torchaudio===0.8.1

ENTRYPOINT ["python3", "submission_runner.py", "--framework=jax", "--workload=mnist_jax", "--submission_path=workloads/mnist/mnist_jax/submission.py"]


