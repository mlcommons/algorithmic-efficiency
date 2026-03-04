# Dockerfile for AlgoPerf environment.
# To build Docker image with both Jax and Pytorch GPU:
# docker build -t <image_name> --build-arg framework=both
# To build a minimal CPU-only image (no GPU dependencies):
# docker build -t <image_name> --build-arg framework=minimal --build-arg BASE_IMAGE=ubuntu:20.04

ARG BASE_IMAGE=nvidia/cuda:12.9.0-cudnn-devel-ubuntu20.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies + Google Cloud CLI
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        apt-transport-https \
        gnupg \
        curl \
        wget \
        tar \
        git \
        ffmpeg \
        unzip \
        pigz \
        vim \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        liblzma-dev \
        libbz2-dev \
        libtcmalloc-minimal4; \
    \
    # Add Google Cloud apt repo + install gcloud (includes gsutil)
    mkdir -p /usr/share/keyrings; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/google-cloud.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/google-cloud.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends google-cloud-cli; \
    \
    rm -rf /var/lib/apt/lists/*

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Download and install Python 3.11
RUN set -eux; \
    cd /tmp; \
    wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz; \
    tar -xvzf Python-3.11.10.tgz; \
    cd Python-3.11.10; \
    ./configure --enable-optimizations; \
    make -j"$(nproc)"; \
    make altinstall

# Create symlinks for python and pip (use 'pip' instead of 'pip3')
RUN ln -s /usr/local/bin/python3.11 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.11 /usr/bin/pip

# Directory setup for input and output
RUN mkdir -p data/ experiment_runs/

# Install Algorithmic efficiency repo
RUN pip install --upgrade pip

ARG branch="main"
ARG framework="both"
ARG git_url=https://github.com/mlcommons/algorithmic-efficiency.git

RUN git clone "$git_url" /algorithmic-efficiency \
    && cd /algorithmic-efficiency \
    && git checkout "$branch"

RUN if [ "$framework" = "jax" ] ; then \
        echo "Installing Jax GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[pytorch_cpu, full]' --extra-index-url https://download.pytorch.org/whl/cpu \
        && pip install -e '.[jax_gpu]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
    elif [ "$framework" = "pytorch" ] ; then \
        echo "Installing Pytorch GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_cpu, full]' \
        && pip install -e '.[pytorch_gpu]' -f https://download.pytorch.org/whl/cu121; \
    elif [ "$framework" = "both" ] ; then \
        echo "Installing Jax GPU and Pytorch GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_gpu, full]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        && pip install -e '.[pytorch_gpu]' -f https://download.pytorch.org/whl/cu121; \
    elif [ "$framework" = "minimal" ] ; then \
        echo "Installing CPU-only (minimal)" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_cpu, pytorch_cpu, full]' --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        echo "Invalid build-arg $framework: framework should be jax, pytorch, both, or minimal." >&2 \
        && exit 1 ; \
    fi

RUN cd /algorithmic-efficiency && git fetch origin && git pull

# Todo: remove this, this is temporary for developing
COPY scripts/startup.sh /algorithmic-efficiency/docker/scripts/startup.sh
RUN chmod a+x /algorithmic-efficiency/docker/scripts/startup.sh

ENTRYPOINT ["bash", "/algorithmic-efficiency/docker/scripts/startup.sh"]
