# Dockerfile for AlgoPerf environment.
# To build Docker image with only Jax GPU installed:
# docker build -t <image_name> --build-arg framework=jax
# To build Docker image with Pytorch GPU installed:
# docker build -t <image_name> --build-arg framework=pytorch

# To build Docker image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Installing machine packages
RUN echo "Setting up machine"
RUN apt-get update
RUN apt-get install -y curl tar
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git python3 pip wget ffmpeg
RUN apt-get install libtcmalloc-minimal4
RUN apt-get install unzip
RUN apt-get install pigz
RUN export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# Install GCP tools
RUN echo "Setting up gsutil"
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-413.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-413.0.0-linux-x86_64.tar.gz
RUN yes | ./google-cloud-sdk/install.sh

# Directory setup for input and output
RUN echo "Setting up directories for data and experiment_runs"
RUN mkdir -p data/
RUN mkdir -p experiment_runs/

# Install Algorithmic efficiency repo
RUN echo "Setting up algorithmic_efficiency repo"
ARG branch="main"
ARG framework="both"
ARG git_url=https://github.com/mlcommons/algorithmic-efficiency.git
RUN git clone $git_url && cd /algorithmic-efficiency 
RUN cd /algorithmic-efficiency && git checkout $branch

RUN if [ "$framework" = "jax" ] ; then \
        echo "Installing Jax GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html' \
        && pip install -e '.[pytorch_cpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'; \
    elif [ "$framework" = "pytorch" ] ; then \
        echo "Installing Pytorch GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_cpu]' \
        && pip install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'; \
    elif [ "$framework" = "both" ] ; then \
        echo "Installing Jax GPU and Pytorch GPU" \
        && cd /algorithmic-efficiency \
        && pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html' \
        && pip install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'; \
    else \
        echo "Invalid build-arg $framework: framework should be either jax, pytorch or both." >&2 \
        && exit 1 ; \
    fi

RUN cd /algorithmic-efficiency && pip install -e '.[full]'

RUN cd /algorithmic-efficiency && pip install -e '.[wandb]'

RUN cd /algorithmic-efficiency && git fetch origin 
RUN cd /algorithmic-efficiency && git pull

# Todo: remove this, this is temporary for developing
COPY scripts/startup.sh /algorithmic-efficiency/docker/scripts/startup.sh
RUN chmod a+x /algorithmic-efficiency/docker/scripts/startup.sh

# Move submissions into algorithmic-efficiency-repo
COPY /home/kasimbeg/algorithmic-efficiency/submissions/submissions_algorithms_v0_5 /algorithmic-efficiency/submissions/

ENTRYPOINT ["bash", "/algorithmic-efficiency/docker/scripts/startup.sh"]
