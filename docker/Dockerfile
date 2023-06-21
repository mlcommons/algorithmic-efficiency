FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Installing machine packages
RUN echo "Setting up machine"
RUN apt-get update
RUN apt-get install -y curl tar
RUN apt-get install -y git python3 pip wget
RUN apt-get install libtcmalloc-minimal4
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
ARG GIT_URL=https://github.com/mlcommons/algorithmic-efficiency.git
RUN git clone $GIT_URL && cd algorithmic-efficiency 

RUN cd algorithmic-efficiency && pip install -e '.[full]'
RUN cd algorithmic-efficiency && pip install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
RUN cd algorithmic-efficiency && pip install -e '.[pytorch_cpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
RUN cd algorithmic-efficiency && pip install -e '.[wandb]'

RUN cd algorithmic-efficiency && git fetch origin 
RUN cd algorithmic-efficiency && git pull

RUN cd algorithmic-efficiency

ENTRYPOINT ["python3", "-m", "algorithmic-efficiency.submission_runner"]
