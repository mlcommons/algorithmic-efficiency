# MLCommons Training Algorithms Dockerfile
#
# docker build https://github.com/mlcommons/algorithmic-efficiency.git

FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

RUN apt-get update

RUN apt-get install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip

# Add user.
RUN useradd -rm -d /home/mlcommons -s /bin/bash -g root -G sudo -u 1001 mlcommons
USER mlcommons

# Set working directory.
WORKDIR /home/mlcommons

# Setup path.
ENV PATH="/home/mlcommons/.local/bin:${PATH}"

# Grab the code.
RUN git clone https://github.com/mlcommons/algorithmic-efficiency.git

# Install python packages.
# We need all of these in the same RUN command so we are in the same dir.
RUN cd /home/mlcommons/algorithmic-efficiency && \
    pip3 install -e ".[pytorch_cpu]" && \
    pip3 install -e ".[jax_gpu]" -f "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" && \
    pip3 install -e ".[full]"

# ENTRYPOINT ["python3" "submission_runner.py"]
CMD ["/bin/bash"]
