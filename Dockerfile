FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ARG UID=1000

RUN apt-get update && \
    apt-get install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   sudo \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip

# add user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u $UID ubuntu

# set working directory
WORKDIR /home/ubuntu/algorithmic-efficiency

# setup path
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# copy files
COPY . /home/ubuntu/algorithmic-efficiency

# install python packages
RUN pip3 install --user .
RUN pip3 install --user .[jax-gpu] -f 'https://storage.googleapis.com/jax-releases/jax_releases.html'
RUN pip3 install --user .[pytorch] -f 'https://download.pytorch.org/whl/torch_stable.html'
RUN pip3 install --user .[jax_core_deps]

# bash
CMD ["/bin/bash"]