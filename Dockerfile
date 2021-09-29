FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip

# add user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 ubuntu
USER ubuntu

# set working directory
WORKDIR /home/ubuntu/algorithmic-efficiency

# setup path
ENV PATH="/home/ubuntu/.local/bin:${PATH}" 

# add requirements
COPY ./requirements.txt /tmp/requirements.txt

# install requirements
RUN pip3 install -r /tmp/requirements.txt

# bash
CMD ["/bin/bash"]