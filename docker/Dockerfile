FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ARG DATASET=ogbg
ARG FRAMEWORK=jax
ARG GIT_URL=https://github.com/sourabh2k15/algorithmic-efficiency.git

COPY ./docker/setup.sh /
COPY ./docker/startup.sh /
COPY ./datasets/dataset_setup.py /

RUN /setup.sh -f $FRAMEWORK -u $GIT_URL -d $DATASET

RUN chmod +x startup.sh
ENTRYPOINT ["/startup.sh"]
