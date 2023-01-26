FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ARG DATASET=ogbg
ARG FRAMEWORK=jax
ARG GIT_URL=https://github.com/sourabh2k15/algorithmic-efficiency.git
ARG SUBMISSION_PATH=reference_algorithms/target_setting_algorithms/jax_nesterov.py
ARG TUNING_SEARCH_SPACE=reference_algorithms/target_setting_algorithms/ogbg/tuning_search_space.json
ARG EXPERIMENT_NAME=docker_ogbg
ARG WORKLOAD=ogbg

ENV FRAMEWORK=$FRAMEWORK
ENV TUNING_SEARCH_SPACE=$TUNING_SEARCH_SPACE
ENV EXPERIMENT_NAME=$EXPERIMENT_NAME
ENV SUBMISSION_PATH=$SUBMISSION_PATH
ENV WORKLOAD=$WORKLOAD

COPY ./docker/setup.sh /
COPY ./docker/startup.sh /
COPY ./datasets/dataset_setup.py /

RUN /setup.sh -f $FRAMEWORK -u $GIT_URL -d $DATASET

RUN chmod +x startup.sh
ENTRYPOINT ["sh", "-c", "/startup.sh -f $FRAMEWORK -s $SUBMISSION_PATH -t $TUNING_SEARCH_SPACE -e $EXPERIMENT_NAME -w $WORKLOAD"]
