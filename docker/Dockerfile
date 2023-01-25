FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

COPY ./docker/setup.sh /
COPY ./docker/startup.sh /
COPY ./datasets/dataset_setup.py /

RUN ["/setup.sh"]

RUN chmod +x startup.sh
ENTRYPOINT ["/startup.sh"]
