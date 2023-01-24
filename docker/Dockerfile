FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

COPY setup.sh /
COPY startup.sh /
COPY download_dataset.py /

RUN ["/setup.sh"]

RUN chmod +x startup.sh
ENTRYPOINT ["/startup.sh"]