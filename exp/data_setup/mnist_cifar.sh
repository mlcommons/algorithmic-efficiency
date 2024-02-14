#!/bin/bash

# activate conda env, export DATA_DIR
source data_setup/set_env.sh

## MNIST
python3 datasets/dataset_setup.py --data_dir $DATA_DIR --mnist --framework pytorch

## CIFAR
python3 datasets/dataset_setup.py --data_dir $DATA_DIR --cifar --framework pytorch