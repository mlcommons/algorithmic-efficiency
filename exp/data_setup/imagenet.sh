#!/bin/bash

# activate conda env, export DATA_DIR
source data_setup/set_env.sh

# URLS (private!)
train_url=<train_url> # task 1&2
val_url=<val_url>

# Ececute python command
python3 datasets/dataset_setup.py \
    --data_dir=$DATA_DIR \
    --imagenet \
    --temp_dir=$DATA_DIR/tmp \
    --imagenet_train_url=$train_url \
    --imagenet_val_url=$val_url \
    --framework=pytorch