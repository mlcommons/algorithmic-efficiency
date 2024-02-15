#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

# Check if exactly two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 train_url val_url"
    exit 1
fi

# URLS (private!)
train_url=$1 # task 1&2
val_url=$2

# Imagenet dataset processsing is resource intensive. 
# To avoid potential ResourcExhausted errors increase the maximum number of open file descriptors:
ulimit -n 8192

# Ececute python command
python3 datasets/dataset_setup.py \
    --data_dir=$DATA_DIR \
    --imagenet \
    --temp_dir=$DATA_DIR/tmp \
    --imagenet_train_url=$train_url \
    --imagenet_val_url=$val_url \
    --framework=pytorch