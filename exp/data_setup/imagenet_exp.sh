#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

train_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
valid_url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar"

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
    --framework=pytorch \
    --interactive_deletion