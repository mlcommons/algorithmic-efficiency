#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

# URLS (private!)
train_url=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar # task 1&2
val_url=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Ececute python command
python3 datasets/dataset_setup.py \
    --data_dir=$DATA_DIR \
    --imagenet \
    --temp_dir=$DATA_DIR/tmp \
    --imagenet_train_url=$train_url \
    --imagenet_val_url=$val_url \
    --framework=pytorch