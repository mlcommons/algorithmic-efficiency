#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

train_url=...
valid_url=...
test_url=...

python datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --fastmri \
    --fastmri_knee_singlecoil_train_url=$train_url \
    --fastmri_knee_singlecoil_val_url=$valid_url \
    --fastmri_knee_singlecoil_test_url=$test_url


# MANUAL DOWNLOAD
# curl -C $train_url --output knee_singlecoil_train.tar.xz
# curl -C $valid_url --output knee_singlecoil_val.tar.xz
# curl -C $test_url --output knee_singlecoil_test.tar.xz
# ...