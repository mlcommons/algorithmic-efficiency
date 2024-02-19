#!/bin/bash

# activate conda env, export DATA_DIR
source exp/data_setup/set_env.sh

## MNIST
python3 datasets/dataset_setup.py \
    --data_dir=$DATA_DIR \
    --temp_dir=$TMP_DIR \
    --mnist \
    --framework=pytorch \
    --interactive_deletion=False