#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars
export DATA_DIR=/fast/najroldi/data

python3 datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --wmt
