#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars
export HOME=/home/najroldi
export DATA_DIR=/fast/najroldi/data

# GPUs (this should coincide with 'request_gpus' in .sub)
export CUDA_VISIBLE_DEVICES=0

# Execute python script
python3 datasets/dataset_setup.py \
    --data_dir $DATA_DIR \
    --temp_dir $DATA_DIR/tmp \
    --criteo1tb 