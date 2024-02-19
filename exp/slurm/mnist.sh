#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars
export CODE_DIR=~/algorithmic-efficiency
export DATA_DIR=~/data
export EXP_DIR=/ptmp/najroldi/exp/algoperf

# Job specific vars
workload=mnist
dataset=mnist
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
trials=1
name="mnist_01"

# GPUs (this should coincide with 'request_gpus' in .sub)
num_gpu=4

# Print GPU infos
nvidia-smi

# Execute python script
python3 \
    $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$DATA_DIR/$dataset \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$trials \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --use_wandb \
    --overwrite