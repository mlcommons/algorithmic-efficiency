#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate alpe

# Env vars
export CODE_DIR=$HOME/algorithmic-efficiency
export EXP_DIR=$HOME/exp/algoperf/exp
export IMAGENET_DIR=/is/cluster/fast/jpiles/imagenet

# Job specific vars
workload=imagenet_resnet
submission='reference_algorithms/paper_baselines/adamw/pytorch/submission.py'
search_space='reference_algorithms/paper_baselines/adamw/tuning_search_space.json'
trials=1
name="imagenet_resnet_seal_01"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
num_gpu=6

# Execute python script
torchrun --redirects 1:0,2:0,3:0,4:0,5:0 \
    --standalone \
    --nproc_per_node=$num_gpu \
    $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$IMAGENET_DIR \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$trials \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --use_wandb \
    --overwrite