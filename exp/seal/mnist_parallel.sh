#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate alpe

# Env vars
export CODE_DIR=$HOME/algorithmic-efficiency
export EXP_DIR=$HOME/exp/algoperf/exp
export DATA_DIR=$HOME/data
export IMAGENET_DIR=/is/cluster/fast/jpiles/imagenet

# Job specific vars
workload=mnist
trials=1
name="mnist_parallel"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
num_gpu=4

# Execute python script
torchrun --redirects 1:0,2:0,3:0 \
    --standalone \
    --nproc_per_node=$num_gpu \
    $CODE_DIR/submission_runner.py \
    --workload=$workload \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$DATA_DIR/MNIST \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json \
    --num_tuning_trials=$trials \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --use_wandb \
    --overwrite