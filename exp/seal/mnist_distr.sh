#!/bin/bash

# add conda TODO: make it more portable!
# source ~/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

# Activate conda environment
conda activate alpe

# Env vars
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=~/exp/algoperf
export DATA_DIR=~/data

# Job specific vars
workload=mnist
trials=1
name="resume_debug_03"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpu=8

# THREADS
export OMP_NUM_THREADS=48

# Execute python script
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
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
    --resume_last_run