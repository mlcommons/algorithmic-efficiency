#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf/results
export DATA_DIR=/fast/najroldi/data
# export IMAGENET_DIR=/home/najroldi/prove/condor/sweep # TODO

dataset=mnist
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
trials=1
name="mnist_vhevk_shell_vars"

# Execute python script
python $CODE_DIR/submission_runner.py \
    --workload=mnist \
    --framework=pytorch \
    --tuning_ruleset=external \
    --data_dir=$DATA_DIR/$dataset \
    --submission_path=$submission \
    --tuning_search_space=$search_space \
    --num_tuning_trials=$trials \
    --experiment_dir=$EXP_DIR  \
    --experiment_name=$name \
    --overwrite \
    --use_wandb