#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

echo "------ $CONDA_DEFAULT_ENV ------"

# Env vars
export CODE_DIR=$HOME/algorithmic-efficiency
export EXP_DIR=$HOME/exp/algoperf/exp
export DATA_DIR=~/data
export CUDA_VISIBLE_DEVICES=0

# Job specific vars
workload=mnist
dataset=MNIST
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
trials=2
name="mnist_check/study_1"

# Print GPU infos
# nvidia-smi

# Execute python script
python3 $CODE_DIR/submission_runner.py \
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
    --overwrite \
    --max_global_steps 50 \
    --save_checkpoints=False