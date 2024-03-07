#!/bin/bash

source ~/.bashrc
conda activate alpe

export CUDA_VISILE_DEVICES=7

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/is/sg2/najroldi/exp
export DATA_DIR=/is/sg2/najroldi/data

# Workload
dataset=MNIST
workload=mnist

# Submission
submission='submissions/nadamw_trapez/dev/adam_mnist.py'
search_space='submissions/nadamw_trapez/dev/space_mnist.json'

# Experiment name, study
base_name="trapez_check"
study=1
num_tuning_trials=1

# Set experient name
experiment_name="${base_name}"

# Execute python script
python \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$num_tuning_trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --overwrite \
  --use_wandb
