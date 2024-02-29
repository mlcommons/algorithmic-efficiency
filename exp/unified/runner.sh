#!/bin/bash

# TODO: bashrc make it non interactive sourcable
source ~/.bashrc
conda activate alpe

# Workload
dataset=$1
workload=$2

# Submission
submission=$3
search_space=$4

# Experiment name, study
base_name=$5
study=$6
num_tuning_trials=$7

# Set config
experiment_name="${base_name}/study_${study}"

# if dataset==librispeech, define tokenizer path, else ""

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --imagenet_v2_data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$name \
  --resume_last_run \
  --use_wandb
