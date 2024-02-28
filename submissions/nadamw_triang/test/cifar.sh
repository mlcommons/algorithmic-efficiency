#!/bin/bash

source ~/.bashrc
conda activate alpe

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=~/exp/algoperf
export DATA_DIR=~/data

# Workload
workload=cifar
dataset=cifar10

# Job specific vars
submission='submissions/nadamw_triang/test/cifar_triang.py'
search_space='reference_algorithms/development_algorithms/cifar/tuning_search_space.json'
name="test_triagular_cifar"
trials=1

# Execute python script
python3 $CODE_DIR/submission_runner.py \
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
  --overwrite \
  --use_wandb

# --use_wandb \
# --save_checkpoints=False \
# --resume_last_run \
# --rng_seed=1996
# --overwrite \
# --max_global_steps 1000 \
