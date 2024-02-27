#!/bin/bash

source ~/.bashrc
conda activate alpe

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=~/exp/algoperf
export DATA_DIR=~/data

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=48
num_gpu=4

# Workload
dataset=ogbg
workload=ogbg

# Job specific vars
submission='reference_algorithms/paper_baselines/nadamw/pytorch/submission.py'
search_space='reference_algorithms/paper_baselines/nadamw/tuning_search_space.json'
name="resume_debug_04"
trials=5

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$num_gpu \
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
