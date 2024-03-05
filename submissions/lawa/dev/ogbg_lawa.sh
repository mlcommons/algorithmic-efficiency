#!/bin/bash

source ~/.bashrc
conda activate alpe

# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=32
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/is/sg2/najroldi/exp/algoperf
export DATA_DIR=/is/sg2/najroldi/data

# Workload
dataset=ogbg
workload=ogbg

# Job specific vars
submission='submissions/lawa/dev/adamw_dev.py'
search_space='submissions/lawa/dev/space_1.json'
name="lawa_dev_01"
trials=1

# Execute python script
# torchrun \
#   --redirects 1:0,2:0,3:0 \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=4 \
python \
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
  --overwrite
