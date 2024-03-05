#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

# export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=32
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf
export DATA_DIR=/fast/najroldi/data

# Workload
dataset=MNIST
workload=mnist

# Job specific vars
submission='submissions/lawa/dev/adamw_mnist.py'
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
