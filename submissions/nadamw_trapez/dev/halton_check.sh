#!/bin/bash

source ~/.bashrc
conda activate alpe

export OMP_NUM_THREADS=8
export CUDA_VISILE_DEVICES=6,7

export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/is/sg2/najroldi/exp
export DATA_DIR=/is/sg2/najroldi/data

# Workload
dataset=MNIST
workload=mnist

# Submission
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='submissions/nadamw_trapez/dev/grid_mnist.json'

# Experiment name, study
base_name="halton_check_DEFAULT_3"
study=1
num_tuning_trials=4

# Set experient name
experiment_name="${base_name}"

# Execute python script
# torchrun \
#   --redirects 1:0,2:0,3:0 \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=4 \
python3 \
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
  --use_wandb \
  --fixed_space=True
