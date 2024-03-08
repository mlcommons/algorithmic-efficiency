#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

export CUDA_VISIBLE_DEVICES=0
# export OMP_NUM_THREADS=32

# Workload
dataset=MNIST
workload=mnist

# Job specific vars
submission='submissions/lawa/dev/adamw_mnist.py'
name="lawa_dev_seed"

# submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
# name="mnist_dev_seed"

search_space='submissions/lawa/dev/space_1.json'
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
  --overwrite \
  --fixed_space \
  --rng_seed=1996 \
  --use_wandb

#resume_last_run