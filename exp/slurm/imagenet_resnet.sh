#!/bin/bash

# provvisorio
source ~/.bashrc

# # add conda TODO: make it more portable!
# source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars
export CODE_DIR=~/algorithmic-efficiency
export DATA_DIR=~/data
export EXP_DIR=/ptmp/najroldi/exp/algoperf

# Job specific vars
workload=imagenet_resnet
dataset=imagenet
submission='reference_algorithms/paper_baselines/adamw/pytorch/submission.py'
search_space='reference_algorithms/paper_baselines/adamw/tuning_search_space.json'
trials=1
name="imagenet_resnet_slurm_01"

# GPUs (this should coincide with 'request_gpus' in .sub)
num_gpu=4

# Print GPU infos
nvidia-smi

# Execute python script
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone \
  --nproc_per_node=$num_gpu \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/imagenet/pytorch \
  --imagenet_v2_data_dir=$DATA_DIR/imagenet/pytorch \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$name \
  --use_wandb \
  --overwrite