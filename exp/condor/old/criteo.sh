#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate alpe

# Env vars
export OMP_NUM_THREADS=48
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf/exp
export DATA_DIR=/fast/najroldi/data

# Workload
dataset=criteo1tb
workload=criteo1tb

# Submission
# submission='prize_qualification_baselines/external_tuning/pytorch_nadamw_full_budget.py'
submission='prize_qualification_baselines/external_tuning/pytorch_nadamw_target_setting.py'

# Search Space
# search_space='prize_qualification_baselines/external_tuning/tuning_search_space.json'
search_space='exp/slurm/nadamw/criteo_search_space.json'

# Experiment name, study
base_name="criteo_tar_aux_dropout"
study=1
num_tuning_trials=1

# Set config
experiment_name="${base_name}/study_${study}"

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
  --num_tuning_trials=$num_tuning_trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$experiment_name \
  --save_intermediate_checkpoints=False \
  --resume_last_run \
  --use_wandb
