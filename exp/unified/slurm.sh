#!/bin/bash

#SBATCH --job-name=criteo_best03_tar
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%j.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
# --- 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500000

# Env vars
export HOME=/home/najroldi
export CODE_DIR=/home/najroldi/algorithmic-efficiency
export EXP_DIR=/fast/najroldi/exp/algoperf/exp
export DATA_DIR=/fast/najroldi/data

# Job specific vars
workload=criteo1tb
dataset=criteo1tb
submission='reference_algorithms/paper_baselines/adamw/pytorch/submission.py'
search_space='reference_algorithms/paper_baselines/adamw/tuning_search_space.json'
name="exp_01"
study=1
num_tuning_trials=1

srun "runner.sh" \
  workload \
  dataset \
  submission \
  search_space \
  base_name \
  study \
  num_tuning_trials
