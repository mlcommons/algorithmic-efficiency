#!/bin/bash

#SBATCH --job-name=wmt_s1
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue
# --- 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500000

source ~/.bashrc
conda activate alpe

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Workload
dataset=wmt
workload=wmt

# Job specific vars
submission='reference_algorithms/paper_baselines/nadamw/pytorch/submission.py'
search_space='reference_algorithms/paper_baselines/nadamw/tuning_search_space.json'
name="nadamw_1/study_1"
trials=5

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
