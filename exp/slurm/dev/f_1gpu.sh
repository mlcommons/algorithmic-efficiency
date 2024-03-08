#!/bin/bash

#SBATCH --job-name=gpu_1
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%x_%j.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%x_%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue

# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

# --- uncomment to use 2 GPUs on a shared node ---
# #SBATCH --gres=gpu:a100:2
# #SBATCH --cpus-per-task=36
# #SBATCH --mem=250000

# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000


source ~/.bashrc
conda activate alpe

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CODE_DIR=~/algorithmic-efficiency
export EXP_DIR=/ptmp/najroldi/exp/algoperf
export DATA_DIR=/ptmp/najroldi/data

# Workload
dataset=fastmri
workload=fastmri

# Submission
submission='prize_qualification_baselines/external_tuning/pytorch_nadamw_target_setting.py'
search_space='prize_qualification_baselines/external_tuning/tuning_search_space.json'

# Experiment name, study
base_name="how_many_GPUS_1"
study=1
num_tuning_trials=1

# Set experient name
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