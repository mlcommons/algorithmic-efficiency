#!/bin/bash

#SBATCH --job-name=criteo1tb_baseline
#SBATCH --error=/ptmp/najroldi/logs/algoperf/err/%j.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/out/%j.out
#SBATCH --time=02:00:00
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
dataset=criteo1tb
workload=criteo1tb

# Submission
submission='prize_qualification_baselines/external_tuning/pytorch_nadamw_target_setting.py'
search_space='exp/slurm/lawa/overhead/baseline.json'

# Experiment name, study
base_name="lawa_overhead_baseline"
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
  --use_wandb \
  --max_global_steps=2000 \
  --rng_seed=1996