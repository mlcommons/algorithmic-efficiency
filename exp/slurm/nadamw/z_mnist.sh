#!/bin/bash

#SBATCH --job-name=mnist_trials_resume
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --time=00:05:00
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
dataset=MNIST
workload=mnist

# Job specific vars
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
name="mnist_trials_resume"
trials=5

# Execute python script
# torchrun --redirects 1:0 \
#   --standalone \
#   --nproc_per_node=$num_gpu \
python3 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$name \
  --resume_last_run
  
  # --overwrite
  # --use_wandb