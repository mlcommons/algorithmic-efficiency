#!/bin/bash -l

#SBATCH --job-name=mnist_02
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err

#SBATCH --time=00:10:00
#SBATCH --ntasks 1
#SBATCH --requeue

# --- default case: use a single GPU on a shared node ---
# #SBATCH --gres=gpu:a100:1
# #SBATCH --cpus-per-task=18
# #SBATCH --mem=125000

# --- uncomment to use 2 GPUs on a shared node ---
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=250000

# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000

srun ~/algorithmic-efficiency/exp/slurm/mnist.sh

## FARLO MEGLIO!!!!
# -l
# cpus?
# request a100
# nvidia mps
# --- is this even useful????? ---
# #SBATCH --partition=gpu
