#!/bin/bash

#SBATCH --job-name=imagenet_resnet_slurm_test_01
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err

#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue

# --- 4 GPUs on a full node ---
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=500000

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun ~/algorithmic-efficiency/exp/slurm/imagenet_resnet.sh