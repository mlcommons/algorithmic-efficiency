#!/bin/bash

#SBATCH --job-name=imagenet_resnet_slurm_check
#SBATCH --output=/u/najroldi/log/algoperf/job_%j.out
#SBATCH --error=/u/najroldi/log/algoperf/job_%j.err

#SBATCH --time=10:00:00
#SBATCH --ntasks 1
#SBATCH --requeue

#SBATCH --cpus-per-task 8
#SBATCH --mem=100M

# Get node with GPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu&^gpu-bw"
# the constraint ensures that we are not reserving gpu-bw

srun ~/algorithmic-efficiency/exp/slurm/prova.sh
