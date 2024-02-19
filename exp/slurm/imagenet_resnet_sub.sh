#!/bin/bash

#SBATCH --job-name=imagenet_resnet_slurm_01
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err

#SBATCH --time=24:00:00
#SBATCH --ntasks 1
#SBATCH --requeue

#SBATCH --cpus-per-task 16
#SBATCH --mem=500000M

# Get node with GPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu&^gpu-bw"
# the constraint ensures that we are not reserving gpu-bw

srun ~/algorithmic-efficiency/exp/slurm/imagenet_resnet.sh