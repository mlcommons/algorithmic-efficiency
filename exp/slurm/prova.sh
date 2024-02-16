#!/bin/bash

#SBATCH --job-name=imagenet_resnet_slurm_check
#SBATCH --output=/u/najroldi/log/algoperf/job_%j.out
#SBATCH --error=/u/najroldi/log/algoperf/job_%j.err

#SBATCH --time=1:00:00

#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --cpus-per-task 16
#SBATCH --mem=200000M

# Get node with GPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

LOG_DIR=~/log/algoperf
srun ~/algorithmic-efficiency/exp/slurm/prova_exe.sh > $LOG_DIR/job.%j.%N.log