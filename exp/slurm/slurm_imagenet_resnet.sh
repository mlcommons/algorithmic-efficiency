#!/bin/bash

#SBATCH --job-name=imagenet_resnet_slurm_check
#SBATCH --output=~/log/algoperf/job.%j.%N.out
#SBATCH --error=~/log/algoperf/job.%j.%N.err

#SBATCH --time=10:00:00

#SBATCH --ntasks 1
#SBATCH --requeue
#SBATCH --cpus-per-task 16
#SBATCH --mem=500G

# Get node with GPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

# Get an A100
#SBATCH --constraint=icelake,gpu

LOG_DIR=~/log/algoperf

srun ~/algorithmic-efficiency/exp/shell/imagenet_resnet.sh  > $LOG_DIR/job.%j.%N.log