#!/bin/bash

#SBATCH --job-name=mnist_01
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err

#SBATCH --time=00:10:00
#SBATCH --ntasks 1
#SBATCH --requeue

#SBATCH --cpus-per-task 4
#SBATCH --mem=1000M

# Get node with GPUs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu&^gpu-bw"
# the last constraint ensures that we are not reserving gpu-bw

srun ~/algorithmic-efficiency/exp/shell/mnist.sh