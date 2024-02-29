#!/bin/bash

#SBATCH --job-name=prova2
#SBATCH --output=/ptmp/najroldi/logs/algoperf/job_%j.out
#SBATCH --error=/ptmp/najroldi/logs/algoperf/job_%j.err

#SBATCH -D ./
#SBATCH --time=00:03:00
#SBATCH --ntasks 1
#SBATCH --requeue

# --- default case: use a single GPU on a shared node ---
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

srun ~/algorithmic-efficiency/exp/slurm/prova2.sh
