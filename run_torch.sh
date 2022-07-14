#!/bin/bash
#SBATCH --job-name=cifar_pytorch_multi
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mincpus=16
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH --partition=p100
#SBATCH --qos=normal
#SBATCH --output=job_%x_%j.out
#SBATCH --error=job_%x_%j.err


. $HOME/envs/mlc_env

mkdir -p workdir_${SLURM_JOB_ID}
cp -r submission_runner.py reference_submissions run_torch.sh  workdir_${SLURM_JOB_ID}/
cd  workdir_${SLURM_JOB_ID}

torchrun --standalone --nnodes=1 --nproc_per_node=4 submission_runner.py \
    --framework=pytorch \
    --workload=imagenet_resnet \
    --submission_path=reference_submissions/imagenet_resnet/imagenet_pytorch/submission.py \
    --tuning_search_space=reference_submissions/imagenet_resnet/tuning_search_space.json \
    --num_tuning_trials=1 \
    --data_dir /scratch/ssd002/datasets/imagenet_pytorch