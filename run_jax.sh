#!/bin/bash
#SBATCH --job-name=cifar_jax_multi
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
cp -r submission_runner.py reference_submissions run_jax.sh  workdir_${SLURM_JOB_ID}/
cd  workdir_${SLURM_JOB_ID}

python submission_runner.py \
    --framework=jax \
    --workload=imagenet_resnet \
    --submission_path=reference_submissions/imagenet_resnet/imagenet_jax/submission.py \
    --tuning_search_space=reference_submissions/imagenet_resnet/tuning_search_space.json \
    --num_tuning_trials=1 \
    --data_dir /scratch/ssd002/datasets/imagenet_tf