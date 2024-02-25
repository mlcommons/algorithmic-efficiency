#!/bin/bash -l

# provvisorio
source ~/.bashrc

# # add conda TODO: make it more portable!
# source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"

# Env vars
export CODE_DIR=~/algorithmic-efficiency
export DATA_DIR=/ptmp/najroldi/data
export EXP_DIR=/ptmp/najroldi/exp/algoperf

# Job specific vars
workload=mnist
dataset=MNIST
submission='reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py'
search_space='reference_algorithms/development_algorithms/mnist/tuning_search_space.json'
trials=1
name="mnist_reume"

# Print GPU infos
# nvidia-smi
num_gpu=2

# Execute python script
# torchrun --redirects 1:0 \
#   --standalone \
#   --nproc_per_node=$num_gpu \
python3 \
  $CODE_DIR/submission_runner.py \
  --workload=$workload \
  --framework=pytorch \
  --tuning_ruleset=external \
  --data_dir=$DATA_DIR/$dataset \
  --submission_path=$submission \
  --tuning_search_space=$search_space \
  --num_tuning_trials=$trials \
  --experiment_dir=$EXP_DIR  \
  --experiment_name=$name \
  --use_wandb \
  --resume_last_run