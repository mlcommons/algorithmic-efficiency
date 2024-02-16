#!/bin/bash

# init conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment 
conda activate alpe

echo "ECCOCI"

# print gpu infos
nvidia-smi

# execute python script
python -c "print('yo from py')"
