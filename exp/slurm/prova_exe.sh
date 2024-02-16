#!/bin/bash

# init conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

echo "ECCOCI"

# print gpu infos
nvidia-smi

# execute python script
python -c "print('yo from py')"
