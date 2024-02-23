#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars

# MPI-IS cluster
# export DATA_DIR=/fast/najroldi/data
# export TMP_DIR=/fast/najroldi/data/tmp

# Workstation
# export DATA_DIR=~/data
# export TMP_DIR=~/data/tmp

# Raven cluster
export DATA_DIR=/ptmp/najroldi/data
export TMP_DIR=/ptmp/najroldi/tmp