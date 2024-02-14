#!/bin/bash

# add conda TODO: make it more portable!
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment TODO: should I use source activate alpe instead?
conda activate alpe

# Env vars

# MPI-IS cluster
# export DATA_DIR=/fast/najroldi/data

# Workstation & MPCDF raven cluster
export DATA_DIR=$HOME/data