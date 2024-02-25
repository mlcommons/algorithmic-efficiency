#!/bin/bash

# sbatch exp/slurm/nadamw/criteo.sh
# sbatch exp/slurm/nadamw/fastmri.sh
sbatch exp/slurm/nadamw/imagenet_resnet.sh
sbatch exp/slurm/nadamw/imagenet_vit.sh
sbatch exp/slurm/nadamw/librispeech_conformer.sh
sbatch exp/slurm/nadamw/librispeech_deepspeech.sh
sbatch exp/slurm/nadamw/ogbg.sh
sbatch exp/slurm/nadamw/wmt.sh