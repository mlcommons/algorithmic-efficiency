#!/bin/bash

echo "imagenet_resnet"
sbatch exp/slurm/parallel_trials/imagenet_resnet.sh
echo

echo "imagenet_vit"
sbatch exp/slurm/parallel_trials/imagenet_vit.sh
echo

echo "librispeech_conformer"
sbatch exp/slurm/parallel_trials/librispeech_conformer.sh
echo

echo "librispeech_deepspeech"
sbatch exp/slurm/parallel_trials/librispeech_deepspeech.sh
echo

echo "ogbg"
sbatch exp/slurm/parallel_trials/ogbg.sh
echo

echo "wmt"
sbatch exp/slurm/parallel_trials/wmt.sh
echo