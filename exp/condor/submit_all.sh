#!/bin/bash

echo "criteo"
condor_submit_bid 25 exp/condor/criteo.sub

echo "wmt"
condor_submit_bid 25 exp/condor/wmt.sub

echo "ogbg"
condor_submit_bid 25 exp/condor/ogbg.sub

echo "librispeech_conformer"
condor_submit_bid 25 exp/condor/librispeech_conformer.sub

# echo "librispeech_deepspeech"
# condor_submit_bid 25 exp/condor/librispeech_deepspeech.sub

# echo "imagenet_resnet"
# condor_submit_bid 25 exp/condor/imagenet_resnet.sub

# echo "imagenet_vit"
# condor_submit_bid 25 exp/condor/imagenet_vit.sub

echo "fastmri"
condor_submit_bid 25 exp/condor/fastmri.sub