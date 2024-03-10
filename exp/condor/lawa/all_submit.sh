#!/bin/bash

echo "criteo"
condor_submit_bid 25 exp/condor/lawa/criteo.sub

echo "fastmri"
condor_submit_bid 25 exp/condor/lawa/fastmri.sub

echo "imagenet_resnet"
condor_submit_bid 25 exp/condor/lawa/imagenet_resnet.sub

echo "imagenet_vit"
condor_submit_bid 25 exp/condor/lawa/imagenet_vit.sub

echo "librispeech_conformer"
condor_submit_bid 25 exp/condor/lawa/librispeech_conformer.sub

echo "librispeech_deepspeech"
condor_submit_bid 25 exp/condor/lawa/librispeech_deepspeech.sub

echo "ogbg"
condor_submit_bid 25 exp/condor/lawa/ogbg.sub

echo "wmt"
condor_submit_bid 25 exp/condor/lawa/wmt.sub