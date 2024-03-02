# Change Log

## algoperf-benchmark-0.1.2 (2024-03-02)
Workload variant additions and fixes:
- Add Deepspeech workload variant
- Fix bugs in Imagenet ResNet, WMT and Criteo1tb variants

Add prize qualification logs for external tuning ruleset.
Note: FastMRI trials with dropout are not yet added due to https://github.com/mlcommons/algorithmic-efficiency/issues/664.

Add missing funcitonality to Docker startup script for self_tuning ruleset.
Add self_tuning ruleset option to script that runs all workloads for scoring.

Datasetup fixes.

Fix tests that check training differences in PyTorch and JAX on GPU.

## algoperf-benchmark-0.1.1 (2024-01-19)
Bug fixes to FastMRI metric calculation and targets.

Added workload variants and targets for ogbg, fastmri, librispeech_conformer, imagenet_resnet, imagenet_vit, criteo1tb to be used as held-out workloads.

## algoperf-benchmark-0.1.0 (2023-11-28)

First release of the AlgoPerf: Training algorithms benchmarking code.
