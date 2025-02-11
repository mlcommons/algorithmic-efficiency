# Change Log

## algoperf-benchmark-0.1.5 (2024-03-26)

- Finalized variant workload targets.
- Fix in random_utils helper function.
- For conformer PyTorch Dropout layers set `inplace=True`.
- Clear CUDA cache at begining of each trial for PyTorch.

## algoperf-benchmark-0.1.4 (2024-03-26)

Upgrade CUDA version to CUDA 12.1:

- Upgrade CUDA version in Dockerfiles that will be used for scoring.
- Update Jax and PyTorch package version tags to use local CUDA installation.

Add flag for completely disabling checkpointing.

- Note that we will run with checkpointing off at scoring time.

Update Deepspeech and Conformer variant target setting configurations.

- Note that variant targets are not final.

Fixed bug in scoring code to take best trial in a study for external-tuning ruleset.

Added instructions for submission.

Changed default number of workers for PyTorch data loaders to 0. Running with >0 may lead to incorrect eval results see <https://github.com/mlcommons/algorithmic-efficiency/issues/732>.

## algoperf-benchmark-0.1.2 (2024-03-04)

Workload variant additions and fixes:

- Add Deepspeech workload variant
- Fix bugs in Imagenet ResNet, WMT and Criteo1tb variants

Add prize qualification logs for external tuning ruleset.
Note: FastMRI trials with dropout are not yet added due to <https://github.com/mlcommons/algorithmic-efficiency/issues/664>.

Add missing funcitonality to Docker startup script for self_tuning ruleset.
Add self_tuning ruleset option to script that runs all workloads for scoring.

Datasetup fixes.

Fix tests that check training differences in PyTorch and JAX on GPU.

## algoperf-benchmark-0.1.1 (2024-01-19)

Bug fixes to FastMRI metric calculation and targets.

Added workload variants and targets for ogbg, fastmri, librispeech_conformer, imagenet_resnet, imagenet_vit, criteo1tb to be used as held-out workloads.

## algoperf-benchmark-0.1.0 (2023-11-28)

First release of the AlgoPerf: Training algorithms benchmarking code.
