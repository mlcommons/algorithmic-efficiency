# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) (additionally tagging whether a change affects the [Code, Docs, Rules, Leaderboard]),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Versioning Policy:**  
AlgoPerf uses a unified versioning scheme: codebase, rules, and leaderboard all share the same `Major.Minor` version. All results produced under the same `Major.Minor` version are comparable. `Patch` versions can be incremented independently for each component to reflect smaller, non-breaking changes to allow some flexibility:

- _Leaderboard_: New submissions or minor fixes to the leaderboard could increment its `Patch` version (e.g., `0.6.0` -> `0.6.1`) as shown in the leaderboard repo.
- _Codebase_: API improvements, bug fixes, or small non-breaking changes in the benchmark code could increment its `Patch` version as reflected in the `algoperf` package version.
- _Documentation/Rules_: Clarifications, typo fixes, or minor updates to the rules/documentation could increment its `Patch` version as shown in the documentation file.

## [0.6.0] - 2025-06-24

Improved and streamlined version of the benchmark which includes important bug fixes, API improvements and benchmark protocol changes following the lessons learned from the first competition.

### Added

- [Code, Rules] Updated API to allow for `prepare_for_eval` function.
- [Docs] Document default dropout values for each workload ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/806)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/786)).
- [Docs] Unified versioning policy section ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/876)).
- [Code] Add the ability to change dropout values during training ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/875)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/753)).

### Changed/Removed

- [Code, Docs] Rename package to `algoperf` ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/833)).
- [Code, Docs] Switch to `ruff` for linting and formatting([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/874)).
- [Code, Rules] Pass `train_state` to `update_params` function ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/790)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/785)).
- [Code, Rules] Reduced number of studies from 5 to 3 ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/876)).
- [Code, Rules] Remove held-out workloads from the benchmark ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/876)).
- [Code] Remove sacrebleu dependency ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/828)).
- [Code] Switch to `pyproject.toml` for package management ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/830)).
- [Code] Update Python version to 3.11 and dependencies accordingly ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/811)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/805)).
- [Rules] Modify the runtime budgets and step hints for each workload ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/838)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/836)).
- [Code] Automatically determine the package version via the latest GitHub tag ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/831)).
- [Code, Docs] Move all algorithms into a dedicated `algorithms` directory ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/876)).

### Fixed

- [Code] Batch norm bug ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/783)/[PR](https://github.com/mlcommons/algorithmic-efficiency/pull/798)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/767)).
- [Code] Fix bug of potentially giving a free evaluation to a submission that goes out of `max_runtime` ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/789)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/719#issuecomment-2328797610)).
- [Code] Fix that models in the self-tuning ruleset will always be initialized with default dropout ([PR](https://github.com/mlcommons/algorithmic-efficiency/pull/873)/[PR](https://github.com/mlcommons/algorithmic-efficiency/pull/875)/[Issue](https://github.com/mlcommons/algorithmic-efficiency/issues/753)).

## [0.5.0] - 2024-03-26

The version of the benchmark used for the first competition.

**Summary:**

- Finalized variant workload targets.
- Fix in random_utils helper function.
- For conformer PyTorch Dropout layers set `inplace=True`.
- Clear CUDA cache at begining of each trial for PyTorch.

**What's changed:**

- update speech variants target setting points by @priyakasimbeg in #727
- set num_workers for librispeech back to 4 by @priyakasimbeg in #736
- [fix] random_utils.py to `_signed_to_unsigned` by @tfaod in #739
- Fix path in helper config for running experiments in bulk. by @priyakasimbeg in #740
- Finalize variants targets by @priyakasimbeg in #738
- Aiming to Fix Conformer OOM by @pomonam in #710
- Lint fixes by @priyakasimbeg in #742
- Add warning for PyTorch data loader num_workers flag. by @priyakasimbeg in #726

## [0.0.4] - 2024-03-26

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

## [0.0.3] - 2024-03-04

Workload variant additions and fixes:

- Add Deepspeech workload variant
- Fix bugs in Imagenet ResNet, WMT and Criteo1tb variants

Add prize qualification logs for external tuning ruleset.
Note: FastMRI trials with dropout are not yet added due to <https://github.com/mlcommons/algorithmic-efficiency/issues/664>.

Add missing funcitonality to Docker startup script for self_tuning ruleset.
Add self_tuning ruleset option to script that runs all workloads for scoring.

Datasetup fixes.

Fix tests that check training differences in PyTorch and JAX on GPU.

## [0.0.2] - 2024-01-19

Bug fixes to FastMRI metric calculation and targets.

Added workload variants and targets for ogbg, fastmri, librispeech_conformer, imagenet_resnet, imagenet_vit, criteo1tb to be used as held-out workloads.

## [0.0.1] - 2023-11-28

First release of the AlgoPerf: Training algorithms benchmarking code.
