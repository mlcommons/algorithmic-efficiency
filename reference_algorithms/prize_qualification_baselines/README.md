# Prize Qualification Baselines

This directory contains the baseline(s) that submissions must beat to qualify for prizes, see the [Scoring Section](/COMPETITION_RULES.md#scoring) of the competition rules.

## Externally Tuned Ruleset

### JAX

The prize qualification baseline submissions for JAX are:

- `reference_algorithms/prize_qualification_baselines/external_tuning/jax_nadamw_target_setting.py`
- `feference_algorithms/prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py`

Example command:

```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=<experiment_name> \
    --workload=<workload> \
    --submission_path=reference_algorithms/prize_qualification_baselines/external_tuning/jax_nadamw_target_setting.py \
    --tuning_search_space=reference_algorithms/prize_qualification_baselines/external_tuning/tuning_search_space.json
```

### PyTorch

The prize qualification baseline submissionss for PyTorch are:

- `reference_algorithms/prize_qualification_baselines/external_tuning/pytorch_nadamw_target_setting.py`
- `feference_algorithms/prize_qualification_baselines/external_tuning/pytorch_nadamw_full_budget.py`

Example command:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=t<experiment_name> \
    --workload=<workload>\
    --submission_path=reference_algorithms/prize_qualification_baselines/external_tuning/pytorch_nadamw_target_setting.py \
    --tuning_search_space=reference_algorithms/prize_qualification_baselines/external_tuning/tuning_search_space.json
```

## Self-tuning Ruleset

### JAX

The prize qualification baseline submissions for jax are:

- `reference_algorithms/prize_qualification_baselines/self_tuning/jax_nadamw_target_setting.py`
- `feference_algorithms/prize_qualification_baselines/self_tuning/jax_nadamw_full_budget.py`

Example command:

```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=<experiment_name> \
    --workload=<workload> \
    --submission_path=reference_algorithms/prize_qualification_baselines/self_tuning/jax_nadamw_target_setting.py \
    --tuning_ruleset=self
```

### PyTorch

The prize qualification baseline submissionss for PyTorch are:

- `reference_algorithms/prize_qualification_baselines/self_tuning/pytorch_nadamw_target_setting.py`
- `feference_algorithms/prize_qualification_baselines/self_tuning/pytorch_nadamw_full_budget.py`

Example command:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=t<experiment_name> \
    --workload=<workload>\
    --submission_path=reference_algorithms/prize_qualification_baselines/self_tuning/pytorch_nadamw_target_setting.py \
    --tuning_ruleset=self
```
