# Baselines

This directory contains the our baseline training algorithm for the AlgoPerf benchmark.
It uses NadamW with a linear warmup and cosine decay learning rate schedule.
There is both a [self-tuning version](./baselines/self_tuning) and an [externally tuned version](./baselines/external_tuning) of this algorithm.
The baseline is implemented in both JAX and PyTorch.

For comparison, we also provide the training logs for the JAX baseline runs in the [`baselines/logs`](./baselines/logs) directory for both rulesets.
For benchmark results of the baseline, see our [Leaderboard](https://github.com/mlcommons/submissions_algorithms/tree/main).

## Externally Tuned Ruleset

The baseline submission for **JAX**:

- `algorithms/baselines/external_tuning/jax_nadamw_full_budget.py`

Example command:

```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=<experiment_name> \
    --workload=<workload> \
    --submission_path=algorithms/baselines/external_tuning/jax_nadamw_target_setting.py \
    --tuning_search_space=algorithms/baselines/external_tuning/tuning_search_space.json
```

The baseline submission for **PyTorch**:

- `algorithms/baselines/external_tuning/pytorch_nadamw_full_budget.py`

Example command:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=t<experiment_name> \
    --workload=<workload>\
    --submission_path=algorithms/baselines/external_tuning/pytorch_nadamw_target_setting.py \
    --tuning_search_space=algorithms/baselines/external_tuning/tuning_search_space.json
```

## Self-tuning Ruleset

The baseline submission for **JAX**:

- `algorithms/baselines/self_tuning/jax_nadamw_full_budget.py`

Example command:

```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=<experiment_name> \
    --workload=<workload> \
    --submission_path=algorithms/baselines/self_tuning/jax_nadamw_target_setting.py \
    --tuning_ruleset=self
```

The baseline submission for **PyTorch**:

- `algorithms/baselines/self_tuning/pytorch_nadamw_full_budget.py`

Example command:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=<data_dir> \
    --experiment_dir=<experiment_dir> \
    --experiment_name=t<experiment_name> \
    --workload=<workload>\
    --submission_path=algorithms/baselines/self_tuning/pytorch_nadamw_target_setting.py \
    --tuning_ruleset=self
```
