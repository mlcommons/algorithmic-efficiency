- Template: This is a template for a new training algorithm submission. It includes all required submission functions as placeholders.
- Target setting algorithms: These runs were used to set the target metric values for the workloads.They are not valid submissions, because they use a different hyperparameter setting per workload. But we include them in order to reproduce how we set the target metric values.
- Paper baselines: These are the baseline submissions for the [external tuning ruleset](../README.md#external-tuning-ruleset) as presented in our paper [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179). They are based on eight different update rules:

  - [Adafactor](/reference_algorithms/paper_baselines/adafactor)
  - [AdamW](/reference_algorithms/paper_baselines/adamw)
  - [LAMB](/reference_algorithms/paper_baselines/lamb)
  - [SGD with Momentum](/reference_algorithms/paper_baselines/momentum)
  - [NadamW](/reference_algorithms/paper_baselines/nadamw)
  - [SGD with Nesterov Momentum](/reference_algorithms/paper_baselines/nesterov)
  - [SAM](/reference_algorithms/paper_baselines/sam)
  - [Shampoo](/reference_algorithms/paper_baselines/shampoo/)

  - Each update rule has two different tuning search spaces, one where the first momentum parameter (often denoted $\beta_1$) is tuned and one where it is set to a fixed value.
  - All paper baselines are implemented in PyTorch and JAX.

- Prize qualification baselines: This directory contains the baseline(s) that submissions had to beat to qualify for prizes of the inaugural competition. For each ruleset there are 2 baselines (`*_target_setting.py` and `*_full_budget.py`).
