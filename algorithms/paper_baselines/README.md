# Baseline Submissions from the "Benchmarking Neural Network Training Algorithms" Paper

This directory contains the baseline submissions for the [external tuning ruleset](../README.md#external-tuning-ruleset) as presented in our paper [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179). They are based on eight different update rules:

- [Adafactor](/reference_algorithms/paper_baselines/adafactor)
- [AdamW](/reference_algorithms/paper_baselines/adamw)
- [LAMB](/reference_algorithms/paper_baselines/lamb)
- [SGD with Momentum](/reference_algorithms/paper_baselines/momentum)
- [NadamW](/reference_algorithms/paper_baselines/nadamw)
- [SGD with Nesterov Momentum](/reference_algorithms/paper_baselines/nesterov)
- [SAM](/reference_algorithms/paper_baselines/sam)
- [Shampoo](/reference_algorithms/paper_baselines/shampoo/)

Each update rule has two different tuning search spaces, one where the first momentum parameter (often denoted $\beta_1$) is tuned and one where it is set to a fixed value.
