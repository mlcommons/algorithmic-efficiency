# Baseline Submissions from the "Benchmarking Neural Network Training Algorithms" Paper

This directory contains the baseline submissions for the [external tuning ruleset](../README.md#external-tuning-ruleset) as presented in our paper [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179). They are based on eight different update rules:

- [AdamW](./adamw)
- [NadamW](./nadamw)
- [Heavy Ball (SGD with Momentum)](./momentum)
- [Nesterov (SGD with Nesterov Momentum)](./nesterov)
- [LAMB](./lamb)
- [Adafactor](./adafactor)
- [SAM (with Adam)](./sam)
- [Shampoo](./shampoo)

Each update rule has two different tuning search spaces, one where the first momentum parameter (often denoted $\beta_1$) is tuned and one where it is set to a fixed value.
