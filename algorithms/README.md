# Training Algorithms Collection

This directory contains different training algorithms used in the project. They range from templates to (historic and current) baselines and algorithms used for target-setting.
Please also see our [Leaderboard](https://github.com/mlcommons/submissions_algorithms/tree/main) which contains all current and previous submissions.

## Algorithm Groups

The algorithms in this subdirectory are organized into the following groups:

### 🧩 [`template`](./template)

This directory provides a `submission.py` template for creating a new AlgoPerf submission. It includes placeholders for all required submission functions. To create a new algorithm, copy the `template/` directory and modify the placeholder implementations with your own.

The submission functions are:

- [`get_batch_size`](./template/submission.py#L13): Returns the batch size for a given workload.
- [`init_optimizer_state`](./template/submission.py#L32): Initializes the optimizer state.
- [`update_params`](./template/submission.py#L56): Updates the model parameters.
- [`prepare_for_eval`](./template/submission.py#L100): Prepares the model for evaluation.
- [`data_selection`](./template/submission.py#L140): Selects a batch of data from the input queue.

### ⚙️ [`development_algorithms`](./development_algorithms)

This directory contains two example algorithms for the `CIFAR-10` and `MNIST` workloads. While these workloads are not part of the official AlgoPerf benchmark, they serve as valuable tools for development and debugging. You can use them to:

- **Verify your setup:** Quickly train a model on `MNIST` or `CIFAR-10` to confirm that your `algoperf` environment is installed and functioning correctly.
- **Understand the submission format:** The provided examples serve as practical templates for new submissions.
  - The `MNIST` algorithm demonstrates a simple _Adam optimizer with a constant learning rate_.
  - The `CIFAR-10` algorithm implements _SGD with a linear warmup and cosine decay learning rate schedule_.

Both algorithms are available in JAX and PyTorch and include corresponding `tuning_search_space.json` files. The `MNIST` algorithm provides both a discrete and a continuous tuning search space.

### 🗃️ [`archived_paper_baselines`](./archived_paper_baselines)

This subdirectory contains the archived baseline submissions for the [external tuning ruleset](../README.md#external-tuning-ruleset), as presented in our paper, ["Benchmarking Neural Network Training Algorithms"](https://arxiv.org/abs/2306.07179). These baselines cover some of the most popular training algorithms, which is why we tested them in our original paper. They can now be used as templates or starting points for future submissions. These baselines are preserved exactly as they were run for our paper experiments and are not intended to be updated. The paper baselines are based on eight different update rules:

- [AdamW](./archived_paper_baselines/adamw)
- [NadamW](./archived_paper_baselines/nadamw)
- [Heavy Ball (SGD with Momentum)](./archived_paper_baselines/momentum)
- [Nesterov (SGD with Nesterov Momentum)](./archived_paper_baselines/nesterov)
- [LAMB](./archived_paper_baselines/lamb)
- [Adafactor](./archived_paper_baselines/adafactor)
- [SAM (with Adam)](./archived_paper_baselines/sam)
- [Shampoo](./archived_paper_baselines/shampoo)

For each update rule, we provide two distinct tuning search spaces:

1. A space where the first momentum parameter (commonly denoted as $\beta_1$) is tuned.
2. A space where the first momentum parameter is fixed.

All paper baselines are implemented in both PyTorch and JAX.

### 📏 [`baselines`](./baselines)

This directory contains the our baseline training algorithm for the AlgoPerf benchmark.
It uses NadamW with a linear warmup and cosine decay learning rate schedule.
There is both a [self-tuning version](./baselines/self_tuning) and an [externally tuned version](./baselines/external_tuning) of this algorithm.
The baseline is implemented in both JAX and PyTorch.

For comparison, we also provide the training logs for the JAX baseline runs in the [`baselines/logs`](./baselines/logs) directory for both rulesets.
For benchmark results of the baseline, see our [Leaderboard](https://github.com/mlcommons/submissions_algorithms/tree/main).

### 🎯 [`target_setting_algorithms`](./target_setting_algorithms)

These algorithms were used to set the target metric values for the workloads.

The procedure was as follows:

1. For each workload, we ran four standard algorithms: AdamW, NadamW, Nesterov Momentum, and Heavy Ball Momentum.
2. For each workload, we then tuned the hyperparameters of all four algorithms using 200 trials using relatively broad search spaces (see Table 8 in [our benchmark paper](https://arxiv.org/abs/2306.07179)).
3. For each workload, we selected the best performing run (combination of algorithm and hyperparameters) across all runs. This configuration is reproduced here as the "target-setting algorithm" for this workload.
4. To finalize the targets, we retrained the workload 20 times, with different random seeds, using this target-setting algorithm. The final validation targets were the median values over these 20 runs, while the test targets were the worst-case test set performance achieved across those 10 repetitions that hit the validation target.

The specific algorithm used for each workload is listed below, the corresponding `tuning_search_space.json` file provides the specific hyperparameter settings used for each workload:

- **Criteo:** `NAdamW`
- **FastMRI:** `Nesterov`
- **ImageNet-Resnet:** `Heavy-ball Momentum`
- **ImageNet-ViT:** `NAdamW`
- **Librispeech-Conformer:** `NAdamW`
- **Librispeech-Deepspeech:** `NAdamW`
- **OGBG:** `Nesterov`
- **WMT:** `NAdamW`

> [!NOTE]
> These are not valid submissions because they use a different hyperparameter setting per workload. However, they are included to ensure the reproducibility of the target-setting process.

## ✔️ Submission Checker

The `submission_checker.py` script can be used to verify that a submission is valid. It checks that the submission file has all the required functions and that they have the correct signatures. It does not check that the algorithm is correct, only that it is a valid submission.

For more information on the submission format, see the [documentation](/DOCUMENTATION.md).
