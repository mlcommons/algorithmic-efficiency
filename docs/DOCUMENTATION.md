# MLCommons™ AlgoPerf: Documentation, Benchmarking Process & FAQs

**Version:** 0.6.0 _(Last updated August 27, 2025)_

> [!IMPORTANT]
>
> **TL;DR:** The MLCommons™ **AlgoPerf: Training Algorithms benchmark is designed to find training algorithms that can train neural networks faster** by rigorously measuring how quickly they reach a specific performance target across a diverse set of deep learning workloads.
> This document provides the technical documentation, benchmarking process, and FAQs for the AlgoPerf benchmark.

## Table of Contents <!-- omit from toc -->

- [Introduction](#introduction)
  - [Motivation](#motivation)
  - [Overview](#overview)
- [Benchmarking Process](#benchmarking-process)
  - [Submissions](#submissions)
    - [Submission API](#submission-api)
    - [Valid Submissions](#valid-submissions)
    - [Runtime Environment and Evaluation](#runtime-environment-and-evaluation)
  - [Tuning Rulesets](#tuning-rulesets)
    - [External Tuning Ruleset](#external-tuning-ruleset)
    - [Self-Tuning Ruleset](#self-tuning-ruleset)
  - [Workloads](#workloads)
    - [Recommended Qualification Set](#recommended-qualification-set)
  - [Scoring](#scoring)
    - [AlgoPerf Benchmark Score via Integrated Performance Profiles](#algoperf-benchmark-score-via-integrated-performance-profiles)
    - [Benchmarking Hardware](#benchmarking-hardware)
    - [Defining Target Performance and `max_runtime`](#defining-target-performance-and-max_runtime)
- [Versioning Policy](#versioning-policy)
  - [Version Freeze](#version-freeze)
- [License and Legal Requirements](#license-and-legal-requirements)
- [FAQs](#faqs)
  - [Setup \& Platform](#setup--platform)
  - [Submitting](#submitting)
  - [Scoring \& Hardware](#scoring--hardware)
- [Disclaimers](#disclaimers)
  - [Shared Data Pipelines between `JAX` and `PyTorch`](#shared-data-pipelines-between-jax-and-pytorch)

## Introduction

### Motivation

Neural networks are powerful models, but they need to be trained to be useful. Training cutting-edge machine learning (ML) models exceeds the compute budgets of many researchers and is a growing cost in industry.
Additionally, when training neural nets, practitioners face many critical yet often opaque decisions: What optimizer to choose? How should its learning rate be tuned? What learning rate schedule should be used? These choices can make or break training, yet the community has lacked a clear, standardized way to identify the state of the art.

To reduce the compute and potentially environmental cost of machine learning models, as well as provide guidance for practitioners, we need more scientifically sound methods for evaluating training speedups due to new algorithms.

Unlike benchmarks focused on hardware or model architecture, AlgoPerf isolates the training algorithm itself, which includes the optimizer, regularization, data selection, and hyperparameters like the learning rate schedule. By standardizing the benchmarking process, AlgoPerf offers a meaningful apples-to-apples comparison of training algorithms.

This document focuses on the **Training Algorithm Track** of the _AlgoPerf benchmark_.

### Overview

The **AlgoPerf: Training Algorithms benchmark** challenges participants to submit training algorithms that accelerate the training of neural networks. The goal is to reach a pre-defined performance target in the shortest possible time ("time-to-result") across a diverse set of workloads. The benchmark is designed to identify general-purpose training algorithms, such as new optimizers, data selection methods, regularization techniques, etc., that provide practical speedups for the broader ML community.

The benchmarking process follows these **key principles**:

- 🎯 **Fixed Target, Model & Hardware:** Submitted training algorithms must train a set of [**fixed models**](#workloads) to a pre-defined validation performance target as fast as possible. All submissions use the same model architecture and are run on the same [**standardized hardware**](#benchmarking-hardware) (currently `8x NVIDIA V100 GPUs`). This isolates the training algorithm's performance and allows a fair apples-to-apples comparison.
- ⏱️ **Time-To-Result:** Submissions are evaluated based on the total wall-clock time required to reach the target, rewarding practical and efficient algorithms.
- 🧠 **Diverse Workloads:** The benchmark includes [**8 diverse deep learning workloads**](#workloads) across domains like image classification, speech recognition, and machine translation. A submission's score is computed by aggregating its performance across all workloads, using [**performance profiles**](#algoperf-benchmark-score-via-integrated-performance-profiles), to ensure general-purpose algorithms.
- 📦 **Fully-Specified Algorithms:** Submissions must be [**complete procedures**](#submission-api) and thus hyperparameter tuning is treated as part of the algorithm. Depending on the [**ruleset**](#tuning-rulesets), submissions may use parallel tuning resources. This ensures that the benchmark measures the _total_ practical cost of a training algorithm and provides practitioners with a complete method, eliminating the guesswork of how to apply it.

To participate, you [**submit a training algorithm**](/README.md#how-to-submit) by implementing a specific set of functions within our API, i.e. the [**submission functions**](#submission-api). All other components, including the model architecture, loss function, and evaluation logic, are fixed. This ensures that any performance gains are directly attributable to your algorithmic innovations.

Submissions can be entered under two distinct rulesets:

1. **External Tuning Ruleset:** This ruleset permits a limited, automated, parallel hyperparameter search for each workload, where the search space is defined by the submitter but must be the same for all workloads. A submission's workload score uses only the fastest tuning trial to reach the target.
2. **Self-Tuning Ruleset:** This ruleset is for hyperparameter-free or fully autonomous algorithms. All workload adaptations or hyperparameter tuning must be performed by the algorithm "on the clock" during a single training run.

A core tenet of the benchmark is to foster the development of broadly applicable methods. Submissions must be able to generalize and are prohibited from using logic or pre-computed solutions specific to any single workload.

## Benchmarking Process

The following sections provide the complete technical specifications of the benchmarking process, starting with what constitutes a [**Submission**](#submissions), followed by the two rulesets handling [**Hyperparameter Tuning**](#tuning-rulesets). The [**Workloads**](#workloads) section outlines the deep learning workloads (i.e. models, datasets, loss functions, etc.) used in the benchmark. Finally, the [**Scoring**](#scoring) section describes the process of computing a submission's final, scalar AlgoPerf score (as well as alternative scoring metrics).

### Submissions

A submission to the _AlgoPerf_ benchmark consists of a `submission.py` file that implements a set of Python functions that define your custom training algorithm. This code will be called by the benchmark harness that manages the overall training and evaluation loop.
The core idea is that a submission replaces specific parts of a standard training pipeline with its own logic to train the _AlgoPerf_ workloads to the target performance as quickly as possible, while adhering to the benchmark's rules.

This section details the functions you must implement (the [**Submission API**](#submission-api)), the most important functions and data provided by the benchmark environment ([**fixed functions**](#fixed-functions)), the [**rules to create a valid submission**](#valid-submissions), as well as the [**runtime environment and evaluation procedure**](#runtime-environment-and-evaluation).

#### Submission API

The submission functions are the [`get_batch_size`](#get_batch_size), [`init_optimizer_state`](#init_optimizer_state), [`update_params`](#update_params), [`prepare_for_eval`](#prepare_for_eval), and [`data_selection`](#data_selection) functions. These functions are the only ones that submitters are allowed to modify.
All other functions are [**fixed functions**](#fixed-functions) and contain among other things the `step_hint`, `_build_input_queue`, `init_model_fn`, `model_fn`, or `loss_fn` functions.
Although a submission might access these fixed functions, e.g., to re-initialize the model after a failed training effort, it is not allowed to modify them.
The trained model will be evaluated in a separate step that does not call any of the submitted code.

> 💡 In principle, submissions are allowed to use the available hardware systems in any data- or model-parallel manner they desire, within the constraints of the submission function APIs. However, in practice, model-parallelism may not be possible with the API. Submitters are allowed to access any framework-specific device information necessary to exploit the hardware.

##### `get_batch_size`

```python
def get_batch_size(workload_name: str) -> int
```

**Purpose:** To specify the training batch size for a given workload.

- This function allows submitters to define a different batch size for each workload to ensure that the training does not run out of memory.
- For example, in advance, submitters can determine, for each workload, the largest batch size that fits into memory of the [benchmarking hardware](#benchmarking-hardware).
- Called once per workload before training begins.

> [!NOTE]
>
> This does not change the _evaluation batch size_ (i.e., the batch size used during the evaluation phase). By design, submitters are not allowed to modify the evaluation batch size, which is set by the benchmarking codebase. However, you can file an issue if you believe that the evaluation batch size of a particular workload is set inappropriately. The working group will review this request and consider adjusting the evaluation batch size in the benchmarking codebase, thus affecting all submitters equally.

##### `init_optimizer_state`

```python
def init_optimizer_state(
    workload: Workload,
    model_params: ParameterContainer,
    model_state: ModelAuxiliaryState,
    hyperparameters: Hyperparameters,
    rng: RandomState
) -> initial_optimizer_state
```

**Purpose:** To initialize the optimizer state, i.e., momentum buffers or defining learning rate schedules.

- It does not involve the [initialization for the model parameters](#fixed-functions), which in this benchmark is considered a fixed function.
- The optimizer state is a dictionary (`Dict[str, Any]`). For a PyTorch submission, any value in this dictionary which is a class instance with internal state has to have a `state_dict()` method implemented to be stored correctly at the training checkpoints.

##### `update_params`

```python
def update_params(
    workload: Workload,
    current_param_container: ParameterContainer,
    current_params_types: ParameterTypeTree,
    model_state: ModelAuxiliaryState,
    hyperparameters: Hyperparameters,
    batch: Dict[str, Tensor],
    loss_type: LossType,
    optimizer_state: OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: RandomState,
    train_state: Optional[Dict[str, Any]] = None
) -> (updated_optimizer_state, updated_params, updated_model_state)
```

**Purpose:** To perform a single training step, i.e., update the model parameters and optimizer state.

- Inside this function, you will typically call the workload's `loss_fn` and `model_fn` to perform a forward and backward pass to get gradients.
  - Uses the `model_fn` of the `workload` in order to decouple the loss from the model so that model outputs (forward passes) can be reused (by storing them in the optimizer state).
- The fixed `init_model_fn` can optionally be called during training, for example, to reinitialize the model after a failed training effort.
- **A call to this function will be considered a step**. The time between a call to this function and the next call to this function will be considered the per-step time.
- A submission can access the elapsed training time and get further information about the evaluation through `train_state`. It may also access the target evaluation metric via the `workload` variable.
- `current_param_container` is the same kind of nested structure as used by `model_fn` which constitutes a nested collection of `float32` arrays, each endowed with information about what kind of parameter that array represents stored in a parallel structure of `current_params_types`.
  - Parameter kind is one of a known list of types, e.g. `{"weights", "biases", "embeddings", "conv_weight", "batch_norm_scale", "batch_norm_bias", ...}`.
- `model_state` holds auxiliary state necessary for some models, such as the current batch norm statistics.
- The loss function will be one of a small set of known possibilities and the update function is allowed to branch on the `loss_type` enum/name.
- The `loss_fn` produces a loss per example and a summed loss (both only for one device), which both can be used.
- Cannot modify the given hyperparameters in a workload-conditional way (please see the [Valid Submissions](#valid-submissions) section). This rule is intended to prohibit circumventing the tuning rules by looking up a pre-tuned optimal set of hyperparameters for each workload. It is not intended to prohibit line searches and other similar techniques.

##### `prepare_for_eval`

```python
def prepare_for_eval(
    workload: Workload,
    current_param_container: ParameterContainer,
    current_params_types: ParameterTypeTree,
    model_state: ModelAuxiliaryState,
    hyperparameters: Hyperparameters,
    loss_type: LossType,
    optimizer_state: OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: RandomState
) -> (updated_optimizer_state, updated_params, updated_model_state)
```

**Purpose:** To prepare the model for evaluation, e.g., for swapping model parameters.

- Arguments are the same as `update_params`, with the only exception of `batch`.
- This function is called when a submission is deemed eligible for an evaluation (see [Evaluation during training](#evaluation-during-training) section).
  - The call to `prepare_for_eval` is timed and its runtime accumulates to the overall submission time.
  - The returned model parameters are evaluated on the validation and test sets, provided that the accumulated submission time does not exceed the maximum runtime after this function call (else the evaluation is skipped and the run is terminated).
- This API supports Polyak averaging and similar methods that implement moving averages of model parameters.
- Allowed to update model state and model parameters.
- Allowed to update state for the optimizer.

##### `data_selection`

```python
def data_selection(
    workload: Workload,
    input_queue: Iterator[Dict[str, Any]],
    optimizer_state: OptimizerState,
    current_param_container: ParameterContainer,
    model_state: ModelAuxiliaryState,
    hyperparameters: Hyperparameters,
    global_step: int,
    rng: RandomState
) -> Dict[str, Tensor]
```

**Purpose:** To select a subset of the training data for the next training step.

- `input_queue` can yield up to the number of elements in the training dataset
- Want to allow for submitters to construct their own data batches from the dataset
- Submissions are allowed to arbitrarily modify the input examples, as long as the modifications are sufficiently generic to be applicable to any workload
- This is only called on the training inputs. **No submitted code will be called during evaluation.**
- Examples of data selection methods include _data echoing_, _curriculum learning_, _bootstrapping_, or _biased sampling_ (based on loss values, so need to store the forward pass in the `optimizer_state`, potentially forward pass of a cheaper proxy model).

##### Fixed Functions

Any function that is not part of the [**Submission API**](#submission-api) (and thus a submission function) is considered a fixed function, which submitters are **not** allowed to modify.
Below, we describe some of the fixed functions to provide a better understanding of the _AlgoPerf_ benchmark API. With the exception of `_build_input_queue`, submitters can call any of these functions (along with any public function in the provided `workload` instance) at any time in their submission functions.

###### Step hint

```python
@property
def step_hint(self) -> int
```

- The `step_hint` function gives the number of global steps the baseline algorithm can perform within the `max_runtime` on a workload. The `step_hint` is therefore dependent on the workload and its (current) `max_runtime`. Note that the baseline algorithms may have reached the target in fewer steps than this, but these were the max number of steps the baseline algorithms used for their learning rate schedules. Submitters can use this to help specify learning rate (or other) schedules.
- The `step_hint` is only a hint. There is no need to use it at all. However, it is often more convenient, e.g. to define the learning rate schedule in terms of the number of steps (instead of the runtime).

###### Data augmentation and preprocessing

```python
def _build_input_queue(
    self,
    data_rng: RandomState,
    split: str,
    data_dir: str,
    global_batch_size: int
) -> Iterator[Dict[str, Tensor]]
```

- The `_build_input_queue` function will be called to produce the iterator over batches that the submitted data selection function consumes. It is responsible for all data reading, shuffling, repeating, preprocessing, and batching. Note that for JAX this should return an iterator over tensors of shape `(num_devices, per_device_batch_size, ...)`, and for PyTorch this should return tensors of shape `(per_device_batch_size, ...)` (assuming PyTorch's [DDP](https://pytorch.org/docs/stable/notes/ddp.html) is used).

###### Model initialization

```python
def init_model_fn(self, rng: RandomState) -> ModelInitState
```

- This function initializes the parameters of the model. While it can be called by the submission (e.g. to restart the model after a failed training effort) it cannot be changed.

###### Forward pass

```python
def model_fn(
    self,
    params: ParameterContainer,
    augmented_and_preprocessed_input_batch: Tensor,
    model_state: ModelAuxiliaryState,
    mode: ForwardPassMode,  # mode \in {train, eval}
    rng: RandomState,
    update_batch_norm: bool,
    dropout_rate: float
) -> (logits_output_batch, new_model_state): Tuple[Tensor, ModelAuxiliaryState]
```

- `params` is whatever the structure is that contains the (`float32`) model parameters. The naming is overloaded due to having to handle the more object-oriented `PyTorch` style and the functional `JAX` style of development. In the `Flax` library (written in `JAX`), this is typically a nested dictionary of `JAX`/`numpy` arrays, but in `PyTorch` this is the `torch.nn.Model`.
- It is possible that `model_parameters` will be endowed with additional information about the kind of each parameter, e.g. "weights" or "bias" or "batch norm", although `model_fn` does not really need that information we might use the same nested structure elsewhere.
- `logits_output_batch` is before the output activation.
- `new_model_state` is for batch norm or similar side effects and will only be updated if `update_batch_norm` is set.
- `dropout_rate` is used in the model forward pass for models that support it. These can be tuned or will default to documented model-specific values (see the [workloads table](#workloads) for the list of defaults). Note that adding additional dropout would be considered changing the model, which is not allowed, but the tuning of dropout in existing dropout layers can be considered a regularizer, so we allow it. There should be at most two dropout rates in a model (if there are more than two we will reuse the same values). The `dropout_rate` can be changed during the training process.

###### Loss function

```python
def loss_fn(
    self,
    # Dense or one-hot labels, or a tuple of (tensor, padding) for speech.
    label_batch: Union[Tuple[Tensor, Tensor], Tensor],
    logits_batch: Union[Tuple[Tensor, Tensor], Tensor],
    mask_batch: Optional[Tensor] = None,
    label_smoothing: float = 0.0
) -> Dict[str, Tensor]  # differentiable
```

- The loss function does **not** include regularization. Instead, regularization can be added by the submissions in the `update_params` function.
- The loss function returns a dict `{'summed': scalar summed loss, 'n_valid_examples': scalar number of valid examples in batch, 'per_example': 1-d array of per-example losses}`.
  - Note that the returned quantities are not synced across devices; this can be done by the user in the `update_params` function.
- Each workload uses one of the following loss functions: {_mean squared error_, _cross-entropy_, _CTC_, or _L1 reconstruction error_}.
  - Your submission must work with all these loss types. We provide the loss type via a workload property in order to let training algorithms depend on the loss function.

#### Valid Submissions

The intention of this benchmark is to identify training algorithm submissions that will be broadly applicable and effective in practical scenarios without customization to the specific [workload](#workloads) (model, dataset, and loss function). Generally useful training algorithms can train models faster and thus require less compute resources, decreasing the cost of machine learning. We want to discourage all submissions that sidestep the purpose of this benchmark. We welcome creative ideas and novel research. Therefore, the API aims to allow a wide variety of submissions. However, in some cases, routines that would be allowed in principle might not be practically feasible to express in the provided framework.

A valid submission must implement **general-purpose training logic** that is expected to work on unseen workloads **without** workload-specific modifications or precomputed lookups.
In order to help clarify which submissions are [allowed](#allowed-submissions) and [disallowed](#disallowed-submissions), we described a few examples below. Two essential questions can help provide a general guideline for whether a submission is allowed or not:

1. What **information** is being used by the submission?
2. What **action** is the submission code taking based on this information?

In general, both parts are needed to decide if a particular piece of code is within the spirit of the rules. For example, it is fine to use the shape information of the model parameters to switch between a low-memory and a high-memory approximation, but it isn't allowed to use this shape as a "fingerprint" to uniquely identify a workload and then use pre-computed hyperparameters for this specific workload. As a rule of thumb, submissions are allowed if it is reasonable to assume that the method will work comparably well on unseen workloads automatically without requiring human engineering labor.

##### Allowed submissions

Submissions are allowed to use the provided model parameter information, e.g. the shapes and types of the layers, if the resulting action works on generic workloads.

<details>
<summary>Examples:</summary>

- Using shape information of the parameters to switch between low-memory and high-memory routines is allowed.
- Using shape information of the parameters to conditionally construct variables to avoid running out of memory, e.g. by approximating larger matrices, is allowed.
- Using the ordering of the parameters to train deeper layers differently, e.g. training them sequentially, is allowed.
- Submissions are allowed to use the layer type to change the update rules, e.g. use a different update rule for all batch normalization layers, or use different sub-routines for each layer type, e.g. compute variances for convolutional layers but not for batch normalization layers.

</details>
<br>

Automatic methods for determining or dynamically setting hyperparameters are allowed if they function on generic workloads.

<details>
<summary>Examples:</summary>

- Submissions are allowed to use automatic procedures for setting hyperparameters, e.g. automated learning rate range tests.
- Inner-loop tuning methods for setting hyperparameters, e.g. line searches, are allowed.
- Changing the batch size dynamically during training.

</details>
<br>

Submissions can also be based on learned training algorithms.

<details>
<summary>Examples:</summary>

- Submissions are allowed to learn the update rule of the training method.
- In the [self-tuning ruleset](#self-tuning-ruleset), submissions could try out a learned list of hyperparameters.

</details>
<br>

Submissions can use additional software dependencies provided they have the intention of supporting new algorithmic and mathematical ideas. The procedure for adding dependencies is described in more detail in the [Software dependencies](#software-dependencies) section.

<details>
<summary>Examples:</summary>

- [`BackPACK`](https://docs.backpack.pt/en/master/index.html) is a `pip` package that hooks into `PyTorch` to extract additional information from the backward pass. An allowed use of `BackPACK` would be to compute batch statistics (e.g. within-batch gradient variances, etc.) to calibrate or auto-tune training algorithms.

</details>

##### Disallowed submissions

Submissions must rely on new algorithmic or mathematical ideas and concepts, and must not use software engineering approaches in order to increase primitive operations in PyTorch, JAX, their dependencies, the operating systems, or the hardware. Submissions may use public APIs in JAX and PyTorch from within the submission function APIs, but may not use APIs to optimize the internals of primitive operations and/or standard dependencies to benefit any submission.

Submissions are not allowed to circumvent the tuning rules by looking up the result of an offline computation that was performed ahead of time.

<details>
<summary>Examples:</summary>

- Submissions are not allowed to look up (pre-trained) model parameters.
- Computing the optimal hyperparameters for every workload offline and having the submission look up those pre-computed values is not allowed. In contrast, finding and hard-coding a single good setting of the hyperparameters that works well across all the workloads simultaneously would be allowed.
- Submissions are not allowed to adjust the hyperparameter search spaces for the external tuning ruleset, such that it differs between the workloads.

</details>
<br>

Submissions may not identify (directly or indirectly) the specific benchmark workload to select special-cased logic or hyperparameters; learned detectors that end up selecting workload-specific behavior are equally disallowed. This would result in highly specific behavior that isn't generally useful. In general, all else being equal, if some submission was written that was extremely effective on a small set of the workloads (and far worse on the rest) and another submission with the opposite performance pattern, we would prefer both submissions to be submitted and tested on **all** workloads.

<details>
<summary>Examples:</summary>

- A hard-coded switching of the update rule based on the workload is not allowed, e.g. using Adam for RNNs and SGD with momentum on CNNs. Although submissions can specialize for certain layer types in generic ways, they should not uniquely identify a model or dataset. In other words, if there are two workloads A and B that both have convolutional layers and fully connected layers the submission shouldn't detect whether it is dealing with A or B specifically and choose Adam for one and SGD with momentum for the other. However, if the updates for all parameters of convolutional layers always used SGD with momentum and the updates for all other layers always used Adam and a workload with both types of layers had mixed updates, that would be fine.
  It is also allowed to make the update rule part of the (external) hyperparameter tuning or determine the optimal update rule during the run, i.e. while "on-the-clock".
- Submissions are not allowed to look up learning rate schedules that are only utilized for specific subsets of the workloads. It is allowed to use one general learning rate schedule, to dynamically adapt the learning rate based on general information such as curvature, or to select the learning rate schedule as part of the (external) hyperparameter tuning.

</details>
<br>

Valid submissions must rely on new algorithmic or mathematical ideas and should not use software engineering approaches to speed up primitive operations in `PyTorch`, `JAX`, their dependencies, the operating system, or the hardware. We recognize that the way a method is implemented will impact its performance in the benchmark. It is generally acceptable to make clever, judicious, and efficient use of public APIs in `JAX` and/or `PyTorch` from within the submission function APIs. It is not acceptable to use these APIs to optimize the internals of primitive operations and standard dependencies in ways that could generally benefit any submission.

<details>
<summary>Examples:</summary>

- Submissions **are allowed** to use `CUDA` streams to schedule operations, e.g., transferring data between CPU and GPU, or among GPUs, while performing other computations.
- Submissions **are not allowed** to use `CUDA` streams or asynchronous operations (e.g., spawning additional threads) to perform additional computations that run during the [untimed evaluations](#evaluation-during-training).
- Submissions **are not allowed** to use faster GPU kernels than other submitters by writing their own, using `TVM`, or using a different version of `cuDNN`/`cuBLAS`.
- Submissions **are not allowed** to skip or reduce system or framework overhead, such as modifying `JAX` to skip internal steps like pytree flattening/unflattening.
- Submissions **are not allowed** to introduce new compiler optimizations, such as modifying `XLA` to perform more or less kernel fusion.

</details>

#### Runtime Environment and Evaluation

##### Evaluation during training

In general, with noisy, non-deterministic training, evaluation frequency can affect training time measurements as more "bites of the apple" potentially allows the training code to exploit instability. We also want to discourage submissions from complicated and unrealistic logic that attempts to guess when training is close to complete and increases the evaluation rate, while not producing a well-sampled training curve at the start of training. Simply allowing submissions complete freedom over evaluation frequency encourages competitors to work to minimize the number of evaluations, which distracts from the primary goal of finding better training algorithms.

Submissions are eligible for an untimed evaluation every `eval_period` seconds. Before proceeding to evaluation, the submission can prepare the model through a call to `prepare_for_eval`, effectively modifying the model parameters/state as well as the optimizer state. Any (optional) additional evaluations performed by the submission code count against the runtime for scoring.
The harness that runs the submission code will attempt to evaluate every `eval_period` seconds by checking between each submission step (call of `update_params`) whether it has been at least `eval_period` seconds since that last evaluation. If so, the submission is given the possibility to prepare for evaluation (through a timed call to `prepare_for_eval`). If the accumulated runtime does not exceed the maximum allowed runtime after the preparation step, the clock is paused, and the submission is evaluated. This means that if calls to `update_params` typically take a lot more than `eval_period` seconds, such submissions will not receive as many untimed evaluations as a submission that had an `update_params` function that took less time. However, for appropriate settings of `eval_period`, we expect this to be quite rare. Submissions are always free to restructure their `update_params` code to split work into two subsequent steps to regain the potential benefits of these untimed model evaluations. For each workload, the `eval_period` will be set such that the total evaluation time is roughly between 10% and 20% of the total training time for the target-setting runs.

##### Software Dependencies

If your submission will have any software dependencies, you must create a `requirements.txt` file in the `/submission` directory. This file must clearly list all software dependencies your submission requires in order to be a valid submission. The file must be "pip-readable" (the dependencies listed can be installed via the `pip install -r requirements.txt` command). You may not modify the package versions of the software dependencies used by the benchmarking codebase, including using a different version of libraries such as PyTorch or JAX from those specified in the benchmark.

We require submissions to use specific versions of `PyTorch`/`JAX` as well as additional dependencies in order to facilitate fair comparisons. Submitters must build on top of these pinned software packages. Additional dependencies can be added as long as they include a comment describing what was added and why. Submitters are free to add dependencies that support new algorithmic and mathematical ideas but they should not circumvent the intention of the benchmark to measure training speedups due to new training methods. For example, software engineering techniques that lead to faster implementations of existing software, e.g. using newer versions of `PyTorch` or `JAX`, are not allowed and these are described in more detail in the [Disallowed submissions](#disallowed-submissions) section.

##### Environment Variables

The benchmark codebase sets environment variables, and submitters are not permitted to modify (or add) environment variables for the software dependencies. However, if you believe a setting is sub-optimal, open an issue with justification; the working group may adjust it. This ensures that all submissions are equally affected by the environment variables and maintains the competition's primary focus on algorithmic improvements.

### Tuning Rulesets

Tuning will be substantially different for the [**external**](#external-tuning-ruleset) and the [**self-tuning ruleset**](#self-tuning-ruleset) and the individual specifications for each will be described in the following.

#### External Tuning Ruleset

For every workload, **$5$ tuning _trials_** are run, and this tuning process is **repeated in $3$ independent _studies_** to capture variance, resulting in $15$ runs overall.
Submitters have to provide a _workload-agnostic search space_, via a `tuning_search_space.json` file.
During scoring, we draw $15$ hyperparameter configurations from this search space using [(quasi)random search](https://arxiv.org/abs/1706.03200) and randomly assign them to the $3$ studies with each $5$ trials.
Instead of independent samples from a search space, submitters can also provide a fixed list of $5$ hyperparameter points, which will be sampled without replacement for each study.

Within each study, we select the fastest trial that reaches the validation target. The median of the three per-study best times is the submission's official _per-workload score_. These $8$ _per-workload runtimes_ are used in the scoring procedure (see the [**Scoring submissions**](#scoring) section). Trials that do not reach the target within `max_runtime` receive $\infty$, (which participates in the median).
Submissions may also perform on-the-clock self-tuning during timed training.

> [!IMPORTANT] Summary
>
> - **Trial**: One training run, with a fixed hyperparameter configuration until the target or `max_runtime` was reached. The first time the validation target is reached in a trial is denoted $\tilde{t}_{ij}$ (a miss scores $\tilde{t}_{ij} = \infty$).
> - **Study**: A set of $5$ trials, each run with distinct hyperparameter points. The studies are independent and capture variance. The study's score is the **fastest** (minimum) time among its trials.
> - **Per-Workload Runtime**: The per-workload runtime is given by the median across the per-study scores, i.e., $t_w \;=\; \operatorname{median}_{j=1..3}\Big(\min_{i=1..5} \; \tilde{t}_{ij}\Big)$, with $\tilde{t}_{ij}$ the score of trial $i$ in study $j$, i.e.
>   $$\tilde{t}_{ij} \;=\;\begin{cases}\text{elapsed seconds to reach target}, & \text{if reached within } \texttt{max\_runtime} \\ \infty, & \text{otherwise} \end{cases}\,.$$

#### Self-Tuning Ruleset

Submissions under this ruleset are not allowed to expose user-defined hyperparameters.
Instead, submissions can either apply one "default" hyperparameter configuration for all workloads (e.g. Adam with default settings), or perform inner-loop tuning during their training run (e.g. SGD with line searches).
All workload adaptations, e.g. inner-loop tuning, will be part of the submission's score.

For each workload, a submission will run for **$3$ independent studies**, and the _per-workload score_ is the median time to reach the validation target, i.e., $t_{s,w} = \operatorname{median}_{j=1..3} \tilde{t}_j$.
To account for the lack of external tuning, submissions have a longer time budget to reach the target performance.
Compared to the [**external tuning ruleset**](#external-tuning-ruleset), the `max_runtime` is $1.5\times$ longer (i.e. multiply the `max_runtimes` from the [**workload overview table**](#workloads) by $1.5$).
As in the [**external tuning ruleset**](#external-tuning-ruleset), any run that fails to achieve the target within this window is assigned an infinite runtime.

### Workloads

For the purposes of the _AlgoPerf: Training Algorithms_ benchmark, we consider a workload the combination of a `dataset`, `model`, `loss_fn`, along with a `target` that is defined over some evaluation `metric`. E.g., `ResNet-50` on `ImageNet` using the `cross-entropy` loss until a target `error` of `22.6%` on the validation set has been reached, would constitute a workload.

The _AlgoPerf: Training Algorithms_ benchmark contains a diverse set of $8$ workloads spanning tasks such as image classification, machine translation, speech recognition, or other typical machine learning tasks. For a single task and dataset there might be multiple models and therefore multiple workloads. As a rough guideline, the entire set of workloads was designed to have a combined runtime of very roughly $100$ hours on the [**benchmarking hardware**](#benchmarking-hardware).

The eight _AlgoPerf Workloads_ are:

|       | **Task**                      | **Dataset** | **Model**   | **Loss** | **Metric** | Validation<br>**Target** | Test<br>**Target** | Max<br>**Runtime** <br>_(in seconds)_ | Default<br>**Dropout**<br>Value             |
| ----- | ----------------------------- | ----------- | ----------- | -------- | ---------- | ------------------------ | ------------------ | ------------------------------------- | ------------------------------------------- |
| **1** | Clickthrough rate prediction  | Criteo 1TB  | DLRMsmall   | CE       | CE (↓)     | 0.123735                 | 0.126041           | 7,703                                 | 0                                           |
| **2** | MRI reconstruction            | fastMRI     | U-Net       | L1       | SSIM (↑)   | 0.723653                 | 0.740633           | 4,430                                 | 0                                           |
| **3** | Image classification          | ImageNet    | ResNet-50   | CE       | ER (↓)     | 0.22569                  | 0.3440             | 66,159                                | None                                        |
| **4** |                               |             | ViT         | CE       | ER (↓)     | 0.22691                  | 0.3481             | 69,768                                | 0                                           |
| **5** | Speech recognition            | LibriSpeech | Conformer   | CTC      | WER (↓)    | 0.085884                 | 0.052981           | 58,015                                | 0.1 (`input`, `attn_res`, `ff_res`); else 0 |
| **6** |                               |             | DeepSpeech  | CTC      | WER (↓)    | 0.119936                 | 0.074143           | 44,405                                | 0.1 (`input`, `ff`); `JAX CudnnLSTM`: 0     |
| **7** | Molecular property prediction | OGBG        | GNN         | CE       | mAP (↑)    | 0.28098                  | 0.268729           | 12,011                                | 0.1                                         |
| **8** | Translation                   | WMT         | Transformer | CE       | BLEU (↑)   | 30.8491                  | 30.7219            | 43,336                                | 0.1 (`main`, `attn`)                        |

> [!NOTE]
> Notes on the default dropout column:
>
> - `None` indicates that the model does not use dropout.
> - `0` or `0.1` indicates that the model uses dropout with a default value of 0.0 or 0.1, respectively.
> - `0.1 (main, attn)` indicates that the model uses dropout with a default value of 0.1 for the main `dropout_rate` and the `attention_dropout_rate`.
> - `0.1 (input, attn_res, ff_res) else 0` indicates that the model uses dropout with a default value of 0.1 for `input_dropout_rate`, `attention_residual_dropout_rate`, and `feed_forward_residual_dropout_rate` and use a default value of 0 for all other dropout rates.
> - `0.1 (input, ff) else 0; JAX CudnnLSTM: 0` indicates that the model uses dropout with a default value of 0.1 for `input_dropout_rate` and `feed_forward_dropout_rate`. For JAX models, the `dropout_rate` is set to 0.0 for the `CudnnLSTM` class.
>
> Dropout defaults are used if not overridden by the submission.

#### Recommended Qualification Set

Because the full _AlgoPerf: Training Algorithms_ benchmark is computationally quite expensive, we also provide a recommendation for a cheaper variant, the _qualification set_.
This _qualification set_ excludes both _ImageNet_ workloads, both _LibriSpeech_ workloads, and the _fastMRI_ workload, leaving **_Criteo 1TB_, _OGBG_, and _WMT_**.
Together, they run in roughly $24$ hours on the [**benchmarking hardware**](#benchmarking-hardware).
To further reduce computational costs, the [**external tuning ruleset**](#external-tuning-ruleset) uses **only one study** (instead of the proposed $3$) on the qualification set. The [**self-tuning ruleset**](#self-tuning-ruleset) will keep the $3$ studies because it is less costly.

> [!NOTE]
>
> The "qualification set" was originally designed as a cheaper benchmark that allowed resource-constrained teams to prove themselves and "qualify" for sponsored compute for the full benchmark. Self-reporting is now optional, but the subset still serves as a cheaper performance estimate, so we're keeping it as a recommendation, including the (historical) name.

### Scoring

Submissions are scored based on the training time needed to reach the target performance on each workload's validation set.
The target metric may match the loss function or use another workload-specific metric such as error rate or BLEU score.
See the [**workload overview table**](#workloads) for the targets and metrics of each workload and the [**Defining target performance**](#defining-target-performance-and-max_runtime) section for how they were determined.
The overall ranking is then determined by the scalar _AlgoPerf Benchmark Score_, which summarizes the _per-workload_ runtimes across all [**workloads**](#workloads), using integrated [**performance profiles**](#algoperf-benchmark-score-via-integrated-performance-profiles), as explained below.

> [!NOTE]
>
> The training time of a submission includes the compilation times for computation graphs and ops that could happen just-in-time during training; all our benchmarks should be fast enough to compile so as not to dramatically impact overall performance.

</br>

> [!NOTE]
>
> The training time until the _test set target_ was reached is not used in the scoring procedure but might be used for additional analysis of the competition results.

#### AlgoPerf Benchmark Score via Integrated Performance Profiles

We will aggregate the _per-workload training times_ of a submission across all workloads using [Performance Profiles](http://www.argmin.net/2018/03/26/performance-profiles/) (originally from [Dolan and Moré](https://arxiv.org/abs/cs/0102001)). Below we surface several relevant definitions from their work for easier readability, before explaining how we integrate the performance profiles to reach a scalar benchmark score that will be used for ranking submissions.

_Notation:_ We have a set $\mathcal{S} = \{s_1, s_2, \dots, s_k\}$ of in total $k$ submissions that we evaluate on a set of $n$ workloads: $\mathcal{W} = \{w_1, w_2, \dots, w_n\}$. For each submission $s$ and each workload $w$ we have a _per-workload runtime_ $t_{s,w} \in [0,\infty)$. This is the median time it took the submission to reach the validation target performance on this particular workload.

##### Computing performance ratios

For all workloads and submissions, we first compute their performance ratio $r$, which is defined for a particular submission $\bar{s}$ and a particular workload $\bar{w}$ to be:

$$r_{\bar{s},\bar{w}} = \frac{t_{\bar{s},\bar{w}}}{\min_{s \in \mathcal{S}} t_{s,\bar{w}}} \in [1,\infty)$$

This performance ratio $r_{s,w}$ expresses the "time spent by submission $s$ on workload $w$" relative to the "time spent by the best submission on this workload". E.g. If a submission takes twice as long on a particular workload compared to the best submission on this workload it will have a performance ratio of $2$. Lower performance ratios are therefore better, with an optimal ratio of $1$ if the given submission is the fastest on this workload.

##### Building performance profiles

Next, we compute how often a submission is within a factor $\tau \in [1,\infty)$ of the optimal submission. For this, we determine the following function for every submission $\bar{s}$:

$$\rho_{\bar{s}}(\tau) = \frac{1}{n} \!\cdot\mkern-28mu \underbrace{\left|\left\{w: \, r_{\bar{s},w}\leq \tau\right\}\right|}_{= \text{number of workloads with}\, r_{\bar{s},w}\leq \tau}$$

In other words, we compute the fraction of workloads where a submission $\bar{s}$ is less than $\tau$ away from the optimal submission. The function $\rho_{\bar{s}}(\tau)$ is monotonically increasing with $\tau$ and bounded between $0$ and $1$.

An example of a performance profiles plot is shown below, where we plot $\rho_{\bar{s}}(\tau)$ for six submissions:

![Example performance profile](/.assets/performance_profiles.png)

##### Integrating performance profiles for the benchmark score

To get the scalar _AlgoPerf Benchmark Score_ that is usable for ranking submissions, we will integrate the performance profiles $\rho_{\bar{s}}(\tau)$ of all submissions to get their _AlgoPerf Benchmark Score_ $B_{\bar{s}}$, with

$$B_{\bar{s}} = \frac{1}{r_{\text{max}}-1} \int_{1}^{r_{\text{max}}} \rho_{\bar{s}}(\tau) \,d\tau \in [0, 1].$$

The upper integration limit will be set to $r_{\text{max}} = 4$ which also serves as the upper limit of the performance profile plot.
This means that any submission that requires more than four times the runtime of the fastest submission will not get any credit on this workload compared to a training algorithm that is unable to successfully train within the maximum allowed runtime budget.
The integral is normalized by the total integration area, such that all _AlgoPerf Benchmark scores_ are between $0$ and $1$, with higher scores being better. A perfect score of $1$ is achieved if a submission is the fastest on all workloads.

##### Alternative scores

Performance profiles and the _AlgoPerf Benchmark Score_ derived from them, take a bit of effort to explain.
However, we believe that they are fairer and well-supported by research in machine learning and the optimization community. To have some simpler to interpret numbers, e.g. for press releases, we will also release a series of alternative scores.

For a given workload $\bar{w}$, we define the "speedup of a submission $\bar{s}$ over submission $\text{ref}$" as $\frac{t_{\text{ref}, \bar{w}}}{t_{\bar{s}, \bar{w}}}$. For example, if a submission was $2\times$ faster than the reference submission, this would be equal to $2$. In addition to the raw $t_{s,w}$ values, we will release the geometric mean of the speedups across all workloads, i.e. $\left(\prod_{w \in \mathcal{W}} \frac{t_{\text{ref}, w}}{t_{\bar{s}, w}}\right)^{\frac{1}{n}}$.

#### Benchmarking Hardware

All officially scored runs will be performed on the same benchmarking hardware to allow for a fair comparison of wall-clock training times.
This benchmarking hardware is chosen to be easily accessible via common cloud computing providers and will likely change with each iteration of the benchmark.
The specs of the benchmarking hardware for this iteration of the benchmark are:

- 8× NVIDIA V100 (16 GB) GPUs
- 240 GB in RAM
- 2 TB in storage (for datasets).

> [!TIP]
> Submitters are no longer required to self-report results to enter the AlgoPerf benchmark.
> Instead, they can open a PR and the working group will score the most promising submissions, see our [**How to Submit**](/README.md#how-to-submit) section for more details.
> If you'd like to self-report results, e.g., for paper experiments or to provide evidence of your submission's performance, it is possible to use a different hardware. However, we strongly recommend to use the same hardware for all algorithms, at least for the scored runs. It is possible to _perform tuning trials on different hardware_, as long as the hardware is consistent for all tuning trials.
> However, in order to compare to the published results, you will have to repeat at least those fastest trials on the benchmarking hardware.
> This allows for a fair comparison to the reported results of other submitters while allowing some flexibility in the hardware.

#### Defining Target Performance and `max_runtime`

This section briefly explains the process to define the target performance for each [**workload**](#workloads), which will be used by both [**tuning rulesets**](#tuning-rulesets) equally. For more details, see [**our benchmark paper**](https://arxiv.org/abs/2306.07179).

For each workload, we take the best performance achievable by one of four standard algorithms (`AdamW`, `NadamW`, `Nesterov Momentum`, and `Heavy Ball Momentum`). These target-setting algorithms will follow the general process of the external tuning ruleset, with a significantly larger tuning budget of $200$ trials to guarantee competitive performance. Once the best algorithm and its hyperparameters are determined, training is repeated $20$ times with this configuration. The median of the best achieved validation errors across seeds is used as the _validation_ target. Out of the $10$ repeated runs that achieved this validation target, we took the worst achieved test error across seeds as our _test_ target. Taking the median validation performance after rerunning the best hyperparameter point prevents our procedure from selecting a lucky outlier.

> [!NOTE]
> The runtime of the target-setting algorithms was chosen to roughly match published results without extending the overall benchmark budget too much.
> The initial `max_runtime` (used in version 0.5 of the benchmark) available to submissions on each workload was $\frac{1}{3}$ longer than the runtime of the target-setting algorithms to allow submissions a bit more time to reach the target on some workloads, if they can make up for it on others. After the initial competition, we have adapted the `max_runtimes` based on the performance of the submissions (see [this issue](https://github.com/mlcommons/algorithmic-efficiency/issues/836)).

## Versioning Policy

_AlgoPerf_ uses a unified versioning scheme: codebase, rules, and leaderboard all share the same `Major.Minor` version. `Patch` versions may differ for minor, non-breaking updates to each component. All results produced under the same `Major.Minor` version are comparable, making it easy to cite "`AlgoPerf v0.X`" and know exactly which set of rules, code, and submissions are being referenced.

- **Codebase:** The version is automatically set from the latest GitHub tag and accessible via `algoperf.__version__`.
- **Rules/Documentation:** This document reflects the unified version shown above.
- **Leaderboard:** The leaderboard in the [**submissions repository**](https://github.com/mlcommons/submissions_algorithms) displays which version of the benchmark was used for scoring.

For detailed information about releases and version history, see our [**README**](../README.md#releases--roadmap) and our [**Changelog**](CHANGELOG.md).

### Version Freeze

To ensure that all submitters can develop their submissions based on the same code that will be utilized for scoring, we freeze the package versions of the codebase dependencies in between benchmark versions. By doing so, we level the playing field for everyone involved, ensuring fairness and consistency in the assessment of submissions. We will try to minimize changes to the benchmark codebase as best as possible.

## License and Legal Requirements

All submissions must be licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
Furthermore, all submitters must sign the following agreements:

- A signed [Contributor License Agreement (CLA) "Corporate CLA"](https://mlcommons.org/en/policies/) of MLCommons.
- _Either_ a membership in MLCommons _or_ a signed [non-member test agreement](https://mlcommons.org/en/policies/).
- A signed trademark license agreement, either the member or the non-member version, as appropriate. These license agreements are available upon request to [support@mlcommons.org](mailto:support@mlcommons.org).

## FAQs

> If your question isn't answered here, please [**contact us**](mailto:algorithms-chairs@mlcommons.org). These FAQs serve to supplement and clarify the rules and documentation described above.

### Setup & Platform

<details>
<summary><strong>My machine only has one GPU. How can I use this repo?</strong></summary>

> You can run this repo on a machine with an arbitrary number of GPUs. However, the default batch sizes of our algorithms collection (e.g. `algorithms/`) are tuned for a machine with 8× NVIDIA V100 (16 GB) GPUs. You may run into OOMs if you run these algorithms with fewer than 8 GPUs. If you run into these issues because you are using a machine with less total GPU memory, please reduce the batch sizes for the submission. Note that your final submission must 'fit' on the [**benchmarking hardware**](#benchmarking-hardware), so if you are using fewer GPUs with higher per-GPU memory, please monitor your memory usage to make sure it will fit on 8× NVIDIA V100 GPUs with 16 GB of VRAM per card.

</details>

<details>
<summary><strong>How do I run this on my SLURM cluster?</strong></summary>

> You may run into issues with `sudo` and `docker` on a SLURM cluster. To run the workloads in a SLURM cluster you can use Apptainer (_formerly Singularity_), see this [**section**](/docs/GETTING_STARTED.md#using-singularityapptainer-instead-of-docker).

</details>

<details>
<summary><strong>How can I run this on my AWS/GCP/Azure cloud project?</strong></summary>

> Depending on your virtual machine, you may have to install the correct GPU drivers and the NVIDIA Docker toolkit. For example, in GCP you will have to do the following.
>
> 1. If you don't have a VM instance yet, we recommend creating a
>    new Compute Instance with the "Deep Learning on Linux" Image in Boot disk options.
> 2. To install the NVIDIA Docker toolkit, you can use [`docker/scripts/cloud-startup.sh`](/docker/scripts/cloud-startup.sh) as a startup script for the VM. This will automate the installation of the NVIDIA GPU Drivers and NVIDIA Docker toolkit.

</details>

### Submitting

<details>
<summary><strong>How do I submit my algorithm to the benchmark?</strong></summary>

> Please see our [**How to Submit**](/README.md#how-to-submit) section. You can submit your algorithm to the benchmark by opening a PR on the [**submission repository**](https://github.com/mlcommons/submissions_algorithms).

</details>

<details>
<summary><strong>Can I submit multiple times?</strong></summary>

> Our benchmark allows multiple submissions as long as they are substantially different. We discourage submitters from creating bulk submissions as this is not in the spirit of the benchmark.

</details>

<details>
<summary><strong>Can my submission span multiple files?</strong></summary>

> Yes, your submission can be structured using multiple files.

</details>

<details>
<summary><strong>Can I install custom dependencies?</strong></summary>

> You may use custom dependencies as long as they do not conflict with any of the pinned packages in [`pyproject.toml`](/pyproject.toml).
>
> To include your custom dependencies in your submission, please include them in a `requirements.txt` file. Please refer to the [**Software dependencies**](#software-dependencies) section of our rules.

</details>

### Scoring & Hardware

<details>
<summary><strong>How can I know if my code can be run on benchmarking hardware?</strong></summary>

> The benchmarking hardware specifications are documented in the [**Benchmarking Hardware Section**](#benchmarking-hardware). We recommend monitoring your submission's memory usage so that it does not exceed the available memory on the benchmarking hardware. We also recommend doing a dry run using a cloud instance.

</details>

<details>
<summary><strong>This benchmark seems computationally expensive. Do I have to run it myself?</strong></summary>

> Submitters are no longer required to self-report results to get on the _AlgoPerf_ leaderboard. Instead, they can open a PR in the [**submission repository**](https://github.com/mlcommons/submissions_algorithms) and the working group will score the most promising submissions, see our [**How to Submit**](/README.md#how-to-submit) section for more details. You can use self-reported results to provide evidence of performance on the benchmark. Even if you fully self-report, we will still verify the scores by rerunning the submission on our setup.

</details>

<details>
<summary><strong>Can I submit previously published training algorithms as submissions?</strong></summary>

> Yes, you may, as long as it isn't an exact copy of an existing submission.
>
> For example, you may submit the Adam optimizer with your particularly effective hyperparameter search space and hyperparameter configuration, as different choices for hyperparameter values and/or search spaces constitute different training algorithms and are potential sources of innovation.
>
> That said, while submitting Adam with some novel heuristic to set various hyperparameters, some especially effective hyperparameter search space, or your single best hyperparameter configuration is fine, avoid making multiple submissions that only differ by their hyperparameter configuration without a convincing justification they are substantially different (see the [**"Can I submit multiple times to the benchmark competition?"**](#submitting) question, above).

</details>

## Disclaimers

### Shared Data Pipelines between `JAX` and `PyTorch`

The `JAX` and `PyTorch` versions of the `Criteo`, `fastMRI`, `LibriSpeech`, `OGBG`, and `WMT` workloads use the same `TensorFlow` input pipelines. Due to differences in how `JAX` and `PyTorch` distribute computations across devices, the `PyTorch` workloads have an additional overhead for these workloads.

Since we use `PyTorch`'s [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) implementation, there is one Python process for each device. Depending on the hardware and the settings of the cluster, running a `TensorFlow` input pipeline in each Python process can lead to errors, since too many threads are created in each process. See [this PR thread](https://github.com/mlcommons/algorithmic-efficiency/pull/85) for more details.
While this issue might not affect all setups, we currently implement a different strategy: we only run the `TensorFlow` input pipeline in one Python process (with `rank == 0`), and [broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast) the batches to all other devices. This introduces additional communication overhead for each batch. See the [implementation for the `WMT` workload](https://github.com/mlcommons/algorithmic-efficiency/blob/main/algoperf/workloads/wmt/wmt_pytorch/workload.py#L215-L288) as an example.
