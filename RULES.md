# MLCommons™ Algorithmic Efficiency Benchmark Rules

**Version:** 0.0.8 *(Last updated 15 August 2022)*

> **TL;DR** New training algorithms and models can make neural net training faster.
> We need a rigorous training time benchmark that measures time to result given a fixed hardware configuration and stimulates algorithmic progress. We propose a [Training Algorithm Track](#training-algorithm-track) and a [Model Track](#model-track) in order to help disentangle optimizer improvements and model architecture improvements. This two-track structure lets us enforce a requirement that new optimizers work well on multiple models and that new models aren't highly specific to particular training hacks.

- [Introduction](#introduction)
- [Training Algorithm Track](#training-algorithm-track)
  - [Submissions](#submissions)
    - [Specification](#specification)
    - [Evaluation during training](#evaluation-during-training)
    - [Valid submissions](#valid-submissions)
  - [Tuning](#tuning)
    - [External tuning ruleset](#external-tuning-ruleset)
    - [Self-tuning ruleset](#self-tuning-ruleset)
  - [Workloads](#workloads)
    - [Public workloads](#public-workloads)
    - [Held-out workloads](#held-out-workloads)
    - [Qualification set](#qualification-set)
  - [Scoring](#scoring)
    - [Competition hardware](#competition-hardware)
    - [Defining target performance](#defining-target-performance)
    - [Competition score using performance profiles](#competition-score-using-performance-profiles)
    - [Awards and prize money](#awards-and-prize-money)
- [Model Track](#model-track)

## Introduction

We need a more scientifically sound methodology for evaluating training speedups due to new algorithms, including both new optimizers and new model architectures. Cutting edge machine learning (ML) models are exceeding the compute budgets of many researchers, and ML compute is becoming a larger and larger cost in industry. To reduce the compute and potentially environmental cost of ML research and practice, we need rigorous benchmarking of efficiency. Such benchmarks will guide us in selecting the best directions to evolve existing techniques and ultimately enable progress toward models that produce not only better results, but better results **at lower cost**.

MLCommons' mission is to build fair and useful benchmarks for measuring training and inference performance of ML hardware, software, and services. Improvements in training speed can come from better hardware, better software stacks, and better algorithms.
To date, the Closed Division of the MLPerf™ Training benchmark has been extremely successful in driving systems innovation by requiring mathematical equivalence to a reference implementation, while still allowing submissions on different hardware. Although the Open Division allows new models and training algorithms, it has several issues that make it inappropriate as a benchmark for progress in training algorithms. By allowing arbitrary hardware, it is impossible to isolate improvements due to algorithms or due to extra computation. Unrestricted hardware makes the competition only accessible to the most well-funded organizations, even if many academic labs and others have interesting algorithms to measure. Finally, even if we could isolate improvements due to particular algorithmic changes and make the benchmark more broadly accessible, there is still no incentive to avoid hyper-specific changes that only help the particular benchmark workload.

In order to drive innovation in machine learning algorithms that reduce the time needed to create useful models, we propose a new set of benchmarks to evaluate the training time for different algorithms (models, optimizers, preprocessing, etc.) on a **fixed hardware configuration** (future iterations can adopt new hardware configurations as needed). Our proposal includes two tracks: (1) a [Model Track](#model-track) and (2) a [Training Algorithm Track](#training-algorithm-track). The goal of the Model Track is to find models that can be trained to achieve the target solution quality (out-of-sample error) in the least amount of time on each benchmark dataset. Similarly, the goal of the Training Algorithm Track is to find training algorithms (optimizers, etc.) that train benchmark models to reach the goal out-of-sample error rate as fast as possible. However, to incentivize practically useful algorithms, in the Training Algorithm Track we require that a single training algorithm simultaneously performs well across all benchmark models and datasets. Although submissions in the Model Track will be inherently dataset-specific, we sharply constrain what parts of the training program can be modified in the Model Track and require submitted models to be easily trainable using standard optimizers. Thus the two-track structure discourages overly specific solutions that aren't generally useful to practitioners and will hopefully produce evidence on the relative returns of speeding up training by finding new models or by developing new training algorithms.

## Training Algorithm Track

The goal of the Training Algorithm Track is to reach the same results faster ("time to result") by using better optimizers, data ordering/weighting schemes, and weight update strategies while producing techniques that work well on a wide variety of models and datasets. We hope to encourage generally useful training algorithms that are not specific to only a small number of particular workloads.

In general, submissions to the Training Algorithm Track will replace specific pieces of a reference implementation in order to produce a training program that reaches the same results faster on as many workloads as possible. The training program has a fixed, high-level structure and competitors are allowed to replace a particular set of functions in the program (the [**submission functions**](#submission-functions)), but must leave all other pieces ([**fixed functions**](#fixed-functions) and high-level structure) of the reference implementation unchanged. The submitted code must perform well on multiple datasets and models simultaneously (a model and dataset pair constitute a [workload](#workloads) for the purposes of this track).

Submissions to the Training Algorithm Track can be entered under two separate rulesets, named [external tuning ruleset](#external-tuning-ruleset) and [self-tuning ruleset](#self-tuning-ruleset), with it being possible to submit to both rulesets. The main difference is that the external tuning ruleset allows moderate, automatic tuning of the optimizer's hyperparameters on each workload, using the submitted workload-agnostic search space. This allows the training algorithm to adapt to a particular task while ensuring that it is not too difficult to tune automatically. Under the Self-tuning Ruleset, there is no external tuning and submissions need to adapt to a particular task autonomously within a single optimization run. Unless otherwise specified, the rules in this section apply to both rulesets (see, for example, the [Tuning](#tuning) Section for the most substantial difference between the rulesets).

The intention is that a training algorithm submission will be broadly applicable and useful without customization to the specific [workload](#workloads) (model, dataset, loss function). We want to discourage detecting the particular workload and doing something highly specific that isn't generally useful. In order to further discourage submissions that overfit to the particular [public benchmark workloads](#public-workloads), submissions must also perform well on one or more [held-out workloads](#held-out-workloads) released after the submission deadline.

### Submissions

A valid submission is a piece of code with the same high-level structure as a reference implementation that can train all benchmark workloads on the [competition hardware](#competition-hardware) (defined in the [Scoring](#scoring) Section but ultimately in the call for submissions). The validation set performance will be checked regularly during training (see the [Evaluation during training](#evaluation-during-training) Section) and training halts when a workload-specific target error has been reached. For each workload, the training time to reach this (validation set) target error will be used as an input to the [scoring process](#scoring) for the submission. Additionally, the test set performance will be probed using the final model parameters to confirm that it also reaches a slightly more generous target performance on this unseen data. Submissions using [external tuning](#external-tuning-ruleset) will be tuned independently for each workload using a single workload-agnostic search space for their specified hyperparameters. Submissions under either tuning ruleset may always self-tune while on the clock.

#### Specification

Any function defined in the reference implementations that isn't a [submission function](#submission-functions) is a [fixed function](#fixed-functions) for the Training Algorithm Track. No submitted code is run to compute the evaluation metrics in the Training Algorithm Track. We just use the final model parameters and the fixed functions from this track at test time.

In principle, submissions are allowed to use the available hardware systems in any data- or model-parallel manner they desire, within the constraints of the submission function APIs. However, in practice, model-parallelism may not be possible with the API. They are allowed to access any framework-specific device information necessary to exploit the hardware.

Submissions provide a [per-workload batch size](#batch-size-getter) to use. Specification of the batch size for each workload is necessary to avoid running out of memory for different workloads. Therefore, submitters can determine this batch size in advance and specify it as part of the submission. For [held-out workloads](#held-out-workloads), the submitted batch size of the most similar [public workload](#public-workloads) will be used (for example, if there is an ImageNet public workload and also a held-out workload with a similarly sized model on similarly sized images, the ImageNet batch size will be used for this held-out workload).

The **submission functions** are the *batch size getter*, *optimizer state initializer*, *variable update*, and *data selection functions*. The *fixed functions* are the *data augmentation/preprocessing*, *model initialization*, *forward pass*, and *loss function*. The trained model will be evaluated in a separate step that does not call any of the submitted code.

##### Fixed functions

###### Data augmentation and preprocessing

```python
def build_input_queue(
    self,
    data_rng: RandomState,
    split: str,
    data_dir: str,
    global_batch_size: int) -> Iterator[Dict[str, Tensor]]:
```

- The `build_input_queue` function will be called to produce the iterator over batches that the submitted data selection function consumes. It is responsible for all data reading, shuffling, repeating, preprocessing, and batching. Note that for Jax this should return an iterator over tensors of shape `(num_devices, per_device_batch_size, ...)`, and for PyTorch this should return tensors of shape `(global_batch_size, ...)`.

###### Model initialization

```python
def init_model_fn(
    rng: RandomState
) -> initial model parameters
```

- Unlike in the [Model Track](#model-track), this function that initializes the parameters of the model, is fixed. While it can be called by the submission (e.g. to restart the model after a failed training effort) it cannot be changed.

###### Forward pass

```python
def model_fn(
    params: ParameterContainer,
    augmented_and_preprocessed_input_batch: Tensor,
    model_state: ModelAuxiliaryState,
    mode: ForwardPassMode,  # mode \in {train, eval}
    rng: RandomState,
    hyperparameters: Hyperparameters,
    update_batch_norm: bool
) -> (logits_output_batch, new_model_state): Tuple[Tensor, ModelAuxiliaryState]
```

- `params` is whatever the structure is that contains the (`float32`) model parameters. The naming is overloaded due to having to handle the more object-oriented `PyTorch` style and the functional `JAX` style of development. In the `Flax` library (written in `JAX`), this is typically a nested dictionary of `JAX`/`numpy` arrays, but in `PyTorch` this is the `torch.nn.Model`.
- It is possible that `model_parameters` will be endowed with additional information about the kind of each parameter, e.g. "weights" or "bias" or "batch norm", although `model_fn` does not really need that information we might use the same nested structure elsewhere
- `logits_output_batch` is before the output activation
- `new_model_state` is for batch norm or similar side effects and will only be updated if `update_batch_norm` is set
- `hyperparameters` will contain only dropout rates, which will be used in the models that support it. These can be tuned or will default to documented model-specific values. Note that adding additional dropout would be considered changing the model, which is not allowed, but the tuning of dropout in existing dropout layers can be considered a regularizer, so we allow it. There should be at most two dropout rates in a model (if there are more than two we will reuse the same values).

###### Loss function

```python
def loss_fn(label_batch, logits_output_batch) -> 1d array of losses per example  # differentiable
```

- Unlike in the [Model Track](#model-track), we will specify the loss function name in order to let training algorithms depend on the loss function. It will be one of {**mean squared error**, **cross-entropy**, **CTC**, or **L1 reconstruction error**}.
  - The optimizer must work with all values of the enum, which will be provided via a property on the workload object that is provided to all submissions functions.
- The loss function does **not** include regularization. Instead, regularization can be added by the submissions in the `update_variables` function.

##### Submission functions

###### Batch size getter

```python
def get_batch_size(workload_name: str) -> int
```

- Submitters define a specific batch size for each [workload](#workloads).
- For example, in advance, they can determine the largest batch size without running out of memory for each workload.
- For the [held-out workloads](#held-out-workloads), the `workload_name` of the closest public workload will be used in this function.

###### Optimizer state initializer

```python
def init_optimizer_state(
    workload: Workload,
    model_params: ParameterContainer,
    model_state: ModelAuxiliaryState,
    hyperparameters: Hyperparameters,
    rng: RandomState
) -> initial_optimizer_state
```

- Allowed to create state for the optimizer
- Does not involve the initialization for the model parameters, which in the Training Algorithm Track, is considered a fixed function, see [Model initialization](#model-initialization).

###### Variable update function

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
    rng: RandomState
) -> (updated_optimizer_state, updated_variables, updated_model_state)
```

- `current_param_container` is the same kind of nested structure as used by `model_fn` which constitutes a nested collection of `float32` arrays, each endowed with information about what kind of parameter that array represents stored in a parallel structure of `current_params_types`.
  - Parameter kind is one of {"weights", "biases", "embeddings", "conv", "batch norm"}
- `model_state` holds auxiliary state necessary for some models, such as the current batch norm statistics
- The loss function will be one of a small set of known possibilities and the update function is allowed to branch on the `loss_fn` enum/name.
- The `loss_fn` produces a loss per example, so the submission code is responsible for summing or averaging
- Allowed to update state for the optimizer
- Uses the `model_fn` of the `workload` in order to decouple the loss from the model so that model outputs (forward passes) can be reused (by storing them in the optimizer state)
- The submission can access the target evaluation metric via the `workload` variable.
- **A call to this function will be considered a step**
  - The time between a call to this function and the next call to this function will be considered the per-step time
- Cannot modify the given hyperparameters in a workload-conditional way (please see the [Valid Submission](#valid-submissions) Section). This rule is intended to prohibit circumventing the tuning rules by looking up a pre-tuned optimal set of hyperparameters for each workload. It is not intended to prohibit line searches and other similar techniques.
  - This will be checked by the spirit jury
- The fixed `init_model_fn` can optionally be called during training, for example, to reinitialize the model after a failed training effort.
- Cannot replace the model parameters with pre-trained ones.
  - This will be checked by the spirit jury.
- This API supports Polyak averaging and similar methods that implement moving averages of model parameters
- Batch norm should work here because the `model_fn` will return updated batch norm moving averages when it is told to with `update_batch_norm`.

###### Data selection

```python
def data_selection(
    workload: Workload,
    input_queue: Iterator[Tuple[Tensor, Tensor]],
    optimizer_state: OptimizerState,
    current_param_container: ParameterContainer,
    hyperparameters: Hyperparameters,
    global_step: int,
    rng: RandomState
) -> Dict[str, Tensor]
```

- `input_queue` can yield up to the number of elements in the training dataset
- Want to allow for submitters to construct their own data batches from the dataset
- Submissions are allowed to arbitrarily modify the input examples, as long as the modifications are sufficiently generic to be applicable to any workload
- This is only called on the training inputs. **No submitted code will be called at eval in the training track.**
- This allows for any of the following methods:
  - Data echoing
  - Curriculum learning
  - Bootstrapping
  - Biased sampling (based on loss values, so need to store the forward pass in the `optimizer_state`, potentially forward pass of a cheaper proxy model)
  - Submissions need batching control

#### Evaluation during training

In general, with noisy, non-deterministic training, evaluation frequency can affect training time measurements as more "bites of the apple" potentially allows the training code to exploit instability. We also want to discourage submissions from complicated and unrealistic logic that attempts to guess when training is close to complete and increases the evaluation rate, while not producing a well-sampled training curve at the start of training. Simply allowing submissions complete freedom over evaluation frequency encourages competitors to work to minimize the number of evaluations, which distracts from the primary goal of finding better training algorithms.

Submissions are eligible for an untimed eval every `eval_period` seconds, run as soon as the current call of `update_params` completes. Any additional evaluations performed by the submission code count against the runtime for scoring. The harness that runs the submission code will attempt to eval every `eval_period` seconds by checking between each submission step (call of `update_params`) whether it has been at least `eval_period` seconds since that last eval and, if so, pausing the clock and running an eval. This means that if calls to `update_params` typically take a lot more than `eval_period` seconds, such submissions will not receive as many untimed evals as a submission that had an `update_params` function that took less time. However, for appropriate settings of `eval_period`, we expect this to be quite rare. For each workload, the `eval_period` will be set such that the total evaluation time is roughly between 10% and 20% of the total training time for the target-setting runs. Submissions are always free to restructure their `update_params` code to split work into two subsequent steps to regain the potential benefits of these untimed model evaluations.

#### Valid submissions

The intention of this benchmark is to identify training algorithm submissions that will be broadly applicable and effective in practical scenarios without customization to the specific [workload](#workloads) (model, dataset, and loss function). Generally useful training algorithms can train models faster and thus require less compute resources, decreasing the cost of machine learning. We want to discourage all submissions that sidestep the purpose of this benchmark.

We reserve the right to disqualify submissions if they clearly violate this spirit of the benchmark, even if those submissions perform well in our benchmark. Unfortunately, we can't easily write rules that make it completely clear if a submission is circumventing the spirit of the benchmark in a way that would encompass all possible cases. Instead, we will have to prohibit these activities in the abstract and defer rulings about specific submissions to a **"spirit [of the rules] jury"** that can hear the justifications of the submitters, inspect the code, and ultimately decide if the spirit of the rules has been violated. The jury might also ask the submitters to explain how the submission was produced, for example, by disclosing their intermediate experiments.

We want to state clearly that we welcome creative ideas and novel research. Therefore, the API aims to allow a wide variety of submissions, however, in some cases, routines that would be allowed in principle might not be practically feasible in the provided framework. The spirit jury, however, will only be invoked for submissions that aim to bypass the core premise of this benchmark since submissions like this would also be irrelevant in practice.

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

- Submission are allowed to learn the update rule of the training method.
- In the [Self-tuning Ruleset](#self-tuning-ruleset), submissions could try out a learned list of hyperparameters.

</details>
<br>

Submissions can use additional software dependencies provided they have the intention of supporting new algorithmic and mathematical ideas. The procedure for adding dependencies is described in more detail in the [Software Dependencies](#software-dependencies) Section.

<details>
<summary>Examples:</summary>

- [`BackPACK`](https://docs.backpack.pt/en/master/index.html) is a `pip` package that hooks into `PyTorch` to extract additional information from the backward pass. An allowed use of `BackPACK` would be to compute batch statistics (e.g. within-batch gradient variances, etc.) to calibrate or auto-tune training algorithms.

</details>

##### Disallowed submissions

Submissions are not allowed to circumvent the tuning rules by looking up the result of an offline computation that was performed ahead of time.

<details>
<summary>Examples:</summary>

- Submissions are not allowed to look up (pre-trained) model parameters.
- Computing the optimal hyperparameters for every public workload offline and having the submission look up those pre-computed values (and finding the closest public workload for a held-out workload) is not allowed. In contrast, finding and hard-coding a single good setting of the hyperparameters that works well across all the workloads simultaneously would be allowed.
- Submissions are not allowed to adjust the hyperparameter search spaces for the external tuning ruleset, such that it differs between the workloads.

</details>
<br>

Submissions are not allowed to detect the particular workload (irrespective of which information they use to this end) in order to use settings that are specified for individual workloads. This would result in highly specific behavior that isn't generally useful. This also extends to learned approaches that ultimately detect specific workloads. In general, all else being equal, if some submission was written that was extremely effective on a small set of the workloads (and far worse on the rest) and another submission with the opposite performance pattern, we would prefer both submissions to be submitted and tested on **all** workloads.

<details>
<summary>Examples:</summary>

- A hard-coded switching of the update rule based on the workload is not allowed, e.g. using Adam for RNNs and SGD with momentum on CNNs. Although submissions can specialize for certain layer types in generic ways, they should not uniquely identify a model or dataset. In other words, if there are two workloads A and B that both have convolutional layers and fully connected layers the submission shouldn't detect whether it is dealing with A or B specifically and choose Adam for one and SGD with momentum for the other. However, if the updates for all parameters of convolutional layers always used SGD with momentum and the updates for all other layers always used Adam and a workload with both types of layers had mixed updates, that would be fine.
It is also allowed to make the update rule part of the (external) hyperparameter tuning or determine the optimal update rule during the run, i.e. while "on-the-clock".
- Submissions are not allowed to look up learning rate schedules that are only utilized for specific subsets of the workloads. It is allowed to use one general learning rate schedule or dynamically adapt the learning rate based on general information such as curvature.

</details>
<br>

It is not allowed to compute any kind of pairwise metrics between the public workloads and the held-out workloads

<details>
<summary>Examples:</summary>

- On a held-out workload, submissions are not allowed to find the nearest neighbor among the public workloads to set any hyperparameter.

</details>
<br>

Valid submissions must rely on new algorithmic or mathematical ideas and should not use software engineering approaches to speed up primitive operations in `PyTorch`, `JAX`, their dependencies, the operating system, or the hardware.

<details>
<summary>Examples:</summary>

- Submitters are not allowed to use faster GPU kernels than other submitters by writing their own, using `TVM`, or using a different version of `cuDNN`/`cuBLAS`.
- Submitters are not allowed to skip or reduce system or framework overhead, such as modifying `JAX` to skip internal steps like pytree flattening/unflattening.
- Submitters are not allowed to reorder the schedule of operations, such as using `CUDA` streams to parallelize GPU kernels.
- Submitters are not allowed to introduce new compiler optimizations, such as modifying `XLA` to perform more or less kernel fusion.
- Submitters are not allowed to have a load-balancing algorithm to vary the amount of work performed on the CPU, GPU, OS subsystems, or compute units such as Tensor cores.
- In general, submissions can make clever, judicious, and efficient use of public APIs in `JAX` and/or `PyTorch` but should not be trying to optimize the internals of primitive operations and standard dependencies.

</details>

##### Software dependencies

We require submissions to use specific versions of `PyTorch`/`JAX` as well as additional dependencies in order to facilitate fair comparisons. Submitters must build on top of these provided software packages, which might be provided as a `Docker` container. Additional dependencies can be added as long as they include a comment describing what was added and why. Submitters are free to add dependencies that support new algorithmic and mathematical ideas but they should not circumvent the intention of the benchmark to measure training speedups due to new training methods. For example, software engineering techniques that lead to faster implementations of existing software, e.g. using newer versions of `PyTorch` or `JAX`, are not allowed and these are described in more detail in the [Disallowed submissions](#disallowed-submissions) Section. In case of doubts, these additional dependencies will be judged by the spirit jury.

### Tuning

Tuning will be substantially different for the [external](#external-tuning-ruleset) and the [self-tuning ruleset](#self-tuning-ruleset) and the individual specifications for each will be described in the following.

#### External tuning ruleset

For each workload, the hyperparameters are tuned using $O=20$ tuning **trials**. To estimate the variance of the results, this tuning will be repeated for $S=5$ **studies**, for a total of $S\cdot O = 100$ different hyperparameter settings. The submitters will provide a workload-agnostic search space and the working group will then return $100$ hyperparameters settings obtained using [(quasi)random search](https://arxiv.org/abs/1706.03200). The working group will also randomly partition these $100$ trials into $5$ studies of $20$ trials each.

In each study, the fastest training time across the $O=20$ settings will be taken into account and the median of these $5$ per-study training times will be the final training time for the submission on this workload (see [Scoring submissions](#scoring) Section). Runs that do not reach the target performance of the evaluation metric have an infinite time. Submissions are always free to perform additional self-tuning while being timed.

#### Self-tuning ruleset

Submissions to this ruleset are not allowed to have user-defined hyperparameters. This ruleset allows both submissions that use the same hyperparameters for all workloads, including the held-out ones (e.g. Adam with default parameters), as well as submissions that perform inner-loop tuning during their training run (e.g. SGD with line searches).

Submissions will run on one instance of the [competition hardware](#competition-hardware). As always, submissions are allowed to perform inner-loop tuning (e.g. for their learning rate) but the tuning efforts will be part of their score. A submission will run *S=5* times and its score will be the median time to reach the target evaluation metric value on the held-out data. Runs that do not reach the target performance of the evaluation metric have an infinite time.

### Workloads

For the purposes of the Training Algorithm Track, we consider the combination of a `dataset`, `model`, and `loss_fn` a workload. E.g., ResNet50 on ImageNet using cross-entropy loss would constitute a workload. The evaluation metric, in this example the misclassification error, is directly implied by the dataset/task. In addition to the [public workload](#public-workloads) set, submissions must also perform well on a set of [held-out workloads](#held-out-workloads). These held-out workloads will be specified after the submission deadline, but their generating process is publicly available with the call for submission.

The submissions will be [scored](#scoring) according to their performance on all workloads, including the public as well as the held-out workloads.

Furthermore, a less computationally expensive subset of the public workloads is collected with the [qualification set](#qualification-set). Submitters without enough compute ressources to self-report on the full set of public and held-out workloads can instead self-report on this smaller qualification set. Well-performing submissions can thereby qualify for computational resources provided by sponsors of the competition to be scored on the full benchmark set.

#### Public workloads

The public workloads are fully specified with the call for submissions. They contain a diverse set of tasks such as image classification, machine translation, speech recognition, or other typical machine learning tasks. For a single tasks there might be multiple models and therefore multiple public workloads. The entire set of public workloads should have a combined runtime of roughly one week on the [competition hardware](#competition-hardware).

The currently nine public workloads are:

|                 | Task                          | Dataset     | Model                        | Loss | Metric   |
|-----------------|-------------------------------|-------------|------------------------------|------|----------|
| **1<br>2**      | Image classification          | ImageNet    | ResNet-50<br>ViT             | CE   | accuracy |
| **3**           | Translation                   | WMT         | Transformer                  | CE   | BLEU     |
| **4<br>5<br>6** | Speech recognition            | LibriSpeech | DeepSpeech<br>Conformer<br>? | CTC  | WER      |
| **7**           | Binary classification         | Criteo 1TB  | DLRMsmall                    | CE   | CE       |
| **8**           | Molecular property prediction | OGBG        | GNN                          | CE   | mAP      |
| **9**           | MRI reconstruction            | fastMRI     | U-Net                        | L1   | SSIM     |

#### Held-out workloads

The held-out workloads function similarly to a holdout test set discouraging submissions that overfit to the [public and known workloads](#public-workloads). Each held-out workload will introduce minor modifications to the data pre-processing and/or model of a public workload. These exact instances of the held-out workloads will be created after the submission deadline. The instructions for creating them will be defined by this working group and made public with the call for submission, to allow the members of this working group to submit as well as ensuring that they do not possess any additional information compared to other submitters.

For each workload in the public workloads, a distribution of possible modifications will be defined. After the submission deadline, a third party will draw a sample from this distribution to generate a held-out workload. Changes could, for example, include changing the number of layers or units (drawn from an interval), swapping the activation function (drawn from a set of applicable functions), or using different data augmentations (drawn from a list of possible pre-processing steps). The sample space should be wide enough to discourage submitters from simply trying them all out, but at the same time should be restricted enough to produce realistic workloads with acceptable achievable performances. If a held-out workload exhibits a significant performance decrease compared to its closest public workload, it might be rejected and instead re-sampled.

The target performance on each held-out workload will be defined by using the performance of the baselines algorithms, [analogously to the public workloads](#defining-target-performance).

#### Qualification set

The qualification set is designed for submitters that may not have the compute resources to self-report on the full set of [public](#public-workloads) and [held-out workloads](#held-out-workloads). They may instead self-report numbers on this smaller qualification set. The best-performing submissions may then qualify for compute sponsorship offering a free evaluation on the full benchmark set and therefore the possibility to win [awards and prize money](#awards-and-prize-money).

The qualification set consists of the same [public workloads](#public-workloads) as mentioned above, except for both workloads on *ImageNet*, the *DeepSpeech* and *Conformer* model on *LibriSpeech*, and the *fastMRI* workload. The remaining four workloads (*WMT*, *LibriSpeech (small)*, *Criteo 1TB*, and *OGBG*) form the qualification set. There are no [held-out workloads](#held-out-workloads) in the qualification set. The qualification set of workloads should have a combined runtime of roughly 24 hours on the [competition hardware](#competition-hardware).

For the [external tuning ruleset](#external-tuning-ruleset), we will only use $1$ study instead of the proposed $5$, when evaluating on the qualification set. The [self-tuning ruleset](#self-tuning-ruleset) will use $5$ studies on the qualification set as well since it is computationally cheaper.

### Scoring

Submissions will be scored based on their required training time to reach the target performance of each workload. The target performance metric can be the same as the loss function but might also be a different workload-specific metric such as the accuracy or BLEU score. The training time includes compilation times for computation graphs and ops that could happen just-in-time during training; all our benchmarks should be fast enough to compile so as not to dramatically impact overall performance. The overall ranking is then determined by summarizing the performances across all [workloads](#workloads), both public and held-out, using [performance profiles](http://www.argmin.net/2018/03/26/performance-profiles/), as explained below.

#### Competition hardware

All scored runs have to be performed on the competition hardware to allow for a fair comparison of training times. The competition hardware has to be chosen to be easily accessible via common cloud computing providers. The exact hardware specification will be specified in the call for submissions and will most likely change with each iteration of the competition. As a placeholder, we are currently planning with 8xV100 GPUs, e.g. the [p3.16xlarge instance on AWS](https://aws.amazon.com/ec2/instance-types/) or the [NVIDIA V100 8 GPUs instance on GCP](https://cloud.google.com/compute/docs/gpus#other_available_nvidia_gpu_models).

#### Defining target performance

A target performance on the validation dataset will be defined for each [workload](#workloads) separately by taking the best performance achievable by one of four standard algorithms (Adam, Nadam, Nesterov Momentum, and Heavy Ball Momentum). These target-setting algorithms will follow the general process of the external tuning ruleset, with a slightly larger tuning budget of $100$ trials to guarantee competitive performance. Once the best hyperparameters are determined, they are repeated $20$ times, with the target performance being the median over those $20$ runs.

Both [tuning rulesets](#tuning) will use the same target performance. The runtime of the target-setting algorithms on each workload will be chosen to match published results and is constrained by the overall time budget of roughly a single week for all public workloads. The `max_runtime` for submissions on each workload is $\frac{1}{3}$ longer than the runtime of the target-setting algorithms.

#### Competition score using performance profiles

We will aggregate the training times of a submission on all workloads using [Performance Profiles](http://www.argmin.net/2018/03/26/performance-profiles/) (originally from [Dolan and Moré](https://arxiv.org/abs/cs/0102001)). Below we surface several relevant definitions from their work for easier readability, before explaining how we integrate the performance profiles to reach a scalar competition score that will be used for ranking submissions.

*Notation:* We have a set $\mathcal{S} = \{s_1, s_2, \dots, s_k\}$ of in total $k$ submissions that we evaluate on a set of $n$ workloads: $\mathcal{W} = \{w_1, w_2, \dots, w_n\}$. For each submission $s$ and each workload $w$ we have a training time score $t_{s,w} \in [0,\infty)$. This is the time it took the submission to reach the target performance on this particular workload.

##### Computing performance ratios

For all workloads and submissions, we first compute their performance ratio $r$, which is defined for a particular submission $\bar{s}$ and a particular workload $\bar{w}$ to be:

$$r_{\bar{s},\bar{w}} = \frac{t_{\bar{s},\bar{w}}}{\min_{s \in \mathcal{S}} t_{s,\bar{w}}} \in [1,\infty)$$

This performance ratio $r(s,w)$ expresses the "time spent by submission $s$ on workload $w$" relative to the "time spent by the best submission on this workload". E.g. If a submission takes twice as long on a particular workload compared to the best submission on this workload it will have a performance ratio of $2$. Lower performance ratios are therefore better, with an optimal ratio of $1$ if the given submission is the fastest on this workload.

##### Building performance profiles

Next, we compute how often a submission is within a factor $\tau \in [1,\infty)$ of the optimal submission. For this, we determine the following function for every submission $\bar{s}$:

$$\rho_{\bar{s}}(\tau) = \left(\frac{1}{n}\right) \cdot \left[\text{number of workloads where}\, r_{\bar{s},w}\leq \tau\right]$$

In other words, we compute the fraction of workloads where a submission $\bar{s}$ is less than $\tau$ away from the optimal submission. The function $\rho_{\bar{s}}(\tau)$ is monotonically increasing with $\tau$ and bounded between $0$ and $1$.

An example of a performance profiles plot is shown below, where we plot $\rho_{\bar{s}}(\tau)$ for seven "submissions":

![Example performance profile](.assets/performance_profiles.png)

##### Integrating performance profiles for the competition score

To get a scalar score that is usable for ranking submissions, we will integrate the performance profiles $\rho_{\bar{s}}(\tau)$ of all submissions to get their competition score $C_{\bar{s}}$, with

$$C_{\bar{s}} = \frac{1}{r_{\text{max}}} \int_{1}^{r_{\text{max}}} \rho_{\bar{s}}(\tau) \,d\tau \in [0, 1].$$

The upper limit of the integral is given by the largest $r(s,w)$ in the competition, i.e. $r_{\text{max}} = \max_{s \in \mathcal{S}} \left(\max_{w \in \mathcal{W}} \left( r_{s,w} \right) \right)$. The integral is normalized by the total integration area, with higher competition scores being better.

##### Alternative scores

Performance profiles and the competition score derived from them, take a bit of effort to explain.
However, we believe that they are fairer and well-supported by research in machine learning and the optimization community. To have some simpler to interpret numbers, e.g. for press releases, we will also release a series of alternative scores.

For a given workload $\bar{w}$, we define the "speedup of a submission $\bar{s}$ over the target-setting reference" as $\frac{t_{\text{ref}, \bar{w}}}{t_{\bar{s}, \bar{w}}}$. For example, if a submission was 2x faster than the target-setting reference, this would be equal to 2. In addition to the raw  $t_{s,w}$ values, we will release the geometric mean of the speedups across all workloads, i.e. $\left(\prod_{w \in \mathcal{W}} \frac{t_{\text{ref}, w}}{t_{\bar{s}, w}}\right)^{\frac{1}{n}}$.

#### Awards and prize money

An awards committee will award a prize for the "*Best Performance*" in each ruleset as well as a "*Jury Award*". The prize for the best-performing submission will be awarded based on the [competition score](#summary-score-using-performance-profiles) on the full benchmark including [held-out workloads](#held-out-workloads). The "*Jury Award*" will favor more out-of-the-box ideas that show great potential, even though the method may not be of practical value with the current landscape of models, software, etc.

The prize money for "*Best Performance*" in a ruleset is 20.000$ each. The winner of the "*Jury Award*" will be awarded 10.000$. We reserve the right to split the prize money and distribute it among multiple submissions.

The chairs of the MLCommons Algorithms Working Group (presently *George Dahl* and *Frank Schneider*) and their institutions (currently *Google Inc.* and the *University of Tübingen*) are ineligible to receive prize money. In addition, all individuals serving on the awards committee and their institutions are ineligible to win prize money (regular attendance at the working group's meetings is a prerequisite for joining the awards committee). A submission with at least one ineligible submitter may still win an award, but the prize money will then be awarded to the top-ranked submission that is eligible for prize money.

## Model Track

🚧 **Coming soon!** 🚧
