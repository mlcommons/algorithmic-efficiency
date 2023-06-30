# MLCommons™ Algorithmic Efficiency

<br />
<p align="center">
<a href="#"><img width="600" img src=".assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2306.07179" target="_blank">Paper (arXiv)</a> •
  <a href="#installation">Installation</a> •
  <a href="RULES.md">Rules</a> •
  <a href="#contributing">Contributing</a> •
  <a href="LICENSE.md">License</a>
</p>

[![CI](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml)
[![Lint](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mlcommons/algorithmic-efficiency/blob/main/LICENSE.md)
[![Code style: yapf](https://img.shields.io/badge/code%20style-yapf-orange)](https://github.com/google/yapf)

---

[MLCommons Algorithmic Efficiency](https://mlcommons.org/en/groups/research-algorithms/) is a benchmark and competition measuring neural network training speedups due to algorithmic improvements in both training algorithms and models. This repository holds the [competition rules](RULES.md) and the benchmark code to run it. For a detailed description of the benchmark design, see our [paper](https://arxiv.org/abs/2306.07179).

# Table of Contents
- [Table of Contents](#table-of-contents)
- [AlgoPerf Benchmark Workloads](#algoperf-benchmark-workloads)
- [Installation](#installation)
   - [Docker](#docker)
- [Getting Started](#getting-started)
- [Rules](#rules)
- [Contributing](#contributing)
- [Citing AlgoPerf Benchmark](#citing-algoperf-benchmark)


## Installation
You can install this package and dependences in a [python virtual environment](#virtual-environment) or use a [Docker container](#install-in-docker) (recommended).

  *TL;DR to install the Jax version for GPU run:*

   ```bash
   pip3 install -e '.[pytorch_cpu]'
   pip3 install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
   pip3 install -e '.[full]'
   ```

  *TL;DR to install the PyTorch version for GPU run:*

   ```bash
   pip3 install -e '.[jax_cpu]'
   pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
   pip3 install -e '.[full]'
   ```
##  Virtual environment
1. Create new environment, e.g. via `conda` or `virtualenv`:

   Python minimum requirement >= 3.8

   ```bash
    sudo apt-get install python3-venv
    python3 -m venv env
    source env/bin/activate
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/mlcommons/algorithmic-efficiency.git
   cd algorithmic-efficiency
   ```

3. Run pip3 install commands above  `algorithmic_efficiency`.

  <details>
  <summary>
  Additional Details
  </summary>
   You can also install the requirements for individual workloads, e.g. via

   ```bash
   pip3 install -e '.[librispeech]'
   ```

   or all workloads at once via

   ```bash
   pip3 install -e '.[full]'
   ```
  </details>

## Docker
We recommend you install the repository by building a Docker image from the Dockerfile in docker/Dockerfile. 
This image will contain the same dependencies we are using to score and run submissions. 

To 

1. Clone this repository:

   ```bash
   git clone https://github.com/mlcommons/algorithmic-efficiency.git
   ```

2. Build Docker

   ```bash
   cd algorithmic-efficiency/ && sudo docker build -t algorithmic-efficiency .
   ```

3. Run Docker

   ```bash
   sudo docker run --gpus all -it --rm -v $PWD:/home/ubuntu/algorithmic-efficiency --ipc=host algorithmic-efficiency
   ```

   Currently docker method installs both PyTorch and JAX


# Getting Started
For instructions on developing and scoring your own algorithm in the benchmark see [Getting Started Document](.)
## Running a workload

See the [`reference_algorithms`](https://github.com/mlcommons/algorithmic-efficiency/tree/main/reference_algorithms) directory for training various algorithm implementations (note that none of these are valid submissions because they have workload-specific logic, so we refer to them as "algorithms" instead of "submissions").

### JAX

```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=/home/znado \
    --experiment_name=baseline \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_jax/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json
```

### PyTorch

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=/home/znado \
    --experiment_name=baseline \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json
```

When using multiple GPUs on a single node it is recommended to use PyTorch's
[distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
To do so, simply replace `python3` by

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N_GPUS
```

where `N_GPUS` is the number of available GPUs on the node. To only see output from the first process, you can run the following to redirect the output from processes 1-7 to a log file:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8
 ```

# Rules

The rules for the MLCommons Algorithmic Efficency benchmark can be found in the seperate [rules document](RULES.md). Suggestions, clarifications and questions can be raised via pull requests.

# Contributing
If you are interested in contributing to the work of the working group, feel free to [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/), open issues, and see the [MLCommons contributing guidelines](CONTRIBUTING.md).


# Note on shared data pipelines between JAX and PyTorch

The JAX and PyTorch versions of the Criteo, FastMRI, Librispeech, OGBG, and WMT workloads are using the same TensorFlow input pipelines. Due to differences in how Jax and PyTorch distribute computations across devices, the PyTorch workloads have an additional overhead for these workloads.

Since we use PyTorch's [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) implementation, there is one Python process for each device. Depending on the hardware and the settings of the cluster, running a TensorFlow input pipeline in each Python process can lead to errors, since too many threads are created in each process. See [this PR thread](https://github.com/mlcommons/algorithmic-efficiency/pull/85) for more details.
While this issue might not affect all setups, we currently implement a different strategy: we only run the TensorFlow input pipeline in one Python process (with `rank == 0`), and [broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast) the batches to all other devices. This introduces an additional communication overhead for each batch. See the [implementation for the WMT workload](https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/wmt/wmt_pytorch/workload.py#L215-L288) as an example.

## Citing AlgoPerf Benchmark
