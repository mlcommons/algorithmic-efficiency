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
- [Installation](#installation)
   - [Python Virtual Environment](#python-virtual-environment)
   - [Docker](#docker)
- [Getting Started](#getting-started)
- [Rules](#rules)
- [Contributing](#contributing)
- [Diclaimers](#disclaimers)
- [FAQS](#faqs)
- [Citing AlgoPerf Benchmark](#citing-algoperf-benchmark)



## Installation
You can install this package and dependences in a [python virtual environment](#virtual-environment) or use a [Docker/Singularity/Apptainer container](#install-in-docker) (recommended).

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
##  Python virtual environment
Note: Python minimum requirement >= 3.8

To set up a virtual enviornment and install this repository
1. Create new environment, e.g. via `conda` or `virtualenv`

   ```bash
    sudo apt-get install python3-venv
    python3 -m venv env
    source env/bin/activate
   ```

2. Clone this repository

   ```bash
   git clone https://github.com/mlcommons/algorithmic-efficiency.git
   cd algorithmic-efficiency
   ```

3. Run pip3 install commands above to install `algorithmic_efficiency`.

<details>
<summary>
Per workload installations
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
We recommend using a Docker container to ensure a similar environment to our scoring and testing environments.
Alternatively, a Singularity/Apptainer container can also be used (see instructions below).


**Prerequisites for NVIDIA GPU set up**: You may have to install the NVIDIA Container Toolkit so that the containers can locate the NVIDIA drivers and GPUs. 
See instructions [here](https://github.com/NVIDIA/nvidia-docker).

### Building Docker Image
1. Clone this repository

   ```bash
   cd ~ && git clone https://github.com/mlcommons/algorithmic-efficiency.git
   ```

2. Build Docker Image
   ```bash
   cd algorithmic-efficiency/docker
   docker build -t <docker_image_name> . --build-arg framework=<framework>
   ```
   The `framework` flag can be either `pytorch`, `jax` or `both`. Specifying the framework will install the framework specific dependencies.
   The `docker_image_name` is arbitrary.


### Running Docker Container (Interactive)
To use the Docker container as an interactive virtual environment, you can run a container mounted to your local data and code directories and execute the `bash` program. This may be useful if you are in the process of developing a submission.
1. Run detached Docker Container. The `container_id` will be printed if the container is running successfully.
   ```bash
   docker run -t -d \
      -v $HOME/data/:/data/ \
      -v $HOME/experiment_runs/:/experiment_runs \
      -v $HOME/experiment_runs/logs:/logs \
      -v $HOME/algorithmic-efficiency:/algorithmic-efficiency \
      --gpus all \
      --ipc=host \
      <docker_image_name> \
      --keep_container_alive true
   ```
   Note: You may have to use double quotes around `algorithmic-efficiency` [path] in the mounting `-v` flag. If the above command fails try replacing the following line:
   ```bash
   -v $HOME/algorithmic-efficiency:/algorithmic-efficiency2 \
   ``` 
   with 
   ```
   -v $HOME"/algorithmic-efficiency:/algorithmic-efficiency" \
   ```
   - Open a bash terminal
   ```bash
   docker exec -it <container_id> /bin/bash
   ```

### Running Docker Container (End-to-end)
To run a submission end-to-end in a containerized environment see [Getting Started Document](./getting_started.md#run-your-submission-in-a-docker-container).

### Using Singularity/Apptainer instead of Docker
Since many compute clusters don't allow the usage of Docker due to securtiy concerns and instead encourage the use of [Singularity/Apptainer](https://github.com/apptainer/apptainer) (formerly Singularity, now called Apptainer), we also provide instructions on how to build an Apptainer container based on the here provided Dockerfile.

To convert the Dockerfile into an Apptainer definition file, we will use [spython](https://github.com/singularityhub/singularity-cli):
```bash
pip3 install spython
cd algorithmic-efficiency/docker
spython recipe Dockerfile &> Singularity.def
```
Now we can build the Apptainer image by running
```bash
singularity build --fakeroot <singularity_image_name>.sif Singularity.def
```
To start a shell session with GPU support (by using the `--nv` flag), we can run
```bash
singularity shell --nv <singularity_image_name>.sif 
```
Similarly to Docker, Apptainer allows you to bind specific paths on the host system and the container by specifying the `--bind` flag, as explained [here](https://docs.sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html).

# Getting Started
For instructions on developing and scoring your own algorithm in the benchmark see [Getting Started Document](./getting_started.md).
## Running a workload
To run a submission directly by running a Docker container, see [Getting Started Document](./getting_started.md#run-your-submission-in-a-docker-container).

From your virtual environment or interactively running Docker container run:

**JAX**

```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=baselines/adamw/jax/submission.py \
    --tuning_search_space=baselines/adamw/tuning_search_space.json
```

**Pytorch**

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=baselines/adamw/jax/submission.py \
    --tuning_search_space=baselines/adamw/tuning_search_space.json
```
<details>
<summary>
Using Pytorch DDP (Recommended)
</summary>

When using multiple GPUs on a single node it is recommended to use PyTorch's [distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html).
To do so, simply replace `python3` by

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N_GPUS
```

where `N_GPUS` is the number of available GPUs on the node. To only see output from the first process, you can run the following to redirect the output from processes 1-7 to a log file:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8
 ```

So the complete command is for example:
```
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 \
submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=baseline \
    --submission_path=baselines/adamw/jax/submission.py \
    --tuning_search_space=baselines/adamw/tuning_search_space.json
```
</details>


# Rules
The rules for the MLCommons Algorithmic Efficency benchmark can be found in the seperate [rules document](RULES.md). Suggestions, clarifications and questions can be raised via pull requests.

# Contributing
If you are interested in contributing to the work of the working group, feel free to [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/), open issues. See our [CONTRIBUTING.md](CONTRIBUTING.md) for MLCommons contributing guidelines and setup and workflow instructions.


# Disclaimers

## Shared data pipelines between JAX and PyTorch

The JAX and PyTorch versions of the Criteo, FastMRI, Librispeech, OGBG, and WMT workloads are using the same TensorFlow input pipelines. Due to differences in how Jax and PyTorch distribute computations across devices, the PyTorch workloads have an additional overhead for these workloads.

Since we use PyTorch's [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) implementation, there is one Python process for each device. Depending on the hardware and the settings of the cluster, running a TensorFlow input pipeline in each Python process can lead to errors, since too many threads are created in each process. See [this PR thread](https://github.com/mlcommons/algorithmic-efficiency/pull/85) for more details.
While this issue might not affect all setups, we currently implement a different strategy: we only run the TensorFlow input pipeline in one Python process (with `rank == 0`), and [broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast) the batches to all other devices. This introduces an additional communication overhead for each batch. See the [implementation for the WMT workload](https://github.com/mlcommons/algorithmic-efficiency/blob/main/algorithmic_efficiency/workloads/wmt/wmt_pytorch/workload.py#L215-L288) as an example.

# FAQS

## Setup and Platform

### My machine only has one GPU. How can I use this repo?
You can run this repo on a machine with an arbitrary number of GPUs. However, the default batch sizes in our reference algorithms `algorithmic-efficiency/baselines` and `algorithmic-efficiency/reference_algorithms` are tuned for a machine with 8 16GB V100 GPUs. You may run into OOMs if you run these algorithms with fewer than 8 GPUs. If you run into these issues because you are using a machine with less total GPU memory, please reduce the batch sizes for the submission. Note that your final submission must 'fit'
on the benchmarking hardware, so if you are using fewer
GPUs with higher per GPU memory, please monitor your memory usage 
to make make sure it will fit on 8xV100 GPUs with 16GB of VRAM per card.

### How do I run this on my SLURM cluster?
You may run into issues with `sudo` and `docker` on a SLURM cluster. To run the workloads in a SLURM cluster you can use Apptainer (previously Singularity), see this [section](using-singularity/apptainer-instead-of-docker).
### How can I run this on my AWS/GCP/Azure cloud project?
 Depending on your virtual machine, you may have to install the correct GPU drivers and the NVIDIA Docker toolkit. For example, in GCP you will have to do the following.
1. If you don't have a VM instance yet, we recommend creating a
new Compute Instance with the "Deep Learning on Linux" Image in Boot disk options. 
2. To install the NVIDIA Docker toolkit, you can use `scripts/cloud-startup.sh` as a startup script for the VM. This will automate the installation of the NVIDIA GPU Drivers and NVIDIA Docker toolkit.

## Submissions
### Can submission be structured using multiple files?
Yes, your submission can be structured using multiple files. 
### Can I install custom dependencies?
You may use custom dependencies as long as they do not conflict with any of the pinned packages in `algorithmic-efficiency/setup.cfg`. 
To include your custom dependencies in your submission, please include them in a requirements.txt file. Please refer to the [Software dependencies](https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#software-dependencies) section of our rules. 
### How can I know if my code can be run on benchmarking hardware?
The benchmarking hardware specifications are documented in the [Getting Started Document](./getting_started.md).
We recommend monitoring your submission's memory usage so that it does not exceed the available memory 
on the competition hardware. We also recommend to do a dry run using a cloud instance.
### Are we allowed to use our own hardware to self-report the results?
You only have to use the competition hardware for runs that are directly involved in the scoring procedure. This includes all runs for the self-tuning ruleset, but only the runs of the best hyperparameter configuration in each study for the external tuning ruleset. For example, you could use your own (different) hardware to tune your submission and identify the best hyperparameter configuration (in each study) and then only run this configuration (i.e. 5 runs, one for each study) on the competition hardware.

# Citing AlgoPerf Benchmark
If you use the **AlgoPerf** Benchmark in your work, please consider citing:

> [George E. Dahl, Frank Schneider, Zachary Nado, et al.<br/>
> **Benchmarking Neural Network Training Algorithms**<br/>
> *arXiv 2306.07179*](http://arxiv.org/abs/2306.07179)

```bibtex
@misc{dahl2023algoperf,
   title={{Benchmarking Neural Network Training Algorithms}},
   author={Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
   year={2023},
   eprint={2306.07179},
   archivePrefix={arXiv},
   primaryClass={cs.LG}
}
```