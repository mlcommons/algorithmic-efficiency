# MLCommons™ Algorithmic Efficiency

<br />
<p align="center">
<a href="#"><img width="600" img src="https://nextcloud.tuebingen.mpg.de/index.php/apps/files_sharing/publicpreview/FisBKmn9ZtmEp9f?x=1272&y=973&a=true&file=mlc_lockup_black_green.png&scalingup=0" alt="MLCommons Logo"/></a>
</p>

<p align="center">
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

> [MLCommons Algorithmic Efficiency](https://mlcommons.org/en/groups/research-algorithms/) is a benchmark and competition measuring neural network training speedups due to algorithmic improvements in both training algorithms and models. This repository holds the [competition rules](RULES.md) and the benchmark code to run it.

## Installation

1. Create new environment, e.g. via `conda` or `virtualenv`:

   Python minimum requirement >= 3.7

   ```bash
    sudo apt-get install python3-venv
    python3 -m venv env
    source env/bin/activate
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/mlcommons/algorithmic-efficiency.git
   ```

3. Install the `algorithmic_efficiency` package:

   ```bash
   pip3 install -e .
   ```

   Depending on the framework you want to use (e.g. `JAX` or `PyTorch`) you need to install them as well. You could either do this manually or by adding the corresponding options:

   **JAX (GPU)**

   ```bash
   pip3 install -e .[jax-gpu] -f 'https://storage.googleapis.com/jax-releases/jax_releases.html'
   ```

   **JAX (CPU)**

   ```bash
   pip3 install -e .[jax-cpu]
   ```

   **PyTorch**

   ```bash
   pip3 install -e .[pytorch] -f 'https://download.pytorch.org/whl/torch_stable.html'
   ```

   **Development**

   To use the development tools such as `pytest` or `pylint` use the `dev` option:

   ```bash
   pip3 install -e .[dev]
   ```

### Docker

Docker is the easiest way to enable PyTorch/JAX GPU support on Linux since only the [NVIDIA® GPU driver](https://github.com/NVIDIA/nvidia-docker/wiki/Frequently-Asked-Questions#how-do-i-install-the-nvidia-driver) is required on the host machine (the NVIDIA® CUDA® Toolkit does not need to be installed).

#### Docker requirements

- Install [Docker](https://docs.docker.com/get-docker/) on your local host machine.

- For GPU support on Linux, [install NVIDIA Docker support](https://github.com/NVIDIA/nvidia-docker).
  - Take note of your Docker version with docker -v. Versions earlier than 19.03 require nvidia-docker2 and the --runtime=nvidia flag. On versions including and after 19.03, you will use the nvidia-container-toolkit package and the --gpus all flag. Both options are documented on the page linked above.

#### Setup

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

   </details>

## Running a workload

### JAX

```bash
python3 algorithmic_efficiency/submission_runner.py --framework=jax --workload=mnist_jax --submission_path=sample_submissions/mnist/mnist_jax/submission.py --tuning_search_space=sample_submissions/mnist/tuning_search_space.json
```

### PyTorch

```bash
python3 algorithmic_efficiency/submission_runner.py --framework=pytorch --workload=mnist_pytorch --submission_path=sample_submissions/mnist/mnist_pytorch/submission.py --tuning_search_space=sample_submissions/mnist/tuning_search_space.json
```

## Rules

The rules for the MLCommons Algorithmic Efficency benchmark can be found in the seperate [rules document](RULES.md). Suggestions, clarifications and questions can be raised via pull requests.

## Contributing

If you are interested in contributing to the work of the working group, feel free to [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/), open issues, and see the [MLCommons contributing guidelines](CONTRIBUTING.md).
