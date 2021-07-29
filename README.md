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

---

> [MLCommons Algorithmic Efficiency](https://mlcommons.org/en/groups/research-algorithms/) is a benchmark and competition measuring neural network training speedups due to algorithmic improvements in both training algorithms and models. This repository holds the [competition rules](RULES.md) and the benchmark code to run it.

## Installation

1. Create new environment, e.g. via `conda` or `virtualenv`:

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
   pip3 install -e .[pytorch]
   ```

   </details>

## Running a workload

### JAX

```bash
python3 submission_runner.py --framework=jax --workload=mnist_jax --submission_path=workloads/mnist/mnist_jax/submission.py
```

### PyTorch

```bash
python3 submission_runner.py --framework=pytorch --workload=mnist_pytorch --submission_path=workloads/mnist/mnist_pytorch/submission.py
```

## Rules

The rules for the MLCommons Algorithmic Efficency benchmark can be found in the seperate [rules document](RULES.md). Suggestions, clarifications and questions can be raised via pull requests.

## Contributing

If you are interested in contributing to the work of the working group, feel free to [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/), open issues, and see the [MLCommons contributing guidelines](CONTRIBUTING.md).
