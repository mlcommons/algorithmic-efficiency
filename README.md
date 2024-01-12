# MLCommons™ AlgoPerf: Training Algorithms Benchmark

<br />
<p align="center">
<a href="#"><img width="600" img src=".assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2306.07179" target="_blank">Paper (arXiv)</a> •
  <a href="/CALL_FOR_SUBMISSIONS.md">Call for Submissions</a> •
  <a href="/GETTING_STARTED.md">Getting Started</a> •
  <a href="/COMPETITION_RULES.md">Competition Rules</a> •
  <a href="/DOCUMENTATION.md">Documentation</a> •
  <a href="/CONTRIBUTING.md">Contributing</a>
</p>

[![CI](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml)
[![Lint](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mlcommons/algorithmic-efficiency/blob/main/LICENSE.md)
[![Code style: yapf](https://img.shields.io/badge/code%20style-yapf-orange)](https://github.com/google/yapf)

---

> *AlgoPerf* is a suite of benchmarks and competitions to measure neural network training speedups due to algorithmic improvements in both training algorithms and models. This is the repository for the *AlgoPerf: Training Algorithms benchmark* and its associated competition. It is developed by the [MLCommons Algorithms Working Group](https://mlcommons.org/en/groups/research-algorithms/). This repository holds the [**competition rules**](/COMPETITION_RULES.md), the [**technical documentation**](/DOCUMENTATION.md) of the benchmark, [**getting started guides**](/GETTING_STARTED.md), and the benchmark code. For a detailed description of the benchmark design, see our [**paper**](https://arxiv.org/abs/2306.07179).

---

> [!IMPORTANT]
> Upcoming Deadline:
> Registration deadline to express non-binding intent to submit: **February 28th, 2024**.\
> **If you consider submitting, please fill out the** (mandatory but non-binding) [**registration form**](https://forms.gle/K7ty8MaYdi2AxJ4N8).

## Table of Contents <!-- omit from toc -->

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Call for Submissions](#call-for-submissions)
  - [Competition Rules](#competition-rules)
  - [Technical Documentation of the Benchmark \& FAQs](#technical-documentation-of-the-benchmark--faqs)
- [Contributing](#contributing)
- [License](#license)
- [Paper and Citing the AlgoPerf Benchmark](#paper-and-citing-the-algoperf-benchmark)

## Installation

You can install this package and dependencies in a [Python virtual environment](/GETTING_STARTED.md#python-virtual-environment) or use a [Docker/Singularity/Apptainer container](/GETTING_STARTED.md#docker) (recommended).
We recommend using a Docker container (or alternatively, a Singularity/Apptainer container) to ensure a similar environment to our scoring and testing environments.
Both options are described in detail in the [**Getting Started**](/GETTING_STARTED.md) document.

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

## Getting Started

For detailed instructions on developing and scoring your own algorithm in the benchmark see the [Getting Started](/GETTING_STARTED.md) document.

*TL;DR running a JAX workload:*

```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=reference_algorithms/paper_baselines/adamw/jax/submission.py \
    --tuning_search_space=reference_algorithms/paper_baselines/adamw/tuning_search_space.json
```

*TL;DR running a PyTorch workload:*

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=reference_algorithms/paper_baselines/adamw/jax/submission.py \
    --tuning_search_space=reference_algorithms/paper_baselines/adamw/tuning_search_space.json
```

## Call for Submissions

The [Call for Submissions](/CALL_FOR_SUBMISSIONS.md) announces the first iteration of the AlgoPerf: Training Algorithms competition based on the benchmark by the same name.

### Competition Rules

The competition rules for the *AlgoPerf: Training Algorithms* competition can be found in the separate [**Competition Rules**](/COMPETITION_RULES.md) document.

### Technical Documentation of the Benchmark & FAQs

We provide additional technical documentation of the benchmark and answer frequently asked questions in a separate [**Documentation**](/DOCUMENTATION.md) page. Suggestions, clarifications and questions can be raised via pull requests, creating an issue, or by sending an email to the [working group](mailto:algorithms@mlcommons.org).

## Contributing

We invite everyone to look through our rules, documentation, and codebase and submit issues and pull requests, e.g. for rules changes, clarifications, or any bugs you might encounter. If you are interested in contributing to the work of the working group and influence the benchmark's design decisions, please [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/) and consider becoming a member of the working group.

Our [**Contributing**](/CONTRIBUTING.md) document provides further MLCommons contributing guidelines and additional setup and workflow instructions.

## License

The *AlgoPerf* codebase is licensed under the [Apache License 2.0](/LICENSE.md).

## Paper and Citing the AlgoPerf Benchmark

In our paper ["Benchmarking Neural Network Training Algorithms"](http://arxiv.org/abs/2306.07179) we motivate, describe, and justify the *AlgoPerf: Training Algorithms* benchmark.

If you are using the *AlgoPerf benchmark*, its codebase, baselines, or workloads, please consider citing our paper:

> [Dahl, Schneider, Nado, et al.<br/>
> **Benchmarking Neural Network Training Algorithms**<br/>
> *arXiv 2306.07179*](http://arxiv.org/abs/2306.07179)

```bibtex
@Misc{Dahl2023AlgoPerf,
  title         = {{Benchmarking Neural Network Training Algorithms}},
  author        = {Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.07179},
}
```
