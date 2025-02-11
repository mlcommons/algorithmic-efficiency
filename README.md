# MLCommons™ AlgoPerf: Training Algorithms Benchmark

<br />
<p align="center">
<a href="#"><img width="600" img src=".assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

<p align="center">
  <a href="https://github.com/mlcommons/submissions_algorithms">Leaderboard</a> •
  <a href="/docs/GETTING_STARTED.md">Getting Started</a> •
  <a href="https://github.com/mlcommons/submissions_algorithms">Submit</a> •
  <a href="/docs/DOCUMENTATION.md">Documentation</a> •
  <a href="/docs/CONTRIBUTING.md">Contributing</a> •
  <a href="https://arxiv.org/abs/2306.07179" target="_blank">Benchmark</a>/<a href="https://openreview.net/forum?id=CtM5xjRSfm" target="_blank">Results</a> Paper
</p>

[![CI](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml)
[![Lint](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml/badge.svg)](https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/mlcommons/algorithmic-efficiency/blob/main/LICENSE.md)
[![Code style: yapf](https://img.shields.io/badge/code%20style-yapf-orange)](https://github.com/google/yapf)
[![Discord](https://dcbadge.vercel.app/api/server/5FPXK7SMt6?style=flat)](https://discord.gg/5FPXK7SMt6)

---

> This is the repository for the *AlgoPerf: Training Algorithms benchmark* measuring neural network training speedups due to algorithmic improvements.
> It is developed by the [MLCommons Algorithms Working Group](https://mlcommons.org/en/groups/research-algorithms/).
> This repository holds the benchmark code, the benchmark's [**technical documentation**](/docs/DOCUMENTATION.md) and [**getting started guides**](/docs/GETTING_STARTED.md). For a detailed description of the benchmark design, see our [**introductory paper**](https://arxiv.org/abs/2306.07179), for the results of the inaugural competition see our [**results paper**](https://openreview.net/forum?id=CtM5xjRSfm).
>
> **See our [AlgoPerf Leaderboard](https://github.com/mlcommons/submissions_algorithms) for the latest results of the benchmark and to submit your algorithm.**
---

> [!IMPORTANT]
> For the next iteration of the AlgoPerf: Training Algorithms benchmark competition, we are switching to a rolling leaderboard, making a few changes to the competition rules, and also run all selected submissions on our hardware. **To submit your algorithm to the next iteration of the benchmark, please see our [How to Submit](#how-to-submit) section and the [submission repository](https://github.com/mlcommons/submissions_algorithms) which hosts the up to date AlgoPerf leaderboard.**

## Table of Contents <!-- omit from toc -->

- [Installation](#installation)
- [Getting Started](#getting-started)
- [How to Submit](#how-to-submit)
  - [Technical Documentation of the Benchmark \& FAQs](#technical-documentation-of-the-benchmark--faqs)
- [Contributing](#contributing)
- [License](#license)
- [Paper and Citing the AlgoPerf Benchmark](#paper-and-citing-the-algoperf-benchmark)

## Installation

> [!TIP]
> **If you have any questions about the benchmark competition or you run into any issues, please feel free to contact us.** Either [file an issue](https://github.com/mlcommons/algorithmic-efficiency/issues), ask a question on [our Discord](https://discord.gg/5FPXK7SMt6) or [join our weekly meetings](https://mlcommons.org/en/groups/research-algorithms/).

You can install this package and dependencies in a [Python virtual environment](/docs/GETTING_STARTED.md#python-virtual-environment) or use a [Docker/Singularity/Apptainer container](/docs/GETTING_STARTED.md#docker) (recommended).
We recommend using a Docker container (or alternatively, a Singularity/Apptainer container) to ensure a similar environment to our scoring and testing environments.
Both options are described in detail in the [**Getting Started**](/docs/GETTING_STARTED.md) document.

*TL;DR to install the Jax version for GPU run:*

```bash
pip3 install -e '.[pytorch_cpu]'
pip3 install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
pip3 install -e '.[full]'
```

*TL;DR to install the PyTorch version for GPU run:*

```bash
pip3 install -e '.[jax_cpu]'
pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'
pip3 install -e '.[full]'
```

## Getting Started

For detailed instructions on developing your own algorithm in the benchmark see the [Getting Started](/docs/GETTING_STARTED.md) document.

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
    --submission_path=reference_algorithms/paper_baselines/adamw/pytorch/submission.py \
    --tuning_search_space=reference_algorithms/paper_baselines/adamw/tuning_search_space.json
```

## How to Submit

Once you have developed your training algorithm, you can submit it to the benchmark by creating a pull request to the [submission repository](https://github.com/mlcommons/submissions_algorithms), which hosts the AlgoPerf leaderboard. The AlgoPerf working group will review your PR. Based on our available resources and the perceived potential of the method, it will be selected for a free evaluation. If selected, we will run your algorithm on our hardware and update the leaderboard with the results.

### Technical Documentation of the Benchmark & FAQs

We provide a technical documentation of the benchmark and answer frequently asked questions in a separate [**Documentation**](/docs/DOCUMENTATION.md) page. This includes which types of submissions are allowed. Please ensure that your submission is compliant with these rules before submitting. Suggestions, clarifications and questions can be raised via pull requests, creating an issue, or by sending an email to the [working group](mailto:algorithms@mlcommons.org).

## Contributing

We invite everyone to look through our rules, documentation, and codebase and submit issues and pull requests, e.g. for rules changes, clarifications, or any bugs you might encounter. If you are interested in contributing to the work of the working group and influence the benchmark's design decisions, please [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/) and consider becoming a member of the working group.

Our [**Contributing**](/docs/CONTRIBUTING.md) document provides further MLCommons contributing guidelines and additional setup and workflow instructions.

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

If you use the results from the first *AlgoPerf competition*, please consider citing the results paper, as well as the relevant submissions:

> [Kasimbeg, Schneider, Eschenhagen, et al.<br/>
> **Accelerating neural network training: An analysis of the AlgoPerf competition**<br/>
> ICLR 2025](https://openreview.net/forum?id=CtM5xjRSfm)

```bibtex
@inproceedings{Kasimbeg2025AlgoPerfResults,
title           = {Accelerating neural network training: An analysis of the {AlgoPerf} competition},
author          = {Kasimbeg, Priya and Schneider, Frank and Eschenhagen, Runa and Bae, Juhan and Sastry, Chandramouli Shama and Saroufim, Mark and Boyuan, Feng and Wright, Less and Yang, Edward Z. and Nado, Zachary and Medapati, Sourabh and Hennig, Philipp and Rabbat, Michael and Dahl, George E.},
booktitle       = {The Thirteenth International Conference on Learning Representations},
year            = {2025},
url             = {https://openreview.net/forum?id=CtM5xjRSfm}
}
```
