# MLCommons™ AlgoPerf: Training Algorithms Benchmark

<br />
<p align="center">
<a href="https://mlcommons.org/en/groups/research-algorithms/"><img width="600" img src=".assets/mlc_logo.png" alt="MLCommons Logo"/></a>
</p>

<p align="center">
  <strong><a href="https://github.com/mlcommons/submissions_algorithms">🏆 Leaderboard</a></strong> •
  <strong><a href="/docs/GETTING_STARTED.md">🚀 Getting Started</a></strong> •
  <strong><a href="https://github.com/mlcommons/submissions_algorithms">📥 Submit</a></strong> •
  <strong><a href="/docs/DOCUMENTATION.md">📖 Docs/Rules</a></strong>
  <br>
  <strong><a href="https://arxiv.org/abs/2306.07179" target="_blank">📜 Benchmark Paper</a></strong> •
  <strong><a href="https://openreview.net/forum?id=CtM5xjRSfm" target="_blank">📊 Results Paper</a></strong>
</p>

<p align="center">
    <a href="https://github.com/mlcommons/algorithmic-efficiency/releases"><img alt="Version" src="https://img.shields.io/github/v/tag/mlcommons/algorithmic-efficiency?style=flat-square&label=Version"></a>
    <a href="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/CI.yml"><img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/mlcommons/algorithmic-efficiency/CI.yml?style=flat-square&logo=github&label=CI"></a>
    <a href="https://github.com/mlcommons/algorithmic-efficiency/actions/workflows/linting.yml"><img alt="Linting Status" src="https://img.shields.io/github/actions/workflow/status/mlcommons/algorithmic-efficiency/linting.yml?style=flat-square&logo=github&label=Linting"></a>
    <a href="https://github.com/astral-sh/ruff"><img alt="Code Style Ruff" src="https://img.shields.io/badge/Code%20Style-Ruff-brightgreen?style=flat-square&logo=ruff"></a>
    <a href="LICENSE.md"><img alt="GitHub License" src="https://img.shields.io/github/license/mlcommons/algorithmic-efficiency?style=flat-square&label=License"></a>
    <a href="https://discord.gg/5FPXK7SMt6"><img alt="Discord" src="https://dcbadge.limes.pink/api/server/5FPXK7SMt6?style=flat-square"></a>
</p>

---

The MLCommons™ **AlgoPerf: Training Algorithms benchmark** is designed to find **training algorithms that can train neural networks faster** by rigorously measuring how quickly they reach a specific performance target across a diverse set of deep learning workloads.

When training neural nets, practitioners face many critical yet often opaque decisions: What optimizer to choose? How should its learning rate be tuned? What learning rate schedule should be used? These choices can make or break training, yet the community has lacked a clear, standardized way to identify the state of the art.
Unlike benchmarks focused on hardware or model architecture, AlgoPerf isolates the **training algorithm** itself, which includes the optimizer, regularization, data selection, and hyperparameters like the learning rate schedule. By standardizing the benchmark process, AlgoPerf offers a meaningful apples-to-apples comparison of training algorithms and follows the following **key principles**:

- 🎯 **Fixed Target, Model & Hardware:** Submitted training algorithms must train the models to a [**pre-defined validation performance target**](/docs/DOCUMENTATION.md#workloads) as fast as possible. All submissions use the same model architecture and are run on the same [**standardized hardware**](/docs/DOCUMENTATION.md#benchmarking-hardware) (8x NVIDIA V100 GPUs). This isolates the training algorithm's performance and allows a fair apples-to-apples comparison.
- ⏱️ **Time-To-Result:** Submissions are evaluated based on their "time-to-result", i.e., the total wall-clock time it takes to reach the workload target. This rewards algorithms that provide practical speed-ups for practitioners.
- 🧠 **Diverse Workloads:** The benchmark includes [**8 diverse deep learning workloads**](/docs/DOCUMENTATION.md#workloads) across domains like image classification, speech recognition, and machine translation. A submission's score is computed by aggregating its performance, using [**performance profiles**](/docs/DOCUMENTATION.md#benchmark-score-using-performance-profiles), across all workloads to ensure general-purpose algorithms.
- 📦 **Fully-Specified Algorithms:** Submissions must be complete procedures and thus hyperparameter tuning is treated as part of the algorithm. Submissions can either provide a search space for automated tuning ([**External tuning ruleset**](/docs/DOCUMENTATION.md#external-tuning-ruleset)) or be hyperparameter-free ([**Self-tuning ruleset**](/docs/DOCUMENTATION.md#self-tuning-ruleset)) with any tuning done automatically and "on the clock". This measures an algorithm's _total_ practical cost and provides practitioners with a complete method, eliminating the guesswork of how to apply it.

> [!IMPORTANT]
>
> **We have moved to a rolling leaderboard!**
> We invite you to submit your algorithm for evaluation, see our [**How to Submit**](#how-to-submit) section and the [**submission repository**](https://github.com/mlcommons/submissions_algorithms). The working group will review your submission and, if selected, run it on our hardware and add your results to the official [**AlgoPerf Leaderboard**](https://github.com/mlcommons/submissions_algorithms). **Note: we are currently focusing our efforts on the self-tuning leaderboard to strengthen its competitiveness.**

---

## Table of Contents <!-- omit from toc -->

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Run a Workload](#run-a-workload)
  - [Develop Your Algorithm](#develop-your-algorithm)
- [How to Submit](#how-to-submit)
- [Rules, Documentation \& FAQ](#rules-documentation--faq)
- [Contributing \& Resources](#contributing--resources)
- [Releases \& Roadmap](#releases--roadmap)
- [Training Algorithm Collection](#training-algorithm-collection)
- [Citing Our Work](#citing-our-work)
- [License](#license)

## Getting Started

Follow these steps to run a baseline algorithm and start developing your own submission.
A more detailed guide can be found in the [**Getting Started**](/docs/GETTING_STARTED.md) document.
If you run into any issues, please feel free to contact us.
Either [**file an issue**](https://github.com/mlcommons/algorithmic-efficiency/issues), ask a question on [**our Discord**](https://discord.gg/5FPXK7SMt6) or [**join our weekly meetings**](https://mlcommons.org/en/groups/research-algorithms/).

### Installation

We recommend using the provided [**Docker container**](/docs/GETTING_STARTED.md#docker) to ensure a reproducible environment similar to our scoring environment.
Alternatively, you can install the package and its dependencies in a Python virtual environment.
Both options are described in more detail in the [**Getting Started**](/docs/GETTING_STARTED.md) document.

_TL;DR: Install for JAX on GPU:_

```bash
pip3 install -e '.[pytorch_cpu]'
pip3 install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
pip3 install -e '.[full]'
```

_TL;DR: Install for PyTorch on GPU:_

```bash
pip3 install -e '.[jax_cpu]'
pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'
pip3 install -e '.[full]'
```

### Run a Workload

Use the `submission_runner.py` to run an experiment, i.e., train a workload using a specific training algorithm.
Here's how to run the AdamW baseline on the `mnist` workload.

_TL;DR: Running a JAX workload:_

```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=algorithms/archived_paper_baselines/adamw/jax/submission.py \
    --tuning_search_space=algorithms/archived_paper_baselines/adamw/tuning_search_space.json
```

_TL;DR: Running a PyTorch workload:_

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=mnist \
    --experiment_dir=$HOME/experiments \
    --experiment_name=my_first_experiment \
    --submission_path=algorithms/archived_paper_baselines/adamw/pytorch/submission.py \
    --tuning_search_space=algorithms/archived_paper_baselines/adamw/tuning_search_space.json
```

### Develop Your Algorithm

Now you're ready to create your own `submission.py`! For detailed instructions, FAQs, and technical details, please refer to our documentation:

- [**Getting Started Guide**](/docs/GETTING_STARTED.md): A detailed walkthrough for developing your algorithm.
- [**Benchmark Documentation**](/docs/DOCUMENTATION.md): The complete technical reference including the "benchmark rules" such as allowed and disallowed submissions, FAQs, and technical details such as the API.

## How to Submit

Ready to see how your algorithm stacks up? Submit it to the official AlgoPerf leaderboard!

1. **Develop Your Algorithm:** Create your training algorithm following the API and "rules" described in our [**documentation**](/docs/DOCUMENTATION.md).
2. **Create a Pull Request:** Fork the [**submissions repository**](https://github.com/mlcommons/submissions_algorithms) and create a pull request with your algorithm.
3. **Review and Evaluation:** The MLCommons Algorithms Working Group will review your PR. Based on its potential and our available resources, it may be selected for a **free, official evaluation** on our hardware.
4. **See Your Results:** If selected, we will run your algorithm and add the results to the [**public leaderboard**](https://github.com/mlcommons/submissions_algorithms).

## Rules, Documentation & FAQ

We provide a technical documentation of the benchmark and answer frequently asked questions regarding the benchmarking protocol in a dedicated [**Documentation**](/docs/DOCUMENTATION.md) page. This includes which types of submissions are allowed, a description of the benchmark API, and the entire benchmarking protocol. Please ensure that your submission is compliant with these rules before submitting. Suggestions, clarifications, and questions can be raised via pull requests, by creating an issue, or by reaching out to the [**working group**](mailto:algorithms@mlcommons.org).

For a detailed description and motivation of the initial benchmark design, please refer to our [**Benchmark Paper**](/docs/DOCUMENTATION.md#benchmark-paper).
For the results of the first AlgoPerf competition, please refer to our [**Competition Results Paper**](/docs/DOCUMENTATION.md#competition-results-paper).
See our [**AlgoPerf Leaderboard**](https://github.com/mlcommons/submissions_algorithms) for the latest results of the benchmark and the option to submit your algorithm.

## Contributing & Resources

AlgoPerf is an open, community-driven project organized by the [MLCommons Algorithms Working Group](https://mlcommons.org/en/groups/research-algorithms/). Whether you want to submit an algorithm, report a bug, or help shape the future of the benchmark, we welcome your contributions.

- 🏆 **Submit Your Algorithm:** Ready to compete? Create a pull request in the [**Submissions Repository**](https://github.com/mlcommons/submissions_algorithms).
- 🐞 **Report a Bug:** Found an issue with the codebase? Please [**file an issue**](https://github.com/mlcommons/algorithmic-efficiency/issues) so we can take a look. This also includes any rules changes or clarifications you would like to see.
- 🛠️ **Contribute to the Codebase:** We actively welcome pull requests! If you're interested in implementing new workloads, adding baselines, or fixing bugs please reach out to us. Our [**Contributing Guide**](/docs/CONTRIBUTING.md) offers further contributing guidelines and additional setup and workflow instructions.
- 👥 **Influence the Benchmark:** To contribute to the benchmark's design and direction, please join the [**weekly working group meetings**](https://mlcommons.org/en/groups/research-algorithms/).
- 💬 **Ask a Question:** Have a question or want to discuss ideas? Join the conversation on our [**Discord Server**](https://discord.gg/5FPXK7SMt6) or [**join our weekly meetings**](https://mlcommons.org/en/groups/research-algorithms/).

## Releases & Roadmap

The AlgoPerf benchmark is an actively evolving project designed to keep pace with the rapidly changing field of machine learning. To ensure clarity and reproducibility, we have adopted a unified versioning system: codebase, rules, and leaderboard all share the same `Major.Minor` version. `Patch` versions may differ for minor updates.
All results produced under the same `Major.Minor` version are comparable, making it easy to cite "`AlgoPerf v0.X`" and know exactly which set of rules, code, and submissions are being referenced.

Here is an overview of our key releases and the future roadmap. For a detailed list of changes in each release, see our [**Changelog**](docs/CHANGELOG.md).

- `v0.5` - Inaugural Competition <br> The benchmark as it was run for the first AlgoPerf competition in 2024. The key findings and analysis from this competition are detailed in our [**ICLR 2025 Results Paper**](https://openreview.net/forum?id=CtM5xjRSfm). It serves as a historical reference.
  - **Leaderboard:** Archived at [**AlgoPerf v0.5 Leaderboard**](https://github.com/mlcommons/submissions_algorithms/tree/main/previous_leaderboards/algoperf_v05).
  - **Rules:** The rules are archived at the [**AlgoPerf v0.5 Documentation**](https://github.com/mlcommons/algorithmic-efficiency/blob/v0.5.0/DOCUMENTATION.md).
- `v0.6` - **Current Version** <br> The active and recommended version of the benchmark. It is an improved and streamlined version that fixes important bugs and modifying the benchmarking protocol based on the lessons learned from the competition. **This is the recommended version for all new submissions.**

  - **Key Changes:**
    - A rolling leaderboard now allows for continuous submissions and updates.
    - Reduced computational cost via removing held-out workloads, 3 repetition studies (down from 5), and adjusted runtime budgets.
    - Includes important bug fixes (e.g., batch norm) and API improvements (e.g., `prepare_for_eval` function).
  - **Leaderboard:** The active (but currently limited) leaderboard can be found at [**AlgoPerf v0.6 Leaderboard**](https://github.com/mlcommons/submissions_algorithms).
  - **Rules:** For the current set of rules see [**AlgoPerf v0.6 Documentation**](/docs/DOCUMENTATION.md).

> 🏗️ `v1.0` (Future) - Planned Long-Term Support Release <br> This will be the next major release of the benchmark and a "long-term support" version.
>
> - **Anticipated Features:**
>   - Migrating from `pmap` to `jit` in JAX for better performance and scalability.
>   - Potentially adding a new language model (LM) workload.
>   - Stronger baselines, especially for the self-tuning leaderboard.

## Training Algorithm Collection

This repository also provides a collection of implemented training algorithms with different purposes. These include [**submission templates**](./algorithms/template), [**development examples**](./algorithms/development_algorithms), [**target-setting algorithms**](./algorithms/target_setting_algorithms), [**historical baselines**](./algorithms/archived_paper_baselines), and [**current baselines**](./algorithms/baselines). For a detailed overview of these algorithms and their organization, please refer to the [`algorithms/README.md`](./algorithms/README.md) file. You can also find all benchmark submissions and their results on the official [**Leaderboard**](https://github.com/mlcommons/submissions_algorithms).
These algorithms provide a starting point for developing your own training algorithm and are a great resource for understanding the AlgoPerf benchmark and its API.

## Citing Our Work

If you use the AlgoPerf benchmark, its codebase, or results in your research, please cite our papers.

**Benchmark Paper:**

In this paper, we motivate, describe, and justify the _AlgoPerf: Training Algorithms_ benchmark.

> [Dahl, Schneider, Nado, et al.<br/> > **Benchmarking Neural Network Training Algorithms**<br/> > _arXiv 2306.07179_](http://arxiv.org/abs/2306.07179)

```bibtex
@Misc{Dahl2023AlgoPerf,
  title         = {{Benchmarking Neural Network Training Algorithms}},
  author        = {Dahl, George E. and Schneider, Frank and Nado, Zachary and Agarwal, Naman and Sastry, Chandramouli Shama and Hennig, Philipp and Medapati, Sourabh and Eschenhagen, Runa and Kasimbeg, Priya and Suo, Daniel and Bae, Juhan and Gilmer, Justin and Peirson, Abel L. and Khan, Bilal and Anil, Rohan and Rabbat, Mike and Krishnan, Shankar and Snider, Daniel and Amid, Ehsan and Chen, Kongtao and Maddison, Chris J. and Vasudev, Rakshith and Badura, Michal and Garg, Ankush and Mattson, Peter},
  year          = {2023},
  archiveprefix = {arXiv},
  eprint        = {2306.07179},
}
```

**Competition Results Paper:**

In this paper, we analyze the results of the first AlgoPerf competition.

> [Kasimbeg, Schneider, Eschenhagen, et al.<br/> > **Accelerating neural network training: An analysis of the AlgoPerf competition**<br/> > _ICLR 2025_](https://openreview.net/forum?id=CtM5xjRSfm)

```bibtex
@inproceedings{Kasimbeg2025AlgoPerfResults,
title           = {Accelerating neural network training: An analysis of the {AlgoPerf} competition},
author          = {Kasimbeg, Priya and Schneider, Frank and Eschenhagen, Runa and Bae, Juhan and Sastry, Chandramouli Shama and Saroufim, Mark and Boyuan, Feng and Wright, Less and Yang, Edward Z. and Nado, Zachary and Medapati, Sourabh and Hennig, Philipp and Rabbat, Michael and Dahl, George E.},
booktitle       = {The Thirteenth International Conference on Learning Representations},
year            = {2025},
url             = {https://openreview.net/forum?id=CtM5xjRSfm}
}
```

## License

The _AlgoPerf_ codebase is licensed under the [**Apache License 2.0**](/LICENSE.md). All AlgoPerf benchmark submissions must likewise be open-source under the same [**Apache License 2.0**](https://www.apache.org/licenses/LICENSE-2.0).

---

<p align="center">
<b>MLCommons™ Algorithms Working Group</b> • <a href="https://mlcommons.org/en/groups/research-algorithms/"><strong>Join us!</strong></a>
</p>
