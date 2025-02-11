# MLCommons™ AlgoPerf: Getting Started

## Table of Contents <!-- omit from toc -->

- [Set Up and Installation](#set-up-and-installation)
  - [Python Virtual Environment](#python-virtual-environment)
  - [Docker](#docker)
    - [Building Docker Image](#building-docker-image)
    - [Running Docker Container (Interactive)](#running-docker-container-interactive)
  - [Using Singularity/Apptainer instead of Docker](#using-singularityapptainer-instead-of-docker)
- [Download the Data](#download-the-data)
- [Develop your Submission](#develop-your-submission)
  - [Set Up Your Directory Structure (Optional)](#set-up-your-directory-structure-optional)
  - [Coding your Submission](#coding-your-submission)
- [Run your Submission](#run-your-submission)
  - [Pytorch DDP](#pytorch-ddp)
  - [Run your Submission in a Docker Container](#run-your-submission-in-a-docker-container)
    - [Docker Tips](#docker-tips)
- [Score your Submission](#score-your-submission)
  - [Running workloads](#running-workloads)
- [Submit your Submission](#submit-your-submission)

## Set Up and Installation

To get started you will have to make a few decisions and install the repository along with its dependencies. Specifically:

1. Decide if you would like to develop your submission in either PyTorch or JAX.
2. Set up your workstation or VM. We recommend to use a setup similar to the [benchmarking hardware](/DOCUMENTATION.md#benchmarking-hardware).
The specs on the benchmarking machines are:
    - 8xV100 16GB GPUs
    - 240 GB in RAM
    - 2 TB in storage (for datasets).
3. Install the `algoperf` package and dependencies either in a [Python virtual environment](#python-virtual-environment) or use a [Docker](#docker) (recommended) or [Singularity/Apptainer container](#using-singularityapptainer-instead-of-docker).

### Python Virtual Environment

> **Prerequisites:**
>
> - Python minimum requirement >= 3.11
> - CUDA 12.1
> - NVIDIA Driver version 535.104.05

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

3. Run the following pip3 install commands based on your chosen framework to install `algoperf` and its dependencies.

    For **JAX**:

    ```bash
    pip3 install -e '.[pytorch_cpu]'
    pip3 install -e '.[jax_gpu]' -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
    pip3 install -e '.[full]'
    ```

    For **PyTorch**

    Note: the below command assumes you have CUDA 12.1 installed locally.
    This is the default in the provided Docker image.
    We recommend you match this CUDA version but if you decide to run
    with a different local CUDA version, please find the appropriate wheel
    url to pass to the `pip install` command for `pytorch`.

    ```bash
    pip3 install -e '.[jax_cpu]'
    pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/cu121'
    pip3 install -e '.[full]'
    ```

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

### Docker

We recommend using a Docker container to ensure a similar environment to our scoring and testing environments. Alternatively, a Singularity/Apptainer container can also be used (see instructions below).

> **Prerequisites:**
>
> - NVIDIA Driver version 535.104.05
> - NVIDIA Container Toolkit so that the containers can locate the NVIDIA drivers and GPUs. See instructions [here](https://github.com/NVIDIA/nvidia-docker).

#### Building Docker Image

1. Clone this repository

   ```bash
   cd ~ && git clone https://github.com/mlcommons/algorithmic-efficiency.git
   ```

2. Build Docker image

   ```bash
   cd algorithmic-efficiency/docker
   docker build -t <docker_image_name> . --build-arg framework=<framework>
   ```

   The `framework` flag can be either `pytorch`, `jax` or `both`. Specifying the framework will install the framework specific dependencies.
   The `docker_image_name` is arbitrary.

#### Running Docker Container (Interactive)

To use the Docker container as an interactive virtual environment, you can run a container mounted to your local data and code directories and execute the `bash` program. This may be useful if you are in the process of developing a submission.

1. Run detached Docker container. The `container_id` will be printed if the container is run successfully.

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

    > Note: You may have to use double quotes around `algorithmic-efficiency` [path] in the mounting `-v` flag. If the above command fails try replacing the following line:
    >
    > ```bash
    > -v $HOME/algorithmic-efficiency:/algorithmic-efficiency2 \
    > ```
    >
    > with
    >
    > ```bash
    > -v $HOME"/algorithmic-efficiency:/algorithmic-efficiency" \
    > ```

2. Open a bash terminal

   ```bash
   docker exec -it <container_id> /bin/bash
   ```

### Using Singularity/Apptainer instead of Docker

Since many compute clusters don't allow the usage of Docker due to securtiy concerns and instead encourage the use of [Singularity/Apptainer](https://github.com/apptainer/apptainer) (formerly Singularity, now called Apptainer), we also provide an Apptainer recipe (located at `docker/Singularity.def`) that can be used to build an image by running

```bash
singularity build --fakeroot <singularity_image_name>.sif Singularity.def
```

Note that this can take several minutes. Then, to start a shell session with GPU support (by using the `--nv` flag), we can run

```bash
singularity shell --bind $HOME/data:/data,$HOME/experiment_runs:/experiment_runs \
    --nv <singularity_image_name>.sif
```

Note the `--bind` flag which, similarly to Docker, allows to bind specific paths on the host system and the container, as explained [here](https://docs.sylabs.io/guides/3.7/user-guide/bind_paths_and_mounts.html).

Also note that we generated `Singularity.def` automatically from the `Dockerfile` using [spython](https://github.com/singularityhub/singularity-cli), as follows:

```bash
pip3 install spython
cd algorithmic-efficiency/docker
python scripts/singularity_converter.py -i Dockerfile -o Singularity.def
```

Users that wish to customize their images are invited to check and modify the `Singularity.def` recipe and the `singularity_converter.py` script.

## Download the Data

The workloads in this benchmark use 6 different datasets across 8 workloads. You may choose to download some or all of the datasets as you are developing your submission, but your submission will be scored across all 8 workloads. For instructions on obtaining and setting up the datasets see [datasets/README](/datasets/README.md#dataset-setup).

## Develop your Submission

To develop a submission you will write a Python module containing your training algorithm. Your training algorithm must implement a set of predefined API methods for the initialization and update steps.

### Set Up Your Directory Structure (Optional)

Make a submissions subdirectory to store your submission modules e.g. `algorithmic-effiency/submissions/my_submissions`.

### Coding your Submission

You can find examples of submission modules under `algorithmic-efficiency/prize_qualification_baselines` and `algorithmic-efficiency/reference_algorithms`. \
A submission for the external ruleset will consist of a submission module and a tuning search space definition.

1. Copy the template submission module `submissions/template/submission.py` into your submissions directory e.g. in `algorithmic-efficiency/my_submissions`.
2. Implement at least the methods in the template submission module. Feel free to use helper functions and/or modules as you see fit. Make sure you adhere to to the competition rules. Check out the guidelines for [allowed submissions](/DOCUMENTATION.md#allowed-submissions), [disallowed submissions](/DOCUMENTATION.md#allowed-submissions) and pay special attention to the [software dependencies rule](/DOCUMENTATION.md#software-dependencies).
3. Add a tuning configuration e.g. `tuning_search_space.json` file to your submission directory. For the tuning search space you can either:
    1. Define the set of feasible points by defining a value for "feasible_points" for the hyperparameters:

        ```JSON
        {
            "learning_rate": {
                "feasible_points": 0.999
                },
        }
        ```

        For a complete example see [tuning_search_space.json](/reference_algorithms/target_setting_algorithms/imagenet_resnet/tuning_search_space.json).

    2. Define a range of values for quasirandom sampling by specifing a `min`, `max` and `scaling` keys for the hyperparameter:

        ```JSON
        {
            "weight_decay": {
                "min": 5e-3, 
                "max": 1.0, 
                "scaling": "log",
                }
        }
        ```

        For a complete example see [tuning_search_space.json](/reference_algorithms/paper_baselines/nadamw/tuning_search_space.json).

## Run your Submission

From your virtual environment or interactively running Docker container run your submission with `submission_runner.py`:  

**JAX**: to score your submission on a workload, from the algorithmic-efficency directory run:

```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=<path_to_experiment_dir>\
    --experiment_name=<experiment_name> \
    --submission_path=submissions/my_submissions/submission.py \
    --tuning_search_space=<path_to_tuning_search_space>
```

**PyTorch**: to score your submission on a workload, from the algorithmic-efficency directory run:

```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

### Pytorch DDP

We recommend using PyTorch's [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) when using multiple GPUs on a single node. You can initialize ddp with torchrun. For example, on single host with 8 GPUs simply replace `python3` in the above command by:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=N_GPUS
```

where `N_GPUS` is the number of available GPUs on the node.

So the complete command is:

```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=N_GPUS \
    submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

### Run your Submission in a Docker Container

The container entrypoint script provides the following flags:

- `--dataset` dataset: can be 'imagenet', 'fastmri', 'librispeech', 'criteo1tb', 'wmt', or 'ogbg'. Setting this flag will download data if `~/data/<dataset>` does not exist on the host machine. Required for running a submission.
- `--framework` framework: can be either 'pytorch' or 'jax'. If you just want to download data, this flag is required for `-d imagenet` since we have two versions of data for imagenet. This flag is also required for running a submission.
- `--submission_path` submission_path: path to submission file on container filesystem. If this flag is set, the container will run a submission, so it is required for running a submission.
- `--tuning_search_space` tuning_search_space: path to file containing tuning search space on container filesystem. Required for running a submission.
- `--experiment_name` experiment_name: name of experiment. Required for running a submission.
- `--workload` workload: can be 'imagenet_resnet', 'imagenet_jax', 'librispeech_deepspeech', 'librispeech_conformer', 'ogbg', 'wmt', 'fastmri' or 'criteo1tb'. Required for running a submission.
- `--max_global_steps` max_global_steps: maximum number of steps to run the workload for. Optional.
- `--keep_container_alive` : can be true or false. If`true` the container will not be killed automatically. This is useful for developing or debugging.

To run the docker container that will run the submission runner run:

```bash
docker run -t -d \
-v $HOME/data/:/data/ \
-v $HOME/experiment_runs/:/experiment_runs \
-v $HOME/experiment_runs/logs:/logs \
--gpus all \
--ipc=host \
<docker_image_name> \
--dataset <dataset> \
--framework <framework> \
--submission_path <submission_path> \
--tuning_search_space <tuning_search_space> \
--experiment_name <experiment_name> \
--workload <workload> \
--keep_container_alive <keep_container_alive>
```

This will print the container ID to the terminal.

#### Docker Tips

To find the container IDs of running containers

```bash
docker ps 
```

To see output of the entrypoint script

```bash
docker logs <container_id> 
```

To enter a bash session in the container

```bash
docker exec -it <container_id> /bin/bash
```

## Score your Submission

To score your submission we will score over all fixed workloads, held-out workloads and studies as described in the rules.
We will sample 1 held-out workload per dataset for a total of 6 held-out workloads and will use the sampled held-out workloads in the scoring criteria for the matching fixed base workloads.
In other words, the total number of runs expected for official scoring is:

- for external tuning ruleset: **350** = (8 (fixed workloads) + 6 (held-out workloads)) x 5 (studies) x 5 (trials)
- for self-tuning ruleset: **70** = (8 (fixed workloads) + 6 (held-out workloads)) x 5 (studies)

### Running workloads

To run workloads for (a mock) scoring you may specify a "virtual" list of held-out workloads. It is important to note that the official set of held-out workloads will be sampled by the competition organizers during scoring time.

An example config for held-out workloads is stored in `scoring/held_workloads_example.json`.
To generate a new sample of held out workloads run:

```bash
python3 generate_held_out_workloads.py --seed <optional_rng_seed> --output_filename <output_filename>
```

To run a number of studies and trials over all workload using Docker containers for each run:

```bash
python scoring/run_workloads.py \
--framework <framework> \
--experiment_name <experiment_name> \
--docker_image_url <docker_image_url> \
--submission_path <sumbission_path> \
--tuning_search_space <submission_path> \
--held_out_workloads_config_path held_out_workloads_example.json \
--num_studies <num_studies>
--seed <rng_seed>
```

Note that to run the above script you will need at least the `jax_cpu` and `pytorch_cpu` installations of the `algorithmic-efficiency` package.

During submission development, it might be useful to do faster, approximate scoring (e.g. without `5` different studies or when some trials are missing) so the scoring scripts allow some flexibility.
To simulate official scoring, pass the `--strict=True` flag in `score_submission.py`. To get the raw scores and performance profiles of group of submissions or single submission:

```bash
python score_submissions.py --submission_directory <directory_with_submissions> --output_dir <output_dir> --compute_performance_profiles
```

We provide the scores and performance profiles for the [paper baseline algorithms](/reference_algorithms/paper_baselines/) in the "Baseline Results" section in [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179).

## Submit your Submission

To submit your submission, please create a PR on the [submission repository](https://github.com/mlcommons/submissions_algorithms). You can find more details in the submission repositories [How to Submit](https://github.com/mlcommons/submissions_algorithms?tab=readme-ov-file#how-to-submit) section. The working group will review your PR and select the most promising submissions for scoring.

**Good Luck!**
