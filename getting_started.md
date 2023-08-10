# Getting Started

Table of Contents:
- [Set up  and installation](#set-up-and-installation)
- [Download the data](#download-the-data)
- [Develop your submission](#develop-your-submission)
- [Run your submission](#run-your-submission)
    - [Docker](#run-your-submission-in-a-docker-container)
- [Score your submission](#score-your-submission)

## Set up and installation
To get started you will have to make a few decisions and install the repository along with its dependencies. Specifically:
1. Decide if you would like to develop your submission in either Pytorch or Jax.
2. Set up your workstation or VM. We recommend to use a setup similar to the [benchmarking hardware](https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#benchmarking-hardware). 
The specs on the benchmarking machines are:
    -  8 V100 GPUs
    - 240 GB in RAM
    - 2 TB in storage (for datasets). 
3. Install the algorithmic package and dependencies, see [Installation](./README.md#installation).

## Download the data
The workloads in this benchmark use 6 different datasets across 8 workloads. You may choose to download some or all of the datasets as you are developing your submission, but your submission will be scored across all 8 workloads. For instructions on obtaining and setting up the datasets see [datasets/README](https://github.com/mlcommons/algorithmic-efficiency/blob/main/datasets/README.md#dataset-setup).


## Develop your submission
To develop a submission you will write a python module containing your optimizer algorithm. Your optimizer must implement a set of predefined API methods for the initialization and update steps.

### Set up your directory structure (Optional)
Make a submissions subdirectory to store your submission modules e.g. `algorithmic-effiency/submissions/my_submissions`.

### Coding your submission
You can find examples of sumbission modules under `algorithmic-efficiency/baselines` and `algorithmic-efficiency/reference_algorithms`. \
A submission for the external ruleset will consist of a submission module and a tuning search space definition.
1. Copy the template submission module `submissions/template/submission.py` into your submissions directory e.g. in `algorithmic-efficiency/my_submissions`.
2. Implement at least the methods in the template submission module. Feel free to use helper functions and/or modules as you see fit. Make sure you adhere to to the competition rules. Check out the guidelines for [allowed submissions](https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#disallowed-submissions), [disallowed submissions](https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#disallowed-submissions) and pay special attention to the [software dependencies rule](https://github.com/mlcommons/algorithmic-efficiency/blob/main/RULES.md#software-dependencies).
3. Add a tuning configuration e.g. `tuning_search_space.json` file to your submission directory. For the tuning search space you can either:
    1. Define the set of feasible points by defining a value for "feasible_points" for the hyperparameters:
    ```
    {
        "learning_rate": {
            "feasible_points": 0.999
            },
    }
    ```
    For a complete example see [tuning_search_space.json](https://github.com/mlcommons/algorithmic-efficiency/blob/main/reference_algorithms/target_setting_algorithms/imagenet_resnet/tuning_search_space.json).

    2. Define a range of values for quasirandom sampling by specifing a `min`, `max` and `scaling` 
    keys for the hyperparameter:
    ```
    {
        "weight_decay": {
            "min": 5e-3, 
            "max": 1.0, 
            "scaling": "log",
            }
    }
    ```
    For a complete example see [tuning_search_space.json](https://github.com/mlcommons/algorithmic-efficiency/blob/main/baselines/nadamw/tuning_search_space.json). 


## Run your submission

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

**Pytorch**: to score your submission on a workload, from the algorithmic-efficency directory run: 
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

#### Pytorch DDP
We recommend using PyTorch's [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 
when using multiple GPUs on a single node. You can initialize ddp with torchrun. 
For example, on single host with 8 GPUs simply replace `python3` in the above command by:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=N_GPUS
```
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

### Run your submission in a Docker container

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

#### Docker Tips ####

To find the container IDs of running containers
```
docker ps 
```

To see output of the entrypoint script
```
docker logs <container_id> 
```

To enter a bash session in the container
```
docker exec -it <container_id> /bin/bash
```

## Score your submission 
To produce performance profile and performance table:
```bash
python3 scoring/score_submission.py --experiment_path=<path_to_experiment_dir> --output_dir=<output_dir>
```

We provide the scores and performance profiles for the baseline algorithms in the "Baseline Results" section in [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179). 


## Good Luck!
