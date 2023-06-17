# Getting Started

## Workspace set up and installation
First, decide where you would like to develop and run your submission algorithms. We recommend using a setup similar to the competition scoring hardware which will consist out of 8 V100 GPUs, has about 350 GB in RAM and 2 TB in storag for dataset storage. For recommendations on setting up your own Cloud VM see [here](https://github.com/mlcommons/algorithmic-efficiency/blob/main/docker/README.md#gcp-integration).

Once you have your VM set up, clone the repo and make sure you have installed the dependencies. If you have not installed the repository dependencies yet see the [Installation](https://github.com/mlcommons/algorithmic-efficiency/blob/main/README.md) instructions.

## Download the data

The workloads in this benchmark use 6 different datasets across 8 workloads. You may choose to download some or all of the datasets as you are developing your submission, but your submission will be scored across all 8 workloads. For instructions on obtaining and setting up the datasets see [datasets/README](https://github.com/mlcommons/algorithmic-efficiency/blob/main/datasets/README.md#dataset-setup).


## Develop your submission
To develop a submission you will write a python module containing your optimizer algorithm. Your optimizer must implement a set of predefined API methods for the initialization and update steps.

### Set up your directory structure (Optional)

1. Make a submissions subdirectory to store your submission modules e.g. `algorithmic-effiency/my_submissions`.
2. Make an experiment directory to store your experiment results e.g. `~/my_experiments`.

### Coding your submission
#TODO(kasimbeg): If we want use this submission_template script where should we store it in the repo?

You can find examples of sumbission modules under `algorithmic-efficiency/baselines` and `algorithmic-efficiency/reference_algorithms`. \
A submission for the external ruleset will consist of a submission module and a tuning search space definition.
1. Clone the submission_template.py script into your submissions directory e.g. in `algorithmic-efficiency/my_submissions`.
2. Implement at least the methods in the submission_template. Feel free to use helper functions and/or modules as you see fit.
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

    2. Define a range of values for halton sampling by specifing a "min", "max" and "scaling" 
        keys for the hyperparameter:
        ```
        {
            "weight_decay": {
                "min": 5e-3, "max": 1.0, "scaling": "log"
                }
        }
        ```
        For a complete example see[tuning_search_space.json](https://github.com/mlcommons/algorithmic-efficiency/blob/main/baselines/nadamw/tuning_search_space.json). 


## Run your submission
TODO(kasimbeg): Add script to run all workloads?

You can evaluate your submission with the `submission_runner.py` module on one workload at a time. 

### JAX submissions
To score your submission on a workload, from the algorithmic-efficency directory run: 
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=mnist \
    --experiment_dir=<path_to_experiment_dir>\
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

### PyTorch submissions
To score your submission on a workload, from the algorithmic-efficency directory run: 
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

#### PyTorch DDP
We recommend using PyTorch's [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) when using multiple GPUs on a single node. You can initialize ddp with torchrun. 
For example, on single host with 8 GPUs simply replace `python3` in the above command by:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=N_GPUS
```
So the complete command is:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=N_GPUS \
submission_runner.py \
    --framework=pytorch \
    --workload=<workload> \
    --experiment_dir=<path_to_experiment_dir> \
    --experiment_name=<experiment_name> \
    --submission_path=<path_to_submission_module> \
    --tuning_search_space=<path_to_tuning_search_space>
```

## Good Luck!


