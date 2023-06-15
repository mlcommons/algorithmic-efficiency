# Getting Started

## Workspace set up and installation
First, decide where you would like to develop and run your submission algorithms. We recommend using a setup similar to the competition scoring hardware which will consist out of 8 V100 GPUs, has about 350 GB in RAM and 2 TB in storag for dataset storage. For recommendations on setting up your own Cloud VM see [here](https://github.com/mlcommons/algorithmic-efficiency/blob/main/docker/README.md#gcp-integration).

Once you have your VM set up, clone the repo and make sure you have installed the dependencies. If you have not installed the repository dependencies yet see the [Installation](https://github.com/mlcommons/algorithmic-efficiency/blob/main/README.md) instructions.

## Download the data

The workloads in this benchmark use 6 different datasets across 8 workloads. You may choose to download some or all of the datasets as you are developing your submission, but your submission will be scored across all 8 workloads. For instructions on obtaining and setting up the datasets see [datasets/README](https://github.com/mlcommons/algorithmic-efficiency/blob/main/datasets/README.md#dataset-setup).


## Develop your submission
To develop a submission you will write a python module containing your optimizer algorithm. Your optimizer must implement a set of predefined API methods for the initialization and update steps.

### Set up your directory structure (Optional)

1. Make a subdirectory to store your submission modules e.g. `algorithmic-effiency/my_submissions`.
2. Make a directory to store your experiment results e.g. `~/my_experiments`.

## 



## Run your submission
You can evaluate your submission with the `submission_runner.py` module on one workload at a time. 

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

#### PyTorch DDP
We recommend using PyTorch's [Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) when using multiple GPUs on a single node. You can initialize ddp with torchrun. 
For example, on single host with 8 GPUs simply replace `python3` in the above command by:
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=N_GPUS
```

## Develop your submission 

## Run your submission

## Running a workload

See the [`reference_algorithms/`](https://github.com/mlcommons/algorithmic-efficiency/tree/main/reference_algorithms) dir for training various algorithm implementations (note that none of these are valid submissions because they have workload-specific logic, so we refer to them as "algorithms" instead of "submissions").



