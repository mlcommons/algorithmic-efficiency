# MLCommons™ AlgoPerf: Contributing

## Table of Contents <!-- omit from toc -->

- [Contributing to MLCommons](#contributing-to-mlcommons)
- [Setup for Contributing](#setup-for-contributing)
  - [Setting up a Linux VM on GCP](#setting-up-a-linux-vm-on-gcp)
  - [Installing GPU Drivers](#installing-gpu-drivers)
  - [Authentication for Google Cloud Container Registry](#authentication-for-google-cloud-container-registry)
- [Installation](#installation)
- [Docker Workflows](#docker-workflows)
  - [Pre-built Images on Google Cloud Container Registry](#pre-built-images-on-google-cloud-container-registry)
  - [Trigger Rebuild and Push of Maintained Images](#trigger-rebuild-and-push-of-maintained-images)
    - [Trigger Build and Push of Images on Other Branch](#trigger-build-and-push-of-images-on-other-branch)
  - [GCP Data and Experiment Integration](#gcp-data-and-experiment-integration)
  - [Downloading Data from GCP](#downloading-data-from-gcp)
  - [Saving Experiments to GCP](#saving-experiments-to-gcp)
  - [Getting Information from a Container](#getting-information-from-a-container)
  - [Mounting Local Repository](#mounting-local-repository)
- [Submitting PRs](#submitting-prs)
- [Testing](#testing)
  - [Style Testing](#style-testing)
  - [Unit and Integration Tests](#unit-and-integration-tests)
  - [Regression Tests](#regression-tests)

## Contributing to MLCommons

We invite everyone to look through our technical documentation and codebase and submit issues and pull requests, e.g. for changes, clarifications, or any bugs you might encounter. If you are interested in contributing to the work of the working group and influence the benchmark's design decisions, please [join the weekly meetings](https://mlcommons.org/en/groups/research-algorithms/) and consider becoming a member of the working group.

The best way to contribute to the MLCommons is to get involved with one of our many project communities. You find more information about getting involved with MLCommons [here](https://mlcommons.org/en/get-involved/#getting-started).

Generally we encourage people to become a MLCommons member if they wish to contribute to MLCommons projects, but outside pull requests are very welcome too.

To get started contributing code, you or your organization needs to sign the MLCommons CLA found at the [MLC policies page](https://mlcommons.org/en/policies/). Once you or your organization has signed the corporate CLA, please fill out this [CLA sign up form](https://forms.gle/Ew1KkBVpyeJDuRw67) form to get your specific GitHub handle authorized so that you can start contributing code under the proper license.

MLCommons project work is tracked with issue trackers and pull requests. Modify the project in your own fork and issue a pull request once you want other developers to take a look at what you have done and discuss the proposed changes. Ensure that cla-bot and other checks pass for your Pull requests.

## Setup for Contributing

### Setting up a Linux VM on GCP

If you want to run containers on GCP VMs or store and retrieve Docker images from the Google Cloud Container Registry, please read ahead.
If you'd like to use a Linux VM, you will have to install the correct GPU drivers and the NVIDIA Docker toolkit.
We recommmend to use the Deep Learning on Linux image. Further instructions are based on that.

### Installing GPU Drivers

You can use the `scripts/cloud-startup.sh` as a startup script for the VM. This will automate the installation of the NVIDIA GPU Drivers and NVIDIA Docker toolkit.

### Authentication for Google Cloud Container Registry

To access the Google Cloud Container Registry, you will have to authenticate to the repository whenever you use Docker.
Use the gcloud credential helper as documented [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#cred-helper).

## Installation

If you have not installed the package and dependencies yet see [Installation](/README.md#installation).

To use the development tools such as `pytest` or `pylint` use the `dev` option:

```bash
pip3 install -e '.[dev]'
pre-commit install
```

To get an installation with the requirements for all workloads and development, use the argument `[full_dev]`.

## Docker Workflows

We recommend developing in our Docker image to ensure a consistent environment between developing, testing and scoring submissions.

To get started see also:

- [Installation with Docker](/GETTING_STARTED.md#docker)
- [Running a submission inside a Docker Container](/GETTING_STARTED.md#run-your-submission-in-a-docker-container)

### Pre-built Images on Google Cloud Container Registry

If you want to maintain or use images stored on our Google Cloud Container Registry read this section.
You will have to use an authentication helper to set up permissions to access the repository:

```bash
ARTIFACT_REGISTRY_URL=us-central1-docker.pkg.dev
gcloud auth configure-docker $ARTIFACT_REGISTRY_URL
```

To pull the latest prebuilt image:

```bash
docker pull us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/<image_name>
```

The naming convention for `image_name` is `algoperf_<framework>_<branch>`.
Currently maintained images on the repository are:

- `algoperf_jax_main`
- `algoperf_pytorch_main`
- `algoperf_both_main`
- `algoperf_jax_dev`
- `algoperf_pytorch_dev`
- `algoperf_both_dev`

To reference the pulled image you will have to use the full `image_path`, e.g.
`us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_main`.

### Trigger Rebuild and Push of Maintained Images

To build and push all images (`pytorch`, `jax`, `both`) on maintained branches (`dev`, `main`).

```bash
bash docker/build_docker_images.sh -b <branch>
```

#### Trigger Build and Push of Images on Other Branch

You can also use the above script to build images from a different branch.

1. Push the branch to `mlcommons/algorithmic-efficiency` repository.
2. Run

   ```bash
   bash docker/build_docker_images.sh -b <branch>
   ```

### GCP Data and Experiment Integration

The Docker entrypoint script can transfer data to and from our GCP buckets on our internal GCP project. If you are an approved contributor you can get access to these resources to automatically download the datasets and upload experiment results.
You can use these features by setting the `--internal_contributor` flag to 'true' for the Docker entrypoint script.

### Downloading Data from GCP

To run a docker container that will only download data (if not found on host)

```bash
docker run -t -d \
-v $HOME/data/:/data/ \
-v $HOME/experiment_runs/:/experiment_runs \
-v $HOME/experiment_runs/logs:/logs \
--gpus all \
--ipc=host \
<image_path> \
--dataset <dataset> \
--framework <framework> \
--keep_container_alive <keep_container_alive> \
--internal_contributor true
```

If `keep_container_alive` is `true` the main process on the container will persist after finishing the data download.
This run command is useful if you are developing or debugging.

### Saving Experiments to GCP

If you set the internal collaborator mode to true
experiments will also be automatically uploaded to our GCP bucket under `gs://mlcommons-runs/<experiment_name`.

Command format

```bash
docker run -t -d \
-v $HOME/data/:/data/ \
-v $HOME/experiment_runs/:/experiment_runs \
-v $HOME/experiment_runs/logs:/logs \
--gpus all \
--ipc=host \
<image_path> \
--dataset <dataset> \
--framework <framework> \
--sumbission_path <submission_path> \
--tuning_search_space <tuning_search_space> \
--experiment_name <experiment_name> \
--workload <workload> \
--keep_container_alive <keep_container_alive>
--internal_contributor true \
```

### Getting Information from a Container

To find the container IDs of running containers

```bash
docker ps 
```

To see the logging output

```bash
docker logs <container_id> 
```

To enter a bash session in the container

```bash
docker exec -it <container_id> /bin/bash
```

### Mounting Local Repository

Rebuilding the docker image can become tedious if
you are making frequent changes to the code.
To have changes in your local copy of the algorithmic-efficiency repo be reflected inside the container you can mount the local repository with the `-v` flag.

```bash
docker run -t -d \
-v $HOME/data/:/data/ \
-v $HOME/experiment_runs/:/experiment_runs \
-v $HOME/experiment_runs/logs:/logs \
-v $HOME/algorithmic-efficiency:/algorithmic-efficiency \
--gpus all \
--ipc=host \
<image_path> \
--keep_container_alive true 
```

## Submitting PRs

New PRs will be merged on the dev branch by default, given that they pass the presubmits.

## Testing

We run tests with GitHub Actions, configured in the [.github/workflows](.github/workflows/) folder.

### Style Testing

We run yapf and linting tests on PRs. You can view and fix offending errors with these instructions.

To run the below commands, use the versions installed via `pip install -e '.[dev]'`.

To automatically fix formatting errors, run the following (*WARNING:* this will edit your code, so it is suggested to make a git commit first!):

```bash
yapf -i -r -vv -p algorithmic_efficiency datasets prize_qualification_baselines reference_algorithms tests *.py
```

To sort all import orderings, run the following:

```bash
isort .
```

To just print out all offending import orderings, run the following:

```bash
isort . --check --diff
```

To print out all offending pylint issues, run the following:

```bash
pylint algorithmic_efficiency
pylint datasets
pylint prize_qualification_baselines
pylint reference_algorithms
pylint submission_runner.py
pylint tests
```

### Unit and Integration Tests

We run unit tests and integration tests as part of the of github actions as well.
You can also use `python tests/reference_algorithm_tests.py` to run a single model update and two model evals for each workload using the reference algorithm in `reference_algorithms/target_setting_algorithms/`.

### Regression Tests

We also have regression tests available in [.github/workflows/regression_tests.yml](.github/workflows/regression_tests.yml) that can be run semi-automatically.
The regression tests are shorter end-to-end submissions run in a containerized environment across all 8 workloads, in both the JAX and PyTorch frameworks.
The regression tests run on self-hosted runners and are triggered for pull requests that target the main branch. Typically these PRs will be from the `dev` branch
so the tests will run containers based on images build from the `dev` branch.
To run a regression test:

1. Build and upload latest Docker images from dev branch.

    ```bash
    bash ~/algorithmic-efficiency/docker/build_docker_images.sh -b dev
    ```

2. Turn on the self-hosted runner.
3. Run the self-hosted runner application for the runner to accept jobs.
4. Open a pull request into mian to trigger the workflow.
