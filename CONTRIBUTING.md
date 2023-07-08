# Contributing

The best way to contribute to the MLCommons is to get involved with one of our many project communities. You find more information about getting involved with MLCommons [here](https://mlcommons.org/en/get-involved/#getting-started).

Generally we encourage people to become a MLCommons member if they wish to contribute to MLCommons projects, but outside pull requests are very welcome too.

To get started contributing code, you or your organization needs to sign the MLCommons CLA found at the [MLC policies page](https://mlcommons.org/en/policies/). Once you or your organization has signed the corporate CLA, please fill out this [CLA sign up form](https://forms.gle/Ew1KkBVpyeJDuRw67) form to get your specific GitHub handle authorized so that you can start contributing code under the proper license.

MLCommons project work is tracked with issue trackers and pull requests. Modify the project in your own fork and issue a pull request once you want other developers to take a look at what you have done and discuss the proposed changes. Ensure that cla-bot and other checks pass for your Pull requests.

# Table of Contents
- [Setup](#setup) 
- [Installation](#installation)
- [Docker workflows](#docker-workflows)
- [Submitting PRs](#submitting-prs)
- [Testing](testing)


# Setup 
## GCP Integration
If you want to run containers on GCP VMs or store and retrieve Docker images from the Google Cloud Container Registry, please read ahead.

## Setting up a Linux VM
If you'd like to use a Linux VM, you will have to install the correct GPU drivers and the NVIDIA Docker toolkit.
We recommmend to use the Deep Learning on Linux image. Further instructions are based on that.

### Installing GPU Drivers
You can use the `scripts/cloud-startup.sh` as a startup script for the VM. This will automate the installation of the
NVIDIA GPU Drivers and NVIDIA Docker toolkit.

### Authentication for Google Cloud Container Registry
To access the Google Cloud Container Registry, you will have to authenticate to the repository whenever you use Docker.
Use the gcloud credential helper as documented [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#cred-helper).

## Setting up a Container Optimized OS VMs on GCP
You may want use a [Container Optimized OS](https://cloud.google.com/container-optimized-os/docs) to run submissions. 
However, the Container Optimized OS does not support CUDA 11.7. If you go down this route,
please adjust the base image in the Dockerfile to CUDA 11.6. 
We don't guarantee compatibility of the `algorithmic_efficiency` package with CUDA 11.6 though.

### Installing GPU Drivers
To install NVIDIA GPU drivers on container optimized OS you can use the `cos` installer.
Follow instructions [here](https://cloud.google.com/container-optimized-os/docs/how-to/run-gpus)

### Authentication for Google Cloud Container Registry
To access the Google Cloud Container Registry, you will have to authenticate to the repository whenever you use Docker.
Use a standalone credential helper as documented [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#cred-helper).

### cloud-init script
You can automate installation GPU Drivers and authentication for Cloud Container Registry with a cloud-init script, by passing
the content of the script as `user-data` in the VMs metadata.

# Installation
If you have not installed the package and dependencies yet see [Installation](./README.md#installation).

To use the development tools such as `pytest` or `pylint` use the `dev` option:

```bash
pip3 install -e '.[dev]'
pre-commit install
```

To get an installation with the requirements for all workloads and development, use the argument `[full_dev]`.



# Docker workflows
We recommend developing in our Docker image to ensure a consistent environment between developing, testing and scoring submissions. 

To get started see:
- [Installation with Docker](./README#docker) 
- [Running a submission inside a Docker Container](./getting_started.md#run-your-submission-in-a-docker-container)

Other resources:
- [Pre-built Images on Google Cloud Container Registry](#pre-built-images-on-google-cloud-container-registry)
- [GCP Data and Experiment Integration](#gcp-integration) 
    - [Downloading Data from GCP](#downloading-data-from-gcp)
    - [Saving Experiments Results to GCP](#saving-experiments-to-gcp)
- [Getting Information from a Container](#getting-information-from-a-container)
- [Mounting local repository](mounting-local-repository)


## Pre-built Images on Google Cloud Container Registry 
If you'd like to maintain or use images stored on our Google Cloud Container Registry read this section.
You will have to use an authentication helper to set up permissions to access the repository:
```
ARTIFACT_REGISTRY_URL=us-central1-docker.pkg.dev
gcloud auth configure-docker $ARTIFACT_REGISTRY_URL
```
Set the project and repo
```
PROJECT=training-algorithms-external
REPO=mlcommons-docker-repo
```

To push built image to artifact registry on GCP 
```

docker tag <image_name> us-central1-docker.pkg.dev/$PROJECT/$REPO/<image_name>
docker push us-central1-docker.pkg.dev/$PROJECT/$REPO/<image_name>
```

To pull the latest image to GCP run:
```
docker pull us-central1-docker.pkg.dev/$PROJECT/$REPO/<image_name>
```

The naming convention for `image_name` is `algoperf_<framework>_<branch>`. 

To build and push all images (`pytorch`, `jax`, `both`) to our GCP artifact registry run.
```
bash docker/build_docker_images.sh -b <branch>
```
We will maintain the dev and main images in this way.

## GCP Data and Experiment Integration
The Docker entrypoint script can communicate with
our GCP buckets on our internal GCP project. If
you are an approved contributor you can get access to these resources to automatically download the datasets and upload experiment results. 
You can use these features by setting the `-i` flag (for internal collaborator) to 'true' for the Docker entrypoint script.

### Downloading Data from GCP
To run a docker container that will only download data (if not found on host)
```
docker run -t -d \
-v $HOME_DIR/data/:/data/ \
-v $HOME_DIR/experiment_runs/:/experiment_runs \
-v $HOME_DIR/experiment_runs/logs:/logs \
--gpus all \
--ipc=host \
<docker_image_name> \
-d <dataset> \
-f <framework> \
-b <debugging_mode> \
-i true
```
If debugging_mode is `true` the main process on the container will persist after finishing the data download.
This run command is useful if you manually want to run a sumbission or look around.

### Saving Experiments to GCP
If you set the internal collaborator mode to true
experiments will also be automatically uploaded to our GCP bucket under `gs://mlcommons-runs/<experiment_name`.

Command format
```
docker run -t -d \
-v $HOME_DIR/data/:/data/ \
-v $HOME_DIR/experiment_runs/:/experiment_runs \
-v $HOME_DIR/experiment_runs/logs:/logs \
--gpus all \
--ipc=host \
<docker_image_name> \
-d <dataset> \
-f <framework> \
-s <submission_path> \
-t <tuning_search_space> \
-e <experiment_name> \
-w <workload> \
-b <debug_mode>
-i true \
```

## Getting Information from a Container
To find the container IDs of running containers
```
docker ps 
```

To see the logging output
```
docker logs <container_id> 
```

To enter a bash session in the container
```
docker exec -it <container_id> /bin/bash
```

## Mounting Local Repository
Rebuilding the docker image can become tedious if
you are making frequent changes to the code.
To have changes in your local copy of the algorithmic-efficiency repo be reflected inside the container you can mount the local repository with the `-v` flag. 
```
docker run -t -d \
-v $HOME_DIR/data/:/data/ \
-v $HOME_DIR/experiment_runs/:/experiment_runs \
-v $HOME_DIR/experiment_runs/logs:/logs \
-v $HOME_DIR/algorithmic-efficiency:/algorithmic-efficiency \
--gpus all \
--ipc=host \
<docker_image_name> \
```
# Submitting PRs 
New PRs will be merged on the dev branch by default, given that they pass the presubmits.

# Testing
We run tests with GitHub Actions, configured in the [.github/workflows](https://github.com/mlcommons/algorithmic-efficiency/tree/main/.github/workflows) folder.

## Style Testing
We run yapf and linting tests on PRs. You can view and fix offending errors with these instructions.

To run the below commands, use the versions installed via `pip install -e '.[dev]'`.

To automatically fix formatting errors, run the following (*WARNING:* this will edit your code, so it is suggested to make a git commit first!):
```bash
yapf -i -r -vv -p algorithmic_efficiency baselines datasets reference_algorithms tests *.py
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
pylint baselines
pylint datasets
pylint reference_algorithms
pylint submission_runner.py
pylint tests
```

## Unit and integration tests
We run unit tests and integration tests as part of the of github actions as well. 
You can also use `python tests/reference_algorithm_tests.py` to run a single model update and two model evals for each workload using the reference algorithm in `reference_algorithms/development_algorithms/`.

## Regression tests
We also have regression tests available in [.github/workflows/regression_tests.yml](https://github.com/mlcommons/algorithmic-efficiency/tree/main/.github/workflows/regression_tests.yml) that can be run semi-automatically.
The regression tests are shorter end-to-end submissions run in a containerized environment across all 8 workloads, in both the jax and pytorch frameworks. 
The regression tests run on self-hosted runners and are triggered for pull requests that target the main branch.
To trigger a regression test:
1. Turn on the self-hosted runner.
2. Run the self-hosted runner application for the runner to accept jobs.
3. Open a pull request into mian to trigger the workflow.
