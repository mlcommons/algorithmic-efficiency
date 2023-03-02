# Docker Instructions

## General 

### Prerequisites
You may have to install the NVIDIA Container Toolkit so that the containers can locate the NVIDIA drivers and GPUs.

If you are working with a GCP VM w Container Optimized OS setup, you will have to mount the NVIDIA drivers and devices on 
`docker run` command (see below).

### Building Image

From `algorithmic-efficiency/docker/scripts` run:
```
docker build -t algo_effiency_image .
```

### Starting container w end-to-end submission runner
To run the docker container that will download data and run a submisison run:
```
docker run -t -d \
-v $HOME_DIR/data/:/data/ \
-v $HOME_DIR/experiment_runs/:/experiment_runs \
--gpus all \
base_image:latest \
-d <dataset> \
-f <framework> \
-s <submission_path> \
-t <tuning_search_space> \
-e <experiment_name> \
-w <workload> \
-b <debugging_mode> \
```
If debugging_mode is `'true'` the main process on the container will persist after finishing the submission runner.
This will print the container ID to the terminal. 

### Starting a container w automated data download
To run a docker container that will only download data:
```
docker run -t -d \
-v $HOME_DIR/data/:/data/ \
-v $HOME_DIR/experiment_runs/:/experiment_runs \
--gpus all
base_image:latest \
-d <dataset> \
-f <framework> \
-b <debugging_mode> \
```
If debugging_mode is `'true'` the main process on the container will persist after finishing the data download.
This run command is for developers who manually want to run a sumbission or look around.

### Interacting with the container
To find the container IDs of running containers run:
```
docker ps 
```

To see the status of the data download or submission runner run: 
```
docker logs <container_id> 
```

To enter a bash session in the container run:
```
docker exec -it <container_id> /bin/bash
```

## GCP Integration
If you want to run containers on GCP VMs or store and retrieve Docker images from the Google Cloud Container Registry, please read ahead.

### Google Cloud Container Registry 
If you'd like to maintain or use images stored the Google Cloud Container Registry read this section.
You will have to use an authentication helper to set up permissions to access the repository:
```
    ARTIFACT_REGISTRY_URL=us-central1-docker.pkg.dev
    gcloud auth configure-docker $ARTIFACT_REGISTRY_URL
```

To Push built image to artifact registry on GCP do this : 
```
    PROJECT=training-algorithms-external
    REPO=mlcommons-docker-repo
    
    docker tag base_image:latest us-central1-docker.pkg.dev/$PROJECT/$REPO/base_image:latest
    docker push us-central1-docker.pkg.dev/$PROJECT/$REPO/base_image:latest
```

To pull the latest image to GCP run:
```
    PROJECT=training-algorithms-external
    REPO=mlcommons-docker-repo
    docker pull us-central1-docker.pkg.dev/$PROJECT/base_image:latest
```

### Setting up a Linux VM
If you'd like to use a Linux VM, you will have to install the correct GPU drivers and the NVIDIA Docker toolkit.
I recommmend to use the Deep Learning VM image from Google Click to Deploy. Further instructions are based on that.

#### Installing GPU Drivers
You can use the `scripts/cloud-startup.sh` as a startup script for the VM. This will automate the installation of the
NVIDIA GPU Drivers and NVIDIA Docker toolkit.

#### Authentication for Google Cloud Container Registry
To access the Google Cloud Container Registry, you will have to authenticate to the repository whenever you use Docker.
Use the gcloud credential helper as documented [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#cred-helper).

### Setting up a Container Optimized OS VMs on GCP
You may want use a [Container Optimized OS](https://cloud.google.com/container-optimized-os/docs) to run submissions. 
However, the Container Optimized OS does not support CUDA 11.7. If you go down this route,
please adjust the base image in the Dockerfile to CUDA 11.6. 
We don't guarantee compatibility of the `algorithmic_efficiency` package with CUDA 11.6 though.

#### Installing GPU Drivers
To install NVIDIA GPU drivers on container optimized OS you can use the `cos` installer.
Follow instructions [here](https://cloud.google.com/container-optimized-os/docs/how-to/run-gpus)

#### Authentication for Google Cloud Container Registry
To access the Google Cloud Container Registry, you will have to authenticate to the repository whenever you use Docker.
Use a standalone credential helper as documented [here](https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling#cred-helper).

#### cloud-init script
You can automate installation GPU Drivers and uthentication for Cloud Container Registry with a cloud-init script, by passing
the content of the script as `user-data` in the VMs metadata.


## Other Tips and tricks

How to avoid sudo for docker ?

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Recommendation : Use a GCP CPU VM to build mlcommons docker image. Do not use cloudshell to build mlcommons docker images as the cloudshell provisioned machine runs out of storage
