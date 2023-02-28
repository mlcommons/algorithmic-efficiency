# Docker Instructions

## General 

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
base_image:latest \
-d <dataset> \
-f <framework> \
-b <debugging_mode> \
```
If debugging_mode is `'true'` the main process on the container will persist after finishing the data download.
This run command is for developers who manually want to run a sumbission.

### Interacting with the container
You can find the container IDs of running containers by running:
```
docker ps 
```

To see the status of the data download or submission runner, run: 
```
docker logs <container_id> 
```

To enter a bash session in the container, run:
```
docker exec -it <container_id> /bin/bash
```

## GCP Integration 

If you're building docker image on a GCP VM (recommended) then do 
```
    ARTIFACT_REGISTRY_URL=us-central1-docker.pkg.dev
    gcloud auth configure-docker $ARTIFACT_REGISTRY_URL
```

To Push built image to artifact registry on GCP do this : 

```
    PROJECT=training-algorithms-external
    REPO=mlcommons-docker-repo
    
    docker tag algo_efficiency_image:latest us-central1-docker.pkg.dev/$PROJECT/$REPO/mlcommons:ogbg
    docker push us-central1-docker.pkg.dev/$PROJECT/$REPO/algo_efficiency_image:latest
```

To pull the latest image to GCP run:
```
    PROJECT=training-algorithms-external
    REPO=mlcommons-docker-repo
    docker pull us-central1-docker.pkg.dev/$PROJECT/algo_efficiency_image:latest
```
This is required when you deploy the built image on a GCP VM

## Other Tips and tricks

How to avoid sudo for docker ?

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Recommendation : Use a GCP CPU VM to build mlcommons docker image. Do not use cloudshell to build mlcommons docker images as the cloudshell provisioned machine runs out of storage
