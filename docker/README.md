Instructions for building a docker image for OGBG : 

```
    cd algorithmic-efficiency
    docker build -t mlcommons:ogbg -f docker/Dockerfile ./
```


To Test the built image you can run :
```
docker run -it mlcommons:ogbg
```

If you want to run docker image in background and run your own commands in it's shell : 

```
docker run -it -d mlcommons:ogbg
docker ps # to get container id

docker exec -it <container_id> /bin/bash 
```
If you're building docker image on a GCP VM (recommended) then do 

```
    ARTIFACT_REGISTRY_URL=us-central1-docker.pkg.dev
    gcloud auth configure-docker $ARTIFACT_REGISTRY_URL
```

To Push built image to artifact registry on GCP do this : 

```
    PROJECT=training-algorithms-external
    REPO=mlcommons-docker-repo
    
    docker tag mlcmmons:ogbg  us-central1-docker.pkg.dev/$PROJECT/$REPO/mlcommons:ogbg
    docker push us-central1-docker.pkg.dev/$PROJECT/$REPO/mlcommons:ogbg
```

This is required when you deploy the built image on a GCP VM

How to avoid sudo for docker ?

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Recommendation : Use a GCP CPU VM to build mlcommons docker image. Do not use cloudshell to build mlcommons docker images as the cloudshell provisioned machine runs out of storage
