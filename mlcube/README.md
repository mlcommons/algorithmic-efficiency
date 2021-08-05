# MLCommons MNIST JAX Workload


### Connect and run
```
ssh eco-13
tmux attach
# do not attach to docker (we need to create docker containers)
source ~/env/bin/activate
cd ~/mlcube_examples/mlc_algo_efficiency_mnist_jax
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download_imagenette.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train_imagenette.yaml
```

### Run MNIST algo workload
```
python3 submission_runner.py --framework=jax --workload=mnist_jax --submission_path=workloads/mnist/mnist_jax/submission.py

python submission_runner.py --flagfile=
```

### Container
#### Build container
```
cd mlc_algo_efficiency_mnist_jax
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml
```
#### Test container
```
docker run -it mlc_algorithms/mlcube_mnist_jax:0.0.1 bash
```

### submission_runner.py FLAGS
```
SEE flags-nope.py
```

# OLD BELOW

# MNIST MLCube

## Create and initialize python environment
```
virtualenv -p python3 ./env && source ./env/bin/activate && pip install mlcube-docker mlcube-singularity mlcube-ssh
```

## Clone MLCube examples and go to MNIST root directory
```
git clone https://github.com/mlperf/mlcube_examples.git && cd ./mlcube_examples/mnist
```

## Run MNIST MLCube on a local machine with Docker runner
```
# Configure MNIST MLCube
mlcube_docker configure --mlcube=. --platform=platforms/docker.yaml

# Run MNIST training tasks: download data and train the model
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/download.yaml
mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content. Then: 
```
sudo rm -r ./workspace/data ./workspace/download_logs ./workspace/model ./workspace/train_logs   
``` 


## Run MNIST MLCube on a local machine with Singularity runner
```
# Configure MNIST MLCube
mlcube_singularity configure --mlcube=. --platform=platforms/singularity.yaml

# Run MNIST training tasks: download data and train the model
mlcube_singularity run --mlcube=. --platform=platforms/singularity.yaml --task=run/download.yaml
mlcube_singularity run --mlcube=. --platform=platforms/singularity.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content. Then:
```
sudo rm -r ./workspace/data ./workspace/download_logs ./workspace/model ./workspace/train_logs   
``` 


## Run MNIST MLCube on a remote machine with SSH runner
Setup passwordless access to a remote machine. Create and/or update your SSH configuration file (`~/.ssh/config`).
Create an alias for your remote machine. This will enable access for tools like `ssh`, `rsync` and `scp` using 
`mlcube-remote` name instead of actual name or IP address. 
```
Host mlcube-remote
    HostName {{IP_ADDRESS}}
    User {{USER_NAME}}
    IdentityFile {{PATH_TO_IDENTITY_FILE}}
```
Remove results of previous runs. Remove all directories in `workspace/` except `workspace/parameters`.

```
# Configure MNIST MLCube
mlcube_ssh configure --mlcube=. --platform=platforms/ssh.yaml

# Run MNIST training tasks: download data and train the model
mlcube_ssh run --mlcube=. --platform=platforms/ssh.yaml --task=run/download.yaml
mlcube_ssh run --mlcube=. --platform=platforms/ssh.yaml --task=run/train.yaml
```
Go to `workspace/` directory and study its content. Then:
```
sudo rm -r ./workspace/data ./workspace/download_logs ./workspace/model ./workspace/train_logs   
``` 
