# MLCube

MLCube™ is a project that reduces friction for machine learning by ensuring that models are easily portable and reproducible.

More information about MLCube™ can be found here:
- https://github.com/mlcommons/mlcube
- https://github.com/mlcommons/mlcube_examples

## Installation
```
pip install mlcube mlcube-docker
```

## Usage on local machine with Docker runner

Build our MLCube container:
```
$ cd algorithmic-efficiency
$ mlcube_docker configure --mlcube=. --platform=mlcube/platforms/docker.yaml
```

Run our tasks:
```
$ cd algorithmic-efficiency/mlcube

# Run MNIST Jax
$ mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/mnist_jax.yaml

# Run MNIST PyTorch
$ mlcube_docker run --mlcube=. --platform=platforms/docker.yaml --task=run/mnist_pytorch.yaml
```

In the `./workspace/` directory you will find output log files.