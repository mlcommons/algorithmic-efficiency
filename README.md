# MLPerf Algorithmic Efficiency

## Installation

```
sudo apt-get install python3-venv
python3 -m venv env
source env/bin/activate
pip3 install -e .[jax-gpu] -f 'https://storage.googleapis.com/jax-releases/jax_releases.html'
```

## Running a workload
```
python3 submission_runner.py --workload=mnist_jax --submission_path=workloads/mnist/mnist_jax/submission.py
```
Note that the current MNIST example uses `tf.data` loaders and the `Flax` library for `Jax` models, but these may not be required dependencies for other implementations of the spec (e.g. PyTorch).

For pytorch example, run:
```
python3 submission_runner.py --workload=mnist_pytorch --submission_path=workloads/mnist/mnist_pytorch/submission.py
```
