# MLPerf Algorithmic Efficiency

## Installation

Install [Bazel](https://docs.bazel.build/versions/master/install-ubuntu.html):
```
sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
sudo mv bazel.gpg /etc/apt/trusted.gpg.d/
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
sudo apt install bazel
```


```
sudo apt-get install python3-venv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

## Running a workload
```
python3 submission_runner.py --workload=mnist --submission_path=workloads/mnist/submission.py
```
Note that the current MNIST example uses `tf.data` loaders and the `Flax` library for `Jax` models, but these may not be required dependencies for other implementations of the spec (e.g. PyTorch).