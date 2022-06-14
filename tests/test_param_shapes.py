import jax
import numpy as np
import pytest
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import \
    ImagenetResNetWorkload as JaxImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import \
    ImagenetResNetWorkload as PyTorchImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetVitWorkload as JaxImagenetViTWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetVitWorkload as PyTorchImagenetViTWorkload
from algorithmic_efficiency.workloads.mnist.mnist_jax.workload import \
    MnistWorkload as JaxMnistWorkload
from algorithmic_efficiency.workloads.mnist.mnist_pytorch.workload import \
    MnistWorkload as PyTorchMnistWorkload
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import \
    WmtWorkload as JaxWmtWorkload
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.workload import \
    WmtWorkload as PyTorchWmtWorkload

WORKLOADS = ['mnist', 'imagenet_resnet', 'imagenet_vit', 'wmt']


# Ideally we would match the shapes layer-wise, but for that we
# have to ensure the exact same order of the shapes and that the
# shapes of the weights of the same layer type actually match between
# Jax and PyTorch, which is not always the case.
@pytest.mark.parametrize('workload', WORKLOADS)
def test_param_shapes(workload):
  jax_workload, pytorch_workload = get_workload(workload)
  # Compare number of parameter tensors of both models.
  jax_param_shapes = jax.tree_leaves(jax_workload.param_shapes.unfreeze())
  pytorch_param_shapes = jax.tree_leaves(pytorch_workload.param_shapes)
  assert len(jax_param_shapes) == len(pytorch_param_shapes)
  # Check if total number of params deduced from shapes match.
  num_jax_params = 0
  num_pytorch_params = 0
  for jax_shape, pytorch_shape in zip(jax_param_shapes, pytorch_param_shapes):
    num_jax_params += np.prod(jax_shape.shape_tuple)
    num_pytorch_params += np.prod(pytorch_shape.shape_tuple)
  assert num_jax_params == num_pytorch_params


def get_workload(workload):
  if workload == 'mnist':
    # Init Jax workload.
    jax_workload = JaxMnistWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchMnistWorkload()
  elif workload == 'imagenet_resnet':
    # Init Jax workload.
    jax_workload = JaxImagenetResNetWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchImagenetResNetWorkload()
  elif workload == 'imagenet_vit':
    # Init Jax workload.
    jax_workload = JaxImagenetViTWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchImagenetViTWorkload()
  elif workload == 'wmt':
    # Init Jax workload.
    jax_workload = JaxWmtWorkload()
    jax_workload._global_batch_size = 128
    # Init PyTorch workload.
    pytorch_workload = PyTorchWmtWorkload()
  else:
    raise ValueError(f'Workload {workload} is not available.')
  _ = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  _ = pytorch_workload.init_model_fn([0])
  return jax_workload, pytorch_workload
