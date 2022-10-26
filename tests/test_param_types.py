import jax
import pytest

from algorithmic_efficiency.workloads.cifar.cifar_jax.workload import \
    CifarWorkload as JaxCifarWorkload
from algorithmic_efficiency.workloads.cifar.cifar_pytorch.workload import \
    CifarWorkload as PyTorchCifarWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax.workload import \
    Criteo1TbDlrmSmallWorkload as JaxCriteoWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.workload import \
    Criteo1TbDlrmSmallWorkload as PyTorchCriteoWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.workload import \
    FastMRIWorkload as JaxFastMRIWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.workload import \
    FastMRIWorkload as PyTorchFastMRIWorkload
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
from algorithmic_efficiency.workloads.ogbg.ogbg_jax.workload import \
    OgbgWorkload as JaxOgbgWorkload
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.workload import \
    OgbgWorkload as PyTorchOgbgWorkload
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import \
    WmtWorkload as JaxWmtWorkload
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.workload import \
    WmtWorkload as PyTorchWmtWorkload

WORKLOADS = [
    'mnist',
    'cifar',
    'criteo1tb',
    'fastmri',
    'imagenet_resnet',
    'imagenet_vit',
    'wmt',
    'ogbg'
]


@pytest.mark.parametrize('workload', WORKLOADS)
def test_param_types(workload):
  jax_workload, pytorch_workload = get_workload(workload)
  # Compare number of parameter tensors of both models.
  jax_param_types = jax.tree_util.tree_leaves(jax_workload.model_params_types)
  pytorch_param_types = jax.tree_util.tree_leaves(
      pytorch_workload.model_params_types)
  assert len(jax_param_types) == len(pytorch_param_types)

  def count_param_types(param_types):
    types_dict = {}
    for t in param_types:
      if t not in types_dict:
        types_dict[t] = 1
      else:
        types_dict[t] += 1
    return types_dict

  jax_param_types_dict = count_param_types(jax_param_types)
  pytorch_param_types_dict = count_param_types(pytorch_param_types)
  assert jax_param_types_dict.keys() == pytorch_param_types_dict.keys()
  # Check if total number of each type match.
  for key in list(jax_param_types_dict.keys()):
    assert jax_param_types_dict[key] == pytorch_param_types_dict[key]


def get_workload(workload):
  if workload == 'mnist':
    # Init Jax workload.
    jax_workload = JaxMnistWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchMnistWorkload()
  elif workload == 'cifar':
    # Init Jax workload.
    jax_workload = JaxCifarWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchCifarWorkload()
  elif workload == 'criteo1tb':
    # Init Jax workload.
    jax_workload = JaxCriteoWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchCriteoWorkload()
  elif workload == 'fastmri':
    # Init Jax workload.
    jax_workload = JaxFastMRIWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchFastMRIWorkload()
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
  elif workload == 'ogbg':
    # Init Jax workload.
    jax_workload = JaxOgbgWorkload()
    # Init PyTorch workload.
    pytorch_workload = PyTorchOgbgWorkload()
  else:
    raise ValueError(f'Workload {workload} is not available.')
  _ = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  _ = pytorch_workload.init_model_fn([0])
  return jax_workload, pytorch_workload
