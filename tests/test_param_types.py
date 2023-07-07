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
    'cifar',
    'criteo1tb',
    'fastmri',
    'imagenet_resnet',
    'imagenet_vit',
    'mnist',
    'ogbg',
    'wmt',
]


def count_param_types(param_types):
  types_dict = {}
  for t in param_types:
    if t not in types_dict:
      types_dict[t] = 1
    else:
      types_dict[t] += 1
  return types_dict


@pytest.mark.parametrize('workload', WORKLOADS)
def test_param_types(workload):
  jax_workload, pytorch_workload = get_workload(workload)
  # Compare number of parameter tensors of both models.
  jax_param_types = jax.tree_util.tree_leaves(jax_workload.model_params_types)
  pytorch_param_types = jax.tree_util.tree_leaves(
      pytorch_workload.model_params_types)
  assert len(jax_param_types) == len(pytorch_param_types)

  jax_param_types_dict = count_param_types(jax_param_types)
  pytorch_param_types_dict = count_param_types(pytorch_param_types)
  # Check if total number of each type match.
  mismatches = ''
  for key in jax_param_types_dict.keys():
    jax_count = jax_param_types_dict.get(key, 0)
    pytorch_count = pytorch_param_types_dict.get(key, 0)
    if jax_count != pytorch_count:
      mismatches += f'\nKey: {key}, Jax {jax_count} != Pytorch {pytorch_count}.'
  if mismatches:
    raise ValueError(f'On workload {workload}, count mismatch: {mismatches}.')


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


if __name__ == '__main__':
  for workload in WORKLOADS:
    test_param_types(workload)
