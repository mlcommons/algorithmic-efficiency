from itertools import zip_longest

import jax
import numpy as np
import pytest

# isort: skip_file
# pylint:disable=line-too-long
from algorithmic_efficiency.workloads.cifar.cifar_jax.workload import CifarWorkload as JaxCifarWorkload
from algorithmic_efficiency.workloads.cifar.cifar_pytorch.workload import CifarWorkload as PyTorchCifarWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax.workload import Criteo1TbDlrmSmallWorkload as JaxCriteoWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.workload import Criteo1TbDlrmSmallWorkload as PyTorchCriteoWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.workload import FastMRIWorkload as JaxFastMRIWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.workload import FastMRIWorkload as PyTorchFastMRIWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import ImagenetResNetWorkload as JaxImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import ImagenetResNetWorkload as PyTorchImagenetResNetWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import ImagenetVitWorkload as JaxImagenetViTWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import ImagenetVitWorkload as PyTorchImagenetViTWorkload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import LibriSpeechConformerWorkload as JaxLibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.workload import LibriSpeechConformerWorkload as PytorchLibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax.workload import LibriSpeechDeepSpeechWorkload as JaxLibriSpeechDeepSpeechWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.workload import LibriSpeechDeepSpeechWorkload as PytorchLibriSpeechDeepSpeechWorkload
from algorithmic_efficiency.workloads.mnist.mnist_jax.workload import MnistWorkload as JaxMnistWorkload
from algorithmic_efficiency.workloads.mnist.mnist_pytorch.workload import MnistWorkload as PyTorchMnistWorkload
from algorithmic_efficiency.workloads.ogbg.ogbg_jax.workload import OgbgWorkload as JaxOgbgWorkload
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.workload import OgbgWorkload as PyTorchOgbgWorkload
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import WmtWorkload as JaxWmtWorkload
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.workload import WmtWorkload as PyTorchWmtWorkload
# pylint:enable=line-too-long

WORKLOADS = [
    'cifar',
    'criteo1tb',
    'fastmri',
    'imagenet_resnet',
    'imagenet_vit',
    # TODO: make tests work for these.
    # 'librispeech_conformer',
    # 'librispeech_deepspeech',
    'mnist',
    'ogbg',
    'wmt',
]


# Ideally we would match the shapes layer-wise, but for that we
# have to ensure the exact same order of the shapes and that the
# shapes of the weights of the same layer type actually match between
# Jax and PyTorch, which is not always the case.
@pytest.mark.parametrize('workload', WORKLOADS)
def test_param_shapes(workload):
  jax_workload, pytorch_workload = get_workload(workload)
  # Compare number of parameter tensors of both models.
  jax_param_shapes = jax.tree_util.tree_leaves(
      jax_workload.param_shapes.unfreeze())
  pytorch_param_shapes = jax.tree_util.tree_leaves(
      pytorch_workload.param_shapes)
  if workload == 'wmt':
    # The PyTorch transformer for WMT is implemented with fused linear layers
    # for the projection of QKV inside of the MultiheadAttention module.
    # Two weight matrices for each of the two self-attention layers less and one
    # less for the encoder-decoder attention layer -> 5 weight matrices less.
    # We have 6 encoder/decoder layers, hence 30 weight matrices less in total.
    assert len(jax_param_shapes) == len(pytorch_param_shapes) + 30
  else:
    assert len(jax_param_shapes) == len(pytorch_param_shapes)
  # Check if total number of params deduced from shapes match.
  num_jax_params = 0
  num_pytorch_params = 0
  for jax_shape, pytorch_shape in zip_longest(jax_param_shapes,
                                              pytorch_param_shapes):
    num_jax_params += np.prod(jax_shape.shape_tuple)
    if pytorch_shape is not None:
      num_pytorch_params += np.prod(pytorch_shape.shape_tuple)
  assert num_jax_params == num_pytorch_params


def get_workload(workload):
  if workload == 'cifar':
    jax_workload = JaxCifarWorkload()
    pytorch_workload = PyTorchCifarWorkload()
  elif workload == 'criteo1tb':
    jax_workload = JaxCriteoWorkload()
    pytorch_workload = PyTorchCriteoWorkload()
  elif workload == 'fastmri':
    jax_workload = JaxFastMRIWorkload()
    pytorch_workload = PyTorchFastMRIWorkload()
  elif workload == 'imagenet_resnet':
    jax_workload = JaxImagenetResNetWorkload()
    pytorch_workload = PyTorchImagenetResNetWorkload()
  elif workload == 'imagenet_vit':
    jax_workload = JaxImagenetViTWorkload()
    pytorch_workload = PyTorchImagenetViTWorkload()
  elif workload == 'librispeech_conformer':
    jax_workload = JaxLibriSpeechConformerWorkload()
    pytorch_workload = PytorchLibriSpeechConformerWorkload()
  elif workload == 'librispeech_deepspeech':
    jax_workload = JaxLibriSpeechDeepSpeechWorkload()
    pytorch_workload = PytorchLibriSpeechDeepSpeechWorkload()
  elif workload == 'mnist':
    jax_workload = JaxMnistWorkload()
    pytorch_workload = PyTorchMnistWorkload()
  elif workload == 'ogbg':
    jax_workload = JaxOgbgWorkload()
    pytorch_workload = PyTorchOgbgWorkload()
  elif workload == 'wmt':
    jax_workload = JaxWmtWorkload()
    pytorch_workload = PyTorchWmtWorkload()
  else:
    raise ValueError(f'Workload {workload} is not available.')
  _ = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  _ = pytorch_workload.init_model_fn([0])
  return jax_workload, pytorch_workload


if __name__ == '__main__':
  for w in WORKLOADS:
    test_param_shapes(w)
