import jax
import pytest

from absl import logging
from algorithmic_efficiency import spec

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
    'librispeech_conformer',
    'librispeech_deepspeech',
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


def _count_mismatches(jax_param_types_dict, pytorch_param_types_dict, keys):
  mismatches = ''
  for key in keys:
    jax_count = jax_param_types_dict.get(key, 0)
    pytorch_count = pytorch_param_types_dict.get(key, 0)
    if jax_count != pytorch_count:
      mismatches += f'\nKey: {key}, Jax {jax_count} != Pytorch {pytorch_count}.'
  return mismatches


def _check_attention_qkv_match(jax_param_types_dict, pytorch_param_types_dict):
  # Sometimes one framework will implement QKV as a single parameter, so we need
  # to make sure there are the same number of QKV params as Q, K, V.
  num_qkv = {
      'jax': jax_param_types_dict.get(spec.ParameterType.ATTENTION_QKV, 0),
      'pytorch': pytorch_param_types_dict.get(
          spec.ParameterType.ATTENTION_QKV, 0),
  }
  num_q = {
      'jax': jax_param_types_dict.get(spec.ParameterType.ATTENTION_Q, 0),
      'pytorch': pytorch_param_types_dict.get(
          spec.ParameterType.ATTENTION_Q, 0),
  }
  num_k = {
      'jax': jax_param_types_dict.get(spec.ParameterType.ATTENTION_K, 0),
      'pytorch': pytorch_param_types_dict.get(
          spec.ParameterType.ATTENTION_K, 0),
  }
  num_v = {
      'jax': jax_param_types_dict.get(spec.ParameterType.ATTENTION_V, 0),
      'pytorch': pytorch_param_types_dict.get(
          spec.ParameterType.ATTENTION_V, 0),
  }
  num_bias = {
      'jax': jax_param_types_dict.get(spec.ParameterType.ATTENTION_BIAS, 0),
      'pytorch': pytorch_param_types_dict.get(
          spec.ParameterType.ATTENTION_BIAS, 0),
  }
  qkv_match = num_qkv['jax'] == num_qkv['pytorch']
  q_match = num_q['jax'] == num_q['pytorch']
  k_match = num_k['jax'] == num_k['pytorch']
  v_match = num_v['jax'] == num_v['pytorch']
  bias_match = num_bias['jax'] == num_bias['pytorch']
  qkv_match = qkv_match and q_match and k_match and v_match and bias_match

  # We subtract 2 * num_qkv from the number of biases because there are 2
  # missing for each of q, k, v.
  jax_qkv_match = (
      num_q['pytorch'] == num_k['pytorch'] == num_v['pytorch'] ==
      num_qkv['jax'] and
      (num_qkv['jax'] != 0 and
       (num_bias['pytorch'] - 2 * num_qkv['jax']) == num_bias['jax']))
  pytorch_qkv_match = (
      num_q['jax'] == num_k['jax'] == num_v['jax'] == num_qkv['pytorch'] and
      (num_qkv['pytorch'] != 0 and
       (num_bias['jax'] - 2 * num_qkv['pytorch']) == num_bias['pytorch']))
  qkv_match = qkv_match or jax_qkv_match or pytorch_qkv_match
  return qkv_match


@pytest.mark.parametrize('workload', WORKLOADS)
def test_param_types(workload_name):
  logging.info(f'Testing workload {workload_name}...')
  jax_workload, pytorch_workload = get_workload(workload_name)

  # Compare number of parameter tensors of both models.
  jax_param_types = jax.tree_util.tree_leaves(jax_workload.model_params_types)
  pytorch_param_types = jax.tree_util.tree_leaves(
      pytorch_workload.model_params_types)

  jax_param_types_dict = count_param_types(jax_param_types)
  pytorch_param_types_dict = count_param_types(pytorch_param_types)

  # Jax fuses LSTM cells together, whereas PyTorch exposes all the weight
  # parameters, and there are two per cell, for each of the forward and backward
  # directional LSTMs, and there are 6 layers of LSTM in librispeech_deepspeech,
  # compared to the 6 Jax LSTM weights.
  #
  # We also subtract an additional 6 biases because the LSTM biases are
  # concatenated to the weights in Jax.
  if workload_name == 'librispeech_deepspeech':
    pytorch_param_types_dict[spec.ParameterType.WEIGHT] -= 3 * 6
    pytorch_param_types_dict[spec.ParameterType.BIAS] -= 3 * 6
    pytorch_param_types_dict[spec.ParameterType.BIAS] -= 6

  # Check if total number of each type match.
  attention_keys = {
      spec.ParameterType.ATTENTION_QKV,
      spec.ParameterType.ATTENTION_Q,
      spec.ParameterType.ATTENTION_K,
      spec.ParameterType.ATTENTION_V,
      spec.ParameterType.ATTENTION_BIAS,
  }
  non_attention_keys = set(jax_param_types_dict.keys()).union(
      set(pytorch_param_types_dict.keys()))
  non_attention_keys -= attention_keys

  mismatches = ''
  mismatches += _count_mismatches(
      jax_param_types_dict, pytorch_param_types_dict, non_attention_keys)
  qkv_match = _check_attention_qkv_match(
      jax_param_types_dict, pytorch_param_types_dict)
  if not qkv_match:
    mismatches += _count_mismatches(
        jax_param_types_dict, pytorch_param_types_dict, attention_keys)
  if mismatches:
    raise ValueError(
        f'On workload {workload_name}, count mismatch: {mismatches}')


def get_workload(workload_name):
  if workload_name == 'cifar':
    jax_workload = JaxCifarWorkload()
    pytorch_workload = PyTorchCifarWorkload()
  elif workload_name == 'criteo1tb':
    jax_workload = JaxCriteoWorkload()
    pytorch_workload = PyTorchCriteoWorkload()
  elif workload_name == 'fastmri':
    jax_workload = JaxFastMRIWorkload()
    pytorch_workload = PyTorchFastMRIWorkload()
  elif workload_name == 'imagenet_resnet':
    jax_workload = JaxImagenetResNetWorkload()
    pytorch_workload = PyTorchImagenetResNetWorkload()
  elif workload_name == 'imagenet_vit':
    jax_workload = JaxImagenetViTWorkload()
    pytorch_workload = PyTorchImagenetViTWorkload()
  elif workload_name == 'librispeech_conformer':
    jax_workload = JaxLibriSpeechConformerWorkload()
    pytorch_workload = PytorchLibriSpeechConformerWorkload()
  elif workload_name == 'librispeech_deepspeech':
    jax_workload = JaxLibriSpeechDeepSpeechWorkload()
    pytorch_workload = PytorchLibriSpeechDeepSpeechWorkload()
  elif workload_name == 'mnist':
    jax_workload = JaxMnistWorkload()
    pytorch_workload = PyTorchMnistWorkload()
  elif workload_name == 'ogbg':
    jax_workload = JaxOgbgWorkload()
    pytorch_workload = PyTorchOgbgWorkload()
  elif workload_name == 'wmt':
    jax_workload = JaxWmtWorkload()
    pytorch_workload = PyTorchWmtWorkload()
  else:
    raise ValueError(f'Workload {workload_name} is not available.')
  _ = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  _ = pytorch_workload.init_model_fn([0])
  return jax_workload, pytorch_workload


if __name__ == '__main__':
  for w in WORKLOADS:
    test_param_types(w)
