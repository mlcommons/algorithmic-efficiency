"""Utilities for looking up and loading workloads."""

import importlib
import inspect
import os.path
from algorithmic_efficiency import spec


# Workload_path will be appended by '_pytorch' or '_jax' automatically.
_WORKLOAD_CLASS_NAMES = {
    # CIFAR.
    'cifar': 'CifarWorkload',

    # Criteo.
    'criteo1tb': 'Criteo1TbDlrmSmallWorkload',
    'criteo1tb_layer_norm': 'Criteo1TbDlrmSmallLayerNormWorkload',
    'criteo1tb_resnet': 'Criteo1TbDlrmSmallResNetWorkload',
    'criteo1tb_test': 'Criteo1TbDlrmSmallTestWorkload',  # Testing only.

    # FastMRI.
    'fastmri': 'FastMRIWorkload',
    'fastmri_model_size': 'FastMRIModelSizeWorkload',
    'fastmri_layer_norm': 'FastMRILayerNormWorkload',
    'fastmri_tanh': 'FastMRITanhWorkload',

    # ImageNet ResNet.
    'imagenet_resnet': 'ImagenetResNetWorkload',
    'imagenet_resnet_bn_scale': 'ImagenetResNetBatchNormScaleWorkload',
    'imagenet_resnet_gelu': 'ImagenetResNetGeluWorkload',
    'imagenet_resnet_silu': 'ImagenetResNetSiluWorkload',

    # ImageNet ViT.
    'imagenet_vit': 'ImagenetVitWorkload',
    'imagenet_vit_glu': 'ImagenetVitGluWorkload',
    'imagenet_vit_map': 'ImagenetVitMapWorkload',
    'imagenet_vit_post': 'ImagenetVitLayerNormWorkload',

    # LibriSpeech Conformer.
    'librispeech_conformer': 'LibriSpeechConformerWorkload',
    'librispeech_conformer_attn_temperature': 'LibriSpeechConformerAttentionTemperatureWorkload',
    'librispeech_conformer_decoder_pre_no_post': 'LibriSpeechConformerLnWorkload',
    'librispeech_conformer_gelu': 'LibriSpeechConformerGeluWorkload',

    # LibriSpeech Deepspeech.
    'librispeech_deepspeech': 'LibriSpeechDeepSpeechWorkload',
    'librispeech_deepspeech_all': 'LibriSpeechDeepSpeechAllWorkload',
    'librispeech_deepspeech_no_residual': 'LibriSpeechDeepSpeechNoResidualWorkload',
    'librispeech_deepspeech_tanh': 'LibriSpeechDeepSpeechTanhWorkload',

    # MNIST.
    'mnist': 'MnistWorkload',

    # OGBG.
    'ogbg': 'OgbgWorkload',
    'ogbg_gelu': 'OgbgGeluWorkload',
    # Also called ogbg_hd_256_256_ld_128_nps_3_bn.
    'ogbg_model_size': 'OgbgModelSizeWorkload',
    'ogbg_silu': 'OgbgSiluWorkload',

    # WMT.
    'wmt': 'WmtWorkload',
    'wmt_attn_temp': 'WmtAttentionTemperatureWorkload',
    'wmt_glu_tanh': 'WmtGluWorkload',
    'wmt_post_ln': 'WmtLayerNormWorkload',
}


_BASE_WORKLOAD_NAMES = [
    'criteo1tb',
    'fastmri',
    'imagenet_resnet',
    'imagenet_vit',
    'librispeech_conformer',
    'librispeech_deepspeech',
    'ogbg',
    'wmt',
]


def workload_names():
  return list(_WORKLOAD_CLASS_NAMES.keys())


def get_base_workload(name):
  if name in _BASE_WORKLOAD_NAMES:
    return name
  for base_name in _BASE_WORKLOAD_NAMES:
    if name.startswith(base_name):
      return base_name
  raise ValueError(f'Could not find base workload for variant {name}.')


def convert_filepath_to_module(path: str):
  base, extension = os.path.splitext(path)

  if extension != '.py':
    raise ValueError(f'Path: {path} must be a python file (*.py)')

  return base.replace('/', '.')


def import_workload(workload_path: str,
                    workload_class_name: str,
                    return_class=False,
                    workload_init_kwargs=None) -> spec.Workload:
  """Import and add the workload to the registry.

  This importlib loading is nice to have because it allows runners to avoid
  installing the dependencies of all the supported frameworks. For example, if
  a submitter only wants to write Jax code, the try/except below will catch
  the import errors caused if they do not have the PyTorch dependencies
  installed on their system.

  Args:
    workload_path: the path to the `workload.py` file to load.
    workload_class_name: the name of the Workload class that implements the
      `Workload` abstract class in `spec.py`.
    return_class: if true, then the workload class is returned instead of the
      instantiated object. Useful for testing when methods need to be overriden.
    workload_init_kwargs: kwargs to pass to the workload constructor.
  """

  # Remove the trailing '.py' and convert the filepath to a Python module.
  workload_path = convert_filepath_to_module(workload_path)

  # Import the workload module.
  workload_module = importlib.import_module(workload_path)
  # Get everything defined in the workload module (including our class).
  workload_module_members = inspect.getmembers(workload_module)
  workload_class = None
  for name, value in workload_module_members:
    if name == workload_class_name:
      workload_class = value
      break
  if workload_class is None:
    raise ValueError(
        f'Could not find member {workload_class_name} in {workload_path}. '
        'Make sure the Workload class is spelled correctly and defined in '
        'the top scope of the module.')
  if return_class:
    return workload_class
  return workload_class(**workload_init_kwargs)


def get_workload(
    name, framework, librispeech_tokenizer_vocab_path=None) -> spec.Workload:
  workload_class_name = _WORKLOAD_CLASS_NAMES[name]
  # Extend path according to framework.
  dataset_name = name.split('_')[0] if '_' in name else name
  workload_subdir = name + '/' + dataset_name + '_' + framework
  workload_path = os.path.join(
      'algorithmic_efficiency/workloads',
      workload_subdir,
      'workload.py')
  workload_init_kwargs = {}
  if librispeech_tokenizer_vocab_path:
    workload_init_kwargs['tokenizer_vocab_path'] = (
        librispeech_tokenizer_vocab_path)
  workload = import_workload(
      workload_path=workload_path,
      workload_class_name=workload_class_name,
      workload_init_kwargs=workload_init_kwargs)
  return workload
