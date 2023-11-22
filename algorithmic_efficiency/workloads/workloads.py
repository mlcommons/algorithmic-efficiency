""" Registry of workload info
"""
import importlib
import inspect
import os

from algorithmic_efficiency import spec

BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'

WORKLOADS = {
    'cifar': {
        'workload_path': 'cifar/cifar', 'workload_class_name': 'CifarWorkload'
    },
    'criteo1tb': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallWorkload',
    },
    'criteo1tb_test': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallTestWorkload',
    },
    'criteo1tb_layernorm': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallLayerNormWorkload'
    },
    'criteo1tb_resnet': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallResNetWorkload'
    },
    'fastmri': {
        'workload_path': 'fastmri/fastmri',
        'workload_class_name': 'FastMRIWorkload',
    },
    'imagenet_resnet': {
        'workload_path': 'imagenet_resnet/imagenet',
        'workload_class_name': 'ImagenetResNetWorkload',
    },
    'imagenet_vit': {
        'workload_path': 'imagenet_vit/imagenet',
        'workload_class_name': 'ImagenetVitWorkload',
    },
    'librispeech_conformer': {
        'workload_path': 'librispeech_conformer/librispeech',
        'workload_class_name': 'LibriSpeechConformerWorkload',
    },
    'librispeech_deepspeech': {
        'workload_path': 'librispeech_deepspeech/librispeech',
        'workload_class_name': 'LibriSpeechDeepSpeechWorkload',
    },
    'mnist': {
        'workload_path': 'mnist/mnist', 'workload_class_name': 'MnistWorkload'
    },
    'ogbg': {
        'workload_path': 'ogbg/ogbg', 'workload_class_name': 'OgbgWorkload'
    },
    'wmt': {'workload_path': 'wmt/wmt', 'workload_class_name': 'WmtWorkload'},
}

BASE_WORKLOADS = [
    'criteo1tb',
    'fastmri',
    ' imagenet_resnet',
    'imagenet_vit',
    'librispeech_conformer',
    'librispeech_deepspeech',
    'ogbg',
    'wmt'
]


def get_base_workload_name(workload_name):
  for base_workload_name in BASE_WORKLOADS:
    if base_workload_name in workload_name:
      return base_workload_name


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
