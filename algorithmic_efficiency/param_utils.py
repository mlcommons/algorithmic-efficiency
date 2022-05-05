"""Utilities for dealing with parameter-related logic like types and shapes."""
from algorithmic_efficiency import spec


def pytorch_param_types(param_shapes):
  if param_shapes is None:
    raise ValueError(
        'This should not happen, workload.init_model_fn() should be called '
        'before workload.model_params_types!')
  param_types = {}
  for name in param_shapes.keys():
    if 'bias' in name:
      param_types[name] = spec.ParameterType.BIAS
    elif 'BatchNorm' in name:
      param_types[name] = spec.ParameterType.BATCH_NORM
    elif 'Conv' in name:
      param_types[name] = spec.ParameterType.CONV_WEIGHT
    else:
      param_types[name] = spec.ParameterType.WEIGHT
  return param_types
