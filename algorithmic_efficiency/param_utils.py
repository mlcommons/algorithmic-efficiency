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
    elif 'Embedding' in name:
      param_types[name] = spec.ParameterType.EMBEDDING
    else:
      param_types[name] = spec.ParameterType.WEIGHT
  return param_types


def jax_param_types(param_tree):
  param_types_dict = {}
  for name, value in param_tree.items():
    if isinstance(value, dict):
      param_types_dict[name] = jax_param_types(value)
    else:
      if 'bias' in name:
        param_types_dict[name] = spec.ParameterType.BIAS
      elif 'BatchNorm' in name:
        param_types_dict[name] = spec.ParameterType.BATCH_NORM
      elif 'Conv' in name:
        param_types_dict[name] = spec.ParameterType.CONV_WEIGHT
      # Note that this is exact equality, not contained in, because
      # flax.linen.Embed names the embedding parameter "embedding"
      # https://github.com/google/flax/blob/main/flax/linen/linear.py#L604.
      elif name == 'embedding':
        param_types_dict[name] = spec.ParameterType.EMBEDDING
      else:
        param_types_dict[name] = spec.ParameterType.WEIGHT
  return param_types_dict
