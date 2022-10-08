"""Utilities for dealing with parameter-related logic like types and shapes."""
import jax

from algorithmic_efficiency import spec


def pytorch_param_shapes(model):
  return {k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()}


def pytorch_param_types(param_shapes):
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


def jax_param_shapes(params):
  return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), params)


def jax_param_types(param_shapes):
  param_types_dict = {}
  for name, value in param_shapes.items():
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
