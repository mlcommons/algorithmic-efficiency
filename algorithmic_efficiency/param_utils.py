"""Utilities for dealing with parameter-related logic like types and shapes."""

from typing import Dict

import flax
import jax
from torch import nn

from algorithmic_efficiency import spec


def pytorch_param_shapes(model: nn.Module) -> Dict[str, spec.ShapeTuple]:
  return {k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()}


def pytorch_param_types(
    param_shapes: Dict[str, spec.ShapeTuple]) -> Dict[str, spec.ParameterType]:
  param_types = {}
  for name in param_shapes.keys():
    if 'bias' in name:
      param_types[name] = spec.ParameterType.BIAS
    elif 'bn' in name:
      param_types[name] = spec.ParameterType.BATCH_NORM
    elif 'conv' in name:
      param_types[name] = spec.ParameterType.CONV_WEIGHT
    elif 'embedding' in name:
      param_types[name] = spec.ParameterType.EMBEDDING
    else:
      param_types[name] = spec.ParameterType.WEIGHT
  return param_types


def jax_param_shapes(
    params: spec.ParameterContainer) -> spec.ParameterShapeTree:
  return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), params)


def jax_param_types(param_shapes: spec.ParameterShapeTree,
                    parent_name: str = '') -> Dict[str, spec.ParameterType]:
  param_types_dict = {}
  for name, value in param_shapes.items():
    if isinstance(value, dict) or isinstance(value, flax.core.FrozenDict):
      param_types_dict[name] = jax_param_types(value, parent_name=name)
    else:
      if 'bias' in name:
        param_types_dict[name] = spec.ParameterType.BIAS
      elif 'batchnorm' in parent_name.lower():
        param_types_dict[name] = spec.ParameterType.BATCH_NORM
      elif 'conv' in parent_name.lower():
        param_types_dict[name] = spec.ParameterType.CONV_WEIGHT
      # Note that this is exact equality, not contained in, because
      # flax.linen.Embed names the embedding parameter "embedding"
      # https://github.com/google/flax/blob/main/flax/linen/linear.py#L604.
      elif 'embedding' in name:
        param_types_dict[name] = spec.ParameterType.EMBEDDING
      else:
        param_types_dict[name] = spec.ParameterType.WEIGHT
  return param_types_dict
