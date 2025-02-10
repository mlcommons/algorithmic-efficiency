"""Utilities for dealing with parameter-related logic like types and shapes."""

from typing import Dict

import flax
import jax
from torch import nn

from algoperf import spec


def pytorch_param_shapes(model: nn.Module) -> Dict[str, spec.ShapeTuple]:
  return {k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()}


def pytorch_param_types(
    param_shapes: Dict[str, spec.ShapeTuple]) -> Dict[str, spec.ParameterType]:
  param_types = {}
  for name in param_shapes.keys():
    if 'bn' in name:
      if 'weight' in name or 'scale' in name:
        param_types[name] = spec.ParameterType.BATCH_NORM_SCALE
      elif 'bias' in name:
        param_types[name] = spec.ParameterType.BATCH_NORM_BIAS
      else:
        raise ValueError(f'Unrecognized batch norm parameter: {name}.')
    elif 'norm' in name or 'ln' in name:
      if 'weight' in name or 'scale' in name:
        param_types[name] = spec.ParameterType.LAYER_NORM_SCALE
      elif 'bias' in name:
        param_types[name] = spec.ParameterType.LAYER_NORM_BIAS
      else:
        raise ValueError(f'Unrecognized layer norm parameter: {name}.')
    elif 'conv' in name:
      if 'bias' in name:
        param_types[name] = spec.ParameterType.BIAS
      else:
        param_types[name] = spec.ParameterType.CONV_WEIGHT
    elif ('embedding' in name or 'embed' in name) and 'weight' in name:
      param_types[name] = spec.ParameterType.EMBEDDING
    elif 'attn' in name or 'attention' in name:
      if 'bias' in name:
        param_types[name] = spec.ParameterType.ATTENTION_BIAS
      elif 'in_proj' in name:
        param_types[name] = spec.ParameterType.ATTENTION_QKV
      elif 'kv_proj' in name:
        param_types[name] = spec.ParameterType.ATTENTION_KV
      elif 'k_proj' in name or 'key' in name:
        param_types[name] = spec.ParameterType.ATTENTION_K
      elif 'q_proj' in name or 'query' in name:
        param_types[name] = spec.ParameterType.ATTENTION_Q
      elif 'v_proj' in name or 'value' in name:
        param_types[name] = spec.ParameterType.ATTENTION_V
      elif 'out' in name and 'weight' in name:
        param_types[name] = spec.ParameterType.ATTENTION_OUT
      elif 'scale' in name:
        param_types[name] = spec.ParameterType.WEIGHT
      else:
        raise ValueError(f'Unrecognized attention parameter: {name}.')
    elif 'bias' in name:
      param_types[name] = spec.ParameterType.BIAS
    else:
      param_types[name] = spec.ParameterType.WEIGHT
  return param_types


def jax_param_shapes(
    params: spec.ParameterContainer) -> spec.ParameterShapeTree:
  return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), params)


def jax_param_types(param_shapes: spec.ParameterShapeTree,
                    parent_name: str = '') -> Dict[str, spec.ParameterType]:
  param_types = {}
  for name, value in param_shapes.items():
    name = name.lower()
    if isinstance(value, dict) or isinstance(value, flax.core.FrozenDict):
      param_types[name] = jax_param_types(
          value, parent_name=parent_name + '/' + name)
    else:
      if 'batchnorm' in parent_name or 'bn' in parent_name:
        if name == 'scale':
          param_types[name] = spec.ParameterType.BATCH_NORM_SCALE
        elif name == 'bias':
          param_types[name] = spec.ParameterType.BATCH_NORM_BIAS
        else:
          raise ValueError(
              f'Unrecognized batch norm parameter: {parent_name}/{name}.')
      elif 'layernorm' in parent_name or 'ln' in parent_name:
        if name == 'scale':
          param_types[name] = spec.ParameterType.LAYER_NORM_SCALE
        elif name == 'bias':
          param_types[name] = spec.ParameterType.LAYER_NORM_BIAS
        else:
          raise ValueError(
              f'Unrecognized layer norm parameter: {parent_name}/{name}.')
      elif 'conv' in parent_name:
        if 'bias' in name:
          param_types[name] = spec.ParameterType.BIAS
        else:
          param_types[name] = spec.ParameterType.CONV_WEIGHT
      # Note that this is exact equality, not contained in, because
      # flax.linen.Embed names the embedding parameter "embedding"
      # https://github.com/google/flax/blob/main/flax/linen/linear.py#L604.
      elif ('embedding' in name or
            ('embedding' in parent_name and name == 'kernel')):
        param_types[name] = spec.ParameterType.EMBEDDING
      elif 'attention' in parent_name:
        if name == 'bias':
          param_types[name] = spec.ParameterType.ATTENTION_BIAS
        elif 'key' in parent_name and name == 'kernel':
          param_types[name] = spec.ParameterType.ATTENTION_K
        elif 'query' in parent_name and name == 'kernel':
          param_types[name] = spec.ParameterType.ATTENTION_Q
        elif 'value' in parent_name and name == 'kernel':
          param_types[name] = spec.ParameterType.ATTENTION_V
        elif 'out' in parent_name and name == 'kernel':
          param_types[name] = spec.ParameterType.ATTENTION_OUT
        elif 'scale' in name:
          param_types[name] = spec.ParameterType.WEIGHT
        elif 'in_proj_weight' in name:
          param_types[name] = spec.ParameterType.ATTENTION_QKV
        else:
          raise ValueError(
              f'Unrecognized attention parameter: {parent_name}/{name}.')
      elif 'bias' in name:
        param_types[name] = spec.ParameterType.BIAS
      else:
        param_types[name] = spec.ParameterType.WEIGHT
  return param_types
