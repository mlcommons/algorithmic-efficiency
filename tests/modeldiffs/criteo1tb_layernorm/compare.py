import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_jax.workload import \
    Criteo1TbDlrmSmallLayerNormWorkload as JaxWorkload
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.workload import \
    Criteo1TbDlrmSmallLayerNormWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def key_transform(k):
  new_key = []
  s_count = None
  layer_norm = False
  print('key')
  print(k)
  for i in k:
    if 'Sequential' in i:
      s_count = int(i.split('_')[1])
      continue
    if 'Embedding' in i:
      return ('embedding_table',)
    if 'Linear' in i:
      i = i.replace('Linear', 'Dense')
      name, count = i.split('_')
      i = name + '_' + str(s_count * 3 + int(count))
    if 'LayerNorm' in i:
      layer_norm = True
      name, count = i.split('_')
      # There is a layernorm on embedding between bottom and top MLP
      if s_count is not None:
        i = name + '_' + str(s_count * 4 + int(count))
      else: 
        i = name + '_' + str(3)
    elif 'weight' in i:
      if layer_norm:
        i = i.replace('weight', 'scale')
      else:
        i = i.replace('weight', 'kernel')
    new_key.append(i)
  print(new_key)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  chunks = []
  for k in sd:
    if 'embedding_chunk' in ''.join(k):
      chunks.append(sd[k].cpu())
    else:
      out[k] = sd[k]
  out[('embedding_table',)] = torch.cat(chunks, dim=0)
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  pyt_batch = {
      'inputs': torch.ones((2, 13 + 26)),
      'targets': torch.randint(low=0, high=1, size=(2,)),
      'weights': torch.ones(2),
  }
  jax_batch = {k: np.array(v) for k, v in pyt_batch.items()}

  # Test outputs for identical weights and inputs.
  pytorch_model_kwargs = dict(
      augmented_and_preprocessed_input_batch=pyt_batch,
      model_state=None,
      mode=spec.ForwardPassMode.EVAL,
      rng=None,
      update_batch_norm=False)

  jax_model_kwargs = dict(
      augmented_and_preprocessed_input_batch=jax_batch,
      mode=spec.ForwardPassMode.EVAL,
      rng=jax.random.PRNGKey(0),
      update_batch_norm=False)

  out_diff(
      jax_workload=jax_workload,
      pytorch_workload=pytorch_workload,
      jax_model_kwargs=jax_model_kwargs,
      pytorch_model_kwargs=pytorch_model_kwargs,
      key_transform=key_transform,
      sd_transform=sd_transform,
      out_transform=None)
