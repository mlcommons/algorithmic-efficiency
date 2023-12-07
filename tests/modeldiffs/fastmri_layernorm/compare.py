import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.workload import \
    FastMRILayerNormWorkload as JaxWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.workload import \
    FastMRILayerNormWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def sd_transform(sd):

  def sort_key(k):
    if k[0] == 'ModuleList_0':
      return (0, *k)
    if k[0] == 'ConvBlock_0':
      return (1, *k)
    if k[0] == 'ModuleList_1':
      return (2, *k)
    if k[0] == 'ModuleList_2':
      return (3, *k)

  keys = sorted(sd.keys(), key=sort_key)
  c = 0
  for idx, k in enumerate(keys):
    print(k)
    new_key = []
    layernorm = False
    for idx2, i in enumerate(k):
      if 'ModuleList' in i or 'Sequential' in i:
        continue
      if i.startswith('ConvBlock'):
        if idx != 0 and keys[idx - 1][:idx2 + 1] != k[:idx2 + 1]:
          c += 1
        i = f'ConvBlock_{c}'
      if 'Conv2d' in i:
        i = i.replace('Conv2d', 'Conv')
      if 'ConvTranspose2d' in i:
        i = i.replace('ConvTranspose2d', 'ConvTranspose')
      if 'GroupNorm' in i:
        i = i.replace('GroupNorm', 'LayerNorm')
        layernorm = True
      if 'weight' in i:
        if layernorm:
          i = i.replace('weight', 'scale')
        else:
          i = i.replace('weight', 'kernel')
      new_key.append(i)
    new_key = tuple(new_key)
    print(new_key)
    sd[new_key] = sd[k]
    del sd[k]
  return sd


key_transform = None
if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 320, 320)

  jax_batch = {'inputs': image.detach().numpy()}
  pyt_batch = {'inputs': image}

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
      key_transform=None,
      sd_transform=sd_transform,
  )
