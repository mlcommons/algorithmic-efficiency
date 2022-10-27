import os

# Disable GPU access for both jax and pytorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flax import jax_utils
import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.fastmri.fastmri_jax.workload import \
    FastMRIWorkload as JaxWorkload
from algorithmic_efficiency.workloads.fastmri.fastmri_pytorch.workload import \
    FastMRIWorkload as PytWorkload
from tests import torch2jax_utils as utils


def sd_transform(sd):

  def sort_key(k):
    if k[0] == 'ModuleList_0':
      return (0, *k)
    if k[0] == 'ConvBlock_0':
      return (1, *k)
    if k[0] == 'ModuleList_1':
      return (2, *k)
    if k[0] == "ModuleList_2":
      return (3, *k)

  keys = sorted(sd.keys(), key=sort_key)
  c = 0
  for idx, k in enumerate(keys):
    new_key = []
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
      if 'weight' in i:
        i = i.replace('weight', 'kernel')
      new_key.append(i)
    new_key = tuple(new_key)
    sd[new_key] = sd[k]
    del sd[k]
  return sd


def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if 'conv' in k_str and 'kernel' in k_str:
    if 'transpose' in k_str:
      return value.permute(2, 3, 0, 1)
    else:
      return value.permute(2, 3, 1, 0)
  elif ('dense' in k_str and 'kernel' in k_str) or ('lstm' in k_str and
                                                    'kernel' in k_str):
    return value.t()
  return value


if __name__ == "__main__":
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pyt_workload = PytWorkload()
  jax_params, model_state = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  pyt_model, _ = pyt_workload.init_model_fn([0])
  jax_params = jax_utils.unreplicate(jax_params).unfreeze()
  model_state = jax_utils.unreplicate(model_state)

  # Map and copy params of pytorch_model to jax_model.
  t2j = utils.Torch2Jax(torch_model=pyt_model, jax_model=jax_params)
  t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 320, 320)

  jax_batch = {"inputs": image.detach().numpy()}
  pyt_batch = {"inputs": image}

  out_p, _ = pyt_workload.model_fn(
    params = pyt_model,
    augmented_and_preprocessed_input_batch=pyt_batch,
    model_state=None,
    mode=spec.ForwardPassMode.EVAL,
    rng=None,
    update_batch_norm=False)

  out_j, _ = jax_workload.model_fn(params=jax_params,
    augmented_and_preprocessed_input_batch=jax_batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.EVAL,
    rng=jax.random.PRNGKey(0),
    update_batch_norm=False)

  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).max(axis=1))
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).min(axis=1))
