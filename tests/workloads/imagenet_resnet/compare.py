import os

# Disable GPU access for both jax and pytorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flax import jax_utils
import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import \
    ImagenetResNetWorkload as JaxWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import \
    ImagenetResNetWorkload as PytWorkload
from tests import torch2jax_utils as utils


def key_transform(k):
  new_key = []
  bn = False
  for i in k:
    bn = bn or "BatchNorm" in i
    if 'ModuleList' in i:
      continue
    if 'Linear' in i:
      if 'NonDynamicallyQuantizableLinear' in i:
        i = 'out'
      else:
        i = i.replace('Linear', 'Dense')
    elif 'Conv2d' in i:
      i = i.replace('Conv2d', 'Conv')
    elif 'BatchNorm2d' in i:
      i = i.replace('BatchNorm2d', 'BatchNorm')
    elif 'MHSAwithQS' in i:
      i = i.replace('MHSAwithQS', 'SelfAttention')
    elif 'weight' in i:
      if bn:
        i = i.replace('weight', 'scale')
      else:
        i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  # pylint: disable=locally-disabled, consider-using-generator
  keys = sorted(sd.keys())
  c = -1
  prev = None
  for k in keys:
    if 'Bottleneck' in ''.join(k):
      if prev is None or prev != k[:2]:
        prev = k[:2]
        c += 1
      new_key = (f'BottleneckResNetBlock_{c}',) + k[2:]
      if 'Sequential' in ''.join(new_key):
        new_key = tuple([
            (i.replace('_0', '_proj') if 'BatchNorm' in i or 'Conv' in i else i)
            for i in new_key
            if 'Sequential' not in i
        ])
      sd[new_key] = sd[k]
      del sd[k]
    elif 'BatchNorm' in k[0] or 'Conv' in k[0]:
      new_key = (k[0].replace('_0', '_init'), *k[1:])
      sd[new_key] = sd[k]
      del sd[k]

  return sd


def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if 'conv' in k_str and 'kernel' in k_str:
    rank = len(value.shape)
    if rank == 3:
      return value.permute(2, 1, 0)
    elif rank == 4:
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
  t2j.key_transform(key_transform)
  t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 3, 224, 224)

  jax_batch = {"inputs": image.permute(0, 2, 3, 1).detach().numpy()}
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
