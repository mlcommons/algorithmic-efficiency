import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_jax.workload import \
    ImagenetResNetWorkload as JaxWorkload
from algorithmic_efficiency.workloads.imagenet_resnet.imagenet_pytorch.workload import \
    ImagenetResNetWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def key_transform(k):
  new_key = []
  bn = False
  for i in k:
    bn = bn or 'BatchNorm' in i
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


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 3, 224, 224)

  jax_batch = {'inputs': image.permute(0, 2, 3, 1).detach().numpy()}
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
      key_transform=key_transform,
      sd_transform=sd_transform,
  )
