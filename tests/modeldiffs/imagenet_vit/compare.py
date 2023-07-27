import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetVitWorkload as JaxWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetVitWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def key_transform(k):
  if 'Conv' in k[0]:
    k = ('embedding', *k[1:])
  elif k[0] == 'Linear_0':
    k = ('pre_logits', *k[1:])
  elif k[0] == 'Linear_1':
    k = ('head', *k[1:])

  new_key = []
  bn = False
  attention = False
  ln = False
  enc_block = False
  for idx, i in enumerate(k):
    bn = bn or 'BatchNorm' in i
    ln = ln or 'LayerNorm' in i
    attention = attention or 'SelfAttention' in i
    if 'ModuleList' in i or 'Sequential' in i:
      continue
    if 'CustomBatchNorm' in i:
      continue
    if 'Linear' in i:
      if attention:
        i = {
            'Linear_0': 'query',
            'Linear_1': 'key',
            'Linear_2': 'value',
            'Linear_3': 'out',
        }[i]
      else:
        i = i.replace('Linear', 'Dense')
    elif 'Conv2d' in i:
      i = i.replace('Conv2d', 'Conv')
    elif 'Encoder1DBlock' in i:
      i = i.replace('Encoder1DBlock', 'encoderblock')
      enc_block = True
    elif 'Encoder' in i:
      i = 'Transformer'
    elif enc_block and 'SelfAttention' in i:
      i = 'MultiHeadDotProductAttention_1'
    elif enc_block and i == 'LayerNorm_1':
      i = 'LayerNorm_2'
    elif enc_block and 'MlpBlock' in i:
      i = 'MlpBlock_3'
    elif idx == 1 and i == 'LayerNorm_0':
      i = 'encoder_layernorm'
    elif 'weight' in i:
      if bn or ln:
        i = i.replace('weight', 'scale')
      else:
        i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


sd_transform = None

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
      sd_transform=None,
  )
