import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetViTPostLNWorkload as JaxWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetViTPostLNWorkload as PytWorkload
from flax import jax_utils
import jax
import numpy as np
import torch

from tests.modeldiffs.torch2jax_utils import Torch2Jax
from tests.modeldiffs.torch2jax_utils import value_transform


#pylint: disable=dangerous-default-value
def torch2jax_with_zeroinit(jax_workload,
              pytorch_workload,
              key_transform=None,
              sd_transform=None,
              init_kwargs=dict(dropout_rate=0.0, aux_dropout_rate=0.0, head_zeroinit=False)):
  jax_params, model_state = jax_workload.init_model_fn(jax.random.PRNGKey(0),
                                                       **init_kwargs)
  pytorch_model, _ = pytorch_workload.init_model_fn([0], **init_kwargs)
  jax_params = jax_utils.unreplicate(jax_params).unfreeze()
  if model_state is not None:
    model_state = jax_utils.unreplicate(model_state)

  if isinstance(
      pytorch_model,
      (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    pytorch_model = pytorch_model.module
  t2j = Torch2Jax(torch_model=pytorch_model, jax_model=jax_params)
  if key_transform is not None:
    t2j.key_transform(key_transform)
  if sd_transform is not None:
    t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()
  return jax_params, model_state, pytorch_model


def out_diff(jax_workload,
             pytorch_workload,
             jax_model_kwargs,
             pytorch_model_kwargs,
             key_transform=None,
             sd_transform=None,
             out_transform=None):
  jax_params, model_state, pytorch_model = torch2jax_with_zeroinit(jax_workload,
                                                     pytorch_workload,
                                                     key_transform,
                                                     sd_transform)
  out_p, _ = pytorch_workload.model_fn(params=pytorch_model,
                                       **pytorch_model_kwargs)
  out_j, _ = jax_workload.model_fn(params=jax_params,
                                   model_state=model_state,
                                   **jax_model_kwargs)
  if out_transform is not None:
    out_p = out_transform(out_p)
    out_j = out_transform(out_j)

  print(np.abs(out_p.detach().numpy() - np.array(out_j)).max())
  print(np.abs(out_p.detach().numpy() - np.array(out_j)).min())


def key_transform(k):
  if 'Conv' in k[0]:
    k = ('conv_patch_extract', *k[1:])
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
    if 'GLU' in i:
        pass
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
