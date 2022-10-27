import os

# Disable GPU access for both jax and pytorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flax import jax_utils
import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetVitWorkload as JaxWorkload
from algorithmic_efficiency.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetVitWorkload as PytWorkload
from tests import torch2jax_utils as utils


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
            'Linear_3': 'out'
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
      i = 'encoder_norm'
    elif 'weight' in i:
      if bn or ln:
        i = i.replace('weight', 'scale')
      else:
        i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  # pylint: disable=locally-disabled, modified-iterating-dict, consider-using-dict-items
  out = {}
  for k in sd:
    if 'Attention' in ''.join(k):
      if 'in_proj' in k[-1]:
        new_key = k[:-1]
        chunks = sd[k].chunk(3)
        for t, c in zip(['query', 'key', 'value'], chunks):
          out[new_key + (t, k[-1].split('_')[-1])] = c
      else:
        out[k] = sd[k]
    elif 'LSTM' in ''.join(k):
      if '_hh_' in ''.join(k):
        chunks = sd[k].chunk(4)
        for t, c in zip(['hi', 'hf', 'hg', 'ho'], chunks):
          out[k[:-1] + ('LSTM',
                        f'LSTMSequenceEncoder_{1 if "reverse" in k[-1] else 0}',
                        'cell',
                        t,
                        k[-1].split('_')[0])] = c
      elif '_ih_' in ''.join(k):
        chunks = sd[k].chunk(4)
        for t, c in zip(['ii', 'if', 'ig', 'io'], chunks):
          out[k[:-1] + ('LSTM',
                        f'LSTMSequenceEncoder_{1 if "reverse" in k[-1] else 0}',
                        'cell',
                        t,
                        k[-1].split('_')[0])] = c
    else:
      out[k] = sd[k]
  keys_to_del = []
  for k in out:
    if (k[-2] in ['ii', 'if', 'ig', 'io']) and k[-1] == 'bias':
      other_bias = k[:-2] + ('h' + k[-2][1], 'bias')
      out[other_bias] = out[other_bias] + out[k]
      keys_to_del.append(k)
  for k in keys_to_del:
    del out[k]
  return out


def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if ('conv' in k_str and 'kernel' in k_str) or ('embedding' in k_str and
                                                 'kernel' in k_str):
    rank = len(value.shape)
    if rank == 3:
      return value.permute(2, 1, 0)
    elif rank == 4:
      return value.permute(2, 3, 1, 0)
    elif rank == 2:
      return value.t()
  elif 'attention' in k_str and 'kernel' in k_str:
    return value.t().reshape(*list(jax_value.shape))
  elif 'attention' in k_str and 'bias' in k_str:
    return value.reshape(*list(jax_value.shape))
  elif ('dense' in k_str and
        'kernel' in k_str) or ('head' in k_str and
                               'kernel' in k_str) or ('pre_logits' in k_str and
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

  if len(out_p.shape) == 4:
    out_p = out_p.permute(0, 2, 3, 1)
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).max(axis=1))
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).min(axis=1))
