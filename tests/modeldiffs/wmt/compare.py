import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.wmt.wmt_jax.workload import \
    WmtWorkload as JaxWorkload
from algorithmic_efficiency.workloads.wmt.wmt_pytorch.workload import \
    WmtWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def key_transform(k):
  new_key = []
  for i in k:
    if 'ModuleList' in i or\
        'TransformerDecoder_' in i or\
        'TransformerEncoder_' in i:
      continue
    if 'Linear' in i:
      if 'NonDynamicallyQuantizableLinear' in i:
        i = 'out'
      else:
        i = i.replace('Linear', 'Dense')
    elif i == 'Decoder_0':
      i = 'decoder'
    elif i == 'Encoder_0':
      i = 'encoder'
    elif 'TransformerEncoderLayer' in i:
      i = i.replace('TransformerEncoderLayer', 'encoderblock')
    elif 'TransformerDecoderLayer' in i:
      i = i.replace('TransformerDecoderLayer', 'encoderdecoderblock')
    elif 'MultiheadAttention' in i:
      i = i.replace('MultiheadAttention', 'SelfAttention')
    elif 'weight' in i:
      i = i.replace('weight', 'kernel')

    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  for k in sd:
    k_str = ''.join(k)
    if 'SelfAttention' in k_str:
      new_key = list(k)
      new_key = [
          i if i != 'SelfAttention_1' else 'MultiHeadDotProductAttention_0'
          for i in new_key
      ]
      if 'SelfAttention_0' in k_str:
        if new_key[-2] == 'Dense_0':
          # qkv
          for name, value in zip(('query', 'key', 'value'), sd[k].chunk(3)):
            out[(*new_key[:-2], name, new_key[-1])] = value
          pass
        elif new_key[-2] == 'Dense_1':
          # out
          out[(*new_key[:-2], 'out', new_key[-1])] = sd[k]
          pass
      else:
        if new_key[-2] == 'Dense_0':
          #q
          out[(*new_key[:-2], 'query', new_key[-1])] = sd[k]
          pass
        elif new_key[-2] == 'Dense_1':
          # kv
          for name, value in zip(('key', 'value'), sd[k].chunk(2)):
            out[(*new_key[:-2], name, new_key[-1])] = value
          pass
        elif new_key[-2] == 'Dense_2':
          # out
          out[(*new_key[:-2], 'out', new_key[-1])] = sd[k]
          pass
    elif 'Dense' in k_str:
      new_key = (*k[:2], 'MlpBlock_0', *k[2:])
      out[new_key] = sd[k]
    elif 'LayerNorm' in k_str:
      new_key = list(k)
      if len(k) == 3:
        if k[0] == 'encoder':
          new_key[1] = 'encoder_layernorm'
        else:
          new_key[1] = 'encoderdecoder_layernorm'
      if k[-1] == 'kernel':
        new_key[-1] = 'scale'
      new_key = tuple(new_key)
      out[new_key] = sd[k]
    elif 'Embedding' in k_str:
      new_key = ('shared_embedding', 'embedding')
      out[new_key] = sd[k]
    else:
      out[k] = sd[k]
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  # Test outputs for identical weights and inputs.
  inp_tokens = torch.randint(low=0, high=32000, size=(2, 256))
  tgt_tokens = torch.randint(low=0, high=32000, size=(2, 256))

  jax_batch = {
      'inputs': inp_tokens.detach().numpy(),
      'targets': tgt_tokens.detach().numpy(),
  }
  pyt_batch = {'inputs': inp_tokens, 'targets': tgt_tokens}

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
