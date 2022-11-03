import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax.workload import \
    LibriSpeechDeepSpeechWorkload as JaxWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.workload import \
    LibriSpeechDeepSpeechWorkload as PytWorkload
from tests.modeldiffs.diff import out_diff


def key_transform(k):
  new_key = []
  bn = False
  for i in k:
    bn = bn or 'BatchNorm' in i
    if 'ModuleList' in i:
      continue
    if 'CustomBatchNorm' in i:
      continue
    if 'Linear' in i:
      if 'NonDynamicallyQuantizableLinear' in i:
        i = 'out'
      else:
        i = i.replace('Linear', 'Dense')
    elif 'Conv1d' in i:
      i = i.replace('Conv1d', 'Conv')
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


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  # Test outputs for identical weights and inputs.
  wave = torch.randn(2, 320000)
  pad = torch.zeros_like(wave)
  pad[0, 200000:] = 1

  jax_batch = {'inputs': (wave.detach().numpy(), pad.detach().numpy())}
  pyt_batch = {'inputs': (wave, pad)}

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
      out_transform=lambda out_outpad: out_outpad[0] *
      (1 - out_outpad[1][:, :, None]))
