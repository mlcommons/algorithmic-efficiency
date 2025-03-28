import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algoperf import spec
from algoperf.workloads.librispeech_conformer.librispeech_jax.workload import \
    LibriSpeechConformerWorkload as JaxWorkload
from algoperf.workloads.librispeech_conformer.librispeech_pytorch.workload import \
    LibriSpeechConformerWorkload as PyTorchWorkload
from tests.modeldiffs.diff import ModelDiffRunner


def key_transform(k):
  new_key = []
  for i in k:
    if 'ModuleList' in i:
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
      i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  for k in sd:
    if 'Attention' in ''.join(k):
      if 'Dense_0' in k[-2]:
        # In-proj
        new_key = k[:-2]
        chunks = sd[k].chunk(3)
        for t, c in zip(['query', 'key', 'value'], chunks):
          out[new_key + (t, k[-1])] = c
      elif 'Dense_1' in k[-2]:
        # Out-proj
        out[(*k[:-2], 'out', k[-1])] = sd[k]
      else:
        out[k] = sd[k]
    else:
      out[k] = sd[k]
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  # Test outputs for identical weights and inputs.
  wave = torch.randn(2, 320000)
  pad = torch.zeros_like(wave)
  pad[0, 200000:] = 1

  jax_batch = {'inputs': (wave.detach().numpy(), pad.detach().numpy())}
  pytorch_batch = {'inputs': (wave, pad)}

  pytorch_model_kwargs = dict(
      augmented_and_preprocessed_input_batch=pytorch_batch,
      model_state=None,
      mode=spec.ForwardPassMode.EVAL,
      rng=None,
      update_batch_norm=False)

  jax_model_kwargs = dict(
      augmented_and_preprocessed_input_batch=jax_batch,
      mode=spec.ForwardPassMode.EVAL,
      rng=jax.random.PRNGKey(0),
      update_batch_norm=False)

  ModelDiffRunner(
      jax_workload=jax_workload,
      pytorch_workload=pytorch_workload,
      jax_model_kwargs=jax_model_kwargs,
      pytorch_model_kwargs=pytorch_model_kwargs,
      key_transform=key_transform,
      sd_transform=sd_transform,
      out_transform=lambda out_outpad: out_outpad[0] *
      (1 - out_outpad[1][:, :, None])).run()
