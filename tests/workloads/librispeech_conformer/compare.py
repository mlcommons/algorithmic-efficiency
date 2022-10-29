import os

# Disable GPU access for both jax and pytorch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flax import jax_utils
import jax
import numpy as np
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.workload import \
    LibriSpeechConformerWorkload as JaxWorkload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.workload import \
    LibriSpeechConformerWorkload as PytWorkload
from tests import torch2jax_utils as utils


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
      if 'in_proj' in k[-1]:
        new_key = k[:-1]
        chunks = sd[k].chunk(3)
        for t, c in zip(['query', 'key', 'value'], chunks):
          out[new_key + (t, k[-1].split('_')[-1])] = c
      else:
        out[k] = sd[k]
    else:
      out[k] = sd[k]
  return out


def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if 'conv' in k_str and 'kernel' in k_str:
    rank = len(value.shape)
    if rank == 3:
      value = value.permute(2, 1, 0)
    elif rank == 4:
      value = value.permute(2, 3, 1, 0)
    elif rank == 2:
      value = value.t()
  elif 'attention' in k_str and 'kernel' in k_str:
    value = value.t().reshape(*list(jax_value.shape))
  elif 'attention' in k_str and 'bias' in k_str:
    value = value.reshape(*list(jax_value.shape))
  elif 'dense' in k_str and 'kernel' in k_str:
    value = value.t()
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
  wave = torch.randn(2, 320000)
  pad = torch.zeros_like(wave)
  pad[0, 200000:] = 1

  jax_batch = {"inputs": (wave.detach().numpy(), pad.detach().numpy())}
  pyt_batch = {"inputs": (wave, pad)}

  (out_p, out_pad_p), _ = pyt_workload.model_fn(
    params = pyt_model,
    augmented_and_preprocessed_input_batch=pyt_batch,
    model_state=None,
    mode=spec.ForwardPassMode.EVAL,
    rng=None,
    update_batch_norm=False)

  (out_j, out_pad_j), _ = jax_workload.model_fn(params=jax_params,
    augmented_and_preprocessed_input_batch=jax_batch,
    model_state=model_state,
    mode=spec.ForwardPassMode.EVAL,
    rng=jax.random.PRNGKey(0),
    update_batch_norm=False)

  labels = torch.randint(low=0,high=1024,size=(2,256))
  label_padding = torch.zeros_like(labels)
  jax_loss = jax_workload.loss_fn((labels.numpy(), label_padding.numpy()), (out_j, out_pad_j))
  pyt_loss = pyt_workload.loss_fn((labels,label_padding),(out_p, out_pad_p))
  print(jax_loss, pyt_loss)

  out_j = out_j * (1 - out_pad_j[:, :, None])
  out_p = out_p * (1 - out_pad_p[:, :, None])
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).max(axis=1))
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).min(axis=1))
  print(
      np.abs(np.array(out_pad_j) - out_pad_p.cpu().detach().numpy()).reshape(
          2, -1).max(axis=1))
  print(
      np.abs(np.array(out_pad_j) - out_pad_p.cpu().detach().numpy()).reshape(
          2, -1).min(axis=1))
