import jax.numpy as jnp
import jax.random as jax_rng
import numpy as np
import torch
import utils

from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    Conformer as JaxConformer
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax.models import \
    ConformerConfig as JaxConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerConfig as PytorchConformerConfig
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    ConformerEncoderDecoder as PytorchConformer
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    initialize


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
      return value.permute(2, 1, 0)
    elif rank == 4:
      return value.permute(2, 3, 1, 0)
    elif rank == 2:
      return value.t()
  elif 'attention' in k_str and 'kernel' in k_str:
    return value.t().reshape(*list(jax_value.shape))
  elif 'attention' in k_str and 'bias' in k_str:
    return value.reshape(*list(jax_value.shape))
  elif 'dense' in k_str and 'kernel' in k_str:
    return value.t()
  return value


if __name__ == "__main__":
  # pylint: disable=locally-disabled, not-callable
  DEVICE = "cpu"  # Allow jax to use GPU

  jax_conformer = JaxConformer(JaxConformerConfig())
  pyt_conformer = PytorchConformer(PytorchConformerConfig()).eval().to(DEVICE)

  # Init Jax model.
  init_rngs = {'params': jax_rng.PRNGKey(0), 'dropout': jax_rng.PRNGKey(1)}
  input_shape = [(320000,), (320000,)]
  fake_input_batch = [jnp.zeros((2, *x), jnp.float32) for x in input_shape]
  jax_model = jax_conformer.init(
      init_rngs, train=False, *fake_input_batch).unfreeze()

  # Init PyTorch model.
  wave = torch.randn(2, 320000).to(DEVICE)
  pad = torch.zeros(2, 320000).to(DEVICE)
  _ = pyt_conformer(wave, pad)
  initialize(pyt_conformer)

  # Map and copy params of pytorch_model to jax_model.
  t2j = utils.Torch2Jax(
      torch_model=pyt_conformer, jax_model=jax_model["params"])
  t2j.key_transform(key_transform)
  t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()

  # Test outputs for identical weights and inputs.
  wave = torch.randn(2, 320000)
  pad = torch.zeros_like(wave)

  out_p, _ = pyt_conformer(wave.to(DEVICE), pad.to(DEVICE))
  out_j,_ = jax_conformer.apply(jax_model,
                                wave.detach().numpy(),
                                pad.detach().numpy(),
                                train=False)

  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).max(axis=1))
  print(
      np.abs(np.array(out_j) - out_p.cpu().detach().numpy()).reshape(
          2, -1).min(axis=1))
