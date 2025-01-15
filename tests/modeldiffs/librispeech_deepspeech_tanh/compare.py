import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algoperf import spec
from algoperf.workloads.librispeech_deepspeech.librispeech_jax.workload import \
    LibriSpeechDeepSpeechNoResNetWorkload as JaxWorkload
from algoperf.workloads.librispeech_deepspeech.librispeech_pytorch.workload import \
    LibriSpeechDeepSpeechNoResNetWorkload as PyTorchWorkload
from tests.modeldiffs.diff import out_diff
from tests.modeldiffs.librispeech_deepspeech.compare import key_transform
from tests.modeldiffs.librispeech_deepspeech.compare import sd_transform

if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

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
