import os

from tests.modeldiffs.diff import ModelDiffRunner
from tests.modeldiffs.imagenet_vit.compare import key_transform

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algoperf import spec
from algoperf.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetVitMapWorkload as JaxWorkload
from algoperf.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetVitMapWorkload as PytWorkload


def sd_transform(sd):
  out = {}
  for k in sd:
    if len(k) > 2 and k[-2] == 'key_value':
      chunk0, chunk1 = sd[k].chunk(2)
      out[(*k[:-2], 'key', k[-1])] = chunk0
      out[(*k[:-2], 'value', k[-1])] = chunk1
    else:
      out[k] = sd[k]
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PytWorkload()

  # Test outputs for identical weights and inputs.
  image = torch.randn(2, 3, 224, 224)

  jax_batch = {'inputs': image.permute(0, 2, 3, 1).detach().numpy()}
  pytorch_batch = {'inputs': image}

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
      sd_transform=sd_transform).run()
