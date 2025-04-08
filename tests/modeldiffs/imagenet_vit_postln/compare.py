import os

from tests.modeldiffs.diff import ModelDiffRunner
from tests.modeldiffs.imagenet_vit.compare import key_transform

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import torch

from algoperf import spec
from algoperf.workloads.imagenet_vit.imagenet_jax.workload import \
    ImagenetVitPostLNWorkload as JaxWorkload
from algoperf.workloads.imagenet_vit.imagenet_pytorch.workload import \
    ImagenetViTPostLNWorkload as PyTorchWorkload

sd_transform = None

if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

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
      sd_transform=None).run()
