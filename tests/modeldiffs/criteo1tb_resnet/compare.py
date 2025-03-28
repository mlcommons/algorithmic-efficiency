import os

# Disable GPU access for both jax and pytorch.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import jax
import jax.numpy as jnp
import numpy as np
import torch

from algoperf import spec
from algoperf.workloads.criteo1tb.criteo1tb_jax.workload import \
    Criteo1TbDlrmSmallResNetWorkload as JaxWorkload
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.workload import \
    Criteo1TbDlrmSmallResNetWorkload as PyTorchWorkload
from tests.modeldiffs.diff import ModelDiffRunner


def key_transform(k):
  new_key = []
  mlp_count = None
  resnet_block_count = None
  mlp_block_count = None
  for i in k:
    if 'Embedding' in i:
      return ('embedding_table',)
    if 'Sequential' in i:
      if mlp_count is None:
        mlp_count = int(i.split('_')[1])
      else:
        mlp_block_count = int(i.split('_')[1])
      continue
    if 'DenseBlock' in i:
      # off set resnet block count by 1
      # since first mlp layer has no resnet connection
      resnet_block_count = int(i.split('_')[1])
      continue
    if 'Linear' in i:
      i = i.replace('Linear', 'Dense')
      name, _ = i.split('_')
      block_count = mlp_block_count if mlp_block_count else resnet_block_count
      i = name + '_' + str(mlp_count * 3 + block_count)
    elif 'weight' in i:
      i = i.replace('weight', 'kernel')
    new_key.append(i)
  return tuple(new_key)


def sd_transform(sd):
  out = {}
  chunks = []
  for k in sd:
    if 'embedding_chunk' in ''.join(k):
      chunks.append(sd[k].cpu())
    else:
      out[k] = sd[k]
  out[('embedding_table',)] = torch.cat(chunks, dim=0)
  return out


if __name__ == '__main__':
  # pylint: disable=locally-disabled, not-callable

  jax_workload = JaxWorkload()
  pytorch_workload = PyTorchWorkload()

  pyt_batch = {
      'inputs': torch.ones((2, 13 + 26)),
      'targets': torch.randint(low=0, high=1, size=(2,)),
      'weights': torch.ones(2),
  }

  init_fake_batch_size = 2
  num_categorical_features = 26
  input_size = 13 + num_categorical_features
  input_shape = (init_fake_batch_size, input_size)
  fake_inputs = jnp.ones(input_shape, jnp.float32)
  jax_batch = {k: np.array(v) for k, v in pyt_batch.items()}
  jax_batch['inputs'] = fake_inputs

  # Test outputs for identical weights and inputs.
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

  ModelDiffRunner(
      jax_workload=jax_workload,
      pytorch_workload=pytorch_workload,
      jax_model_kwargs=jax_model_kwargs,
      pytorch_model_kwargs=pytorch_model_kwargs,
      key_transform=key_transform,
      sd_transform=sd_transform,
      out_transform=None).run()
