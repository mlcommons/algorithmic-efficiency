from flax import jax_utils
import jax
import numpy as np
import torch

from tests.modeldiffs.torch2jax_utils import Torch2Jax
from tests.modeldiffs.torch2jax_utils import value_transform


#pylint: disable=dangerous-default-value
def torch2jax(jax_workload,
              pytorch_workload,
              key_transform=None,
              sd_transform=None,
              init_kwargs=dict(dropout_rate=0.0, aux_dropout_rate=0.0)):
  jax_params, model_state = jax_workload.init_model_fn(jax.random.PRNGKey(0),
                                                       **init_kwargs)
  pytorch_model, _ = pytorch_workload.init_model_fn([0], **init_kwargs)
  jax_params = jax_utils.unreplicate(jax_params).unfreeze()
  if model_state is not None:
    model_state = jax_utils.unreplicate(model_state)

  if isinstance(
      pytorch_model,
      (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
    pytorch_model = pytorch_model.module
  # Map and copy params of pytorch_model to jax_model.
  t2j = Torch2Jax(torch_model=pytorch_model, jax_model=jax_params)
  if key_transform is not None:
    t2j.key_transform(key_transform)
  if sd_transform is not None:
    t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()
  return jax_params, model_state, pytorch_model


def out_diff(jax_workload,
             pytorch_workload,
             jax_model_kwargs,
             pytorch_model_kwargs,
             key_transform=None,
             sd_transform=None,
             out_transform=None):
  jax_params, model_state, pytorch_model = torch2jax(jax_workload,
                                                     pytorch_workload,
                                                     key_transform,
                                                     sd_transform)
  out_p, _ = pytorch_workload.model_fn(params=pytorch_model,
                                       **pytorch_model_kwargs)
  out_j, _ = jax_workload.model_fn(params=jax_params,
                                   model_state=model_state,
                                   **jax_model_kwargs)
  if out_transform is not None:
    out_p = out_transform(out_p)
    out_j = out_transform(out_j)

  max_diff = np.abs(out_p.detach().numpy() - np.array(out_j)).max()
  min_diff = np.abs(out_p.detach().numpy() - np.array(out_j)).min()

  logging.info(f'Max fprop difference between jax and pytorch: {max_diff}')
  logging.info(f'Min fprop difference between jax and pytorch: {min_diff}')
