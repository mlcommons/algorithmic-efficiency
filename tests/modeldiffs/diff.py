from flax import jax_utils
import jax
import numpy as np

from tests.modeldiffs.torch2jax_utils import Torch2Jax
from tests.modeldiffs.torch2jax_utils import value_transform


def out_diff(jax_workload,
             pytorch_workload,
             jax_model_kwargs,
             pytorch_model_kwargs,
             key_transform=None,
             sd_transform=None,
             out_transform=None):
  jax_params, model_state = jax_workload.init_model_fn(jax.random.PRNGKey(0))
  pytorch_model, _ = pytorch_workload.init_model_fn([0])
  jax_params = jax_utils.unreplicate(jax_params).unfreeze()
  model_state = jax_utils.unreplicate(model_state)

  # Map and copy params of pytorch_model to jax_model.
  t2j = Torch2Jax(torch_model=pytorch_model, jax_model=jax_params)
  if key_transform is not None:
    t2j.key_transform(key_transform)
  if sd_transform is not None:
    t2j.sd_transform(sd_transform)
  t2j.value_transform(value_transform)
  t2j.diff()
  t2j.update_jax_model()

  out_p, _ = pytorch_workload.model_fn(params=pytorch_model,
                                       **pytorch_model_kwargs)
  out_j, _ = jax_workload.model_fn(params=jax_params,
                                   model_state=model_state,
                                   **jax_model_kwargs)
  if out_transform is not None:
    out_p = out_transform(out_p)
    out_j = out_transform(out_j)

  print(np.abs(out_p.detach().numpy() - np.array(out_j)).max())
  print(np.abs(out_p.detach().numpy() - np.array(out_j)).min())
