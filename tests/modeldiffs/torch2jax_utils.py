from collections import Counter
import pprint


def jax_like_pytorch_statedict(model, state_dict, keys=None):
  if keys is None:
    keys = []
  c = Counter()
  children = list(model.children())

  for k, v in model.named_parameters():
    if '.' not in k:
      state_dict[(*keys, k)] = v
  for i in children:
    num_params = sum(p.numel() for p in i.parameters() if p.requires_grad)
    if num_params != 0:
      name = i.__class__.__name__
      k = f'{name}_{c[name]}'
      c[name] += 1
      jax_like_pytorch_statedict(i, state_dict, keys + [k])


def flatten(jm, ret, keys=None):
  if keys is None:
    keys = []
  for k in jm:
    if isinstance(jm[k], dict):
      flatten(jm[k], ret, keys + [k])
    else:
      ret[tuple(keys + [k])] = jm[k]


def value_transform(k, value, jax_value):
  k_str = ''.join(k).lower()
  if ('conv' in k_str and 'kernel' in k_str) or \
    ('embedding' in k_str and 'kernel' in k_str):
    if 'transpose' in k_str:
      # Assumes 2D ConvTranspose with stride equal to kernel_size.
      return value.reshape(value.shape[0], value.shape[1],
                           -1).flip(-1).permute(2, 0,
                                                1).reshape(*jax_value.shape)
    else:
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
  elif ('dense' in k_str and 'kernel' in k_str) or \
    ('lstm' in k_str and 'kernel' in k_str) or \
    ('head' in k_str and 'kernel' in k_str) or \
    ('pre_logits' in k_str and 'kernel' in k_str):
    value = value.t()
  return value


class Torch2Jax:

  def __init__(self, torch_model, jax_model):
    self.torch_model = torch_model
    self.jax_model = jax_model

    self.pytorch_sd = {}
    jax_like_pytorch_statedict(torch_model, self.pytorch_sd)

    self.flattened_jax_model = {}
    flatten(jax_model, self.flattened_jax_model)

  def key_transform(self, k_transform_fn):
    self.pytorch_sd = {
        k_transform_fn(k): self.pytorch_sd[k] for k in self.pytorch_sd
    }

  def value_transform(self, v_transform_fn):
    self.pytorch_sd = {
        k: v_transform_fn(k, self.pytorch_sd[k], self.flattened_jax_model[k])
        for k in self.pytorch_sd
    }

  def sd_transform(self, sd_transform_fn):
    self.pytorch_sd = sd_transform_fn(self.pytorch_sd)

  def diff(self):
    j_p = set(self.flattened_jax_model.keys()) - set(self.pytorch_sd.keys())
    p_j = set(self.pytorch_sd.keys()) - set(self.flattened_jax_model.keys())
    pj = set(self.pytorch_sd.keys()) & set(self.flattened_jax_model.keys())
    print(f'Keys in jax but not in pytorch: {len(j_p)}')
    pprint.pprint(sorted(list(j_p)))

    print(f'Keys in pytorch but not in jax: {len(p_j)}')
    pprint.pprint(sorted(list(p_j)))

    print(f'Common keys: {len(pj)}')

    if len(pj) == len(self.pytorch_sd):
      count = 0
      for k in self.pytorch_sd:
        s_p = list(self.pytorch_sd[k].shape)
        s_j = list(self.flattened_jax_model[k].shape)
        if s_p == s_j:
          count += 1
        else:
          print('Difference in pytorch and jax key:')
          print(k, s_p, s_j)
      print(f'Number of values with identical shapes: {count}')

  def update_jax_model(self):
    for k in self.flattened_jax_model:
      d = self.jax_model
      for i in k[:-1]:
        d = d[i]
      d[k[-1]] = self.pytorch_sd[k].detach().cpu().numpy()
