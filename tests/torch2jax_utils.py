from collections import Counter
import pprint


def jax_like_pytorch_sd(m, ret, keys=None):
  if keys is None:
    keys = []
  c = Counter()
  children = list(m.children())

  for k, v in m.named_parameters():
    if '.' not in k:
      ret[tuple(keys + [k])] = v
  for i in children:
    num_params = sum(p.numel() for p in i.parameters() if p.requires_grad)
    if num_params != 0:
      name = i.__class__.__name__
      k = f"{name}_{c[name]}"
      c[name] += 1
      jax_like_pytorch_sd(i, ret, keys + [k])


def flatten(jm, ret, keys=None):
  if keys is None:
    keys = []
  for k in jm:
    if 'dict' not in str(type(jm[k])).lower():
      ret[tuple(keys + [k])] = jm[k]
    else:
      flatten(jm[k], ret, keys + [k])


class Torch2Jax:

  def __init__(self, torch_model, jax_model):
    self.torch_model = torch_model
    self.jax_model = jax_model

    self.pytorch_sd = {}
    jax_like_pytorch_sd(torch_model, self.pytorch_sd)

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
    print(f"Keys in jax but not in pytorch: {len(j_p)}")
    pprint.pprint(sorted(list(j_p)))

    print(f"Keys in pytorch but not in jax: {len(p_j)}")
    pprint.pprint(sorted(list(p_j)))

    print(f"Common keys: {len(pj)}")

    if len(pj) == len(self.pytorch_sd):
      count = 0
      for k in self.pytorch_sd:
        s_p = list(self.pytorch_sd[k].shape)
        s_j = list(self.flattened_jax_model[k].shape)
        if s_p == s_j:
          count += 1
        else:
          print(k, s_p, s_j)
      print(f"Number of values with identical shapes: {count}")

  def update_jax_model(self):
    for k in self.flattened_jax_model:
      d = self.jax_model
      for i in k[:-1]:
        d = d[i]
      d[k[-1]] = self.pytorch_sd[k].detach().cpu().numpy()
