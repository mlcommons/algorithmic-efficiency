import numpy as np

# Annoyingly, RandomState(seed) requires seed to be in [0, 2 ** 32 - 1] (an
# unsigned int), while RandomState.randint only accepts and returns signed ints.
MAX_INT32 = 2 ** 31
MIN_INT32 = -MAX_INT32


def _signed_to_unsigned(seed):
  if isinstance(seed, int):
    return seed + 2 ** 32 if seed < 0 else seed
  if isinstance(seed, list):
    return [s + 2 ** 32 if s < 0 else s for s in seed]
  if isinstance(seed, np.ndarray):
    return np.array([s + 2 ** 32 if s < 0 else s for s in seed.tolist()])


def fold_in(seed, data):
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  new_seed = rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32)
  return [new_seed, data]


def split(seed, num=2):
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  return rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=[num, 2])


def PRNGKey(seed: int):
  return split(seed, num=2)[0]
