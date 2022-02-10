"""Proxy functions in front of the Jax RNG API or a compatible Numpy RNG API."""
from absl import flags
from absl import logging
import numpy as np

try:
  import jax.random as jax_rng
except (ImportError, ModuleNotFoundError):
  logging.warning(
      'Could not import jax.random for the submission runner, falling back to '
      'numpy random_utils.')
  jax_rng = None

FLAGS = flags.FLAGS

# Annoyingly, RandomState(seed) requires seed to be in [0, 2 ** 32 - 1] (an
# unsigned int), while RandomState.randint only accepts and returns signed ints.
MAX_INT32 = 2**31
MIN_INT32 = -MAX_INT32


def _signed_to_unsigned(seed):
  if isinstance(seed, int):
    return seed + 2**32 if seed < 0 else seed
  if isinstance(seed, list):
    return [s + 2**32 if s < 0 else s for s in seed]
  if isinstance(seed, np.ndarray):
    return np.array([s + 2**32 if s < 0 else s for s in seed.tolist()])


def _fold_in(seed, data):
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  new_seed = rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32)
  return [new_seed, data]


def _split(seed, num=2):
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  return rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=[num, 2])


def _prng_key(seed: int):
  return split(seed, num=2)[0]


# It is usually bad practice to use FLAGS outside of the main() function, but
# the alternative is having to pipe the framework flag to all functions that may
# need it, which seems unnecessarily cumbersome.
def _check_jax_install():
  if jax_rng is None:
    raise ValueError(
        'Must install jax to use the jax RNG library, or use PyTorch and pass '
        '--framework=pytorch to use the Numpy version instead.')


def fold_in(seed, data):
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.fold_in(seed, data)
  return _fold_in(seed, data)


def split(seed, num=2):
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.split(seed, num)
  return _split(seed, num)


def prng_key(seed):
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.PRNGKey(seed)
  return _prng_key(seed)
