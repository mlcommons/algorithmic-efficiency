"""Proxy functions in front of the Jax RNG API or a compatible Numpy RNG API."""

from typing import Union

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

# SALT constants
_SALT1 = np.random.RandomState(seed=5).randint(
    MIN_INT32, MAX_INT32, dtype=np.int32)
_SALT2 = np.random.RandomState(seed=6).randint(
    MIN_INT32, MAX_INT32, dtype=np.int32)

SeedType = Union[int, list, np.ndarray]


def _signed_to_unsigned(seed: SeedType) -> SeedType:
  if isinstance(seed, int):
    return seed + 2**32 if seed < 0 else seed
  if isinstance(seed, list):
    return [s + 2**32 if s < 0 else s for s in seed]
  if isinstance(seed, np.ndarray):
    return np.array([s + 2**32 if s < 0 else s for s in seed.tolist()])


def _fold_in(seed: SeedType, data: int) -> SeedType:
  a = np.random.RandomState(seed=_signed_to_unsigned(seed ^ _SALT1)).randint(
      MIN_INT32, MAX_INT32, dtype=np.int32)
  b = np.random.RandomState(seed=_signed_to_unsigned(data ^ _SALT2)).randint(
      MIN_INT32, MAX_INT32, dtype=np.int32)
  c = np.random.RandomState(seed=_signed_to_unsigned(a ^ b)).randint(
      MIN_INT32, MAX_INT32, dtype=np.int32)
  return c


def _split(seed: SeedType, num: int = 2) -> SeedType:
  rng = np.random.RandomState(seed=_signed_to_unsigned(seed))
  return rng.randint(MIN_INT32, MAX_INT32, dtype=np.int32, size=[num])


def _PRNGKey(seed: SeedType) -> SeedType:  # pylint: disable=invalid-name
  return split(seed, num=2)[0]


# It is usually bad practice to use FLAGS outside of the main() function, but
# the alternative is having to pipe the framework flag to all functions that may
# need it, which seems unnecessarily cumbersome.
def _check_jax_install() -> None:
  if jax_rng is None:
    raise ValueError(
        'Must install jax to use the jax RNG library, or use PyTorch and pass '
        '--framework=pytorch to use the Numpy version instead.')


def _bits(seed: SeedType) -> int:
  rng = np.random.RandomState(_signed_to_unsigned(seed))
  b = rng.bytes(4)
  return int.from_bytes(b, byteorder='little')


def fold_in(seed: SeedType, data: int) -> SeedType:
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.fold_in(seed, data)
  return _fold_in(seed, data)


def split(seed: SeedType, num: int = 2) -> SeedType:
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.split(seed, num)
  return _split(seed, num)


def PRNGKey(seed: SeedType) -> SeedType:  # pylint: disable=invalid-name
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.PRNGKey(seed)
  return _PRNGKey(seed)


def bits(seed: SeedType) -> int:
  if FLAGS.framework == 'jax':
    _check_jax_install()
    return jax_rng.bits(seed)
  return _bits(seed)
