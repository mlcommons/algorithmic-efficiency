"""
Test algoperf.jax_utils.Dropout by comparing to flax.linen.Dropout
Run it as: pytest <path to this module>
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jax.tree_util import tree_leaves, tree_map, tree_structure

from algoperf.jax_utils import Dropout

SEED = 1996
DEFAULT_DROPOUT = 0.5


def pytrees_are_equal(a, b, rtol=1e-5, atol=1e-8):
  """
  A custom function to check if two PyTrees are equal, handling floats with
  a tolerance.
  """
  if tree_structure(a) != tree_structure(b):
    return False

  def leaf_comparator(x, y):
    # Use allclose for floating-point JAX arrays
    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
      return jnp.allclose(x, y, rtol=rtol, atol=atol)
    # Use standard equality for everything else
    else:
      return x == y

  comparison_tree = tree_map(leaf_comparator, a, b)
  all_equal = all(tree_leaves(comparison_tree))

  return all_equal


class LegacyDropoutModel(nn.Module):
  dropout_rate: float = DEFAULT_DROPOUT

  @nn.compact
  def __call__(self, x, train):
    return nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)


class DropoutModel(nn.Module):
  @nn.compact
  def __call__(self, x, train, dropout_rate=DEFAULT_DROPOUT):
    return Dropout(rate=dropout_rate, deterministic=not train)(
      x, rate=dropout_rate
    )


class DropoutTest(parameterized.TestCase):
  @parameterized.named_parameters(
    dict(testcase_name='Dropout, p=0.0, train', dropout_rate=0.0, mode='train'),
    dict(testcase_name='Dropout, p=0.0, eval', dropout_rate=0.0, mode='eval'),
    dict(testcase_name='Dropout, p=0.1, train', dropout_rate=0.1, mode='train'),
    dict(testcase_name='Dropout, p=0.1, eval', dropout_rate=0.1, mode='eval'),
  )
  def test_forward(self, dropout_rate, mode):
    """Compare forward pass of Dropout layer to flax.linen.Dropout in train and
    eval mode.
    """

    # initialize models
    rng, dropout_rng = jax.random.split(jax.random.key(SEED), 2)
    fake_batch = jnp.ones((10,))
    orig_model = LegacyDropoutModel(dropout_rate=dropout_rate)
    cust_model = DropoutModel()

    initial_variables_original = orig_model.init(
      {'params': rng}, fake_batch, train=False
    )
    initial_variables_custom = cust_model.init(
      {'params': rng}, fake_batch, train=False
    )

    assert pytrees_are_equal(
      initial_variables_original, initial_variables_custom, rtol=1e-6
    )

    # forward pass
    x = jnp.ones((10,))

    train = mode == 'train'
    y1 = orig_model.apply(
      initial_variables_original, x, train=train, rngs={'dropout': dropout_rng}
    )
    y2 = cust_model.apply(
      initial_variables_custom,
      x,
      train=train,
      dropout_rate=dropout_rate,
      rngs={'dropout': dropout_rng},
    )

    assert jnp.allclose(y1, y2, atol=1e-3, rtol=1e-3)

  @parameterized.named_parameters(
    dict(testcase_name='Dropout, p=0.0, train', dropout_rate=0.0, mode='train'),
    dict(testcase_name='Dropout, p=0.0, eval', dropout_rate=0.0, mode='eval'),
    dict(testcase_name='Dropout, p=0.1, train', dropout_rate=0.1, mode='train'),
    dict(testcase_name='Dropout, p=0.1, eval', dropout_rate=0.1, mode='eval'),
  )
  def test_dropout_update(self, dropout_rate, mode):
    """Call forward pass of Dropout layer with two different dropout rates
    and check that the output matches to flax.linen.Dropout in train and
    eval mode.
    """
    # init model
    rng, dropout_rng = jax.random.split(jax.random.key(SEED), 2)
    fake_batch = jnp.ones((10,))
    orig_model = LegacyDropoutModel(dropout_rate=dropout_rate)
    cust_model = DropoutModel()

    initial_variables_original = orig_model.init(
      {'params': rng}, fake_batch, train=False
    )

    initial_variables_custom = cust_model.init(
      {'params': rng}, fake_batch, train=False
    )

    assert pytrees_are_equal(
      initial_variables_original, initial_variables_custom, rtol=1e-6
    )

    # forward pass
    x = jnp.ones((10,))

    train = mode == 'train'
    y1 = orig_model.apply(
      initial_variables_original, x, train=train, rngs={'dropout': dropout_rng}
    )

    _ = cust_model.apply(
      initial_variables_custom,
      x,
      train=train,
      dropout_rate=0.9,
      rngs={'dropout': dropout_rng},
    )

    y2 = cust_model.apply(
      initial_variables_custom,
      x,
      train=train,
      dropout_rate=dropout_rate,
      rngs={'dropout': dropout_rng},
    )
    assert jnp.allclose(y1, y2, atol=1e-3, rtol=1e-3)

  @parameterized.named_parameters(
    dict(testcase_name='Dropout, p=0.0, train', dropout_rate=0.0, mode='train'),
    dict(testcase_name='Dropout, p=0.0, eval', dropout_rate=0.0, mode='eval'),
    dict(testcase_name='Dropout, p=0.1, train', dropout_rate=0.1, mode='train'),
    dict(testcase_name='Dropout, p=0.1, eval', dropout_rate=0.1, mode='eval'),
  )
  def test_jitted_updates(self, dropout_rate, mode):
    """Compare jitted updates with dropout."""

    # initialize models
    rng, dropout_rng = jax.random.split(jax.random.key(SEED), 2)
    fake_batch = jnp.ones((10,))
    orig_model = LegacyDropoutModel(dropout_rate=dropout_rate)
    cust_model = DropoutModel()

    initial_variables_original = orig_model.init(
      {'params': rng}, fake_batch, train=False
    )
    initial_variables_custom = cust_model.init(
      {'params': rng}, fake_batch, train=False
    )

    assert pytrees_are_equal(
      initial_variables_original, initial_variables_custom, rtol=1e-6
    )

    # forward pass
    x = jnp.ones((10,))

    train = mode == 'train'
    jitted_original_apply = jax.jit(
      partial(orig_model.apply), static_argnames=['train']
    )
    jitted_custom_apply = jax.jit(
      partial(cust_model.apply), static_argnames=['train']
    )

    for d in [i * 0.1 * dropout_rate for i in range(0, 11)]:
      y1 = jitted_original_apply(
        initial_variables_original,
        x,
        train=train,
        rngs={'dropout': dropout_rng},
      )

    for d in [i * 0.1 * dropout_rate for i in range(0, 11)]:
      y2 = jitted_custom_apply(
        initial_variables_custom,
        x,
        train=train,
        dropout_rate=d,
        rngs={'dropout': dropout_rng},
      )
    assert jnp.allclose(y1, y2, atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
