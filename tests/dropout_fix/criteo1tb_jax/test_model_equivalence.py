"""
Runs fwd pass with random input for our DLRM models and compares outputs.
Run it as:
  python3 tests/dropout_fix/criteo1tb_jax/test_model_equivalence.py
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
# import equinox as eqx

from jax.tree_util import tree_structure, tree_leaves, tree_map


def pytrees_are_equal(a, b, rtol=1e-5, atol=1e-8):
  """
    A custom function to check if two PyTrees are equal, handling floats with a tolerance.
    """
  # 1. Check if the structures are the same
  if tree_structure(a) != tree_structure(b):
    return False

  # 2. Define a comparison function for leaves
  def leaf_comparator(x, y):
    # Use allclose for floating-point JAX arrays
    if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
      return jnp.allclose(x, y, rtol=rtol, atol=atol)
    # Use standard equality for everything else
    else:
      return x == y

  # 3. Map the comparison function over the trees and check if all results are True
  # We also need to flatten the results of the tree_map and check if all are True
  comparison_tree = tree_map(leaf_comparator, a, b)
  all_equal = all(tree_leaves(comparison_tree))

  return all_equal

from algoperf.workloads.criteo1tb.criteo1tb_jax.models_ref import \
    DLRMResNet as OriginalDLRMResNet
from algoperf.workloads.criteo1tb.criteo1tb_jax.models_ref import \
    DlrmSmall as OriginalDlrmSmall
from algoperf.workloads.criteo1tb.criteo1tb_jax.models import \
    DLRMResNet as CustomDLRMResNet
from algoperf.workloads.criteo1tb.criteo1tb_jax.models import \
    DlrmSmall as CustomDlrmSmall

BATCH, DENSE, SPARSE = 16, 13, 26
FEATURES = DENSE + SPARSE
VOCAB = 1000
DEVICE = 'cuda'
SEED = 1996


class ModelEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='DLRMResNet, p=0.0',
          model='dlrm_resnet',
          dropout_rate=0.0),
      dict(
          testcase_name='DlrmSmall, p=0.0',
          model='dlrm_small',
          dropout_rate=0.0),
      dict(
          testcase_name='DLRMResNet, p=0.1',
          model='dlrm_resnet',
          dropout_rate=0.1),
      dict(
          testcase_name='DlrmSmall, p=0.1',
          model='dlrm_small',
          dropout_rate=0.1),
  )
  def test_forward(self, model, dropout_rate):
    OrigCls, CustCls = (
        (OriginalDLRMResNet, CustomDLRMResNet)
        if model == 'dlrm_resnet'
        else (OriginalDlrmSmall, CustomDlrmSmall)
    )

    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)
    fake_batch = jnp.ones((2, 39))
    assert dropout_rate == 0.1
    orig_model = OrigCls(vocab_size=VOCAB, dropout_rate=dropout_rate)
    cust_model = CustCls(vocab_size=VOCAB)
    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)
    assert pytrees_are_equal(
        initial_params_original, initial_params_custom, rtol=1e-6)

    x = jax.random.normal(data_rng, shape=(BATCH, FEATURES))

    for mode in ('train', 'eval'):
      train = mode == 'train'
      y1 = orig_model.apply(
          initial_params_original,
          x,
          train=train,
          rngs={'dropout': dropout_rng})
      y2 = cust_model.apply(
          initial_params_custom,
          x,
          train=train,
          dropout_rate=dropout_rate,
          rngs={'dropout': dropout_rng})

      assert jnp.allclose(y1, y2, atol=1e-3, rtol=1e-3)

  @parameterized.named_parameters(
      dict(testcase_name='DLRMResNet, default', model='dlrm_resnet'),
      dict(testcase_name='DlrmSmall, default', model='dlrm_small'),
  )
  def test_default_dropout(self, model):
    """Test default dropout_rate."""
    OrigCls, CustCls = (
        (OriginalDLRMResNet, CustomDLRMResNet)
        if model == 'dlrm_resnet'
        else (OriginalDlrmSmall, CustomDlrmSmall)
    )
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)
    fake_batch = jnp.ones((2, 39))
    orig_model = OrigCls(vocab_size=VOCAB)
    cust_model = CustCls(vocab_size=VOCAB)
    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    x = jax.random.normal(data_rng, shape=(BATCH, FEATURES))

    for mode in ('train', 'eval'):
      train = mode == 'train'
      y1 = orig_model.apply(
          initial_params_original,
          x,
          train=train,
          rngs={'dropout': dropout_rng})
      y2 = cust_model.apply(
          initial_params_custom, x, train=train, rngs={'dropout': dropout_rng})

      assert jnp.allclose(y1, y2, atol=0, rtol=0)


if __name__ == '__main__':
  absltest.main()
