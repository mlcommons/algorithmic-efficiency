"""
Runs fwd pass with random input for OGBG
"""

import os

import jraph

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from algoperf.workloads.ogbg.ogbg_jax.models_ref import \
    GNN as OrigCls
from algoperf.workloads.ogbg.ogbg_jax.models import \
    GNN as CustCls

# Model / test hyper-params
SEED = 1994

class ModeEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='OGBG, p=0.0',
          dropout_rate=0.0),
      dict(
          testcase_name='OGBG, p=0.1',
          dropout_rate=0.1),
  )
  def test_forward(self, dropout_rate):
    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(num_outputs=128, dropout_rate=dropout_rate)
    cust_model = CustCls(num_outputs=128)

    fake_batch = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, 128)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, 128)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

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
      dict(testcase_name='OGBG, default'),
  )
  def test_default_dropout(self):
    """Test default dropout_rate."""


    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(num_outputs=128)
    cust_model = CustCls(num_outputs=128)

    fake_batch = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, 128)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jraph.GraphsTuple(
        n_node=jnp.asarray([1]),
        n_edge=jnp.asarray([1]),
        nodes=jnp.ones((1, 9)),
        edges=jnp.ones((1, 3)),
        globals=jnp.zeros((1, 128)),
        senders=jnp.asarray([0]),
        receivers=jnp.asarray([0]))

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