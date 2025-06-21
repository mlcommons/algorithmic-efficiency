"""
Runs fwd pass with random input for FASTMRI U-Net models and compares outputs.
Run it as:
  python3 tests/dropout_fix/imagenet_vit_jax/test_model_equivalence.py
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from algoperf.workloads.wmt.wmt_jax.models import TransformerConfig as CustClsConfig
from algoperf.workloads.wmt.wmt_jax.models import Transformer as CustCls

from algoperf.workloads.wmt.wmt_jax.models_ref import TransformerConfig as OrigClsConfig
from algoperf.workloads.wmt.wmt_jax.models_ref import Transformer as OrigCls


# Model / test hyper-params
SEED = 1994

class ModeEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='WMT, p=0.0',
          dropout_rate=0.0),
      dict(
          testcase_name='WMT p=0.1',
          dropout_rate=0.1),
  )
  def test_forward(self, dropout_rate):

    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(OrigClsConfig)
    cust_model = CustCls(CustClsConfig)

    init_fake_batch_size = 8
    input_shape = (init_fake_batch_size, 256)
    target_shape = (init_fake_batch_size, 256)

    initial_params_original = orig_model.init({'params': rng},
                                              jnp.ones(input_shape, jnp.float32),
                                              jnp.ones(target_shape, jnp.float32),
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            jnp.ones(input_shape, jnp.float32),
                                            jnp.ones(target_shape, jnp.float32),
                                            train=False)

    # fwd

    for mode in ('train', 'eval'):
      train = mode == 'train'
      y1 = orig_model.apply(
          initial_params_original,
          jnp.ones(input_shape, jnp.float32),
          jnp.ones(target_shape, jnp.float32),
          train=train,
          rngs={'dropout': dropout_rng},
          mutable=['batch_stats'],)
      y2 = cust_model.apply(
          initial_params_custom,
          jnp.ones(input_shape, jnp.float32),
          jnp.ones(target_shape, jnp.float32),
          train=train,
          dropout_rate=dropout_rate,
          rngs={'dropout': dropout_rng},
          mutable=['batch_stats'])

      for i in range(len(y1)):
        assert jnp.allclose(y1[i], y2[i])



  @parameterized.named_parameters(
      dict(testcase_name='WMT, default'),
  )
  def test_default_dropout(self):
    """Test default dropout_rate."""
    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(OrigClsConfig)
    cust_model = CustCls(CustClsConfig)

    init_fake_batch_size = 8
    input_shape = (init_fake_batch_size, 256)
    target_shape = (init_fake_batch_size, 256)

    initial_params_original = orig_model.init({'params': rng},
                                              jnp.ones(input_shape, jnp.float32),
                                              jnp.ones(target_shape, jnp.float32),
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            jnp.ones(input_shape, jnp.float32),
                                            jnp.ones(target_shape, jnp.float32),
                                            train=False)

    # fwd
    x = [jax.random.normal(data_rng, (2, *x)) for x in INPUT_SHAPE]

    for mode in ('train', 'eval'):
      train = mode == 'train'
      y1 = orig_model.apply(
          initial_params_original,
          jnp.ones(input_shape, jnp.float32),
          jnp.ones(target_shape, jnp.float32),
          train=train,
          rngs={'dropout': dropout_rng}, mutable=['batch_stats'])
      y2 = cust_model.apply(
          initial_params_custom,
          jnp.ones(input_shape, jnp.float32),
          jnp.ones(target_shape, jnp.float32),
          train=train, rngs={'dropout': dropout_rng},
          mutable=['batch_stats'])

      print(jax.tree.map(lambda x: x.shape, y1))
      
      for i in range(len(y1)):
        assert jnp.allclose(y1[i], y2[i])


if __name__ == '__main__':
  absltest.main()