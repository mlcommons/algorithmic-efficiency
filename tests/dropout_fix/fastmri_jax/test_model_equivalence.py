"""
Runs fwd pass with random input for FASTMRI U-Net models and compares outputs.
Run it as:
  python3 tests/dropout_fix/fastmri_pytorch/test_model_equivalence.py
"""

import os


from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
# import equinox as eqx


from algoperf.workloads.fastmri.fastmri_jax.models_ref import \
    UNet as OriginalUNet
from algoperf.workloads.fastmri.fastmri_jax.models import \
    UNet as CustomUNet

BATCH, IN_CHANS, H, W = 4, 1, 256, 256
OUT_CHANS, C, LAYERS = 1, 32, 4
SEED = 1996


class ModelEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='UNet, p=0.0',
          dropout_rate=0.0),
      dict(
          testcase_name='UNet, p=0.1',
          dropout_rate=0.1),
  )
  def test_forward(self, dropout_rate):
    OrigCls, CustCls = (OriginalUNet, CustomUNet)


    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    kwargs = dict(num_pool_layers = LAYERS, num_channels=IN_CHANS)
    orig_model = OrigCls(**kwargs)
    cust_model = CustCls(**kwargs)

    fake_batch = jnp.ones((BATCH, IN_CHANS, H, W))

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jax.random.normal(data_rng, shape=(BATCH, H, W))

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
      dict(testcase_name='UNet, default'),
  )
  def test_default_dropout(self):
    """Test default dropout_rate."""
    OrigCls, CustCls = (OriginalUNet, CustomUNet)


    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    kwargs = dict(num_pool_layers=LAYERS, 
                  num_channels=IN_CHANS,
                  )
    orig_model = OrigCls(**kwargs)
    cust_model = CustCls(**kwargs)

    fake_batch = jnp.ones((2, IN_CHANS, H, W))

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jax.random.normal(data_rng, shape=(BATCH, H, W))

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
