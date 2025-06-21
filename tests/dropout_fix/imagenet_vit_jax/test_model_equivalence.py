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

from algoperf.workloads.imagenet_vit.imagenet_jax.models_ref import \
    ViT as OriginalVit
from algoperf.workloads.imagenet_vit.imagenet_jax.models import \
    ViT as CustomVit

# Model / test hyper-params
INPUT_SHAPE = (2, 224, 124, 3)
SEED = 1994

class ImageNetVitModeEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='ViT, p=0.0',
          dropout_rate=0.0),
      dict(
          testcase_name='ViT, p=0.1',
          dropout_rate=0.1),
  )
  def test_forward(self, dropout_rate):
    OrigCls, CustCls = (OriginalVit, CustomVit)


    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls()
    cust_model = CustCls()

    fake_batch = jnp.ones(INPUT_SHAPE)

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jax.random.normal(data_rng, shape=INPUT_SHAPE)

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
    OrigCls, CustCls = (OriginalVit, CustomVit)


    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls()
    cust_model = CustCls()

    fake_batch = jnp.ones(INPUT_SHAPE)

    initial_params_original = orig_model.init({'params': rng},
                                              fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            fake_batch,
                                            train=False)

    # fwd
    x = jax.random.normal(data_rng, INPUT_SHAPE)

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