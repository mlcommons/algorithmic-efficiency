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

from algoperf.workloads.librispeech_deepspeech.librispeech_jax.models import DeepspeechConfig as CustClsConfig
from algoperf.workloads.librispeech_deepspeech.librispeech_jax.models import Deepspeech as CustCls

from algoperf.workloads.librispeech_deepspeech.librispeech_jax.models_ref import DeepspeechConfig as OrigClsConfig
from algoperf.workloads.librispeech_deepspeech.librispeech_jax.models_ref import Deepspeech as OrigCls


# Model / test hyper-params
INPUT_SHAPE = [(3200,), (3200,)]
SEED = 1994

class ModeEquivalenceTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Conformer, p=0.0',
          dropout_rate=0.0),
      dict(
          testcase_name='Conformer, p=0.1',
          dropout_rate=0.1),
  )
  def test_forward(self, dropout_rate):

    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(OrigClsConfig)
    cust_model = CustCls(CustClsConfig)

    fake_batch = [jnp.zeros((2, *x), jnp.float32) for x in INPUT_SHAPE]

    initial_params_original = orig_model.init({'params': rng},
                                              *fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            *fake_batch,
                                            train=False)

    # fwd
    x = [jax.random.normal(data_rng, (2, *x)) for x in INPUT_SHAPE]

    for mode in ('train', 'eval'):
      train = mode == 'train'
      (y1, _), _ = orig_model.apply(
          initial_params_original,
          *x,
          train=train,
          rngs={'dropout': dropout_rng},
          mutable=['batch_stats'],)
      (y2, _), _ = cust_model.apply(
          initial_params_custom,
          *x,
          train=train,
          dropout_rate=dropout_rate,
          rngs={'dropout': dropout_rng},
          mutable=['batch_stats'])

      assert jnp.allclose(y1, y2)



  @parameterized.named_parameters(
      dict(testcase_name='Conformer, default'),
  )
  def test_default_dropout(self):
    """Test default dropout_rate."""
    # init model
    rng, data_rng, dropout_rng = jax.random.split(jax.random.key(SEED), 3)

    orig_model = OrigCls(OrigClsConfig)
    cust_model = CustCls(CustClsConfig)

    fake_batch = [jnp.zeros((2, *x), jnp.float32) for x in INPUT_SHAPE]

    initial_params_original = orig_model.init({'params': rng},
                                               *fake_batch,
                                              train=False)
    initial_params_custom = cust_model.init({'params': rng},
                                            *fake_batch,
                                            train=False)

    # fwd
    x = [jax.random.normal(data_rng, (2, *x)) for x in INPUT_SHAPE]

    for mode in ('train', 'eval'):
      train = mode == 'train'
      (y1, _), _ = orig_model.apply(
          initial_params_original,
          *x,
          train=train,
          rngs={'dropout': dropout_rng}, mutable=['batch_stats'])
      (y2, _), _ = cust_model.apply(
          initial_params_custom, *x, train=train, rngs={'dropout': dropout_rng}, mutable=['batch_stats'])

      
      assert jnp.allclose(y1, y2)


if __name__ == '__main__':
  absltest.main()