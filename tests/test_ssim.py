"""Test for the equality of the SSIM calculation in Jax and PyTorch."""

import os
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import torch

from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.fastmri.fastmri_jax.ssim import \
    _uniform_filter as _jax_uniform_filter
from algoperf.workloads.fastmri.fastmri_jax.ssim import \
    ssim as jax_ssim
from algoperf.workloads.fastmri.fastmri_pytorch.ssim import \
    _uniform_filter as _pytorch_uniform_filter
from algoperf.workloads.fastmri.fastmri_pytorch.ssim import \
    ssim as pytorch_ssim

# Make sure no GPU memory is preallocated to Jax.
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
DEVICE = pytorch_setup()[2]


def _create_fake_im(height: int, width: int) -> Tuple[jnp.array, torch.Tensor]:
  fake_im = np.random.randn(height, width)
  jax_fake_im = jnp.asarray(fake_im)
  pytorch_fake_im = torch.as_tensor(fake_im, device=DEVICE)
  return jax_fake_im, pytorch_fake_im


def _create_fake_batch(
    batch_size: int, height: int, width: int
) -> Tuple[Tuple[jnp.array, jnp.array], Tuple[torch.Tensor, torch.Tensor]]:
  logits = np.random.randn(batch_size, height, width)
  targets = np.random.randn(batch_size, height, width)
  jax_logits = jnp.asarray(logits)
  jax_targets = jnp.asarray(targets)
  pytorch_logits = torch.as_tensor(logits, device=DEVICE)
  pytorch_targets = torch.as_tensor(targets, device=DEVICE)
  return (jax_logits, jax_targets), (pytorch_logits, pytorch_targets)


class SSIMTest(parameterized.TestCase):
  """Test for equivalence of SSIM and _uniform_filter implementations in Jax
  and PyTorch."""

  @parameterized.named_parameters(
      dict(testcase_name='fastmri_im', height=320, width=320),
      dict(testcase_name='uneven_even_im', height=31, width=16),
      dict(testcase_name='even_uneven_im', height=42, width=53),
  )
  def test_uniform_filter(self, height: int, width: int) -> None:
    jax_im, pytorch_im = _create_fake_im(height, width)
    jax_result = np.asarray(_jax_uniform_filter(jax_im))
    torch_result = _pytorch_uniform_filter(pytorch_im).cpu().numpy()
    assert np.allclose(jax_result, torch_result, atol=1e-6)

  @parameterized.named_parameters(
      dict(
          testcase_name='fastmri_batch', batch_size=256, height=320, width=320),
      dict(
          testcase_name='uneven_even_batch', batch_size=8, height=31, width=16),
      dict(
          testcase_name='even_uneven_batch', batch_size=8, height=42, width=53),
  )
  def test_ssim(self, batch_size: int, height: int, width: int) -> None:
    jax_inputs, pytorch_inputs = _create_fake_batch(batch_size, height, width)
    jax_ssim_result = jax_ssim(*jax_inputs)
    pytorch_ssim_result = pytorch_ssim(*pytorch_inputs)
    self.assertEqual(jax_ssim_result.shape, pytorch_ssim_result.shape)
    assert np.allclose(
        jax_ssim_result.sum().item(),
        pytorch_ssim_result.sum().item(),
        atol=1e-6)


if __name__ == '__main__':
  absltest.main()
