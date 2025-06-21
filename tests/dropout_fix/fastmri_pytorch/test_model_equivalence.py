"""
Runs fwd pass with random input for FASTMRI U-Net models and compares outputs.
Run it as:
  python3 tests/dropout_fix/fastmri_pytorch/test_model_equivalence.py
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.testing import assert_close

from algoperf.workloads.fastmri.fastmri_pytorch.models import \
    UNet as OriginalUNet
from algoperf.workloads.fastmri.fastmri_pytorch.models_dropout import \
    UNet as CustomUNet

BATCH, IN_CHANS, H, W = 4, 1, 256, 256
OUT_CHANS, C, LAYERS = 1, 32, 4
DEVICE = 'cuda'
TORCH_COMPILE = False
SEED = 1996

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


class FastMRIModeEquivalenceTest(parameterized.TestCase):

  def fwd_pass(self, orig, cust, dropout_rate):
    x = torch.randn(BATCH, IN_CHANS, H, W, device=DEVICE)
    for mode in ('train', 'eval'):
      getattr(orig, mode)()
      getattr(cust, mode)()
      torch.manual_seed(0)
      y1 = orig(x)
      torch.manual_seed(0)
      y2 = cust(x, dropout_rate)
      assert_close(y1, y2, atol=0, rtol=0)
      if mode == 'eval':  # one extra test: omit dropout at eval
        torch.manual_seed(0)
        y2 = cust(x)
        assert_close(y1, y2, atol=0, rtol=0)

  @parameterized.named_parameters(
      dict(testcase_name='p=0.0', dropout_rate=0.0),
      dict(testcase_name='p=0.1', dropout_rate=0.1),
      dict(testcase_name='p=0.7', dropout_rate=0.7),
      dict(testcase_name='p=1.0', dropout_rate=1.0),
  )
  def test_dropout_values(self, dropout_rate):
    """Test different values of dropout_rate."""

    torch.manual_seed(SEED)
    orig = OriginalUNet(
        IN_CHANS, OUT_CHANS, C, LAYERS, dropout_rate=dropout_rate).to(DEVICE)

    torch.manual_seed(SEED)
    cust = CustomUNet(IN_CHANS, OUT_CHANS, C, LAYERS).to(DEVICE)

    cust.load_state_dict(orig.state_dict())  # sync weights
    if TORCH_COMPILE:
      orig = torch.compile(orig)
      cust = torch.compile(cust)

    self.fwd_pass(orig, cust, dropout_rate)

  @parameterized.named_parameters(
      dict(testcase_name='default', use_tanh=False, use_layer_norm=False),
      dict(testcase_name='tanh', use_tanh=True, use_layer_norm=False),
      dict(testcase_name='layer_norm', use_tanh=False, use_layer_norm=True),
      dict(testcase_name='both', use_tanh=True, use_layer_norm=True),
  )
  def test_arch_configs(self, use_tanh, use_layer_norm):
    """Test different architecture configurations, fixed dropout_rate."""
    dropout_rate = 0.1

    torch.manual_seed(SEED)
    orig = OriginalUNet(
        IN_CHANS,
        OUT_CHANS,
        C,
        LAYERS,
        dropout_rate=dropout_rate,
        use_tanh=use_tanh,
        use_layer_norm=use_layer_norm).to(DEVICE)

    torch.manual_seed(SEED)
    cust = CustomUNet(
        IN_CHANS,
        OUT_CHANS,
        C,
        LAYERS,
        use_tanh=use_tanh,
        use_layer_norm=use_layer_norm).to(DEVICE)

    cust.load_state_dict(orig.state_dict())  # sync weights
    if TORCH_COMPILE:
      orig = torch.compile(orig)
      cust = torch.compile(cust)

    self.fwd_pass(orig, cust, dropout_rate)

  @parameterized.named_parameters(
      dict(testcase_name=''),)
  def test_default_dropout(self):
    """Test default dropout_rate."""

    torch.manual_seed(SEED)
    orig = OriginalUNet(IN_CHANS, OUT_CHANS, C, LAYERS).to(DEVICE)
    torch.manual_seed(SEED)
    cust = CustomUNet(IN_CHANS, OUT_CHANS, C, LAYERS).to(DEVICE)
    cust.load_state_dict(orig.state_dict())  # sync weights

    x = torch.randn(BATCH, IN_CHANS, H, W, device=DEVICE)
    for mode in ('train', 'eval'):
      getattr(orig, mode)()
      getattr(cust, mode)()
      torch.manual_seed(0)
      y1 = orig(x)
      torch.manual_seed(0)
      y2 = cust(x)
      assert_close(y1, y2, atol=0, rtol=0)


if __name__ == '__main__':
  absltest.main()
