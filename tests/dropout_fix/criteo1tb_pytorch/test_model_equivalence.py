"""
Runs fwd pass with random input for our DLRM models and compares outputs.
Run it as:
  python3 tests/dropout_fix/criteo1tb_pytorch/test_model_equivalence.py
"""

import os

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch.testing import assert_close

from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models import \
    DLRMResNet as OriginalDLRMResNet
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models import \
    DlrmSmall as OriginalDlrmSmall
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models_dropout import \
    DLRMResNet as CustomDLRMResNet
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models_dropout import \
    DlrmSmall as CustomDlrmSmall

BATCH, DENSE, SPARSE = 16, 13, 26
FEATURES = DENSE + SPARSE
VOCAB = 1000
DEVICE = 'cuda'
SEED = 1996

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


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
      dict(
          testcase_name='DLRMResNet, p=1.0',
          model='dlrm_resnet',
          dropout_rate=1.0),
      dict(
          testcase_name='DlrmSmall, p=1.0',
          model='dlrm_small',
          dropout_rate=1.0),
  )
  def test_forward(self, model, dropout_rate):
    OrigCls, CustCls = (
        (OriginalDLRMResNet, CustomDLRMResNet)
        if model == 'dlrm_resnet'
        else (OriginalDlrmSmall, CustomDlrmSmall)
    )

    torch.manual_seed(SEED)
    orig = OrigCls(vocab_size=VOCAB, dropout_rate=dropout_rate).to(DEVICE)

    torch.manual_seed(SEED)
    cust = CustCls(vocab_size=VOCAB).to(DEVICE)

    x = torch.randn(BATCH, FEATURES, device=DEVICE)

    for mode in ('train', 'eval'):
      getattr(orig, mode)()
      getattr(cust, mode)()
      torch.manual_seed(SEED)
      y1 = orig(x)
      torch.manual_seed(SEED)
      y2 = cust(x, dropout_rate)
      assert_close(y1, y2, atol=0, rtol=0)
      if mode == 'eval':  # one extra test: omit dropout at eval
        torch.manual_seed(SEED)
        y2 = cust(x)
        assert_close(y1, y2, atol=0, rtol=0)

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

    torch.manual_seed(SEED)
    orig = OrigCls(vocab_size=VOCAB).to(DEVICE)
    torch.manual_seed(SEED)
    cust = CustCls(vocab_size=VOCAB).to(DEVICE)

    x = torch.randn(BATCH, FEATURES, device=DEVICE)
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
