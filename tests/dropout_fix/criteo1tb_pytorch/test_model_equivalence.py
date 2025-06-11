"""
Runs fwd pass with random input for our DLRM models and compares outputs.
Run it as:
  python3 tests/dropout_fix/criteo1tb_pytorch/test_model_equivalence.py
"""

from absl.testing import absltest, parameterized
from torch.testing import assert_close
import torch
import os

from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models import (
    DLRMResNet as OriginalDLRMResNet,
    DlrmSmall  as OriginalDlrmSmall,
)
from algoperf.workloads.criteo1tb.criteo1tb_pytorch.models_dropout import (
    DLRMResNet as CustomDLRMResNet,
    DlrmSmall  as CustomDlrmSmall,
)


BATCH, DENSE, SPARSE = 16, 13, 26
FEATURES = DENSE + SPARSE
VOCAB = 1000
DEVICE = 'cuda'
TORCH_COMPILE = False
SEED = 1996

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class ModelEquivalenceTest(parameterized.TestCase):

    @parameterized.named_parameters(
    dict(testcase_name='DLRMResNet, p=None', model='dlrm_resnet', dropout_rate=None),
    dict(testcase_name='DlrmSmall, p=None',  model='dlrm_small', dropout_rate=None),
    dict(testcase_name='DLRMResNet, p=0.0', model='dlrm_resnet', dropout_rate=0.0),
    dict(testcase_name='DlrmSmall, p=0.0',  model='dlrm_small', dropout_rate=0.0),
    dict(testcase_name='DLRMResNet, p=0.1', model='dlrm_resnet', dropout_rate=0.1),
    dict(testcase_name='DlrmSmall, p=0.1',  model='dlrm_small', dropout_rate=0.1),
    dict(testcase_name='DLRMResNet, p=1.0', model='dlrm_resnet', dropout_rate=1.0),
    dict(testcase_name='DlrmSmall, p=1.0',  model='dlrm_small', dropout_rate=1.0),
    )
    def test_forward(self, model, dropout_rate):
        OrigCls, CustCls = (
            (OriginalDLRMResNet, CustomDLRMResNet)
            if model == 'dlrm_resnet'
            else (OriginalDlrmSmall, CustomDlrmSmall)
        )

        # Test initalizing custom model with a None dropout_rate
        for custom_init_dropout_rate in [dropout_rate, None]:

            torch.manual_seed(SEED)
            orig = OrigCls(vocab_size=VOCAB, dropout_rate=dropout_rate)
            orig.to(DEVICE)

            torch.manual_seed(SEED)
            cust = CustCls(vocab_size=VOCAB, dropout_rate=custom_init_dropout_rate)
            cust.to(DEVICE)

            if TORCH_COMPILE:
              orig = torch.compile(orig); cust = torch.compile(cust)
            
            x = torch.randn(BATCH, FEATURES, device=DEVICE)

            for mode in ('train', 'eval'):
                getattr(orig, mode)(); getattr(cust, mode)()
                torch.manual_seed(SEED); y1 = orig(x)
                torch.manual_seed(SEED); y2 = cust(x, dropout_rate)
                assert_close(y1, y2, atol=0, rtol=0)
        

if __name__ == '__main__':
    absltest.main()
