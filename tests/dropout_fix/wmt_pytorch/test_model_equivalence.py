"""
Runs fwd pass with random input for WMT Transformer models and compares outputs.
Run with:
  python3 tests/dropout_fix/wmt_pytorch/test_model_equivalence.py
"""

import os
import random

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import torch
from torch.testing import assert_close

from algoperf.workloads.wmt.wmt_pytorch.models import \
    Transformer as OriginalModel
from algoperf.workloads.wmt.wmt_pytorch.models_dropout import \
    Transformer as CustomModel

B, SRC_LEN, TGT_LEN, NTOK = 16, 80, 80, 32_000
DEVICE = "cuda"
SEED = 1996

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def _rand_tokens(bs, seqlen):
    return torch.randint(1, NTOK, (bs, seqlen), device=DEVICE)


class TransformerEquivalenceTest(parameterized.TestCase):

    @parameterized.named_parameters(
        # NOTE: removed dropout=1.0 since it will generate nan in scaled_dot_product_attention
        dict(testcase_name="0.0", dropout_rate=0.0, compile=False),
        dict(testcase_name="0.2", dropout_rate=0.2, compile=False),
        dict(testcase_name="0.7", dropout_rate=0.7, compile=False),
        dict(testcase_name="p=0.0_compile", dropout_rate=0.0, compile=True),
        dict(testcase_name="p=0.2_compile", dropout_rate=0.2, compile=True),
        dict(testcase_name="p=0.7_compile", dropout_rate=0.7, compile=True),
    )
    def test_dropout_value(self, dropout_rate, compile):

        orig = OriginalModel(
          dropout_rate=dropout_rate, 
          attention_dropout_rate=dropout_rate
        ).to(DEVICE)
        cust = CustomModel().to(DEVICE)
        
        orig.load_state_dict(cust.state_dict())  # sync weights
        
        if compile:
          orig = torch.compile(orig)
          cust = torch.compile(cust)

        src = _rand_tokens(B, SRC_LEN)
        tgt = _rand_tokens(B, TGT_LEN)

        for mode in ("train", "eval"):
            getattr(orig, mode)()
            getattr(cust, mode)()

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y1 = orig(src, tgt)

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y2 = cust(src, tgt, dropout_rate=dropout_rate)
            
            assert_close(y1, y2, atol=0, rtol=0)
            
            if mode == 'eval':  # one extra test: omit dropout at eval
                torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
                y2 = cust(src, tgt)
                assert_close(y1, y2, atol=0, rtol=0)


    @parameterized.named_parameters(
        dict(testcase_name="default", compile=False),
        dict(testcase_name="default_compile", compile=True),
    )
    def test_default(self, compile):

        orig = OriginalModel().to(DEVICE)
        cust = CustomModel().to(DEVICE)

        orig.load_state_dict(cust.state_dict())  # sync weights
        
        if compile:
          orig = torch.compile(orig)
          cust = torch.compile(cust)

        src = _rand_tokens(B, SRC_LEN)
        tgt = _rand_tokens(B, TGT_LEN)

        for mode in ("train", "eval"):
            getattr(orig, mode)()
            getattr(cust, mode)()

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y1 = orig(src, tgt)

            torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
            y2 = cust(src, tgt)
            
            assert_close(y1, y2, atol=0, rtol=0)
            

if __name__ == "__main__":
    absltest.main()
