"""
Runs fwd pass with random input for WMT Transformer models and compares outputs.
Run with:
  python3 tests/dropout_fix/wmt_pytorch/test_model_equivalence.py
"""

from absl.testing import absltest, parameterized
from torch.testing import assert_close
import torch, os, random, numpy as np

from algoperf.workloads.wmt.wmt_pytorch.models import (
    Transformer as OriginalModel,
)
from algoperf.workloads.wmt.wmt_pytorch.models_dropout import (
    Transformer as CustomModel,
)

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
        # NOTE: removed dropout=1.0 will generate nan in scaled_dot_product_attention

        dict(testcase_name="None", dropout_rate=None, compile=False),
        dict(testcase_name="0.0", dropout_rate=0.0, compile=False),
        dict(testcase_name="0.2", dropout_rate=0.2, compile=False),
        dict(testcase_name="0.7", dropout_rate=0.7, compile=False),
        
        dict(testcase_name="p=None, compile", dropout_rate=None, compile=True),
        dict(testcase_name="p=0.0, compile", dropout_rate=0.0, compile=True),
        dict(testcase_name="p=0.2, compile", dropout_rate=0.2, compile=True),
        dict(testcase_name="p=0.7, compile", dropout_rate=0.7, compile=True),
    )
    def test_forward(self, dropout_rate, compile):
      
        # Test initalizing custom model with a None dropout_rate
        for custom_init_dropout_rate in [None, dropout_rate]:

            orig = OriginalModel(
              dropout_rate=dropout_rate, 
              attention_dropout_rate=dropout_rate
            ).to(DEVICE)
            cust = CustomModel(
              dropout_rate=custom_init_dropout_rate
            ).to(DEVICE)
            
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


if __name__ == "__main__":
    absltest.main()
