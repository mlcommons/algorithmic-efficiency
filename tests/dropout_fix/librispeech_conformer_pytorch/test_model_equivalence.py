"""
Runs fwd pass with random input for LIBRISPEECH Conformer models and compares outputs.
Run with:
  python3 tests/dropout_fix/librispeech_conformer_pytorch/test_model_equivalence.py

`dropout_rate` controls the following args:
- `attention_residual_dropout_rate` (if None, 0.1
- `conv_residual_dropout_rate` (if None, 0.0)
- `feed_forward_residual_dropout_rate`  (if None, 0.1)
- `input_dropout_rate` (if None, 0.1)
"""

from absl.testing import absltest, parameterized
from torch.testing import assert_close
import torch
import os

from algoperf.workloads.librispeech_conformer.librispeech_pytorch.models import (
  ConformerConfig as OriginalConfig,
  ConformerEncoderDecoder as OriginalModel
)
from algoperf.workloads.librispeech_conformer.librispeech_pytorch.models_dropout import( 
  ConformerConfig as CustomConfig,
  ConformerEncoderDecoder as CustomModel,
)

N_LAYERS = 3
B, T = 32, 36_000
DEVICE = 'cuda'

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True)
SEED = 1996


class ConformerEquivalenceTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='p=0.0', dropout_rate=0.0),
        dict(testcase_name='p=0.2', dropout_rate=0.2),
        dict(testcase_name='p=0.7', dropout_rate=0.7),
        dict(testcase_name='p=1.0', dropout_rate=1.0),
    )
    def test_forward(self, dropout_rate):

        torch.manual_seed(SEED)
        orig = OriginalModel(
          OriginalConfig(
            num_encoder_layers=N_LAYERS,
            attention_residual_dropout_rate=dropout_rate,
            conv_residual_dropout_rate=dropout_rate,
            feed_forward_residual_dropout_rate=dropout_rate,
            input_dropout_rate=dropout_rate,
        )).to(DEVICE)
        
        torch.manual_seed(SEED)
        cust = CustomModel(
          CustomConfig(
            num_encoder_layers=N_LAYERS
          )
        ).to(DEVICE)

        orig.load_state_dict(cust.state_dict())  # sync weights

        x = torch.randn(B, T, device=DEVICE)
        paddings = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)

        for mode in ('train', 'eval'):
            getattr(orig, mode)()
            getattr(cust, mode)()

            torch.manual_seed(SEED)
            y1, p1 = orig(x, paddings)
            torch.manual_seed(SEED)
            y2, p2 = cust(x, paddings, dropout_rate=dropout_rate)
            
            assert_close(y1, y2, atol=0, rtol=0)
            assert_close(p1, p2, atol=0, rtol=0)

            if mode == 'eval':  # one extra test: omit dropout at eval
                torch.manual_seed(SEED)
                y2, p2 = cust(x, paddings)
                assert_close(y1, y2, atol=0, rtol=0)
                assert_close(p1, p2, atol=0, rtol=0)

    @parameterized.named_parameters(
      dict(testcase_name=''),
    )
    def test_default_dropout(self):
        """Test default dropout_rate."""

        torch.manual_seed(SEED)
        orig = OriginalModel(OriginalConfig(num_encoder_layers=N_LAYERS)).to(DEVICE)
        torch.manual_seed(SEED)
        cust = CustomModel(CustomConfig(num_encoder_layers=N_LAYERS)).to(DEVICE)
        orig.load_state_dict(cust.state_dict())
        
        x = torch.randn(B, T, device=DEVICE)
        paddings = torch.zeros(B, T, dtype=torch.float32, device=DEVICE)
        for mode in ('train', 'eval'):
            getattr(orig, mode)()
            getattr(cust, mode)()
            
            torch.manual_seed(SEED)
            y1, p1 = orig(x, paddings)
            torch.manual_seed(SEED)
            y2, p2 = cust(x, paddings)
            
            assert_close(y1, y2, atol=0, rtol=0)
            assert_close(p1, p2, atol=0, rtol=0)

if __name__ == '__main__':
    absltest.main()
