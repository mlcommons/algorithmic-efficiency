"""
Runs fwd pass with random input for FASTMRI U-Net models and compares outputs.
Run it as:
  python3 tests/dropout_fix/imagenet_vit_pytorch/test_model_equivalence.py
"""

from absl.testing import absltest, parameterized
from torch.testing import assert_close
import torch
import os
import itertools

from algoperf.workloads.imagenet_vit.imagenet_pytorch.models import ViT as OriginalVit
from algoperf.workloads.imagenet_vit.imagenet_pytorch.models_dropout import ViT as CustomVit

# Model / test hyper-params
BATCH, C, H, W = 4, 3, 224, 224 # input shape (N,C,H,W)
WIDTH, DEPTH, HEADS = 256, 4, 8
DROPOUT_RATE = None
DEVICE = 'cuda'
SEED = 1996

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

class ImageNetVitModeEquivalenceTest(parameterized.TestCase):

    def fwd_pass(self, orig, cust, dropout_rate):
        x = torch.randn(BATCH, C, H, W, device=DEVICE)
        for mode in ('train', 'eval'):
            getattr(orig, mode)(); getattr(cust, mode)()
            torch.manual_seed(0); y1 = orig(x)
            torch.manual_seed(0); y2 = cust(x, dropout_rate)
            assert_close(y1, y2, atol=0, rtol=0)
            if mode == 'eval':  # one extra test: omit dropout at eval
                torch.manual_seed(0); y2 = cust(x, dropout_rate)
                assert_close(y1, y2, atol=0, rtol=0)

    @parameterized.named_parameters(
        dict(testcase_name='p=0.0', dropout_rate=0.0),
        dict(testcase_name='p=0.1', dropout_rate=0.1),
        dict(testcase_name='p=0.6', dropout_rate=0.6),
        dict(testcase_name='p=1.0', dropout_rate=1.0),
    )
    def test_dropout_values(self, dropout_rate):
        """Test different dropout_rates."""
          
        torch.manual_seed(SEED)
        orig = OriginalVit(
            width=WIDTH,
            depth=DEPTH,
            num_heads=HEADS,
            dropout_rate=dropout_rate,
        ).to(DEVICE)

        torch.manual_seed(SEED)
        cust = CustomVit(
            width=WIDTH,
            depth=DEPTH,
            num_heads=HEADS,
        ).to(DEVICE)
        
        cust.load_state_dict(orig.state_dict())  # sync weights
        self.fwd_pass(orig, cust, dropout_rate)


    @parameterized.named_parameters([
        dict(
            testcase_name=f"GLU={use_glu}_LN={use_post_ln}_MAP={use_map}",
            use_glu=use_glu,
            use_post_ln=use_post_ln,
            use_map=use_map,
        )
        for use_glu, use_post_ln, use_map in itertools.product([False, True], repeat=3)
    ])
    def test_arch(self, use_glu, use_post_ln, use_map):
        """Test different architecture configurations, fixed dropout_rate."""
        dropout_rate = 0.1

        torch.manual_seed(SEED)
        orig = OriginalVit(
            width=WIDTH,
            depth=DEPTH,
            num_heads=HEADS,
            use_glu=use_glu,
            use_post_layer_norm=use_post_ln,
            use_map=use_map,
            dropout_rate=dropout_rate,
        ).to(DEVICE)

        torch.manual_seed(SEED)
        cust = CustomVit(
            width=WIDTH,
            depth=DEPTH,
            num_heads=HEADS,
            use_glu=use_glu,
            use_post_layer_norm=use_post_ln,
            use_map=use_map,
        ).to(DEVICE)
        
        cust.load_state_dict(orig.state_dict())  # sync weights
        self.fwd_pass(orig, cust, dropout_rate)

    @parameterized.named_parameters(
      dict(testcase_name=''),
    )
    def test_default_dropout(self):
        """Test default dropout_rate."""

        torch.manual_seed(SEED)
        orig = OriginalVit(width=WIDTH, depth=DEPTH, num_heads=HEADS).to(DEVICE)
        torch.manual_seed(SEED)
        cust = CustomVit(width=WIDTH, depth=DEPTH, num_heads=HEADS).to(DEVICE)
        cust.load_state_dict(orig.state_dict())  # sync weights
        
        x = torch.randn(BATCH, C, H, W, device=DEVICE)
        for mode in ('train', 'eval'):
            getattr(orig, mode)(); getattr(cust, mode)()
            torch.manual_seed(0); y1 = orig(x)
            torch.manual_seed(0); y2 = cust(x)
            assert_close(y1, y2, atol=0, rtol=0)


if __name__ == '__main__':
    absltest.main()
