"""ImageNet ViT workload."""

from typing import Dict, Iterator, Optional

from algoperf import spec
from algoperf.workloads.imagenet_resnet.workload import BaseImagenetResNetWorkload


def decode_variant(variant: str) -> Dict[str, int]:
  """Converts a string like 'B/32' into a params dict."""
  v, patch = variant.split('/')

  return {
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      'width': {
          'Ti': 192,
          'S': 384,
          'M': 512,
          'B': 768,
          'L': 1024,
          'H': 1280,
          'g': 1408,
          'G': 1664,
      }[v],
      'depth': {
          'Ti': 12,
          'S': 12,
          'M': 12,
          'B': 12,
          'L': 24,
          'H': 32,
          'g': 40,
          'G': 48,
      }[v],
      'mlp_dim': {
          'Ti': 768,
          'S': 1536,
          'M': 2048,
          'B': 3072,
          'L': 4096,
          'H': 5120,
          'g': 6144,
          'G': 8192,
      }[v],
      'num_heads': {
          'Ti': 3, 'S': 6, 'M': 8, 'B': 12, 'L': 16, 'H': 16, 'g': 16, 'G': 16
      }[v],
      'patch_size': (int(patch), int(patch)),
  }


class BaseImagenetVitWorkload(BaseImagenetResNetWorkload):

  @property
  def validation_target_value(self) -> float:
    return 1 - 0.22691  # 0.77309

  @property
  def test_target_value(self) -> float:
    return 1 - 0.3481  # 0.6519

  @property
  def use_post_layer_norm(self) -> bool:
    """Whether to use layer normalization after the residual branch."""
    return False

  @property
  def use_map(self) -> bool:
    """Whether to use multihead attention pooling."""
    return False

  @property
  def use_glu(self) -> bool:
    """Whether to use GLU in the MLPBlock."""
    return False

  @property
  def eval_batch_size(self) -> int:
    return 2048

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 69_768  # ~19.4 hours

  @property
  def eval_period_time_sec(self) -> int:
    return 7 * 60  # 7 mins.

  def _build_dataset(
      self,
      data_rng: spec.RandomState,
      split: str,
      data_dir: str,
      global_batch_size: int,
      cache: Optional[bool] = None,
      repeat_final_dataset: Optional[bool] = None,
      use_mixup: bool = False,
      use_randaug: bool = False) -> Iterator[Dict[str, spec.Tensor]]:
    # We use mixup and Randaugment for ViT workloads.
    use_mixup = use_randaug = split == 'train'
    return super()._build_dataset(data_rng,
                                  split,
                                  data_dir,
                                  global_batch_size,
                                  cache,
                                  repeat_final_dataset,
                                  use_mixup,
                                  use_randaug)

  @property
  def step_hint(self) -> int:
    """Max num steps the baseline algo was given to reach the target."""
    return 186_666
