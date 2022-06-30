"""ImageNet ViT workload."""

from algorithmic_efficiency.workloads.imagenet_resnet.workload import \
    BaseImagenetResNetWorkload


def decode_variant(variant):
  """Converts a string like 'B/32' into a params dict."""
  v, patch = variant.split('/')

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      'width': {
          'Ti': 192,
          'S': 384,
          'M': 512,
          'B': 768,
          'L': 1024,
          'H': 1280,
          'g': 1408,
          'G': 1664
      }[v],
      'depth': {
          'Ti': 12,
          'S': 12,
          'M': 12,
          'B': 12,
          'L': 24,
          'H': 32,
          'g': 40,
          'G': 48
      }[v],
      'mlp_dim': {
          'Ti': 768,
          'S': 1536,
          'M': 2048,
          'B': 3072,
          'L': 4096,
          'H': 5120,
          'g': 6144,
          'G': 8192
      }[v],
      'num_heads': {
          'Ti': 3, 'S': 6, 'M': 8, 'B': 12, 'L': 16, 'H': 16, 'g': 16, 'G': 16
      }[v],  # pylint:enable=line-too-long
      'patch_size': (int(patch), int(patch))
  }


class BaseImagenetVitWorkload(BaseImagenetResNetWorkload):

  @property
  def target_value(self):
    return 0.76

  @property
  def max_allowed_runtime_sec(self):
    return 111600  # 31 hours

  @property
  def eval_period_time_sec(self):
    return 6000  # 100 mins
