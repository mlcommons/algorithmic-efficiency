"""ImageNet ViT workload."""

from algorithmic_efficiency.workloads.imagenet_resnet.workload import \
    BaseImagenetResNetWorkload


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
