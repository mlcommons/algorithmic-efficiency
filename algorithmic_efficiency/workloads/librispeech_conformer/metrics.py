from clu import metrics
import flax
import numpy as np


def average_ctc_loss():
  """Returns a clu.Metric that computes average CTC loss taking padding into account.
  """

  @flax.struct.dataclass
  class _Metric(metrics.Metric):
    """Applies `fun` and computes the average."""
    total: np.float32
    weight: np.float32

    @classmethod
    def from_model_output(cls, normalized_loss, **_):
      return cls(total=normalized_loss, weight=1.0)

    def merge(self, other):
      return type(self)(
          total=self.total + other.total, weight=self.weight + other.weight)

    def compute(self):
      return self.total / self.weight

  return _Metric


def get_metrics_bundle():
  return metrics.Collection.create(ctc_loss=average_ctc_loss())
