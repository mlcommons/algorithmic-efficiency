import json

from absl import logging
import numpy as np


class EarlyStopping:
  """Stop training early if not improving.

  Args:
    config: A dict containing the arguments for this class or a str of a path
      to a JSON file containing the config.

  Usage example:
  ```
  $ python submission_runner.py \
      --early_stopping_config="early_stopping_config.json"
  ```

  Config dict example:
  ```
  {
    "metric_name": "validation/loss",
    "min_delta": 0,
    "patience": 0,
    "min_steps": 0,
    "max_steps": null,
    "mode": "min",
    "baseline": null
  }
  ```

  Config:
    metric_name: str, metric to track. "validation/loss", "train/accuracy", etc.
    min_delta: float, minimum change in the monitored quantity to qualify as an
      improvement, i.e. an absolute change of less or equal to than min_delta,
      will count as no improvement. Defaults to 0.
    patience: int, number of model evaluations with no improvement after which
      training will be stopped. Defaults to 0.
    min_steps: int, stop is never requested if step count is less than this
      value. Defaults to 0.
    max_steps: int, always stop if step count is greater than this value.
    mode: One of {"min", "max"}. In min mode, training will stop when the
      quantity monitored has stopped decreasing. In "max" mode it will stop when
      the quantity monitored has stopped increasing.
    baseline: float, baseline value for the monitored quantity. Training will
      stop if the model doesn't show improvement over the baseline.
  """

  def __init__(self, config):
    if config and isinstance(config, dict):
      self.enabled = True
    elif config and isinstance(config, str):
      with open(config, 'r') as file:
        config = json.load(file)
        self.enabled = True
    else:
      config = {}
      self.enabled = False

    self.metric_name = config.get('metric_name')
    self.min_delta = config.get('min_delta', 0)
    self.patience = config.get('patience', 0)
    self.min_steps = config.get('min_steps', 0)
    self.max_steps = config.get('max_steps', None)
    self.mode = config.get('mode', 'min')
    self.baseline_score = config.get('baseline', None)
    self.no_change_count = 0

    try:
      assert (self.mode in ['min', 'max'])
    except:
      logging.error(
          'Failed to parse early_stopping config. Please check "mode" setting.')
      raise
    if self.mode == 'min':
      self.compare_fn = lambda a, b: np.less(a, b - self.min_delta)
      self.best_score = np.Inf
    elif self.mode == 'max':
      self.compare_fn = lambda a, b: np.greater(a, b + self.min_delta)
      self.best_score = -np.Inf
    if self.baseline_score:
      self.best_score = self.baseline_score

  def early_stop_check(self, metrics: dict, step_count: int):
    """Returns True if it is time to stop."""
    if not self.enabled:
      return False
    if self.max_steps and step_count > self.max_steps:
      logging.warning('Early stop due to exceeding max steps.')
      return True

    current_score = metrics[self.metric_name]

    if self.compare_fn(current_score, self.best_score):
      self.best_score = current_score
      self.no_change_count = 0
      return False
    else:
      if (self.no_change_count >= self.patience and
          step_count >= self.min_steps):
        logging.warning('Early stop due to no improvement.')
        return True
      else:
        self.no_change_count += 1
        return False