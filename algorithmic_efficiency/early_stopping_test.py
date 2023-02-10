"""Tests for early_stopping.py."""

from absl.testing import absltest
import early_stopping


class EarlyStoppingTest(absltest.TestCase):
  """Tests for early_stopping.py."""

  def test_early_stopping_mode_min(self):
    step = 1
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 0,
        "min_steps": 0,
        "max_steps": None,
        "mode": "min",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_mode_max(self):
    step = 1
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 0,
        "min_steps": 0,
        "max_steps": None,
        "mode": "max",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_min_delta(self):
    step = 1
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0.1,
        "patience": 0,
        "min_steps": 0,
        "max_steps": None,
        "mode": "min",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 0.9}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 0.5}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 0.6}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_patience(self):
    step = 1
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 1,
        "min_steps": 0,
        "max_steps": None,
        "mode": "max",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_min_steps(self):
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 0,
        "min_steps": 2,
        "max_steps": None,
        "mode": "max",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    step = 0
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    step = 1
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    step = 2
    eval_result = {'loss': 0.0}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_max_steps(self):
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 0,
        "min_steps": 0,
        "max_steps": 2,
        "mode": "min",
        "baseline": None
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    step = 1
    eval_result = {'loss': 1.0}
    self.assertEqual(early_stop_check(eval_result, step), False)
    step = 2
    eval_result = {'loss': 0.9}
    self.assertEqual(early_stop_check(eval_result, step), False)
    step = 3
    eval_result = {'loss': 0.8}
    self.assertEqual(early_stop_check(eval_result, step), True)

  def test_early_stopping_baseline(self):
    step = 1
    early_stopping_config = {
        "metric_name": "loss",
        "min_delta": 0,
        "patience": 0,
        "min_steps": 0,
        "max_steps": None,
        "mode": "min",
        "baseline": 1
    }
    early_stop_check = early_stopping.EarlyStopping(
        early_stopping_config).early_stop_check

    eval_result = {'loss': 1.1}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 1}
    self.assertEqual(early_stop_check(eval_result, step), True)
    eval_result = {'loss': 0.9}
    self.assertEqual(early_stop_check(eval_result, step), False)


if __name__ == '__main__':
  absltest.main()