"""Tests for logging_utils.py."""

import collections
import json
import os
import shutil

from absl.testing import absltest
import pandas as pd

from algorithmic_efficiency import logging_utils
from algorithmic_efficiency.workloads.mnist.mnist_jax.workload import \
    MnistWorkload


class LoggingUtilsTest(absltest.TestCase):
  """Tests for logging_utils.py."""

  @classmethod
  def setUpClass(self):
    self.logging_dir = './test_logging_utils_temp'
    self.workload = MnistWorkload()
    self.workload_name = 'mnist'
    submission_path = 'baselines/mnist/mnist_jax/submission.py'
    tuning_ruleset = 'external'
    tuning_search_space = 'baselines/mnist/tuning_search_space.json'
    num_tuning_trials = 2
    extra_metadata = ['key=value']

    # Create recorder class
    self.recorder = logging_utils.Recorder(self.workload, self.workload_name, self.logging_dir,
                                    submission_path, tuning_ruleset,
                                    tuning_search_space, num_tuning_trials, extra_metadata=extra_metadata)

  @classmethod
  def tearDownClass(self):
    shutil.rmtree(self.logging_dir, ignore_errors=True)
    pass

  def test_workload_results_file_exists(self):
    # Assert the workload name 'mnist' has been saved to 'workload_results.json'
    output_file = os.path.join(self.logging_dir, self.workload_name, 'workload_results.json')
    with open(output_file, 'r') as f:
      results = json.load(f)
    self.assertEqual(results['workload'], self.workload_name)

  def test_no_op_class(self):
    logging_utils.NoOpRecorder()

  def test_save_eval(self):
    learning_rate =  0.01
    hyperparameters = collections.namedtuple('Hyperparamters', ['learning_rate'])
    hyperparameters = hyperparameters(learning_rate)
    trial_idx = 1
    global_step = 1
    batch_size = 1024
    latest_eval_result = {'loss': 0.1}
    global_start_time = '2022-04-02T21:55:31.206899'
    accumulated_submission_time = 1.0
    goal_reached = False
    is_time_remaining = False
    training_complete = False

    # Save a mock model evaluation
    self.recorder.save_eval(self.workload, hyperparameters, trial_idx, global_step,
                     batch_size, latest_eval_result, global_start_time,
                     accumulated_submission_time, goal_reached,
                     is_time_remaining, training_complete)

    # Assert the workload name 'mnist' has been saved to 'eval_results.csv'
    output_file = os.path.join(self.logging_dir, self.workload_name, 'trial_%s' % trial_idx, 'eval_results.csv')
    df = pd.read_csv(output_file)
    self.assertEqual(df.iloc[0].workload, self.workload_name)

  def test_trial_complete(self):
    learning_rate =  0.01
    hyperparameters = collections.namedtuple('Hyperparamters', ['learning_rate'])
    hyperparameters = hyperparameters(learning_rate)
    trial_idx = 1
    global_step = 1
    batch_size = 1024
    latest_eval_result = {'loss': 0.1}
    global_start_time = '2022-04-02T21:55:31.206899'
    accumulated_submission_time = 1.0
    goal_reached = False
    is_time_remaining = False
    training_complete = False

    # Save a mock model evaluation
    self.recorder.trial_complete(self.workload, hyperparameters, trial_idx, global_step, batch_size, latest_eval_result, global_start_time,
                     accumulated_submission_time, goal_reached,
                     is_time_remaining, training_complete)

    # Assert the workload name 'mnist' has been saved to 'trial_results.csv'
    output_file = os.path.join(self.logging_dir, self.workload_name, 'trial_%s' % trial_idx, 'trial_results.json')
    with open(output_file, 'r') as f:
      results = json.load(f)
    self.assertEqual(results['workload'], self.workload_name)

  def test_workload_complete(self):
    score = 100
    self.recorder.workload_complete(score)

    # Assert the score '100' has been saved to 'workload_results.json'
    output_file = os.path.join(self.logging_dir, self.workload_name, 'workload_results.json')
    with open(output_file, 'r') as f:
      results = json.load(f)
    self.assertEqual(results['score'], score)


  def test_eval_frequency_override(self):
    if self.workload_name == 'mnist':
      batch_size = 1024 # Note: there are 58 steps in an epoch for mnist

    eval_frequency_override = '1 step'
    global_step = 1
    eval_requested = self.recorder.check_eval_frequency_override(
        eval_frequency_override, self.workload, global_step, batch_size)
    self.assertTrue(eval_requested)

    eval_frequency_override = '2 step'
    global_step = 1
    eval_requested = self.recorder.check_eval_frequency_override(eval_frequency_override, self.workload, global_step, batch_size)
    self.assertFalse(eval_requested)

    eval_frequency_override = '1 epoch'
    global_step = 1
    eval_requested = self.recorder.check_eval_frequency_override(
        eval_frequency_override, self.workload, global_step, batch_size)
    self.assertTrue(eval_requested)

    eval_frequency_override = '1 epoch'
    global_step = 2
    eval_requested = self.recorder.check_eval_frequency_override(
        eval_frequency_override, self.workload, global_step, batch_size)
    self.assertFalse(eval_requested)

    eval_frequency_override = '1 epoch'
    global_step = 58
    eval_requested = self.recorder.check_eval_frequency_override(
        eval_frequency_override, self.workload, global_step, batch_size)
    self.assertTrue(eval_requested)

    eval_frequency_override = '1 epoch'
    global_step = 59
    eval_requested = self.recorder.check_eval_frequency_override(
        eval_frequency_override, self.workload, global_step, batch_size)
    self.assertFalse(eval_requested)

    eval_frequency_override = '-1 epoch'
    with self.assertRaises(ValueError):
      eval_requested = self.recorder.check_eval_frequency_override(
          eval_frequency_override, self.workload, global_step, batch_size)

if __name__ == '__main__':
  absltest.main()
