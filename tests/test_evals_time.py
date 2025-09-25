"""
Module for evaluating timing consistency in MNIST workload training.

This script runs timing consistency tests for PyTorch and JAX implementations of an MNIST training workload.
It ensures that the total reported training time aligns with the sum of submission, evaluation, and logging times.
"""

import os
import sys
import copy
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from absl import logging
from collections import namedtuple
from algoperf import halton
from algoperf import random_utils as prng
from algoperf.profiler import PassThroughProfiler
from algoperf.workloads import workloads
import submission_runner
import reference_algorithms.development_algorithms.mnist.mnist_pytorch.submission as submission_pytorch
import reference_algorithms.development_algorithms.mnist.mnist_jax.submission as submission_jax
import jax.random as jax_rng

FLAGS = flags.FLAGS
FLAGS(sys.argv)

class Hyperparameters:
    """
    Defines hyperparameters for training.
    """
    def __init__(self):
        self.learning_rate = 0.0005
        self.one_minus_beta_1 = 0.05
        self.beta2 = 0.999
        self.weight_decay = 0.01
        self.epsilon = 1e-25
        self.label_smoothing = 0.1
        self.dropout_rate = 0.1

class CheckTime(parameterized.TestCase):
    """
    Test class to verify timing consistency in MNIST workload training.
    
    Ensures that submission time, evaluation time, and logging time sum up to approximately the total wall-clock time.
    """
    rng_seed = 0

    @parameterized.named_parameters(
        dict(
            testcase_name='mnist_pytorch',
            framework='pytorch',
            init_optimizer_state=submission_pytorch.init_optimizer_state,
            update_params=submission_pytorch.update_params,
            data_selection=submission_pytorch.data_selection,
            rng=prng.PRNGKey(rng_seed)
        ),
        dict(
            testcase_name='mnist_jax',
            framework='jax',
            init_optimizer_state=submission_jax.init_optimizer_state,
            update_params=submission_jax.update_params,
            data_selection=submission_jax.data_selection,
            rng=jax_rng.PRNGKey(rng_seed)
        )
    )
    def test_train_once_time_consistency(self, framework, init_optimizer_state, update_params, data_selection, rng):
        """
        Tests the consistency of timing metrics in the training process.

        Ensures that:
        - The total logged time is approximately the sum of submission, evaluation, and logging times.
        - The expected number of evaluations occurred within the training period.
        """
        workload_metadata = copy.deepcopy(workloads.WORKLOADS["mnist"])
        workload_metadata['workload_path'] = os.path.join(
            workloads.BASE_WORKLOADS_DIR,
            workload_metadata['workload_path'] + '_' + framework,
            'workload.py'
        )
        workload = workloads.import_workload(
            workload_path=workload_metadata['workload_path'],
            workload_class_name=workload_metadata['workload_class_name'],
            workload_init_kwargs={}
        )

        Hp = namedtuple("Hp", ["dropout_rate", "learning_rate", "one_minus_beta_1", "weight_decay", "beta2", "warmup_factor", "epsilon"])
        hp1 = Hp(0.1, 0.0017486387539278373, 0.06733926164, 0.9955159689799007, 0.08121616522670176, 0.02, 1e-25)

        accumulated_submission_time, metrics = submission_runner.train_once(
            workload=workload,
            workload_name="mnist",
            global_batch_size=32,
            global_eval_batch_size=256,
            data_dir='~/tensorflow_datasets',  # Dataset location
            imagenet_v2_data_dir=None,
            hyperparameters=hp1,
            init_optimizer_state=init_optimizer_state,
            update_params=update_params,
            data_selection=data_selection,
            rng=rng,
            rng_seed=0,
            profiler=PassThroughProfiler(),
            max_global_steps=500,
            prepare_for_eval=None
        )

        # Calculate total logged time
        total_logged_time = (
            metrics['eval_results'][-1][1]['total_duration']
            - (accumulated_submission_time +
               metrics['eval_results'][-1][1]['accumulated_logging_time'] +
               metrics['eval_results'][-1][1]['accumulated_eval_time'])
        )

        # Set tolerance for floating-point precision errors
        tolerance = 10
        self.assertAlmostEqual(total_logged_time, 0, delta=tolerance,
                               msg="Total wallclock time does not match the sum of submission, eval, and logging times.")

        # Verify expected number of evaluations
        expected_evals = int(accumulated_submission_time // workload.eval_period_time_sec)
        self.assertTrue(expected_evals <= len(metrics['eval_results']) + 2,
                        f"Number of evaluations {len(metrics['eval_results'])} exceeded the expected number {expected_evals + 2}.")

if __name__ == '__main__':
    absltest.main()
