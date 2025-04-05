"""Tests for sharding consistency in JAX workloads.

Specifically this will test the model_init functions, and input_pipeline.
"""
import copy
import os
import sys

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from algoperf.profiler import PassThroughProfiler
import submission_runner
from algoperf.workloads.workloads import import_workload
from algoperf.workloads.workloads import BASE_WORKLOADS_DIR
from algoperf.workloads.workloads import WORKLOADS

FLAGS = flags.FLAGS
# Needed to avoid UnparsedFlagAccessError
# (see https://github.com/google/model_search/pull/8).
FLAGS(sys.argv)

FRAMEWORK = 'jax' # Can extend to pytorch later


test_case = dict(testcase_name='test_ogbg',
                workload='ogbg')


class SubmissionRunnerTest(parameterized.TestCase):
  """Tests for reference submissions."""


  @parameterized.named_parameters(test_case)
  def test_invariance(self, workload_name):
    workload_name = 'ogbg'
    dataset_dir = f'/data/{workload_name}'
    workload_metadata = copy.deepcopy(WORKLOADS[workload_name])
    workload_metadata['workload_path'] = os.path.join(BASE_WORKLOADS_DIR,
                                                    workload_metadata['workload_path'] + '_' + FRAMEWORK,
                                                    'workload.py')
    workload = import_workload(workload_path=workload_metadata['workload_path'],
                                workload_class_name=workload_metadata['workload_class_name'],
                                workload_init_kwargs={})

    rng = jax.random.PRNGKey(0)
    initial_params, model_state = workload.init_model_fn(rng)
    data_iter = workload._build_input_queue(rng, 'train', dataset_dir, 32)
    batch = next(data_iter)
    inputs = batch['inputs']

    def forward_pass(params,
                    batch,
                    model_state,
                    rng,):
        logits, _ = workload.model_fn(initial_params, 
                                batch, 
                                model_state, 
                                spec.ForwardPassMode.TRAIN, 
                                rng, 
                                update_batch_norm=True)
        return logits

    forward_pass_jitted = jax.jit(forward_pass,
                            in_shardings=(jax_sharding_utils.get_replicate_sharding(),
                                            jax_sharding_utils.get_batch_dim_sharding(),
                                            jax_sharding_utils.get_replicate_sharding(),
                                            jax_sharding_utils.get_replicate_sharding(),
                                            ),
                            out_shardings=jax_sharding_utils.get_batch_dim_sharding())

    logits = forward_pass(initial_params,
                        batch,
                        model_state,
                        rng,)

    logits_jitted = forward_pass_jitted(initial_params,
                        batch,
                        model_state,
                        rng,)

    jax.debug.visualize_array_sharding(logits_jitted)

    equal = jnp.allclose(logits, logits_jitted, atol=1e-6)


if __name__ == '__main__':
  absltest.main()