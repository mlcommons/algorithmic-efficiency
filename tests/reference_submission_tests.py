"""Test that each reference submission can run a train and eval step.

Assumes that each reference submission is using the external tuning ruleset and
that it is defined in:
"reference_submissions/{workload}/{workload}_{framework}/submission.py"
"reference_submissions/{workload}/tuning_search_space.json".
"""
import copy
import importlib
import json
import os

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

from algorithmic_efficiency import halton
from algorithmic_efficiency import random_utils as prng
import submission_runner

FLAGS = flags.FLAGS


def _test_submission(
    workload_name, framework, submission_path, search_space_path, data_dir):
  FLAGS.framework = framework
  workload_metadata = copy.deepcopy(submission_runner.WORKLOADS[workload_name])
  workload_metadata['workload_path'] = os.path.join(
      submission_runner.BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + '_' + framework,
      'workload.py')
  workload = submission_runner.import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'])

  submission_module_path = submission_runner.convert_filepath_to_module(
      submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  get_batch_size = submission_module.get_batch_size
  global_batch_size = get_batch_size(workload_name)

  # Get a sample hyperparameter setting.
  with open(search_space_path, 'r', encoding='UTF-8') as search_space_file:
    hyperparameters = halton.generate_search(
        json.load(search_space_file), 1)[0]
    print(hyperparameters)
    return

  rng = prng.PRNGKey(0)
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)
  model_params, model_state = workload.init_model_fn(model_init_rng)
  return
  input_queue = workload.build_input_queue(
      data_rng, 'train', data_dir=data_dir, global_batch_size=global_batch_size)
  optimizer_state = init_optimizer_state(workload,
                                         model_params,
                                         model_state,
                                         hyperparameters,
                                         opt_init_rng)

  global_step = 0
  data_select_rng, update_rng, eval_rng = prng.split(rng, 3)
  (selected_train_input_batch,
   selected_train_label_batch,
   selected_train_mask_batch) = data_selection(workload,
                                               input_queue,
                                               optimizer_state,
                                               model_params,
                                               hyperparameters,
                                               global_step,
                                               data_select_rng)
  _, model_params, model_state = update_params(
      workload=workload,
      current_param_container=model_params,
      current_params_types=workload.model_params_types,
      model_state=model_state,
      hyperparameters=hyperparameters,
      input_batch=selected_train_input_batch,
      label_batch=selected_train_label_batch,
      mask_batch=selected_train_mask_batch,
      loss_type=workload.loss_type,
      optimizer_state=optimizer_state,
      eval_results=[],
      global_step=global_step,
      rng=update_rng)

  eval_result = workload.eval_model(model_params,
                                    global_batch_size,
                                    model_state,
                                    eval_rng,
                                    data_dir)
  print(eval_result)




class ReferenceSubmissionTest(parameterized.TestCase):
  """Tests for reference submissions."""

  def test_submission(self):
    # Example: /home/znado/algorithmic-efficiency/tests
    self_location = os.path.dirname(os.path.realpath(__file__))
    # Example: /home/znado/algorithmic-efficiency
    repo_location = '/'.join(self_location.split('/')[:-1])
    references_dir = f'{repo_location}/reference_submissions'
    for workload_name in os.listdir(references_dir):
      workload_dir = f'{repo_location}/reference_submissions/{workload_name}'
      search_space_path = f'{workload_dir}/tuning_search_space.json'
      for framework in ['jax', 'pytorch']:
        submission_dir = f'{workload_dir}/{workload_name}_{framework}'
        if os.path.exists(submission_dir):
          submission_path = (
              f'reference_submissions/{workload_name}/'
              f'{workload_name}_{framework}/submission.py')
          data_dir = None # DO NOT SUBMIT
          logging.info(f'Testing {workload_name} in {framework}.')
          _test_submission(
              workload_name,
              framework,
              submission_path,
              search_space_path,
              data_dir)



if __name__ == '__main__':
  absltest.main()
