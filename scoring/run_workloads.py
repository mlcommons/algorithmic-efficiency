"""
Example Usage:
python run_workloads.py --framework jax \
--experiment_name my_first_experiment \
--docker_image_url <url_for_docker_image> \
--tag <some_docker_tag> \
--run_percentage 10 \
--submission_path <path_to_submission_py_file> \
--tuning_search_space <path_to_tuning_search_space_json> 
"""

import datetime
import json
import os
import struct
import subprocess
import time

from absl import app
from absl import flags
from absl import logging

from algoperf import random_utils as prng
from algoperf.workloads.workloads import get_base_workload_name
import docker

flags.DEFINE_string(
    'docker_image_url',
    'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev',
    'URL to docker image')
flags.DEFINE_integer(
    'run_percentage',
    100,
    'Percentage of max num steps to run for.'
    'Must set the flag enable_step_budget to True for this to take effect.')
flags.DEFINE_string('experiment_name',
                    'my_experiment',
                    'Name of top sub directory in experiment dir.')
flags.DEFINE_boolean('rsync_data',
                     True,
                     'Whether or not to transfer the data from GCP w rsync.')
flags.DEFINE_boolean('local', False, 'Mount local algorithmic-efficiency repo.')
flags.DEFINE_string(
    'submission_path',
    'prize_qualification_baselines/external_tuning/jax_nadamw_full_budget.py',
    'Path to reference submission.')
flags.DEFINE_string(
    'tuning_search_space',
    'prize_qualification_baselines/external_tuning/tuning_search_space.json',
    'Path to tuning search space.')
flags.DEFINE_string('framework', 'jax', 'Can be either PyTorch or JAX.')
flags.DEFINE_boolean(
    'dry_run',
    False,
    'Whether or not to actually run the docker containers. '
    'If False, simply print the docker run commands. ')
flags.DEFINE_enum(
    'tuning_ruleset',
    'external',
    enum_values=['external', 'self'],
    help='Can be either external of self.')
flags.DEFINE_integer('num_studies', 5, 'Number of studies to run')
flags.DEFINE_integer('study_start_index', None, 'Start index for studies.')
flags.DEFINE_integer('study_end_index', None, 'End index for studies.')
flags.DEFINE_integer('num_tuning_trials', 5, 'Number of tuning trials.')
flags.DEFINE_integer('hparam_start_index',
                     None,
                     'Start index for tuning trials.')
flags.DEFINE_integer('hparam_end_index', None, 'End index for tuning trials.')
flags.DEFINE_integer('seed', None, 'Random seed for evaluating a submission.')
flags.DEFINE_integer('submission_id',
                     0,
                     'Submission ID to generate study and hparam seeds.')
flags.DEFINE_string('held_out_workloads_config_path',
                    None,
                    'Path to config containing held-out workloads')
flags.DEFINE_string(
    'workload_metadata_path',
    None,
    'Path to config containing dataset and maximum number of steps per workload.'
    'The default values of these are set to the full budgets as determined '
    'via the target-setting procedure. '
    'We provide workload_metadata_external_tuning.json and '
    'workload_metadata_self_tuning.json as references.'
    'Note that training will be interrupted at either the set maximum number '
    'of steps or the fixed workload maximum run time, whichever comes first. '
    'If your algorithm has a smaller per step time than our baselines '
    'you may want to increase the number of steps per workload.')
flags.DEFINE_string(
    'workloads',
    None,
    'String representing a comma separated list of workload names.'
    'If not None, only run this workload, else run all workloads in workload_metadata_path.'
)
flags.DEFINE_string('additional_requirements_path',
                    None,
                    'Path to requirements.txt if any.')
flags.DEFINE_integer(
    'max_steps',
    None,
    'Maximum number of steps to run. Must set flag enable_step_budget.'
    'This flag takes precedence over the run_percentage flag.')
flags.DEFINE_bool(
    'enable_step_budget',
    False,
    'Flag that has to be explicitly set to override time budgets to step budget percentage.'
)

FLAGS = flags.FLAGS


def read_held_out_workloads(filename):
  with open(filename, "r") as f:
    held_out_workloads = json.load(f)
  return held_out_workloads


def container_running():
  docker_client = docker.from_env()
  containers = docker_client.containers.list()
  if len(containers) == 0:
    return False
  else:
    return True


def kill_containers():
  docker_client = docker.from_env()
  containers = docker_client.containers.list()
  for container in containers:
    container.kill()


def gpu_is_active():
  output = subprocess.check_output([
      'nvidia-smi',
      '--query-gpu=utilization.gpu',
      '--format=csv,noheader,nounits'
  ])
  return any(int(x) > 0 for x in output.decode().splitlines())


def wait_until_container_not_running(sleep_interval=5 * 60):
  # check gpu util
  # if the gpu has not been utilized for 30 minutes kill the
  gpu_last_active = datetime.datetime.now().timestamp()

  while container_running():
    # check if gpus have been inactive > 45 min and if so terminate container
    if gpu_is_active():
      gpu_last_active = datetime.datetime.now().timestamp()
    if (datetime.datetime.now().timestamp() - gpu_last_active) > 45 * 60:
      kill_containers(
          "Killing container: GPUs have been inactive > 45 minutes...")
    time.sleep(sleep_interval)
  return


def main(_):
  framework = FLAGS.framework
  experiment_name = FLAGS.experiment_name
  docker_image_url = FLAGS.docker_image_url
  submission_path = FLAGS.submission_path
  tuning_search_space = FLAGS.tuning_search_space
  num_studies = FLAGS.num_studies
  num_tuning_trials = FLAGS.num_tuning_trials
  hparam_start_index_flag = ''
  hparam_end_index_flag = ''
  if FLAGS.hparam_start_index:
    hparam_start_index_flag = f'--hparam_start_index {FLAGS.hparam_start_index} '
  if FLAGS.hparam_end_index:
    hparam_end_index_flag = f'--hparam_end_index {FLAGS.hparam_end_index} '
  study_start_index = FLAGS.study_start_index if FLAGS.study_start_index else 0
  if FLAGS.study_end_index is not None:
    study_end_index = FLAGS.study_end_index
  else:
    study_end_index = num_studies - 1

  additional_requirements_path_flag = ''
  if FLAGS.additional_requirements_path:
    additional_requirements_path_flag = f'--additional_requirements_path {FLAGS.additional_requirements_path} '

  submission_id = FLAGS.submission_id

  rng_seed = FLAGS.seed

  if not rng_seed:
    rng_seed = struct.unpack('I', os.urandom(4))[0]

  logging.info('Using RNG seed %d', rng_seed)
  rng_key = (prng.fold_in(prng.PRNGKey(rng_seed), hash(submission_id)))

  with open(FLAGS.workload_metadata_path) as f:
    workload_metadata = json.load(f)

  # Get list of all possible workloads
  workloads = [w for w in workload_metadata.keys()]

  # Read heldout workloads
  if FLAGS.held_out_workloads_config_path:
    held_out_workloads = read_held_out_workloads(
        FLAGS.held_out_workloads_config_path)
    workloads = workloads + held_out_workloads

  # Filter workloads if explicit workloads specified
  if FLAGS.workloads is not None:
    workloads = list(
        filter(lambda x: x in FLAGS.workloads.split(','), workloads))
    if len(workloads) != len(FLAGS.workloads.split(',')):
      unmatched_workloads = set(FLAGS.workloads.split(',')) - set(workloads)
      raise ValueError(f'Invalid workload name {unmatched_workloads}')

  rng_subkeys = prng.split(rng_key, num_studies)

  for study_index, rng_subkey in zip(range(study_start_index, study_end_index + 1), rng_subkeys):
    print('-' * 100)
    print('*' * 40, f'Starting study {study_index + 1}/{num_studies}', '*' * 40)
    print('-' * 100)
    study_dir = os.path.join(experiment_name, f'study_{study_index}')

    # For each runnable workload check if there are any containers running and if not launch next container command
    for workload in workloads:
      run_key = prng.fold_in(rng_subkey, hash(workload))
      run_seed = run_key[0]  # arbitrary
      base_workload_name = get_base_workload_name(workload)
      wait_until_container_not_running()
      os.system(
          "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")  # clear caches
      print('=' * 100)
      dataset = workload_metadata[base_workload_name]['dataset']
      max_steps_flag = ''
      if FLAGS.enable_step_budget:
        run_fraction = FLAGS.run_percentage / 100.
        if FLAGS.max_steps is None:
          max_steps = int(workload_metadata[base_workload_name]['max_steps'] *
                          run_fraction)
        else:
          max_steps = FLAGS.max_steps
        max_steps_flag = f'-m {max_steps}'

      mount_repo_flag = ''
      if FLAGS.local:
        mount_repo_flag = '-v /home/kasimbeg/algorithmic-efficiency:/algorithmic-efficiency '
      command = ('docker run -t -d -v /home/kasimbeg/data/:/data/ '
                 '-v /home/kasimbeg/experiment_runs/:/experiment_runs '
                 '-v /home/kasimbeg/experiment_runs/logs:/logs '
                 f'{mount_repo_flag}'
                 '--gpus all --ipc=host '
                 f'{docker_image_url} '
                 f'-d {dataset} '
                 f'-f {framework} '
                 f'-s {submission_path} '
                 f'-w {workload} '
                 f'-e {study_dir} '
                 f'{max_steps_flag} '
                 f'--num_tuning_trials {num_tuning_trials} '
                 f'--rng_seed {run_seed} '
                 f'{additional_requirements_path_flag}'
                 '-c false '
                 '-o true '
                 '-i true ')

      # Append tuning ruleset flags
      tuning_ruleset_flags = ''
      if FLAGS.tuning_ruleset == 'external':
        tuning_ruleset_flags += f'--tuning_ruleset {FLAGS.tuning_ruleset} '
        tuning_ruleset_flags += f'-t {tuning_search_space} '
        tuning_ruleset_flags += f'{hparam_start_index_flag} '
        tuning_ruleset_flags += f'{hparam_end_index_flag} '
      else:
        tuning_ruleset_flags += f'--tuning_ruleset {FLAGS.tuning_ruleset} '

      command += tuning_ruleset_flags

      if not FLAGS.dry_run:
        print('Running docker container command')
        print('Container ID: ')
        return_code = os.system(command)
      else:
        return_code = 0
      if return_code == 0:
        print(
            f'SUCCESS: container for {framework} {workload} launched successfully'
        )
        print(f'Command: {command}')
        print(f'Results will be logged to {experiment_name}')
      else:
        print(
            f'Failed: container for {framework} {workload} failed with exit code {return_code}.'
        )
        print(f'Command: {command}')
      wait_until_container_not_running()
      os.system(
          "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")  # clear caches

      print('=' * 100)


if __name__ == '__main__':
  flags.mark_flag_as_required('workload_metadata_path')
  app.run(main)
