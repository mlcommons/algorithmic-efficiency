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

import json
import os
import struct
import time

from absl import app
from absl import flags
from absl import logging

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency.workloads.workloads import get_base_workload_name
import docker

flags.DEFINE_string(
    'docker_image_url',
    'us-central1-docker.pkg.dev/training-algorithms-external/mlcommons-docker-repo/algoperf_jax_dev',
    'URL to docker image')
flags.DEFINE_integer('run_percentage',
                     100,
                     'Percentage of max num steps to run for.')
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
flags.DEFINE_boolean('dry_run',
                     False,
                     'Whether or not to actually run the command')
flags.DEFINE_integer('num_studies', 5, 'Number of studies to run')
flags.DEFINE_integer('study_start_index', None, 'Start index for studies.')
flags.DEFINE_integer('study_end_index', None, 'End index for studies.')
flags.DEFINE_integer('num_tuning_trials', 5, 'Number of tuning trials.')
flags.DEFINE_integer('hparam_start_index',
                     None,
                     'Start index for tuning trials.')
flags.DEFINE_integer('hparam_end_index', None, 'End index for tuning trials.')
flags.DEFINE_integer('seed', None, 'Random seed for scoring.')
flags.DEFINE_integer('submission_id',
                     0,
                     'Submission ID to generate study and hparam seeds.')
flags.DEFINE_string('held_out_workloads_config_path',
                    None,
                    'Path to config containing held-out workloads')

FLAGS = flags.FLAGS

DATASETS = ['imagenet', 'fastmri', 'ogbg', 'wmt', 'librispeech', 'criteo1tb']

WORKLOADS = {
    'imagenet_resnet': {'max_steps': 186_666, 'dataset': 'imagenet'},
    'imagenet_vit': {'max_steps': 186_666, 'dataset': 'imagenet'},
    'fastmri': {'max_steps': 36_189, 'dataset': 'fastmri'},
    'ogbg': {'max_steps': 80_000, 'dataset': 'ogbg'},
    'wmt': {'max_steps': 133_333, 'dataset': 'wmt'},
    'librispeech_deepspeech': {'max_steps': 48_000, 'dataset': 'librispeech'},
    'criteo1tb': {'max_steps': 10_666, 'dataset': 'criteo1tb'},
    'librispeech_conformer': {'max_steps': 80_000, 'dataset': 'librispeech'},
}


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


def wait_until_container_not_running(sleep_interval=5 * 60):
  while container_running():
    time.sleep(sleep_interval)
  return


def main(_):
  framework = FLAGS.framework
  run_fraction = FLAGS.run_percentage / 100.
  experiment_name = FLAGS.experiment_name
  docker_image_url = FLAGS.docker_image_url
  submission_path = FLAGS.submission_path
  tuning_search_space = FLAGS.tuning_search_space
  num_studies = FLAGS.num_studies
  num_tuning_trials = FLAGS.num_tuning_trials
  hparam_start_index = FLAGS.hparam_start_index
  hparam_end_index = FLAGS.hparam_end_index
  study_start_index = FLAGS.study_start_index if FLAGS.study_start_index else 0
  study_end_index = FLAGS.study_end_index if FLAGS.study_end_index else num_studies - 1
  submission_id = FLAGS.submission_id
  rng_seed = FLAGS.seed

  if not rng_seed:
    rng_seed = struct.unpack('I', os.urandom(4))[0]

  logging.info('Using RNG seed %d', rng_seed)
  rng_key = (prng.fold_in(prng.PRNGKey(rng_seed), submission_id))

  workloads = [w for w in WORKLOADS.keys()]

  # Read held-out workloads
  if FLAGS.held_out_workloads_config_path:
    held_out_workloads = read_held_out_workloads(
        FLAGS.held_out_workloads_config_path)
    workloads = workloads + held_out_workloads

  rng_subkeys = prng.split(rng_key,
                           num_studies)[:num_studies:]

  for study_index, rng_subkey in zip(range(study_start_index, study_end_index), rng_subkeys):
    print('-' * 100)
    print('*' * 40, f'Starting study {study_index}/{num_studies}', '*' * 40)
    print('-' * 100)
    study_dir = os.path.join(experiment_name, f'study_{study_index}')

    # For each runnable workload check if there are any containers running and if not launch next container command
    for workload in workloads:
      rng_subkey, run_key = prng.split(rng_subkey)
      run_seed = run_key[0]  # arbitrary
      base_workload_name = get_base_workload_name(workload)
      wait_until_container_not_running()
      os.system(
          "sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")  # clear caches
      print('=' * 100)
      dataset = WORKLOADS[base_workload_name]['dataset']
      max_steps = int(WORKLOADS[base_workload_name]['max_steps'] * run_fraction)
      mount_repo_flag = ''
      if FLAGS.local:
        mount_repo_flag = '-v $HOME/algorithmic-efficiency:/algorithmic-efficiency '
      command = ('docker run -t -d -v $HOME/data/:/data/ '
                 '-v $HOME/experiment_runs/:/experiment_runs '
                 '-v $HOME/experiment_runs/logs:/logs '
                 f'{mount_repo_flag}'
                 '--gpus all --ipc=host '
                 f'{docker_image_url} '
                 f'-d {dataset} '
                 f'-f {framework} '
                 f'-s {submission_path} '
                 f'-w {workload} '
                 f'-t {tuning_search_space} '
                 f'-e {study_dir} '
                 f'-m {max_steps} '
                 f'--num_tuning_trials {num_tuning_trials} '
                 f'--hparam_start_index {hparam_start_index} '
                 f'--hparam_end_index {hparam_end_index} '
                 f'--rng_seed {run_seed} '
                 '-c false '
                 '-o true '
                 '-i true ')
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

  app.run(main)
