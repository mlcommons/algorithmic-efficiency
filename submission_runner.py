r"""Run a submission on a single workload.

Example command:

# pylint: disable=line-too-long
python3 submission_runner.py \
    --workload=mnist \
    --framework=jax \
    --submission_path=reference_algorithms/development_algorithms/mnist/mnist_jax/submission.py \
    --tuning_ruleset=external \
    --tuning_search_space=reference_algorithms/development_algorithms/mnist/tuning_search_space.json \
    --num_tuning_trials=3 \
    --experiment_dir=/home/znado/experiment_dir \
    --experiment_name=baseline
"""

import datetime
import importlib
import inspect
import json
import os
import struct
import time
from typing import Any, Dict, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import jax
import tensorflow as tf
import torch
import torch.distributed as dist

from algorithmic_efficiency import checkpoint_utils
from algorithmic_efficiency import halton
from algorithmic_efficiency import logger_utils
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.pytorch_utils import pytorch_init
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.pytorch_utils import sync_ddp_time

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')

# TODO(znado): make a nicer registry of workloads that lookup in.
BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'

# Workload_path will be appended by '_pytorch' or '_jax' automatically.
WORKLOADS = {
    'cifar': {
        'workload_path': 'cifar/cifar', 'workload_class_name': 'CifarWorkload'
    },
    'criteo1tb': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallWorkload',
    },
    'criteo1tb_test': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallTestWorkload',
    },
    'fastmri': {
        'workload_path': 'fastmri/fastmri',
        'workload_class_name': 'FastMRIWorkload',
    },
    'imagenet_resnet': {
        'workload_path': 'imagenet_resnet/imagenet',
        'workload_class_name': 'ImagenetResNetWorkload',
    },
    'imagenet_vit': {
        'workload_path': 'imagenet_vit/imagenet',
        'workload_class_name': 'ImagenetVitWorkload',
    },
    'librispeech_conformer': {
        'workload_path': 'librispeech_conformer/librispeech',
        'workload_class_name': 'LibriSpeechConformerWorkload',
    },
    'librispeech_deepspeech': {
        'workload_path': 'librispeech_deepspeech/librispeech',
        'workload_class_name': 'LibriSpeechDeepSpeechWorkload',
    },
    'mnist': {
        'workload_path': 'mnist/mnist', 'workload_class_name': 'MnistWorkload'
    },
    'ogbg': {
        'workload_path': 'ogbg/ogbg', 'workload_class_name': 'OgbgWorkload'
    },
    'wmt': {'workload_path': 'wmt/wmt', 'workload_class_name': 'WmtWorkload'},
}

flags.DEFINE_string(
    'submission_path',
    None,
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string(
    'workload',
    None,
    help=f'The name of the workload to run.\n Choices: {list(WORKLOADS.keys())}'
)
flags.DEFINE_enum(
    'tuning_ruleset',
    'external',
    enum_values=['external', 'self'],
    help='Which tuning ruleset to use.')
flags.DEFINE_string(
    'tuning_search_space',
    None,
    'The path to the JSON file describing the external tuning search space.')
flags.DEFINE_integer('num_tuning_trials',
                     1,
                     'The number of external hyperparameter trials to run.')
flags.DEFINE_string('data_dir', '~/data', 'Dataset location.')
flags.DEFINE_string('imagenet_v2_data_dir',
                    '~/data',
                    'Dataset location for ImageNet-v2.')
flags.DEFINE_enum(
    'framework',
    None,
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')
flags.DEFINE_string('librispeech_tokenizer_vocab_path',
                    '',
                    'Location to librispeech tokenizer.')

flags.DEFINE_string(
    'experiment_dir',
    None,
    'The root directory to store all experiments. '
    'It is required and the directory should have '
    'an absolute path rather than a relative path.')
flags.DEFINE_string('experiment_name', None, 'Name of the experiment.')
flags.DEFINE_boolean(
    'save_intermediate_checkpoints',
    True,
    'Whether to save any intermediate checkpoints. '
    'If False, it will only keep the latest checkpoint.')
flags.DEFINE_boolean('resume_last_run',
                     None,
                     'Whether to resume the experiment from its last run.')
flags.DEFINE_boolean(
    'append_timestamp',
    False,
    'If True, the current datetime will be appended to the experiment name. '
    'Useful for guaranteeing a unique experiment dir for new runs.')
flags.DEFINE_boolean('use_wandb',
                     False,
                     'Whether to use Weights & Biases logging.')
flags.DEFINE_boolean('profile', False, 'Whether to produce profiling output.')
flags.DEFINE_integer('max_global_steps',
                     None,
                     'Maximum number of update steps.')
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def convert_filepath_to_module(path: str):
  base, extension = os.path.splitext(path)

  if extension != '.py':
    raise ValueError(f'Path: {path} must be a python file (*.py)')

  return base.replace('/', '.')


def import_workload(workload_path: str,
                    workload_class_name: str,
                    return_class=False,
                    workload_init_kwargs=None) -> spec.Workload:
  """Import and add the workload to the registry.

  This importlib loading is nice to have because it allows runners to avoid
  installing the dependencies of all the supported frameworks. For example, if
  a submitter only wants to write Jax code, the try/except below will catch
  the import errors caused if they do not have the PyTorch dependencies
  installed on their system.

  Args:
    workload_path: the path to the `workload.py` file to load.
    workload_class_name: the name of the Workload class that implements the
      `Workload` abstract class in `spec.py`.
    return_class: if true, then the workload class is returned instead of the
      instantiated object. Useful for testing when methods need to be overriden.
    workload_init_kwargs: kwargs to pass to the workload constructor.
  """

  # Remove the trailing '.py' and convert the filepath to a Python module.
  workload_path = convert_filepath_to_module(workload_path)

  # Import the workload module.
  workload_module = importlib.import_module(workload_path)
  # Get everything defined in the workload module (including our class).
  workload_module_members = inspect.getmembers(workload_module)
  workload_class = None
  for name, value in workload_module_members:
    if name == workload_class_name:
      workload_class = value
      break
  if workload_class is None:
    raise ValueError(
        f'Could not find member {workload_class_name} in {workload_path}. '
        'Make sure the Workload class is spelled correctly and defined in '
        'the top scope of the module.')
  if return_class:
    return workload_class
  return workload_class(**workload_init_kwargs)


def train_once(
    workload: spec.Workload,
    global_batch_size: int,
    global_eval_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: Optional[spec.Hyperparameters],
    rng: spec.RandomState,
    profiler: Profiler,
    max_global_steps: int = None,
    log_dir: Optional[str] = None,
    save_checkpoints: Optional[bool] = True
) -> Tuple[spec.Timing, Dict[str, Any]]:
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

  # Workload setup.
  logging.info('Initializing dataset.')
  with profiler.profile('Initializing dataset'):
    input_queue = workload._build_input_queue(
        data_rng,
        'train',
        data_dir=data_dir,
        global_batch_size=global_batch_size)
  logging.info('Initializing model.')
  with profiler.profile('Initializing model'):
    dropout_rate = None
    aux_dropout_rate = None
    if hasattr(hyperparameters, 'dropout_rate'):
      dropout_rate = hyperparameters.dropout_rate
    if hasattr(hyperparameters, 'aux_dropout_rate'):
      aux_dropout_rate = hyperparameters.aux_dropout_rate
    model_params, model_state = workload.init_model_fn(
        model_init_rng, dropout_rate, aux_dropout_rate)
  logging.info('Initializing optimizer.')
  with profiler.profile('Initializing optimizer'):
    optimizer_state = init_optimizer_state(workload,
                                           model_params,
                                           model_state,
                                           hyperparameters,
                                           opt_init_rng)
  logging.info('Initializing metrics bundle.')
  # Bookkeeping.
  train_state = {
      'validation_goal_reached': False,
      'test_goal_reached': False,
      'is_time_remaining': True,
      'last_eval_time': 0,
      'training_complete': False,
      'accumulated_submission_time': 0,
      'accumulated_eval_time': 0,
      'accumulated_logging_time': 0,
  }
  global_step = 0
  eval_results = []
  preemption_count = 0

  # Loggers and checkpoint setup.
  logging.info('Initializing checkpoint and logger.')
  if log_dir is not None:
    # If the checkpoint exists, load from the checkpoint.
    (optimizer_state,
     model_params,
     model_state,
     train_state,
     eval_results,
     global_step,
     preemption_count) = checkpoint_utils.maybe_restore_checkpoint(
         FLAGS.framework,
         optimizer_state,
         model_params,
         model_state,
         train_state,
         eval_results,
         global_step,
         preemption_count,
         checkpoint_dir=log_dir)
    meta_data = logger_utils.get_meta_data(workload)
    meta_file_name = os.path.join(log_dir, f'meta_data_{preemption_count}.json')
    logging.info(f'Saving meta data to {meta_file_name}.')
    logger_utils.write_json(meta_file_name, meta_data)
    flag_file_name = os.path.join(log_dir, f'flags_{preemption_count}.json')
    logging.info(f'Saving flags to {flag_file_name}.')
    logger_utils.write_json(flag_file_name, flags.FLAGS.flag_values_dict())
    metrics_logger = logger_utils.set_up_loggers(log_dir,
                                                 flags.FLAGS,
                                                 hyperparameters)
    workload.attach_metrics_logger(metrics_logger)

  global_start_time = time.time()
  if USE_PYTORCH_DDP:
    # Make sure all processes start training at the same time.
    global_start_time = sync_ddp_time(global_start_time, DEVICE)

  logging.info('Starting training loop.')
  goals_reached = (
      train_state['validation_goal_reached'] and
      train_state['test_goal_reached'])
  while train_state['is_time_remaining'] and \
      not goals_reached and \
      not train_state['training_complete']:

    train_step_start_time = time.time()

    step_rng = prng.fold_in(rng, global_step)
    data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)

    if USE_PYTORCH_DDP:
      train_step_start_time = sync_ddp_time(train_step_start_time, DEVICE)
    with profiler.profile('Data selection'):
      batch = data_selection(workload,
                             input_queue,
                             optimizer_state,
                             model_params,
                             model_state,
                             hyperparameters,
                             global_step,
                             data_select_rng)
    try:
      with profiler.profile('Update parameters'):
        optimizer_state, model_params, model_state = update_params(
            workload=workload,
            current_param_container=model_params,
            current_params_types=workload.model_params_types,
            model_state=model_state,
            hyperparameters=hyperparameters,
            batch=batch,
            loss_type=workload.loss_type,
            optimizer_state=optimizer_state,
            eval_results=eval_results,
            global_step=global_step,
            rng=update_rng)
    except spec.TrainingCompleteError:
      train_state['training_complete'] = True
    global_step += 1
    if (max_global_steps is not None) and (global_step == max_global_steps):
      train_state['training_complete'] = True

    train_step_end_time = time.time()
    if USE_PYTORCH_DDP:
      train_step_end_time = sync_ddp_time(train_step_end_time, DEVICE)

    train_state['accumulated_submission_time'] += (
        train_step_end_time - train_step_start_time)
    train_state['is_time_remaining'] = (
        train_state['accumulated_submission_time'] <
        workload.max_allowed_runtime_sec)
    # Check if submission is eligible for an untimed eval.
    if ((train_step_end_time - train_state['last_eval_time']) >=
        workload.eval_period_time_sec or train_state['training_complete']):
      with profiler.profile('Evaluation'):
        try:
          eval_start_time = time.time()
          if USE_PYTORCH_DDP:
            eval_start_time = sync_ddp_time(eval_start_time, DEVICE)
          latest_eval_result = workload.eval_model(global_eval_batch_size,
                                                   model_params,
                                                   model_state,
                                                   eval_rng,
                                                   data_dir,
                                                   imagenet_v2_data_dir,
                                                   global_step)
          # Check if targets reached
          train_state['validation_goal_reached'] = (
              workload.has_reached_validation_target(latest_eval_result) or
              train_state['validation_goal_reached'])
          train_state['test_goal_reached'] = (
              workload.has_reached_test_target(latest_eval_result) or
              train_state['test_goal_reached'])
          # Save last eval time
          eval_end_time = time.time()
          if USE_PYTORCH_DDP:
            eval_end_time = sync_ddp_time(eval_end_time, DEVICE)

          # Accumulate eval time
          train_state[
              'accumulated_eval_time'] += eval_end_time - eval_start_time

          # Add times to eval results for logging
          latest_eval_result['score'] = (
              train_state['accumulated_submission_time'])
          latest_eval_result[
              'total_duration'] = eval_end_time - global_start_time
          latest_eval_result['accumulated_submission_time'] = train_state[
              'accumulated_submission_time']
          latest_eval_result['accumulated_eval_time'] = train_state[
              'accumulated_eval_time']
          latest_eval_result['accumulated_logging_time'] = train_state[
              'accumulated_logging_time']
          time_since_start = latest_eval_result['total_duration']
          logging.info(f'Time since start: {time_since_start:.2f}s, '
                       f'\tStep: {global_step}, \t{latest_eval_result}')
          eval_results.append((global_step, latest_eval_result))

          logging_start_time = time.time()
          if USE_PYTORCH_DDP:
            logging_start_time = sync_ddp_time(logging_start_time, DEVICE)
          if log_dir is not None:
            metrics_logger.append_scalar_metrics(
                latest_eval_result,
                global_step=global_step,
                preemption_count=preemption_count)
            if save_checkpoints:
              checkpoint_utils.save_checkpoint(
                  framework=FLAGS.framework,
                  optimizer_state=optimizer_state,
                  model_params=model_params,
                  model_state=model_state,
                  train_state=train_state,
                  eval_results=eval_results,
                  global_step=global_step,
                  preemption_count=preemption_count,
                  checkpoint_dir=log_dir,
                  save_intermediate_checkpoints=FLAGS
                  .save_intermediate_checkpoints)
          logging_end_time = time.time()
          if USE_PYTORCH_DDP:
            logging_end_time = sync_ddp_time(logging_end_time, DEVICE)

          train_state['last_eval_time'] = logging_end_time
          train_state['accumulated_logging_time'] += (
              logging_end_time - logging_start_time)

        except RuntimeError as e:
          logging.exception(f'Eval step {global_step} error.\n')
          if 'out of memory' in str(e):
            logging.warning('Error: GPU out of memory during eval during step '
                            f'{global_step}, error : {str(e)}.')
            if torch.cuda.is_available():
              torch.cuda.empty_cache()

  metrics = {'eval_results': eval_results, 'global_step': global_step}

  if log_dir is not None:
    metrics_logger.append_scalar_metrics(
        {'score': train_state['accumulated_submission_time']},
        global_step=global_step,
        preemption_count=preemption_count)
    metrics_logger.finish()
    checkpoint_utils.save_checkpoint(
        framework=FLAGS.framework,
        optimizer_state=optimizer_state,
        model_params=model_params,
        model_state=model_state,
        train_state=train_state,
        eval_results=eval_results,
        global_step=global_step,
        preemption_count=preemption_count,
        checkpoint_dir=log_dir,
        save_intermediate_checkpoints=FLAGS.save_intermediate_checkpoints)

  return train_state['accumulated_submission_time'], metrics


def score_submission_on_workload(workload: spec.Workload,
                                 workload_name: str,
                                 submission_path: str,
                                 data_dir: str,
                                 tuning_ruleset: str,
                                 profiler: Optional[Profiler] = None,
                                 max_global_steps: Optional[int] = None,
                                 imagenet_v2_data_dir: Optional[str] = None,
                                 tuning_search_space: Optional[str] = None,
                                 num_tuning_trials: Optional[int] = None,
                                 log_dir: Optional[str] = None):
  # Expand paths because '~' may not be recognized
  data_dir = os.path.expanduser(data_dir)
  if imagenet_v2_data_dir:
    imagenet_v2_data_dir = os.path.expanduser(imagenet_v2_data_dir)

  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  global_batch_size = submission_module.get_batch_size(workload_name)
  # n_gpus has to be set here, because we cannot call the first Jax operation
  # before pytorch_init().
  n_gpus = max(N_GPUS, jax.local_device_count())
  if global_batch_size % n_gpus != 0:
    raise ValueError(
        f'The global batch size ({global_batch_size}) has to be divisible by '
        f'the number of GPUs ({n_gpus}).')
  if hasattr(submission_module, 'get_eval_batch_size'):
    # If the user specifies the eval batch size, use the provided one.
    global_eval_batch_size = submission_module.get_eval_batch_size(
        workload_name)
  else:
    global_eval_batch_size = workload.eval_batch_size
  if global_eval_batch_size % n_gpus != 0:
    raise ValueError(
        f'The global eval batch size ({global_eval_batch_size}) has to be '
        f'divisible by the number of GPUs ({n_gpus}).')

  if tuning_ruleset == 'external':
    # If the submission runner is responsible for hyperparameter tuning, load in
    # the search space and generate a list of randomly selected hyperparameter
    # settings from it.
    if tuning_search_space is None:
      raise ValueError(
          'Must provide a tuning search space JSON file when using external '
          'tuning.')
    with open(tuning_search_space, 'r', encoding='UTF-8') as search_space_file:
      tuning_search_space = halton.generate_search(
          json.load(search_space_file), num_tuning_trials)
    all_timings = []
    all_metrics = []
    for hi, hyperparameters in enumerate(tuning_search_space):
      # Generate a new seed from hardware sources of randomness for each trial.
      rng_seed = struct.unpack('I', os.urandom(4))[0]
      logging.info('Using RNG seed %d', rng_seed)
      rng = prng.PRNGKey(rng_seed)
      # Because we initialize the PRNGKey with only a single 32 bit int, in the
      # Jax implementation this means that rng[0] is all zeros, which means this
      # could lead to unintentionally reusing the same seed of only rng[0] were
      # ever used. By splitting the rng into 2, we mix the lower and upper 32
      # bit ints, ensuring we can safely use either rng[0] or rng[1] as a random
      # number.
      rng, _ = prng.split(rng, 2)
      logging.info(f'--- Tuning run {hi + 1}/{num_tuning_trials} ---')

      tuning_dir_name = None
      if log_dir is not None:
        tuning_dir_name = os.path.join(log_dir, f'trial_{hi + 1}')
        logging.info(f'Creating tuning directory at {tuning_dir_name}.')
        logger_utils.makedir(tuning_dir_name)

        # If existing hyperparameter exists, use saved
        # hyperparameters for consistency.
        hyperparameters = logger_utils.write_hparams(hyperparameters,
                                                     tuning_dir_name)
        tuning_search_space[hi] = hyperparameters

      with profiler.profile('Train'):
        if 'imagenet' not in workload_name:
          imagenet_v2_data_dir = None
        timing, metrics = train_once(workload, global_batch_size,
                                     global_eval_batch_size,
                                     data_dir, imagenet_v2_data_dir,
                                     init_optimizer_state,
                                     update_params, data_selection,
                                     hyperparameters, rng,
                                     profiler,
                                     max_global_steps,
                                     tuning_dir_name)
      all_timings.append(timing)
      all_metrics.append(metrics)
    score = min(all_timings)
    for ti in range(num_tuning_trials):
      logging.info(f'Tuning trial {ti + 1}/{num_tuning_trials}')
      logging.info(f'Hyperparameters: {tuning_search_space[ti]}')
      logging.info(f'Metrics: {all_metrics[ti]}')
      logging.info(f'Timing: {all_timings[ti]}')
      logging.info('=' * 20)
  else:
    rng_seed = struct.unpack('q', os.urandom(8))[0]
    rng = prng.PRNGKey(rng_seed)
    # If the submission is responsible for tuning itself, we only need to run it
    # once and return the total time.
    with profiler.profile('Train'):
      score, _ = train_once(
          workload, global_batch_size, global_eval_batch_size,
          data_dir, imagenet_v2_data_dir,
          init_optimizer_state, update_params, data_selection,
          None, rng, profiler, max_global_steps, log_dir)
  return score


def main(_):
  if FLAGS.profile:
    profiler = Profiler()
  else:
    profiler = PassThroughProfiler()

  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  workload_metadata = WORKLOADS[FLAGS.workload]
  # Extend path according to framework.
  workload_metadata['workload_path'] = os.path.join(
      BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + f'_{FLAGS.framework}',
      'workload.py')
  workload_init_kwargs = {}
  if FLAGS.librispeech_tokenizer_vocab_path:
    workload_init_kwargs['tokenizer_vocab_path'] = (
        FLAGS.librispeech_tokenizer_vocab_path)
  workload = import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'],
      workload_init_kwargs=workload_init_kwargs)

  experiment_name = FLAGS.experiment_name
  if experiment_name and FLAGS.append_timestamp:
    experiment_name += datetime.datetime.now().strftime('-%Y-%m-%d-%H-%M-%S')
  logging_dir_path = logger_utils.get_log_dir(FLAGS.experiment_dir,
                                              FLAGS.workload,
                                              FLAGS.framework,
                                              experiment_name,
                                              FLAGS.resume_last_run)

  score = score_submission_on_workload(
      workload=workload,
      workload_name=FLAGS.workload,
      submission_path=FLAGS.submission_path,
      data_dir=FLAGS.data_dir,
      tuning_ruleset=FLAGS.tuning_ruleset,
      profiler=profiler,
      max_global_steps=FLAGS.max_global_steps,
      imagenet_v2_data_dir=FLAGS.imagenet_v2_data_dir,
      tuning_search_space=FLAGS.tuning_search_space,
      num_tuning_trials=FLAGS.num_tuning_trials,
      log_dir=logging_dir_path)
  logging.info(f'Final {FLAGS.workload} score: {score}')

  if FLAGS.profile:
    logging.info(profiler.summary())

  if USE_PYTORCH_DDP:
    # Cleanup.
    dist.destroy_process_group()


if __name__ == '__main__':
  flags.mark_flag_as_required('workload')
  flags.mark_flag_as_required('framework')
  flags.mark_flag_as_required('submission_path')
  flags.mark_flag_as_required('experiment_dir')
  flags.mark_flag_as_required('experiment_name')
  app.run(main)
