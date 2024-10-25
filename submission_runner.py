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
import gc
import importlib
from inspect import signature
import itertools
import json
import os
import struct
import time
from types import MappingProxyType
from typing import Any, Dict, Optional, Tuple

from absl import app
from absl import flags
from absl import logging
import jax
import torch
import torch.distributed as dist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Disables tensorRT, cuda warnings.
import tensorflow as tf

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# it unavailable to JAX.
tf.config.set_visible_devices([], 'GPU')

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
from algorithmic_efficiency.workloads import workloads

# disable only for deepspeech if it works fine for other workloads.
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

# TODO(znado): make a nicer registry of workloads that lookup in.
BASE_WORKLOADS_DIR = workloads.BASE_WORKLOADS_DIR

# Workload_path will be appended by '_pytorch' or '_jax' automatically.
WORKLOADS = workloads.WORKLOADS

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
                    None,
                    'Dataset location for ImageNet-v2.')
flags.DEFINE_string('librispeech_tokenizer_vocab_path',
                    '',
                    'Location to librispeech tokenizer.')

flags.DEFINE_enum(
    'framework',
    None,
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')
flags.DEFINE_boolean(
    'torch_compile',
    True,
    'Whether to use `torch.compile` to JIT-compile PyTorch code. '
    'This will only take effect when `framework`==pytorch.')

flags.DEFINE_string(
    'experiment_dir',
    None,
    'The root directory to store all experiments. '
    'It is required and the directory should have '
    'an absolute path rather than a relative path.')
flags.DEFINE_string('experiment_name', None, 'Name of the experiment.')
flags.DEFINE_boolean(
    'save_checkpoints',
    True,
    'Whether or not to save checkpoints of the model and optimizer '
    'at every eval and after training.')
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
flags.DEFINE_boolean(
    'overwrite',
    False,
    'Whether to overwrite the experiment with identical experiment_dir and'
    'experiment_name.')
flags.DEFINE_integer(
    'hparam_start_index',
    None,
    'Start index to slice set of hyperparameters in tuning search space.')
flags.DEFINE_integer(
    'hparam_end_index',
    None,
    'End index to slice set of hyperparameters in tuning search space.')
flags.DEFINE_integer(
    'rng_seed',
    None,
    'Value of rng seed. If None, a random seed will'
    'be generated from hardware.')
flags.DEFINE_boolean('set_pytorch_max_split_size',
                     False,
                     'If true, set pytorch max_split_size_mb to 256')
flags.DEFINE_integer(
    'pytorch_eval_num_workers',
    0,
    'Number of workers for ImageNet PyTorch evaluation data loaders.'
    'WARNING: Setting pytorch_eval_num_workers != 0, will result '
    'in incorrect evals currently, see issues/732.')
FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()


def _get_time():
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  return time.time()


def _get_time_ddp():
  torch.cuda.synchronize()
  t = time.time()
  return sync_ddp_time(t, DEVICE)


if USE_PYTORCH_DDP:
  get_time = _get_time_ddp
else:
  get_time = _get_time


def _reset_cuda_mem():
  if FLAGS.framework == 'pytorch' and torch.cuda.is_available():
    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def train_once(
    workload: spec.Workload,
    workload_name: str,
    global_batch_size: int,
    global_eval_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: Optional[spec.Hyperparameters],
    rng_seed: int,
    rng: spec.RandomState,
    profiler: Profiler,
    max_global_steps: int = None,
    log_dir: Optional[str] = None,
    save_checkpoints: Optional[bool] = True
) -> Tuple[spec.Timing, Dict[str, Any]]:
  _reset_cuda_mem()
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

  # Workload setup.
  logging.info('Initializing dataset.')
  if hasattr(workload, '_eval_num_workers'):
    # Set the number of workers for PyTorch evaluation data loaders
    # (not all workloads have them).
    workload.eval_num_workers = FLAGS.pytorch_eval_num_workers
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
    if FLAGS.framework == 'pytorch' and FLAGS.torch_compile:
      compile_error_workloads = [
          'librispeech_conformer',
          'ogbg',
          'criteo1tb',
          'imagenet_vit',
      ]
      eager_backend_workloads = ['librispeech_deepspeech']
      aot_eager_backend_workloads = []
      loss_compilation_workloads = [
          'fastmri', 'librispeech_deepspeech', 'ogbg', 'wmt'
      ]
      base_workload = workloads.get_base_workload_name(workload_name)
      if base_workload in compile_error_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding without `torch.compile`.')
      elif base_workload in eager_backend_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding with `backend=eager`.')
        model_params = torch.compile(model_params, backend='eager')
      elif base_workload in aot_eager_backend_workloads:
        logging.warning(
            'These workloads cannot be fully compiled under current '
            'PyTorch version. Proceeding with `backend=aot_eager`.')
        model_params = torch.compile(model_params, backend='aot_eager')
      else:
        logging.info('Performing `torch.compile`.')
        model_params = torch.compile(model_params)
      if base_workload in loss_compilation_workloads:
        workload.loss_fn = torch.compile(workload.loss_fn)
  logging.info('Initializing optimizer.')
  with profiler.profile('Initializing optimizer'):
    optimizer_state = init_optimizer_state(workload,
                                           model_params,
                                           model_state,
                                           hyperparameters,
                                           opt_init_rng)
  logging.info('Initializing metrics bundle.')

  # Check if 'train_state' is in the function signature
  needs_train_state = 'train_state' in signature(update_params).parameters

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
      'last_step_end_time': None,
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
    meta_file_name = os.path.join(log_dir, f'meta_data_{preemption_count}.json')
    logging.info(f'Saving meta data to {meta_file_name}.')
    meta_data = logger_utils.get_meta_data(workload, rng_seed)
    logger_utils.write_json(meta_file_name, meta_data)
    flag_file_name = os.path.join(log_dir, f'flags_{preemption_count}.json')
    logging.info(f'Saving flags to {flag_file_name}.')
    logger_utils.write_json(flag_file_name, flags.FLAGS.flag_values_dict())
    metrics_logger = None
    if RANK == 0:
      metrics_logger = logger_utils.set_up_loggers(log_dir,
                                                   flags.FLAGS,
                                                   hyperparameters)
      workload.attach_metrics_logger(metrics_logger)

  global_start_time = get_time()
  train_state['last_step_end_time'] = global_start_time

  logging.info('Starting training loop.')
  goals_reached = (
      train_state['validation_goal_reached'] and
      train_state['test_goal_reached'])
  while train_state['is_time_remaining'] and \
      not goals_reached and \
      not train_state['training_complete']:

    step_rng = prng.fold_in(rng, global_step)
    data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)

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
            rng=update_rng,
            **({'train_state': MappingProxyType(train_state)}
               if needs_train_state else {}))
    except spec.TrainingCompleteError:
      train_state['training_complete'] = True
    global_step += 1
    if (max_global_steps is not None) and (global_step == max_global_steps):
      train_state['training_complete'] = True

    train_step_end_time = get_time()

    train_state['accumulated_submission_time'] += (
        train_step_end_time - train_state['last_step_end_time'])
    # Use 3x the runtime budget for the self-tuning ruleset.
    max_allowed_runtime_sec = (
        workload.max_allowed_runtime_sec if FLAGS.tuning_ruleset == 'external'
        else 3 * workload.max_allowed_runtime_sec)
    train_state['is_time_remaining'] = (
        train_state['accumulated_submission_time'] < max_allowed_runtime_sec)
    # Check if submission is eligible for an untimed eval.
    if ((train_step_end_time - train_state['last_eval_time']) >=
        workload.eval_period_time_sec or train_state['training_complete']):
      with profiler.profile('Evaluation'):
        del batch
        _reset_cuda_mem()

        try:
          eval_start_time = get_time()
          latest_eval_result = workload.eval_model(global_eval_batch_size,
                                                   model_params,
                                                   model_state,
                                                   eval_rng,
                                                   data_dir,
                                                   imagenet_v2_data_dir,
                                                   global_step)
          # Check if targets reached.
          # Note that this is one of the stopping conditions for the length of
          # a training run. To score the run we only consider the time
          # to validation target retrospectively.
          train_state['validation_goal_reached'] = (
              workload.has_reached_validation_target(latest_eval_result) or
              train_state['validation_goal_reached'])
          train_state['test_goal_reached'] = (
              workload.has_reached_test_target(latest_eval_result) or
              train_state['test_goal_reached'])
          goals_reached = (
              train_state['validation_goal_reached'] and
              train_state['test_goal_reached'])
          # Save last eval time.
          eval_end_time = get_time()
          train_state['last_eval_time'] = eval_end_time

          # Accumulate eval time.
          train_state[
              'accumulated_eval_time'] += eval_end_time - eval_start_time

          # Add times to eval results for logging.
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

          logging_start_time = get_time()

          if log_dir is not None and RANK == 0:
            metrics_logger.append_scalar_metrics(
                latest_eval_result,
                global_step=global_step,
                preemption_count=preemption_count,
                is_eval=True,
            )
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

          logging_end_time = get_time()
          train_state['accumulated_logging_time'] += (
              logging_end_time - logging_start_time)

          _reset_cuda_mem()

        except RuntimeError as e:
          logging.exception(f'Eval step {global_step} error.\n')
          if 'out of memory' in str(e):
            logging.warning('Error: GPU out of memory during eval during step '
                            f'{global_step}, error : {str(e)}.')
            _reset_cuda_mem()

    train_state['last_step_end_time'] = get_time()

  metrics = {'eval_results': eval_results, 'global_step': global_step}

  if log_dir is not None and RANK == 0:
    metrics_logger.append_scalar_metrics(
        {'score': train_state['accumulated_submission_time']},
        global_step=global_step,
        preemption_count=preemption_count)
    metrics_logger.finish()
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
                                 log_dir: Optional[str] = None,
                                 save_checkpoints: Optional[bool] = True,
                                 hparam_start_index: Optional[bool] = None,
                                 hparam_end_index: Optional[bool] = None,
                                 rng_seed: Optional[int] = None):
  # Expand paths because '~' may not be recognized
  data_dir = os.path.expanduser(data_dir)
  if imagenet_v2_data_dir:
    imagenet_v2_data_dir = os.path.expanduser(imagenet_v2_data_dir)

  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = workloads.convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  try:
    global_batch_size = submission_module.get_batch_size(workload_name)
  except ValueError:
    base_workload_name = workloads.get_base_workload_name(workload_name)
    global_batch_size = submission_module.get_batch_size(base_workload_name)
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
    all_timings = {}
    all_metrics = {}
    tuning_search_space_iter = itertools.islice(
        enumerate(tuning_search_space), hparam_start_index, hparam_end_index)
    for hi, hyperparameters in tuning_search_space_iter:
      # Generate a new seed from hardware sources of randomness for each trial.
      if not rng_seed:
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
        timing, metrics = train_once(workload, workload_name,
                                     global_batch_size,
                                     global_eval_batch_size,
                                     data_dir, imagenet_v2_data_dir,
                                     init_optimizer_state,
                                     update_params, data_selection,
                                     hyperparameters,
                                     rng_seed,
                                     rng,
                                     profiler,
                                     max_global_steps,
                                     tuning_dir_name,
                                     save_checkpoints=save_checkpoints,)
      all_timings[hi] = timing
      all_metrics[hi] = metrics
      logging.info(f'Tuning trial {hi + 1}/{num_tuning_trials}')
      logging.info(f'Hyperparameters: {tuning_search_space[hi]}')
      logging.info(f'Metrics: {all_metrics[hi]}')
      logging.info(f'Timing: {all_timings[hi]}')
      num_evals = len(all_metrics[hi]['eval_results'])
      logging.info(f'Total number of evals: {num_evals}')
      logging.info('=' * 20)
    score = min(all_timings)
  else:
    if tuning_search_space is not None:
      raise ValueError(
          'Cannot provide a tuning search space when using self tuning.')
    if not rng_seed:
      rng_seed = struct.unpack('q', os.urandom(8))[0]
    rng = prng.PRNGKey(rng_seed)
    # If the submission is responsible for tuning itself, we only need to run it
    # once and return the total time.
    if log_dir is not None:
      log_dir = os.path.join(log_dir, 'trial_1')
      logging.info(f'Creating directory at {log_dir}.')
      logger_utils.makedir(log_dir)
    with profiler.profile('Train'):
      score, _ = train_once(
          workload, workload_name, global_batch_size, global_eval_batch_size,
          data_dir, imagenet_v2_data_dir,
          init_optimizer_state, update_params, data_selection,
          None, rng_seed, rng, profiler, max_global_steps, log_dir,
          save_checkpoints=save_checkpoints)
  return score


def main(_):
  if FLAGS.profile:
    profiler = Profiler()
  else:
    profiler = PassThroughProfiler()

  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  # TODO: remove once issue resolved.
  if FLAGS.pytorch_eval_num_workers != 0:
    logging.warning(
        'WARNING: Setting pytorch_eval_num_workers != 0, will result '
        'in incorrect evals currently, see issues/732.')

  workload_metadata = WORKLOADS[FLAGS.workload]

  # Prevent OOM on librispeech conformer.
  base_workload = workloads.get_base_workload_name(FLAGS.workload)
  if base_workload == 'librispeech_conformer':
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'

  if FLAGS.set_pytorch_max_split_size:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

  # Extend path according to framework.
  workload_metadata['workload_path'] = os.path.join(
      BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + f'_{FLAGS.framework}',
      'workload.py')
  workload_init_kwargs = {}
  if FLAGS.librispeech_tokenizer_vocab_path:
    workload_init_kwargs['tokenizer_vocab_path'] = (
        FLAGS.librispeech_tokenizer_vocab_path)
  workload = workloads.import_workload(
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
                                              FLAGS.resume_last_run,
                                              FLAGS.overwrite)

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
      log_dir=logging_dir_path,
      save_checkpoints=FLAGS.save_checkpoints,
      hparam_start_index=FLAGS.hparam_start_index,
      hparam_end_index=FLAGS.hparam_end_index,
      rng_seed=FLAGS.rng_seed)
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
