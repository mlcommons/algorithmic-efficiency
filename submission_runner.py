r"""Run a submission on a single workload.

Example command:

python3 submission_runner.py \
    --workload=mnist \
    --framework=jax \
    --submission_path=reference_submissions/mnist/mnist_jax/submission.py \
    --tuning_ruleset=external \
    --tuning_search_space=reference_submissions/mnist/tuning_search_space.json \
    --num_tuning_trials=3
"""
import importlib
import inspect
import json
import os
import struct
import time
from typing import Dict, Iterator, List, Tuple, Optional
import functools
import jax.lax as lax
import jax.numpy as jnp
import flax

from absl import app
from absl import flags
from absl import logging
import optax 
import flax.linen as nn
import flax.jax_utils as jax_utils

from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models

from algorithmic_efficiency.workloads.librispeech_conformer import \
    input_pipeline

from algorithmic_efficiency import halton
from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.profiler import PassThroughProfiler
from algorithmic_efficiency.profiler import Profiler
from algorithmic_efficiency.pytorch_utils import pytorch_init
from algorithmic_efficiency.pytorch_utils import pytorch_setup
import jax 

# Setup JAX so it preallocates less GPU memory by default.
#os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.7'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'


_GRAD_CLIP_EPS = 1e-6

# TODO(znado): make a nicer registry of workloads that lookup in.
BASE_WORKLOADS_DIR = 'algorithmic_efficiency/workloads/'

# Workload_path will be appended by '_pytorch' or '_jax' automatically.
WORKLOADS = {
    'cifar': {
        'workload_path': 'cifar/cifar', 'workload_class_name': 'CifarWorkload'
    },
    'criteo1tb': {
        'workload_path': 'criteo1tb/criteo1tb',
        'workload_class_name': 'Criteo1TbDlrmSmallWorkload'
    },
    'fastmri': {
        'workload_path': 'fastmri/fastmri',
        'workload_class_name': 'FastMRIWorkload'
    },
    'imagenet_resnet': {
        'workload_path': 'imagenet_resnet/imagenet',
        'workload_class_name': 'ImagenetResNetWorkload'
    },
    'imagenet_vit': {
        'workload_path': 'imagenet_vit/imagenet',
        'workload_class_name': 'ImagenetVitWorkload'
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
    'reference_submissions/librispeech_deepspeech/librispeech_jax/submission.py',
    'The relative path of the Python file containing submission functions. '
    'NOTE: the submission dir must have an __init__.py file!')
flags.DEFINE_string(
    'workload',
    'librispeech_deepspeech',
    help=f'The name of the workload to run.\n Choices: {list(WORKLOADS.keys())}'
)
flags.DEFINE_enum(
    'tuning_ruleset',
    'external',
    enum_values=['external', 'self'],
    help='Which tuning ruleset to use.')
flags.DEFINE_string(
    'tuning_search_space',
    'reference_submissions/librispeech_deepspeech/tuning_search_space.json',
    'The path to the JSON file describing the external tuning search space.')
flags.DEFINE_integer('num_tuning_trials',
                     1,
                     'The number of external hyperparameter trials to run.')

flags.DEFINE_integer('num_train_steps',
                     10,
                     'The number of training steps to run.')
flags.DEFINE_string('data_dir', '/mnt/disks/librispeech_processed/work_dir/data', 'Dataset location')
flags.DEFINE_string('imagenet_v2_data_dir',
                    '~/tensorflow_datasets/',
                    'Dataset location for ImageNet-v2.')
flags.DEFINE_enum(
    'framework',
    'jax',
    enum_values=['jax', 'pytorch'],
    help='Whether to use Jax or Pytorch for the submission. Controls among '
    'other things if the Jax or Numpy RNG library is used for RNG.')
flags.DEFINE_boolean('profile', False, 'Whether to produce profiling output.')
flags.DEFINE_string('summary_log_dir',
                    'reference_submissions/librispeech_deepspeech/librispeech_jax/summaries',
                    'Location to dump tensorboard summaries.')
flags.DEFINE_string('tokenizer_vocab_path',
                    '/mnt/disks/librispeech_processed/spm_model.vocab',
                    'Location to read tokenizer from.')

FLAGS = flags.FLAGS
USE_PYTORCH_DDP, RANK, DEVICE, _ = pytorch_setup()

def convert_filepath_to_module(path: str):
  base, extension = os.path.splitext(path)

  if extension != '.py':
    raise ValueError(f'Path: {path} must be a python file (*.py)')

  return base.replace('/', '.')

def import_workload(workload_path: str,
                    workload_class_name: str,
                    return_class=False) -> spec.Workload:
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
  return workload_class()

# def init_model_fn(model_init_rng):
#   params_rng, dropout_rng = jax.random.split(model_init_rng, 2)

#   inputs = jnp.zeros((2, 320000))
#   input_paddings = jnp.zeros((2, 320000))
  
#   vars = model_class.init(
#     {'params': params_rng, 'dropout': dropout_rng}, 
#     inputs, 
#     input_paddings, 
#     train=True)

#   batch_stats = vars['batch_stats']
#   params = vars['params']

#   return params, batch_stats


# config = models.DeepspeechConfig()
# model_class = models.Deepspeech(config)

# def rsqrt_schedule(
#     init_value: float,
#     shift: int = 0):
#   def schedule(count):
#     return init_value * (count + shift)**-.5 * shift**.5

#   return schedule

# def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
#   """Creates a rsqrt schedule with linear warmup."""
#   return optax.join_schedules([
#       optax.linear_schedule(
#           init_value=0, end_value=learning_rate, transition_steps=warmup_steps),
#       rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
#   ],
#   boundaries=[warmup_steps])

# def init_optimizer_state(params):
#   learning_rate_init=0.002
#   warmup_steps = 5

#   learning_rate_fn = create_learning_rate_schedule(learning_rate_init, warmup_steps)
#   optimizer_init_fn, optimizer_update_fn = optax.adamw(
#       learning_rate_fn, 
#       b1=0.9, 
#       b2=0.98, 
#       eps=1e-9, 
#       weight_decay=0.1)
#   optimizer_state = optimizer_init_fn(params)
#   return optimizer_state, optimizer_update_fn


# def update_step(
#   batch, 
#   params, 
#   batch_stats, 
#   optimizer_state, 
#   workload, global_step, hyperparameters, opt_update_fn, rng):
  
#   # lr = workload.get_learning_rate(global_step, hyperparameters)
#   def _loss_fn(params):
#     """loss function used for training."""
#     params_rng, dropout_rng = jax.random.split(rng, 2)
#     (logits, logit_paddings), new_batch_stats = workload.model_fn(
#         params,
#         batch,
#         batch_stats,
#         spec.ForwardPassMode.TRAIN,
#         {'params' : params_rng, 'dropout' : dropout_rng})

#     loss = workload.loss_fn(batch['targets'],(logits, logit_paddings))
#     return loss, new_batch_stats


#   grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#   (loss, new_model_state), grad = grad_fn(params)
#   loss, grad = lax.pmean((loss, grad), axis_name='batch')
#   print('loss inside pmap = ', loss)

#   updates, new_optimizer_state = opt_update_fn(grad, optimizer_state, params)
#   print('applying updates')
#   updated_params = optax.apply_updates(params, updates)
#   print('after applying updates')

#   return updated_params, new_model_state, new_optimizer_state, loss


# def update_params(
#     workload, 
#     params,
#     batch_stats,
#     hyperparameters,
#     batch,
#     optimizer_state,
#     eval_results: List[Tuple[int, float]],
#     global_step: int,
#     rng):
#   """Return (updated_optimizer_state, updated_params)."""
#   del eval_results

#   print('in update params in submission')
#   per_device_rngs = jax.random.split(rng, jax.local_device_count())
#   print('before pmapped_update')
#   # unfrozen = params.unfreeze()
#   print('params shape = ', params['Dense_0']['kernel'].shape)
#   # @functools.partial(
#   #   jax.pmap,
#   #   axis_name='batch',
#   #   in_axes=(None, None, 0, 0, 0, None, 0, 0, None),
#   #   static_broadcasted_argnums=(0, 1))

#   optimizer_state, opt_update_fn = optimizer_state

#   update_fn = functools.partial(
#     update_step, 
#     opt_update_fn=opt_update_fn,
#     global_step=global_step,
#     workload=workload,
#     hyperparameters=hyperparameters,
#     rng=rng)

#   pmapped_update_step = jax.pmap(update_fn, axis_name='batch', in_axes=(0,0,0,0))
#   new_params, new_batch_stats, new_optimizer_state, loss = pmapped_update_step(batch, params, batch_stats, optimizer_state)

#   #new_params = jax_utils.unreplicate(new_params)

#   print('after applying updates inside submission')
#   print('loss = ', loss.mean())
#   print('updated params shape = ', new_params['Dense_0']['kernel'].shape)

#   return (new_optimizer_state, opt_update_fn), new_params, new_batch_stats
  
# def shard(batch, n_devices=None):
#   if n_devices is None:
#     n_devices = jax.local_device_count()

#   # Otherwise, the entries are arrays, so just reshape them.
#   def _shard_array(array):
#     return array.reshape((n_devices, -1) + array.shape[1:])

#   return jax.tree_map(_shard_array, batch)

# Example reference implementation showing how to use the above functions
# together.
def train_once(
    workload: spec.Workload,
    global_batch_size: int,
    data_dir: str,
    imagenet_v2_data_dir: str,
    init_optimizer_state: spec.InitOptimizerFn,
    update_params: spec.UpdateParamsFn,
    data_selection: spec.DataSelectionFn,
    hyperparameters: Optional[spec.Hyperparameters],
    rng: spec.RandomState,
    profiler: Profiler,
    log_dir: Optional[str] = None,
    tokenizer_vocab_path: Optional[str] = None,
    num_train_steps:int = None
) -> Tuple[spec.Timing, spec.Steps]:
  data_rng, opt_init_rng, model_init_rng, rng = prng.split(rng, 4)

  # Workload setup.
  logging.info('Initializing dataset.')
  with profiler.profile('Initializing dataset'):
    input_queue = workload.build_input_queue(
        data_rng,
        'train',
        data_dir=data_dir,
        global_batch_size=global_batch_size)
  logging.info('Initializing model.')
  with profiler.profile('Initializing model'):
    model_params, model_state = workload.init_model_fn(model_init_rng)
  logging.info('Initializing optimizer.')
  if log_dir:
    logging.info('Initializing tensorboard summary writer')
    workload.create_summary_writer(log_dir)

  with profiler.profile('Initializing optimizer'):
    optimizer_state = init_optimizer_state(workload,
                                           model_params,
                                           model_state,
                                           hyperparameters,
                                           opt_init_rng)
  # optimizer_state, opt_update_fn = init_optimizer_state(model_params)
  # replicated_optimizer_state = flax.jax_utils.replicate(optimizer_state)

  # Bookkeeping.
  goal_reached = False
  is_time_remaining = True
  last_eval_time = 0
  accumulated_submission_time = 0
  eval_results = []
  global_step = 0
  training_complete = False
  global_start_time = time.time()

  logging.info('Starting training loop.')
  while global_step < num_train_steps:
    step_rng = prng.fold_in(rng, global_step)
    data_select_rng, update_rng, eval_rng = prng.split(step_rng, 3)
    start_time = time.time()

    with profiler.profile('Data selection'):
      batch = data_selection(workload,
                             input_queue,
                             optimizer_state,
                             model_params,
                             hyperparameters,
                             global_step,
                             data_select_rng)
    try:
      with profiler.profile('Update parameters'):
        print('before call to update parameters')
        optimizer_state, model_params, model_state = update_params(
          workload=workload,
          current_param_container=model_params,
          current_params_types=workload.model_params_types,
          model_state=model_state,
          hyperparameters=hyperparameters,
          batch=batch,
          optimizer_state=optimizer_state,
          loss_type=workload.loss_type,
          eval_results=eval_results,
          global_step=global_step,
          rng=update_rng)
        print('after update_params in submission_runner.py')
        global_step += 1
    except spec.TrainingCompleteError:
      training_complete = True
  
  metrics = {'eval_results': eval_results, 'global_step': global_step}
  return 0, metrics


def score_submission_on_workload(workload: spec.Workload,
                                 workload_name: str,
                                 submission_path: str,
                                 data_dir: str,
                                 imagenet_v2_data_dir: str,
                                 profiler: Profiler,
                                 tuning_ruleset: str,
                                 tuning_search_space: Optional[str] = None,
                                 num_tuning_trials: Optional[int] = None,
                                 log_dir: Optional[str] = None,
                                 tokenizer_vocab_path: Optional[str] = None, 
                                 num_train_steps:int = None):
  # Remove the trailing '.py' and convert the filepath to a Python module.
  submission_module_path = convert_filepath_to_module(submission_path)
  submission_module = importlib.import_module(submission_module_path)

  init_optimizer_state = submission_module.init_optimizer_state
  update_params = submission_module.update_params
  data_selection = submission_module.data_selection
  global_batch_size = submission_module.get_batch_size(workload_name)

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
      rng = prng.PRNGKey(rng_seed)
      # Because we initialize the PRNGKey with only a single 32 bit int, in the
      # Jax implementation this means that rng[0] is all zeros, which means this
      # could lead to unintentionally reusing the same seed of only rng[0] were
      # ever used. By splitting the rng into 2, we mix the lower and upper 32
      # bit ints, ensuring we can safely use either rng[0] or rng[1] as a random
      # number.
      rng, _ = prng.split(rng, 2)
      logging.info('--- Tuning run %d/%d ---', hi + 1, num_tuning_trials)
      with profiler.profile('Train'):
        if 'imagenet' not in workload_name:
          imagenet_v2_data_dir = None
        timing, metrics = train_once(workload, global_batch_size,
                                     data_dir, imagenet_v2_data_dir,
                                     init_optimizer_state,
                                     update_params,
                                     data_selection,
                                     hyperparameters, rng, profiler, log_dir,
                                     tokenizer_vocab_path,
                                     num_train_steps)
      all_timings.append(timing)
      all_metrics.append(metrics)
    score = min(all_timings)
    for ti in range(num_tuning_trials):
      logging.info('Tuning trial %d/%d', ti + 1, num_tuning_trials)
      logging.info('Hyperparameters: %s', tuning_search_space[ti])
      logging.info('Metrics: %s', all_metrics[ti])
      logging.info('Timing: %s', all_timings[ti])
      logging.info('=' * 20)
  else:
    rng_seed = struct.unpack('q', os.urandom(8))[0]
    rng = prng.PRNGKey(rng_seed)
    # If the submission is responsible for tuning itself, we only need to run it
    # once and return the total time.
    with profiler.profile('Train'):
      score, _ = train_once(
          workload, global_batch_size, data_dir,
          imagenet_v2_data_dir,
          init_optimizer_state,
          update_params,
          data_selection,
          None, rng, profiler)
  # TODO(znado): record and return other information (number of steps).
  return score


def main(_):
  if FLAGS.profile:
    profiler = Profiler()
  else:
    profiler = PassThroughProfiler()

  if FLAGS.framework == 'pytorch':
    pytorch_init(USE_PYTORCH_DDP, RANK, profiler)

  workload_metadata = WORKLOADS[FLAGS.workload]
  # extend path according to framework
  workload_metadata['workload_path'] = os.path.join(
      BASE_WORKLOADS_DIR,
      workload_metadata['workload_path'] + '_' + FLAGS.framework,
      'workload.py')
  workload = import_workload(
      workload_path=workload_metadata['workload_path'],
      workload_class_name=workload_metadata['workload_class_name'])

  score = score_submission_on_workload(workload,
                                       FLAGS.workload,
                                       FLAGS.submission_path,
                                       FLAGS.data_dir,
                                       FLAGS.imagenet_v2_data_dir,
                                       profiler,
                                       FLAGS.tuning_ruleset,
                                       FLAGS.tuning_search_space,
                                       FLAGS.num_tuning_trials,
                                       FLAGS.summary_log_dir,
                                       FLAGS.tokenizer_vocab_path,
                                       FLAGS.num_train_steps)
  logging.info('Final %s score: %f', FLAGS.workload, score)

  if FLAGS.profile:
    logging.info(profiler.summary())

  if USE_PYTORCH_DDP:
    # cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)