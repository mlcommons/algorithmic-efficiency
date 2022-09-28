import os
import time

import flax
import flax.linen as nn
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf

from algorithmic_efficiency.workloads.librispeech_conformer import \
    input_pipeline
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import \
    models

tf.config.set_visible_devices([], 'GPU')
print(jax.local_device_count())
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

BATCH_SIZE = 8
GRAD_CLIP = False
NUM_TRAINING_STEPS = 1

_GRAD_CLIP_EPS = 1e-6

config = models.DeepspeechConfig()
model = models.Deepspeech(config)

print(model)

inputs = jnp.zeros((2, 320000))
input_paddings = jnp.zeros((2, 320000))

rng = jax.random.PRNGKey(0)
params_rng, data_rng, dropout_rng = jax.random.split(rng, 3)

print('initing model .....')
init_start = time.time()
vars = model.init({'params': params_rng, 'dropout': dropout_rng},
                  inputs,
                  input_paddings,
                  train=True)

print('inited model')
init_end = time.time()
print('init time taken = ', init_end - init_start)

batch_stats = vars['batch_stats']
params = vars['params']
warmup_steps = 2
learning_rate_init = 0.002


def rsqrt_schedule(init_value, shift=0):

  def schedule(count):
    return init_value * (count + shift)**-.5 * shift**.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0, end_value=learning_rate, transition_steps=warmup_steps),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
  ],
                              boundaries=[warmup_steps])


learning_rate_fn = create_learning_rate_schedule(
    learning_rate=learning_rate_init, warmup_steps=warmup_steps)


optimizer_init_fn, optimizer_update_fn = optax.adamw(
    learning_rate_fn,
    b1=0.9,
    b2=0.98,
    eps=1e-9,
    weight_decay=0.1
)
optimizer_state = optimizer_init_fn(params)
print('optimizer initialized')


def shard(batch, n_devices=None):
  if n_devices is None:
    n_devices = jax.local_device_count()

  # Otherwise, the entries are arrays, so just reshape them.
  def _shard_array(array):
    return array.reshape((n_devices, -1) + array.shape[1:])

  return jax.tree_map(_shard_array, batch)


replicated_params = flax.jax_utils.replicate(params)
replicated_batch_stats = flax.jax_utils.replicate(batch_stats)
replicated_optimizer_state = flax.jax_utils.replicate(optimizer_state)

num_train_step = 0
print('starting training loop')

fake_batch = {
    'inputs': np.zeros((8, 320000)),
    'input_paddings': np.zeros((8, 320000)),
    'targets': np.zeros((8, 256)),
    'target_paddings': np.zeros((8, 256))
}

real_batch = {}

ds = input_pipeline.get_librispeech_dataset(
    'train-clean-100+train-clean-360+train-other-500',
    '/mnt/disks/librispeech_processed/work_dir/data',
    data_rng,
    False,
    BATCH_SIZE,
    NUM_TRAINING_STEPS)


def get_iterator():
  for batch in iter(ds):
    yield batch


def l2_regularization(params, l2_decay_rank_threshold):
  """Computes the squared l2 norm of the given parameters.

  This function will only filter for parameters with
  rank >= l2_decay_rank_threshold. So if this threshold is set to 2, then all
  1d (and lower) parameter arrays, including all bias and batch norm params,
  will be ignored in this computation.


  Args:
    params: Pytree containing parameters.
    l2_decay_rank_threshold: The calculation will only include parameters with
       param.ndim >= l2_decay_rank_threshold. Set to 2 to ignore all bias and
       batch_norm params in the model.

  Returns:
    weight_l2: the squared l2 norm of all params matching the threshold.
  """
  weight_penalty_params = jax.tree_leaves(params)
  weight_l2 = sum(
      jnp.sum(x**2)
      for x in weight_penalty_params
      if x.ndim >= l2_decay_rank_threshold)
  return weight_l2


iterator = get_iterator()

for i in range(NUM_TRAINING_STEPS):
  batch = next(iterator)
  batch = jax.tree_map(lambda x: x._numpy(), batch)

  def update_step(batch, params, batch_stats, optimizer_state):

    def loss_fn(params, batch_stats):
      inputs, input_paddings = batch['inputs']
      # input_paddings = batch['input_paddings']

      # print(batch['inputs'].shape)
      # print(batch['input_paddings'].shape)

      (logits, logit_paddings), new_batch_stats = model.apply({
          'params': params,
          'batch_stats': batch_stats
      }, inputs, input_paddings, train=True, rngs = {'dropout': dropout_rng}, mutable=['batch_stats'])

      targets, target_paddings = batch['targets']
      # target_paddings = batch['target_paddings']

      logprobs = nn.log_softmax(logits)
      per_seq_loss = optax.ctc_loss(logprobs,
                                    logit_paddings,
                                    targets,
                                    target_paddings)
      normalizer = jnp.sum(1 - target_paddings)
      normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)

      return normalized_loss, new_batch_stats

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, new_batch_stats), grad = grad_fn(params, batch_stats)
    loss_val, grad = lax.pmean((loss_val, grad), axis_name='batch')

    grad_clip = 5.0
    grad_norm = jnp.sqrt(l2_regularization(grad, 0))

    if GRAD_CLIP:
      grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
      grad_scaling_factor = jax.lax.clamp(
          min=0.0, x=grad_scaling_factor, max=1.0)

      grad = jax.tree_map(lambda x: x * grad_scaling_factor, grad)

    updates, new_optimizer_state = optimizer_update_fn(grad, optimizer_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_batch_stats, new_optimizer_state, loss_val, grad_norm

  train_start = time.time()
  pmapped_update_step = jax.pmap(
      update_step, axis_name='batch', in_axes=(0, 0, 0, 0))

  sharded_batch = shard(batch)
  replicated_params, replicated_batch_stats, replicated_optimizer_state, loss_val, grad_norm = pmapped_update_step(sharded_batch, replicated_params, replicated_batch_stats, replicated_optimizer_state)

  grad_norm.block_until_ready()
  train_end = time.time()

  print('{}) loss_value = {}, grad_norm = {}, step_time = {}'.format(
      num_train_step,
      loss_val.mean(),
      grad_norm.mean(),
      train_end - train_start))
  num_train_step = num_train_step + 1
