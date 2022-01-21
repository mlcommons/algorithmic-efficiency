"""Training algorithm track submission functions for WMT."""
import functools
from typing import Iterator, List, Tuple

from flax import jax_utils
from flax import linen as nn
from flax import optim
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import spec

from . import models


def get_batch_size(workload_name):
  batch_sizes = {"wmt_jax": 16}
  return batch_sizes[workload_name]


def create_learning_rate_scheduler(
    factors="constant * linear_warmup * rsqrt_decay",
    base_learning_rate=0.5,
    warmup_steps=1000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * rsqrt_normalized_decay: divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: string, factors separated by "*" that defines the schedule.
    base_learning_rate: float, the starting constant for the lr schedule.
    warmup_steps: int, how many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {"learning_rate": float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split("*")]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == "constant":
        ret *= base_learning_rate
      elif name == "linear_warmup":
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == "rsqrt_decay":
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "rsqrt_normalized_decay":
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == "decay_every":
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == "cosine_decay":
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError("Unknown factor %s." % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def train_step(optimizer,
               batch,
               config,
               learning_rate_fn,
               label_smoothing=0.1,
               dropout_rng=None):
  """Perform a single training step."""
  # X_position and X_segmentation are needed only when using "packed examples"
  # where multiple sequences are packed into the same example with this
  # metadata.
  # if such features are not present they are ignored and the example is
  # treated like a normal, unpacked sequence example.
  train_keys = [
      "inputs", "targets", "inputs_position", "targets_position",
      "inputs_segmentation", "targets_segmentation"
  ]
  (inputs, targets, inputs_positions, targets_positions, inputs_segmentation,
   targets_segmentation) = [batch.get(k, None) for k in train_keys]

  weights = jnp.where(targets > 0, 1, 0).astype(jnp.float32)

  dropout_rng = jax.random.fold_in(dropout_rng, optimizer.state.step)

  def loss_fn(params):
    """loss function used for training."""
    logits = models.Transformer(config).apply(
        {"params": params},
        inputs,
        targets,
        inputs_positions=inputs_positions,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        rngs={"dropout": dropout_rng})

    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) +
        (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
    soft_targets = common_utils.onehot(
        targets, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
    loss = loss - normalizing_constant

    loss = loss * weights
    normalizing_factor = weights.sum()

    mean_loss = loss.sum() / normalizing_factor
    return mean_loss, logits

  step = optimizer.state.step
  lr = learning_rate_fn(step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  _, grad = grad_fn(optimizer.target)
  grad = jax.lax.pmean(grad, "batch")
  new_optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

  return new_optimizer


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparamters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  del model_state
  del rng
  del workload

  optimizer_def = optim.Adam(
      learning_rate=hyperparameters.learning_rate,
      beta1=1.0 - hyperparameters.one_minus_beta_1,
      beta2=0.98,
      eps=hyperparameters.epsilon)
  optimizer = optimizer_def.create(model_params)

  # Replicate optimizer.
  optimizer = jax_utils.replicate(optimizer)

  learning_rate_fn = create_learning_rate_scheduler(
      base_learning_rate=hyperparameters.learning_rate, warmup_steps=1000)

  # compile multidevice versions of train.
  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          config=models.TransformerConfig(
              dropout_rate=hyperparameters.dropout_rate,
              attention_dropout_rate=hyperparameters.attention_dropout_rate),
          learning_rate_fn=learning_rate_fn),
      axis_name="batch",
      donate_argnums=(0,))

  return optimizer, p_train_step


def update_params(
    workload: spec.Workload,
    current_param_container: spec.ParameterContainer,
    current_params_types: spec.ParameterTypeTree,
    model_state: spec.ModelAuxiliaryState,
    hyperparameters: spec.Hyperparamters,
    input_batch: spec.Tensor,
    label_batch: spec.Tensor,
    # This will define the output activation via `output_activation_fn`.
    loss_type: spec.LossType,
    optimizer_state: spec.OptimizerState,
    eval_results: List[Tuple[int, float]],
    global_step: int,
    rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params)."""
  del workload
  del current_param_container
  del current_params_types
  del eval_results
  del global_step
  del model_state
  del loss_type
  del hyperparameters
  del label_batch

  optimizer, p_train_step = optimizer_state
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  optimizer = p_train_step(optimizer, input_batch, dropout_rng=dropout_rngs)

  return (optimizer, p_train_step), optimizer.target, None


# Not allowed to update the model parameters, hyperparameters, global step, or
# optimzier state.
def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Tuple[spec.Tensor, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   hyperparameters: spec.Hyperparamters, global_step: int,
                   rng: spec.RandomState) -> Tuple[spec.Tensor, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.

  Each element of the queue is a single training example and label.

  We left out `current_params_types` because we do not believe that it would
  # be necessary for this function.

  Return a tuple of input label batches.
  """
  del optimizer_state
  del current_param_container
  del global_step
  del rng
  del hyperparameters
  del workload

  return common_utils.shard(jax.tree_map(np.asarray, next(input_queue))), None

