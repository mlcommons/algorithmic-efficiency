"""Submission file for a SGD with Nesterov optimizer in Jax."""

import functools
import inspect
from typing import Callable, Iterable, Union

from flax import jax_utils
import jax
import jax.numpy as jnp
import optax

from algorithmic_efficiency import spec


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a Nesterov optimizer and a learning rate schedule."""
  del model_params
  del model_state
  del rng

  # Create optimizer.
  params_zeros_like = jax.tree_map(lambda s: jnp.zeros(s.shape_tuple),
                                   workload.param_shapes)
  opt_init_fn, opt_update_fn = static_inject_hyperparams(sgd)(
      learning_rate=0.0,  # Manually injected on each train step.
      weight_decay=hyperparameters.l2,
      momentum=hyperparameters.beta1,
      nesterov=True)
  optimizer_state = opt_init_fn(params_zeros_like)

  # Create learning rate schedule.
  lr_schedule_fn = create_lr_schedule_fn(hyperparameters)

  optimizer_state = {
      'optimizer_state': jax_utils.replicate(optimizer_state),
      'lr_schedule_fn': lr_schedule_fn
  }

  return optimizer_state, opt_update_fn


def create_lr_schedule_fn(
    hyperparameters: spec.Hyperparameters) -> Callable[[int], float]:
  schedule_hparams = {
      'schedule': 'polynomial_warmup',
      'power': 1,
      'base_lr': hyperparameters.learning_rate,
      'decay_steps_factor': hyperparameters.decay_steps_factor,
      'end_factor': hyperparameters.end_factor,
      'warmup_steps': hyperparameters.warmup_steps
  }
  lr_schedule_fn = prepend_linear_warmup(schedule_hparams,
                                         hyperparameters.num_steps,
                                         polynomial_schedule)
  return lr_schedule_fn


# Forked from
# github.com/google/init2winit/blob/master/init2winit/optimizer_lib/utils.py.
def static_inject_hyperparams(
    inner_factory: Callable[..., optax.GradientTransformation],
    injectable_args: Union[str, Iterable[str]] = ('learning_rate',)
) -> Callable[..., optax.GradientTransformation]:
  """Wrapper for `optax.inject_hyperparams` making all args static by default.
  This wrapper resolves two issues:
  1. If anyone adds an optional argument to an `optax` optimizer, code
     will break because `optax.inject_hyperparams` will pass 0.0.
  2. Optimizers like `adafactor` have arguments that are not boolean, but are
     used in boolean statements, which leads to ConcretizationTypeErrors.
  Args:
    inner_factory: a function that returns the inner
      ``optax.GradientTransformation`` given the hyperparameters.
    injectable_args: a string or iterable of strings specifying which callable
      parameters **are** schedules.
  Returns:
    A callable that returns a ``optax.GradientTransformation``. This callable
    accepts the same arguments as ``inner_factory`` and you may provide
    schedules for the args listed in `injectable_args`.
  """

  injectable_args = ({injectable_args} if isinstance(injectable_args, str) else
                     set(injectable_args))
  inner_signature = inspect.signature(inner_factory)

  @functools.wraps(inner_factory)
  def wrapped_transform(*args, **kwargs) -> optax.GradientTransformation:
    bound_arguments = inner_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    static_args = set(bound_arguments.arguments.keys()) - injectable_args

    return optax.inject_hyperparams(inner_factory, static_args)(*args, **kwargs)

  return wrapped_transform


# Forked from github.com/google/init2winit/blob/master/init2winit/ (cont. below)
# optimizer_lib/optimizers.py.
def sgd(learning_rate, weight_decay, momentum=None, nesterov=False):
  r"""A customizable gradient descent optimizer.
  NOTE: We apply weight decay **before** computing the momentum update.
  This is equivalent to applying WD after for heavy-ball momentum,
  but slightly different when using Nesterov accelleration. This is the same as
  how the Flax optimizers handle weight decay
  https://flax.readthedocs.io/en/latest/_modules/flax/optim/momentum.html.
  Args:
    learning_rate: The learning rate. Expected as the positive learning rate,
      for example `\alpha` in `w -= \alpha * u` (as opposed to `\alpha`).
    weight_decay: The weight decay hyperparameter.
    momentum: The momentum hyperparameter.
    nesterov: Whether or not to use Nesterov momentum.
  Returns:
    An optax gradient transformation that applies weight decay and then one of a
    {SGD, Momentum, Nesterov} update.
  """
  return optax.chain(
      optax.add_decayed_weights(weight_decay),
      optax.sgd(
          learning_rate=learning_rate, momentum=momentum, nesterov=nesterov))


# Forked from github.com/google/init2winit/blob/master/init2winit/schedules.py.
def prepend_linear_warmup(schedule_hparams,
                          max_training_updates,
                          base_lr_schedule):
  """Models the base_lr_schedule to include a warmup phase.
  The returned schedule will have the following form:
  if step < hps.warmup_steps:
     lr = (step / warmup_steps) ** warmup_power * base_lr
  otherwise:
    lr = base_lr_schedule(step - hps.warmup_steps)
    where the max train steps input to base_lr_schedule is
    max_train_steps - hps.warmup_steps.
  Effectively, what this does is the first warmup_steps will be linear warmup
  (if power =1), followed by what the base_lr_schedule would be if called with
  max_train_steps - warmup_steps. The default value for warmup_power is 1
  meaning linear warmup
  Args:
    schedule_hparams: Must include all required hparams needed in
      base_lr_schedule. Additionally we require warmup_steps, warmup_power to
      be added.
    max_training_updates: Full number of model updates to be used in training.
    base_lr_schedule: One of the schedule functions defined in this module.
      Must satisfy the API of -
      base_lr_schedule(schedule_hparams, max_training_updates) -> returns lr_fn.
  Returns:
    A function mapping global_step to learning rate.
  """

  # grab warmup hparams
  schedule_hparams = dict(schedule_hparams)  # convert to dict so we can pop
  warmup_steps = schedule_hparams.pop('warmup_steps')
  warmup_power = schedule_hparams.pop('warmup_power', 1)
  base_lr = schedule_hparams['base_lr']

  base_lr_fn = base_lr_schedule(schedule_hparams,
                                max_training_updates - warmup_steps)

  def lr_fn(t):
    if t < warmup_steps:
      return ((t / warmup_steps)**warmup_power) * base_lr
    step = t - warmup_steps
    return base_lr_fn(step)

  return lr_fn


# Forked from github.com/google/init2winit/blob/master/init2winit/schedules.py.
def polynomial_schedule(schedule_hparams, max_training_updates):
  """Same behavior as tf.train.polynomial_decay.
  Supports either decay_steps or decay_steps_factor, but providing both is an
  error.
  Args:
    schedule_hparams: Relevant hparams are schedule,
      base_lr, end_factor, power, and one of decay_steps or
      decay_steps_factor.
    max_training_updates: Only used when decay_steps_factor is provided.
  Returns:
    lr_fn: A function mapping global_step to lr.
  """
  expected_keys = ['schedule', 'base_lr', 'end_factor', 'power']
  if 'decay_steps' in schedule_hparams:
    expected_keys.append('decay_steps')
    decay_steps = schedule_hparams.decay_steps
  else:
    expected_keys.append('decay_steps_factor')
    decay_steps = int(max_training_updates *
                      schedule_hparams['decay_steps_factor'])
  if set(schedule_hparams.keys()) != set(expected_keys):
    raise ValueError(
        'Provided schedule_hparams keys are invalid. Recieved: {}, Expected: {}'
        .format(sorted(schedule_hparams.keys()), sorted(expected_keys)))

  end_learning_rate = schedule_hparams['base_lr'] * schedule_hparams[
      'end_factor']

  def lr_fn(t):
    step = min(decay_steps, t)
    decayed_learning_rate = (schedule_hparams['base_lr'] -
                             end_learning_rate) * (1 - step / decay_steps)**(
                                 schedule_hparams['power']) + end_learning_rate
    return decayed_learning_rate

  return lr_fn
