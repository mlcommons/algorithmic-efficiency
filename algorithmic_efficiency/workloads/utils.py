import flax.linen as nn
import jax


def print_jax_model_summary(model, fake_inputs):
  """Prints a summary of the jax module."""
  tabulate_fn = nn.tabulate(
      model,
      jax.random.PRNGKey(0),
      console_kwargs={
          'force_terminal': False, 'force_jupyter': False, 'width': 240
      },
  )
  print(tabulate_fn(fake_inputs, train=False))
