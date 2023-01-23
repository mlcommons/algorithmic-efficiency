import jax

print('Checking GPU presence in JAX')
print(jax.local_device_count())
print(jax.device_count())

rng = jax.random.PRNGKey(0)
data_rng, shuffle_rng = jax.random.split(rng, 2)
