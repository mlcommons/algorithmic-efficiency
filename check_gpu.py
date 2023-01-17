import jax

print('Checking GPU presence in JAX')
print(jax.local_device_count())
