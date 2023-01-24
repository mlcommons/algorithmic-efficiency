import jax

print('JAX identified %d GPU devices' % jax.local_device_count())
print('Generating RNG seed for CUDA sanity check ... ')
rng = jax.random.PRNGKey(0)
data_rng, shuffle_rng = jax.random.split(rng, 2)

if jax.local_device_count() == 8 and data_rng is not None:
    print('Woohoo 8 GPUs present and CUDA works!!')
