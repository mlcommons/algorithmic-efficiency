import jax 
import jax.numpy as jnp
import jax.lax as lax
import flax
import tensorflow_datasets as tfds 


print('num_devices = ', jax.local_device_count())
print('devices = ', jax.devices())

# train_ds_builder = tfds.builder('librispeech')
# print(train_ds_builder.info)

with jax.profiler.trace(log_dir="./jax-trace"):
    data = jnp.arange(10).astype(jnp.float32)
    data = jnp.reshape(data, (1, 5, 2)) # (N = 1) x (W = 5) x (C = 2)

    kernel = jnp.array([[[1, 0], [0, 1]]], dtype=jnp.float32) # (O = 1) x (I = 1) x (W = 2)
    print('data shape = ', data.shape)
    print('kernel shape = ', kernel.shape)

    print('data = ', data)
    print('kernel = ', kernel)

    dn = lax.conv_dimension_numbers(data.shape,     # only ndim matters, not shape
                                    kernel.shape,  # only ndim matters, not shape 
                                    ('NWC', 'OIW', 'NWC'))  # the important bit
    print(dn)

    out = lax.conv_general_dilated(data,    # lhs = image tensor
                                kernel, # rhs = conv kernel tensor
                                (1,),  # window strides
                                'VALID', # padding mode
                                (1,),  # lhs/image dilation
                                (1,),  # rhs/kernel dilation
                                dn)     # dimension_numbers = lhs, rhs, out dimension permutation

    print('out shape = ', out.shape)
    print('convolution result = ', out)