import torch
import jax
import numpy as np
import jax.numpy as jnp 

from algorithmic_efficiency.workloads.librispeech_conformer.input_pipeline import \
    LibriSpeechDataset

from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax import librispeech_preprocessor as preprocessor
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_jax import model as model 

from algorithmic_efficiency import data_utils

def shard(batch, n_devices=None):
    if n_devices is None:
      n_devices = max(torch.cuda.device_count(), jax.local_device_count())

    # Otherwise, the entries are arrays, so just reshape them.
    def _shard_array(array):
      return array.reshape((n_devices, -1) + array.shape[1:])

    return jax.tree_map(_shard_array, batch)

def maybe_pad_batch(batch, desired_batch_size, padding_value=0.0):
  """Zero pad the batch on the right to desired_batch_size.

  All keys in the batch dictionary will have their corresponding arrays
  padded. Will return a dictionary with the same keys.

  Args:
    batch: A dictionary mapping keys to arrays. We assume that inputs is
    one of the keys.
    desired_batch_size: All arrays in the dict will be padded to have
    first dimension equal to desired_batch_size.
    padding_value: value to be used as padding.

  Returns:
    A dictionary mapping the same keys to the padded batches. Additionally
    we add a key representing weights, to indicate how the batch was padded.
  """
  batch_axis = 0
  inputs, input_paddings = batch['inputs']
  targets, target_paddings = batch['targets']

  batch_size = inputs.shape[batch_axis]
  batch_pad = desired_batch_size - batch_size

  # Most batches will not need padding so we quickly return to avoid slowdown.
  if batch_pad == 0:
    new_batch = jax.tree_map(lambda x: x, batch)
    return new_batch

  def zero_pad(ar, pad_axis):
    pw = [(0, 0)] * ar.ndim
    pw[pad_axis] = (0, batch_pad)
    return np.pad(ar, pw, mode='constant', constant_values=padding_value)

  padded_batch = {
      'inputs': (zero_pad(inputs, batch_axis),
                  zero_pad(input_paddings, batch_axis)),
      'targets': (zero_pad(targets, batch_axis),
                  zero_pad(target_paddings, batch_axis))
  }
  return padded_batch

if __name__ == '__main__':
    split ='train'
    global_batch_size = 1
    data_dir='/data/data'

    if split == 'train':
      split = 'train-clean-100+train-clean-360+train-other-500'
      train = True
    elif split == 'eval_train':
      split = 'train-clean-100'
    elif split == 'validation':
      split = 'dev-clean+dev-other'
    elif split == 'test':
      split = 'test-clean'

    ds = LibriSpeechDataset(split=split, data_dir=data_dir)
    ds_iter_batch_size = global_batch_size
    sampler = None

    dataloader = torch.utils.data.DataLoader(
      ds,
      batch_size=ds_iter_batch_size,
      shuffle=train,
      sampler=sampler,
      num_workers=4,
      prefetch_factor=10, 
      pin_memory=False,
      drop_last=train,)

    dataloader = data_utils.cycle(
      dataloader, custom_sampler=False, use_mixup=False)
    
    rng = jax.random.PRNGKey(0)
      
    i = 0
    for batch in iter(dataloader):
      inputs, input_paddings = batch['inputs']
      targets, target_paddings = batch['targets']

      numpy_batch =  {
        'inputs': (inputs.numpy(), input_paddings.numpy()),
        'targets': (targets.numpy(), target_paddings.numpy()),
      }

      padded_batch = maybe_pad_batch(numpy_batch, global_batch_size, padding_value=1.0)
      # # sharded_padded_batch = shard(padded_batch)

      inputs, input_paddings = padded_batch['inputs']
      print('inputs shape before = ', inputs.shape)

      inputs = jnp.array(inputs)
      input_paddings = jnp.array(input_paddings)

      preprocessing_config = preprocessor.LibrispeechPreprocessingConfig()
      fft_layer = preprocessor.MelFilterbankFrontend(
          preprocessing_config,
          per_bin_mean=preprocessor.LIBRISPEECH_MEAN_VECTOR,
          per_bin_stddev=preprocessor.LIBRISPEECH_STD_VECTOR)

      
      fft_vars = fft_layer.init(rng, inputs, input_paddings)
      inputs, input_paddings = fft_layer.apply(fft_vars, inputs, input_paddings) 

      print('inputs shape after FFT = ', inputs.shape)
      config = model.DeepspeechConfig()
      subsampling_layer = model.Subsample(config)
      
      subsampling_vars = subsampling_layer.init(rng, inputs, input_paddings, train=False)
      inputs, input_paddings = subsampling_layer.apply(subsampling_vars, inputs, input_paddings, False, rngs={'params':rng, 'dropout': rng})

      print('inputs shape after = ', inputs.shape)
      print('input_paddings shape after = ', input_paddings.shape)

      inputs = jnp.reshape(inputs, -1)
      input_paddings = jnp.reshape(input_paddings, -1)

      padded_batch['inputs'] = (inputs, input_paddings) 
      np.savez("sharded_padded_batch_{}".format(i), **padded_batch)
      i = i + 1

      if i == 3:
        break
