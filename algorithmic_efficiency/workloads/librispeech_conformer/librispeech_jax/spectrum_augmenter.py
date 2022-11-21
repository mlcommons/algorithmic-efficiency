"""A flax layer to do data augmentation for audio signals as
described in https://arxiv.org/abs/1904.08779.

Code based on:
github.com/tensorflow/lingvo/blob/master/lingvo/jax/layers/spectrum_augmenter.py
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class SpecAug(nn.Module):
  """Layer performs masking prodecure along time and frequency axis.

  The procedure is detailed in https://arxiv.org/abs/1904.08779.
  This is an essential component in speech recognition models that helps achieve
  better word error rates.
  """
  freq_mask_count: int = 1
  freq_mask_max_bins: int = 15 
  time_mask_count: int = 1
  time_mask_max_frames: int = 50
  time_mask_max_ratio: float = 1.0
  time_masks_per_frame: float = 0.0
  use_dynamic_time_mask_max_frames: bool = False

  def setup(self):
    self.rng = self.make_rng('dropout')

  def _get_mask(self,
                batch_size,
                choose_range,
                mask_size,
                global_seed,
                max_length=None,
                masks_per_frame=0.0,
                multiplicity=1,
                max_ratio=1.0):
    # Sample lengths for multiple masks.
    if max_length and max_length > 0:
      max_length = jnp.tile(max_length, (batch_size,))
    else:
      max_length = choose_range * max_ratio
    masked_portion = jax.random.uniform(
        key=global_seed,
        shape=(batch_size, multiplicity),
        minval=0.0,
        maxval=1.0)
    masked_frame_size = jnp.einsum('b,bm->bm', max_length,
                                   masked_portion).astype(jnp.int32)
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way was biased
    # (shorter sequence may over-masked.)
    choose_range = jnp.tile(choose_range[:, None], [1, multiplicity])
    length_bound = (max_ratio * choose_range).astype(jnp.int32)
    length = jnp.minimum(masked_frame_size, jnp.maximum(length_bound, 1))

    # Choose starting point.
    random_start = jax.random.uniform(
        key=global_seed, shape=(batch_size, multiplicity), maxval=1.0)
    start_with_in_valid_range = random_start * (choose_range - length + 1)
    start = start_with_in_valid_range.astype(jnp.int32)
    end = start + length - 1

    # Shift starting and end point by small value.
    delta = 0.1
    start = jnp.expand_dims(start - delta, -1)
    start = jnp.tile(start, [1, 1, mask_size])
    end = jnp.expand_dims(end + delta, -1)
    end = jnp.tile(end, [1, 1, mask_size])

    # Construct pre-mask of shape (batch_size, multiplicity, mask_size).
    diagonal = jnp.expand_dims(jnp.expand_dims(jnp.arange(mask_size), 0), 0)
    diagonal = jnp.tile(diagonal, [batch_size, multiplicity, 1])
    pre_mask = jnp.minimum(diagonal < end, diagonal > start)

    # Sum masks with appropriate multiplicity.
    if masks_per_frame > 0:
      multiplicity_weights = jnp.tile(
          jnp.expand_dims(jnp.arange(multiplicity, dtype=jnp.int32), 0),
          [batch_size, 1])
      multiplicity_tensor = masks_per_frame * choose_range
      multiplicity_weights = (multiplicity_weights <
                              multiplicity_tensor).astype(jnp.int32)
      pre_mask = jnp.einsum('bmt,bm->bt', pre_mask, multiplicity_weights)
    else:
      pre_mask = jnp.einsum('bmt->bt', pre_mask)
    mask = 1.0 - (pre_mask > 0).astype(jnp.int32)

    return mask

  def _time_mask(self, inputs, length, global_seed):
    # Get time masking parameters.
    time_mask_max_frames = self.time_mask_max_frames
    use_dynamic_time_mask_max_frames = self.use_dynamic_time_mask_max_frames
    multiplicity = self.time_mask_count
    max_ratio = self.time_mask_max_ratio

    # If maximum mask length is zero, do nothing.
    if ((time_mask_max_frames == 0 and not use_dynamic_time_mask_max_frames) or
        max_ratio <= 0.0):
      return inputs
    if multiplicity == 0:
      return inputs
    batch_size, time_length, _ = inputs.shape

    # When using dynamic time mask size, discard upper-bound on
    # maximum allowed frames for time mask.
    if use_dynamic_time_mask_max_frames:
      time_mask_max_frames = None
    # Create masks in time direction and apply.
    block_arrays = self._get_mask(
        batch_size,
        choose_range=length,
        mask_size=time_length,
        global_seed=global_seed,
        max_length=time_mask_max_frames,
        masks_per_frame=self.time_masks_per_frame,
        multiplicity=multiplicity,
        max_ratio=max_ratio)

    outputs = jnp.einsum('bxy,bx->bxy', inputs, block_arrays)
    return outputs

  def _frequency_mask(self, inputs, global_seed):
    # Mask parameters.
    freq_mask_max_bins = self.freq_mask_max_bins
    multiplicity = self.freq_mask_count

    # If masking length or count is zero, do nothing.
    if freq_mask_max_bins == 0 or multiplicity == 0:
      return inputs

    # Arguments to pass to mask generator.
    batch_size, _, num_freq = inputs.shape
    choose_range = jnp.tile(num_freq, (batch_size,))
    # Create masks in frequency direction and apply.
    block_arrays = self._get_mask(
        batch_size,
        choose_range=choose_range,
        mask_size=num_freq,
        global_seed=global_seed,
        max_length=freq_mask_max_bins,
        masks_per_frame=0.0,
        multiplicity=multiplicity,
        max_ratio=1.0)

    outputs = jnp.einsum('bxy,by->bxy', inputs, block_arrays)
    return outputs

  @nn.compact
  def __call__(self, inputs, paddings):
    lengths = jnp.einsum('bh->b', 1 - paddings).astype(jnp.int32)

    inputs = self._time_mask(inputs, lengths, global_seed=self.rng)
    inputs = self._frequency_mask(inputs, global_seed=self.rng)

    return inputs, paddings
