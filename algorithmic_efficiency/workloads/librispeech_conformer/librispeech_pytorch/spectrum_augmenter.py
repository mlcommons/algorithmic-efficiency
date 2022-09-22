"""
This is a pytorch implementation mirroring:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/spectrum_augmenter.py
"""

import torch
from torch import nn


class SpecAug(nn.Module):
  """Layer performs masking prodecure along time and frequency axis.

    The procedure is detailed in https://arxiv.org/abs/1904.08779.
    This is an essential component in speech recognition models that
    helps achieve better word error rates.
    """

  def __init__(self,
               freq_mask_count: int = 1,
               freq_mask_max_bins: int = 15,
               time_mask_count: int = 1,
               time_mask_max_frames: int = 50,
               time_mask_max_ratio: float = 1.0,
               time_masks_per_frame: float = 0.0,
               use_dynamic_time_mask_max_frames: bool = False):
    super().__init__()

    self.freq_mask_count = freq_mask_count
    self.freq_mask_max_bins = freq_mask_max_bins
    self.time_mask_count = time_mask_count
    self.time_mask_max_frames = time_mask_max_frames
    self.time_mask_max_ratio = time_mask_max_ratio
    self.time_masks_per_frame = time_masks_per_frame
    self.use_dynamic_time_mask_max_frames = use_dynamic_time_mask_max_frames

  def next_prng_key(self, name='dropout'):
    return self.make_rng(name)

  def _get_mask(self,
                batch_size,
                choose_range,
                mask_size,
                max_length=None,
                masks_per_frame=0.0,
                multiplicity=1,
                max_ratio=1.0,
                device='cpu'):
    # Sample lengths for multiple masks.
    if max_length and max_length > 0:
      max_length = max_length * torch.ones(batch_size, device=device)
    else:
      max_length = choose_range * max_ratio
    masked_portion = torch.rand(batch_size, multiplicity, device=device)
    masked_frame_size = torch.einsum('b,bm->bm', max_length,
                                     masked_portion).long()
    # Make sure the sampled length was smaller than max_ratio * length_bound.
    # Note that sampling in this way was biased
    # (shorter sequence may over-masked.)
    choose_range = torch.tile(choose_range[:, None], [1, multiplicity])
    length_bound = (max_ratio * choose_range).long()
    length = torch.minimum(masked_frame_size, length_bound.clamp(min=1))

    # Choose starting point.
    random_start = torch.rand(batch_size, multiplicity, device=device)
    start_with_in_valid_range = random_start * (choose_range - length + 1)
    start = start_with_in_valid_range.long()
    end = start + length - 1

    # Shift starting and end point by small value.
    delta = 0.1
    start = (start - delta)[..., None]
    start = torch.tile(start, [1, 1, mask_size])
    end = (end + delta)[..., None]
    end = torch.tile(end, [1, 1, mask_size])

    # Construct pre-mask of shape (batch_size, multiplicity, mask_size).
    diagonal = torch.arange(mask_size, device=device).reshape(1, 1, -1)
    diagonal = torch.tile(diagonal, [batch_size, multiplicity, 1])
    pre_mask = torch.minimum(diagonal < end, diagonal > start)

    # Sum masks with appropriate multiplicity.
    if masks_per_frame > 0:
      multiplicity_weights = torch.tile(
          torch.arange(multiplicity, device=device).long()[None, ...],
          [batch_size, 1])
      multiplicity_tensor = masks_per_frame * choose_range
      multiplicity_weights = (multiplicity_weights < multiplicity_tensor).long()
      pre_mask = torch.einsum('bmt,bm->bt', pre_mask, multiplicity_weights)
    else:
      pre_mask = torch.einsum('bmt->bt', pre_mask)
    mask = 1.0 - (pre_mask > 0).long()

    return mask

  def _time_mask(self, inputs, length):
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
        max_length=time_mask_max_frames,
        masks_per_frame=self.time_masks_per_frame,
        multiplicity=multiplicity,
        max_ratio=max_ratio,
        device=inputs.device)

    outputs = torch.einsum('bxy,bx->bxy', inputs, block_arrays)
    return outputs

  def _frequency_mask(self, inputs):
    # Mask parameters.
    freq_mask_max_bins = self.freq_mask_max_bins
    multiplicity = self.freq_mask_count

    # If masking length or count is zero, do nothing.
    if freq_mask_max_bins == 0 or multiplicity == 0:
      return inputs

    # Arguments to pass to mask generator.
    batch_size, _, num_freq = inputs.shape
    choose_range = num_freq * torch.ones(batch_size, device=inputs.device)
    # Create masks in frequency direction and apply.
    block_arrays = self._get_mask(
        batch_size,
        choose_range=choose_range,
        mask_size=num_freq,
        max_length=freq_mask_max_bins,
        masks_per_frame=0.0,
        multiplicity=multiplicity,
        max_ratio=1.0,
        device=inputs.device)

    outputs = torch.einsum('bxy,by->bxy', inputs, block_arrays)
    return outputs

  def forward(self, inputs, paddings):
    lengths = torch.einsum('bh->b', 1 - paddings).long()
    inputs = self._time_mask(inputs, lengths)
    inputs = self._frequency_mask(inputs)
    return inputs, paddings
