import torch
import numpy as np


# from NVIDIA DL Examples:
# github.com/NVIDIA/DeepLearningExamples/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
def fast_collate(batch):
  imgs = [img[0] for img in batch]
  targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
  w = imgs[0].size[0]
  h = imgs[0].size[1]
  tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8).contiguous(
      memory_format=torch.contiguous_format)
  for i, img in enumerate(imgs):
    nump_array = np.asarray(img, dtype=np.uint8)
    if nump_array.ndim < 3:
      nump_array = np.expand_dims(nump_array, axis=-1)
    nump_array = np.rollaxis(nump_array, 2)

    tensor[i] += torch.from_numpy(nump_array.copy())

  return tensor, targets


# from https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
def cycle(iterable):
  iterator = iter(iterable)
  while True:
    try:
      yield next(iterator)
    except StopIteration:
      iterator = iter(iterable)

