
import torch

from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader

trainset_path = "/fast/najroldi/data/lm/slim_pajama/new_sp_15B_tokens/train"
vocab_size = 50280
seq_len = 2048
sampler = 'sequential'
sampler_seed = None
num_workers = 4

train_set = load_from_disk(trainset_path)  # <class 'datasets.arrow_dataset.Dataset'>

"""
>>> type(train_set)
<class 'datasets.arrow_dataset.Dataset'>

>>> len(train_set)
7501407

>>> train_set[0]
{'input_ids': tensor([ 5166,    20,  1639,  ...,   275,   253, 19992])}

>>> type(train_set[0]['input_ids'])
<class 'torch.Tensor'>

# In PyTorch we do:
trainloader = DataLoader(
    train_set,
    sampler = ...,
    batch_size = ...,
    num_workers = ...,
    pin_memory = ...,
  )

# PyTorch’s DataLoader expects an iterable dataset, 
# which means it calls __getitem__() and __len__() on train_set.

"""

