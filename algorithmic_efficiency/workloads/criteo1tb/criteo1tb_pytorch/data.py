import os 
import math

import numpy as np
import torch
from torch.utils.data import Dataset

def data_collate_fn(batch_data, device="cuda", orig_stream=None):
    """Split raw batch data to features and labels
    Args:
        batch_data (Tensor): One batch of data from CriteoBinDataset.
        device (torch.device): Output device. If device is GPU, split data on GPU is much faster.
        orig_stream (torch.cuda.Stream): CUDA stream that data processing will be run in.
    Returns:
        numerical_features (Tensor):
        categorical_features (Tensor):
        click (Tensor):
    """
    if not isinstance(batch_data, torch.Tensor):
        # Distributed pass
        if batch_data[1] is not None:
            numerical_features = torch.log(batch_data[1].to(device, non_blocking=True) + 1.).squeeze()
        else:
            # There are codes rely on numerical_features' dtype
            numerical_features = torch.empty(batch_data[0].shape[0], 13, dtype=torch.float32, device=device)
        if batch_data[2] is not None:
            categorical_features = batch_data[2].to(device, non_blocking=True)
        else:
            categorical_features = None
        click = batch_data[0].to(device, non_blocking=True).squeeze()
    else:
        batch_data = batch_data.to(device, non_blocking=True).split([1, 13, 26], dim=1)
        numerical_features = torch.log(batch_data[1].to(torch.float32) + 1.).squeeze()
        categorical_features = batch_data[2].to(torch.long)
        click = batch_data[0].to(torch.float32).squeeze()

    # record_stream() prevents data being unintentionally reused. Aslo NOTE that it may not work
    # with num_works >=1 in the DataLoader when use this data_collate_fn() as collate function.
    if orig_stream is not None:
        numerical_features.record_stream(orig_stream)
        if categorical_features is not None:
            categorical_features.record_stream(orig_stream)
        click.record_stream(orig_stream)

    return numerical_features, categorical_features, click


def prefetcher(load_iterator, prefetch_stream):
    def _prefetch():
        with torch.cuda.stream(prefetch_stream):
            try:
                data_batch = next(load_iterator)
            except StopIteration:
                return None

        return data_batch

    next_data_batch = _prefetch()

    while next_data_batch is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        data_batch = next_data_batch
        next_data_batch = _prefetch()
        yield data_batch

def _dist_permutation(size):
    """Generate permutation for dataset shuffle

        Args:
            size (int): Size and high value of permutation
        Returns:
            permutation (ndarray):
    """
    if torch.distributed.get_world_size() > 1:
        # To guarantee all ranks have the same same permutation, generating it from rank 0 and sync
        # to other rank by writing to disk
        permutation_file = "/tmp/permutation.npy"

        if int(os.environ['LOCAL_RANK'])== 0:
            np.save(permutation_file, np.random.permutation(size))

        torch.distributed.barrier()
        permutation = np.load(permutation_file)
    else:
        permutation = np.random.permutation(size)

    return permutation




class CriteoBinDataset(Dataset):
    """Binary version of criteo dataset.
    Main structure is copied from reference. With following changes:
    - Removed unnecessary things, like counts_file which is not really used in training.
    - _transform_features is removed, doing it on GPU is much faster.
    """
    def __init__(self, data_file, batch_size=1, bytes_per_feature=4, shuffle=False):
        # dataset. single target, 13 dense features, 26 sparse features
        self.tad_fea = 1 + 13
        self.tot_fea = 1 + 13 + 26

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.tot_fea * batch_size)

        self.num_batches = math.ceil(os.path.getsize(data_file) / self.bytes_per_batch)

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb', buffering=0)

        if shuffle:
            self.permutation = _dist_permutation(self.num_batches - 1)
        else:
            self.permutation = None


    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.permutation is not None and idx != self.num_batches - 1:
            idx = self.permutation[idx]
        self.file.seek(idx * self.bytes_per_batch, 0)
        raw_data = self.file.read(self.bytes_per_batch)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array).view((-1, self.tot_fea))

        return tensor

    def __del__(self):
        self.file.close()
