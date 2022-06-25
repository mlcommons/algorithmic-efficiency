
import os 
import sys
sys.path.insert(0, os.getcwd())

from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.workload import Criteo1TbDlrmSmallPytorchWorkload 
from absl.testing import absltest

import torch
import time 

from torch.distributed.elastic.multiprocessing.errors import record


def init_distributed_mode(backend="nccl", use_gpu=True):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_RANK' in os.environ and 'OMPI_COMM_WORLD_SIZE' in os.environ:
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Not using distributed mode')
        return None, 1, None

    if use_gpu:
        torch.cuda.set_device(gpu)

    print('| distributed init (rank {})'.format(rank), flush=True)
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank, init_method='env://')

    return rank, world_size, gpu



@record
def main():
    rank, world_size, gpu = init_distributed_mode()
    wk = Criteo1TbDlrmSmallPytorchWorkload()
    a = wk.build_input_queue(
            23,
            "train",
            "/dataset",
            131072*8,
            #1*8,
            #1, # this is supposed to be global batch, needs to fix that to local batch
            2,
            )

    t1 = time.time()
    for _ in range(50):
        print(f"### batch: {_} ###")
        print(next(a))
        #next(a)
        print("################")
    print(f"batchtime for {_} batches: {time.time() - t1}")

if __name__ == '__main__':
    main()
