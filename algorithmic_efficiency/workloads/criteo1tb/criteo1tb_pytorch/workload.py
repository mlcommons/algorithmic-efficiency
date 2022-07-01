import contextlib
import os 
from typing import Any, Callable, List, Optional, Type, Union, Dict, Tuple
import functools
import sys
import time
import math

#import numpy as np
import torch
from torch import nn 
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from algorithmic_efficiency import data_utils
from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
import algorithmic_efficiency.random_utils as prng
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.dlrm_small_model import DlrmSmall
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.input_pipeline import CriteoBinDataset, data_collate_fn, prefetcher
#from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch.utils.distributed import DistributedSampler
from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch import metrics 



PYTORCH_DDP = 'LOCAL_RANK' in os.environ 
RANK = int(os.environ['LOCAL_RANK']) if PYTORCH_DDP else 0
DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


_NUM_DENSE_FEATURES = 13
_VOCAB_SIZES = [1024 * 128] * 26 

class Criteo1TbDlrmSmallWorkload(spec.Workload):
    """
    Criteo1TB DLRM-Small Pytorch workload.
    """

    def __init__(self):
        self._eval_iters = {}
        self._param_shapes = None
        self._param_types = None
    
    def has_reached_goal(self, eval_result: float) -> bool:
        return eval_result['validation/auc_roc'] > self.target_value
    
    @property
    def target_value(self):
        return 0.8

    @property
    def target_value(self):
        return 0.8

    @property
    def loss_type(self):
        return spec.LossType.SIGMOID_CROSS_ENTROPY
    
    @property
    def num_train_examples(self):
        return 4_195_197_692
    
    @property
    def num_eval_train_examples(self):
        return 100_000
    
    @property
    def num_validation_examples(self):
        return 131072 * 8  # TODO(znado): finalize the validation split size.
    
    @property
    def num_test_examples(self):
        return None
    
    @property
    def train_mean(self):
        return 0.0
    
    @property
    def train_stddev(self):
        return 1.0
    
    @property
    def max_allowed_runtime_sec(self):
        return 6 * 60 * 60
    
    @property
    def eval_period_time_sec(self):
        return 20 * 60

    @property 
    def param_shapes(self):
        if self._param_shapes is None:
            raise ValueError("This should not happen, workload.init_model_fn() should be called \
          before workload.param_shapes!")
        return self._param_shapes

    @property
    def model_params_types(self):
        """The shapes of the parameters in the workload model."""
        if self._param_types is None:
          self._param_types = param_utils.pytorch_param_types(self._param_shapes)
        return self._param_types



    def _get_data_loader(self, 
                        data_rng: spec.RandomState,
                        split: str,
                        data_dir: str, 
                        global_batch_size: int, 
                        ):
        is_train = split =="train"


        if PYTORCH_DDP:
            batch_size = global_batch_size//N_GPUS
            if batch_size == 0:
                raise ValueError("global_bs can't be 0")
        else:
           batch_size = global_batch_size 

        return_device = DEVICE
        data_loader_args = dict(
                batch_size = None,
                num_workers = 0,
                pin_memory = False, 
                generator = torch.Generator().manual_seed(int(data_rng[0])),
                collate_fn=functools.partial(data_collate_fn, device=return_device, orig_stream=torch.cuda.current_stream()))


        if is_train:
            train_data_set_bin = os.path.join(data_dir, "train_data.bin")
            dataset_train = CriteoBinDataset(train_data_set_bin, batch_size = batch_size, shuffle=False)
            if PYTORCH_DDP:
                sampler = DistributedSampler(
                    dataset_train, num_replicas=N_GPUS, rank=RANK, shuffle=False, drop_last=False)
                data_loader_args.update({'sampler':sampler})
            data_loader_train = torch.utils.data.DataLoader(dataset_train, **data_loader_args)
            return data_loader_train

        else:
            test_data_set_bin = os.path.join(data_dir, "test_data.bin")
            dataset_test = CriteoBinDataset(test_data_set_bin, batch_size = batch_size)
            if PYTORCH_DDP:
                sampler = data_utils.DistributedEvalSampler(
                    dataset_test, num_replicas=N_GPUS, rank=RANK, shuffle=False)
                data_loader_args.update({'sampler':sampler})
            data_loader_test = torch.utils.data.DataLoader(dataset_test, **data_loader_args)
            return data_loader_test

    
    
    def build_input_queue(self, 
                          data_rng: spec.RandomState, 
                          split: str, 
                          data_dir: str,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
    
        data_stream = torch.cuda.Stream()

        if split == "train":
            data_loader_train = self._get_data_loader(data_rng, split, data_dir, global_batch_size)
            for step, (numerical_features, categorical_features, click) in enumerate(prefetcher(iter(data_loader_train), data_stream)):
                torch.cuda.current_stream().wait_stream(data_stream)
            #for step, (numerical_features, categorical_features, click) in enumerate(iter(data_loader_train)):
                
                #print(f"##################################### num_fea: {numerical_features.shape}################")
                #print(f"##################################### cat_fea: {categorical_features.shape}################")
                #print(f"##################################### click: {click.shape}################")

                yield {
                        'inputs': (numerical_features, categorical_features),
                        'targets': click 
                      }
        else:
            data_loader_test = self._get_data_loader(data_rng, split, data_dir, global_batch_size) 
            for step, (numerical_features, categorical_features, click) in enumerate(prefetcher(iter(data_loader_test), data_stream)):
                torch.cuda.current_stream().wait_stream(data_stream)
                yield {
                         'inputs': (numerical_features, categorical_features),
                         'targets': click
                      }






        
    def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
        #print(rng)
        torch.random.manual_seed(rng[0])
        #np.random.seed(rng[0])
        #np.set_printoptions(precision=4)
        torch.set_printoptions(precision=4)
        torch.manual_seed(rng[0])
    
        torch.cuda.manual_seed_all(rng[0])
        torch.backends.cudnn.deterministic = True

        model = DlrmSmall(
            vocab_sizes=_VOCAB_SIZES,
            total_vocab_sizes=sum(_VOCAB_SIZES),
            num_dense_features=_NUM_DENSE_FEATURES
        )

        self._param_shapes = {
                k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
                }

        model.to(DEVICE)

        if N_GPUS > 1:
            if PYTORCH_DDP:
                model = DDP(model, device_ids=[RANK], output_device=RANK)
            else:
                model = torch.nn.DataParallel(model)
        return model, None

    def model_fn(
            self, 
            params: spec.ParameterContainer,
            augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
            model_state: spec.ModelAuxiliaryState,
            mode: spec.ForwardPassMode,
            rng: spec.RandomState,
            update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        
        del model_state
        del rng
        del update_batch_norm

        model = params

        if mode == spec.ForwardPassMode.EVAL:
            model.eval()

        if mode == spec.ForwardPassMode.TRAIN:
            model.train()
            
        contexts = {
                spec.ForwardPassMode.EVAL: torch.no_grad,
                spec.ForwardPassMode.TRAIN: contextlib.nullcontext 
        }

        with contexts[mode]():
            logits_batch = model(augmented_and_preprocessed_input_batch['inputs'], None)

        return logits_batch, None


    def output_activation_fn(self,
                              logits_batch: spec.Tensor,
                              loss_type: spec.LossType) -> spec.Tensor:
        pass 


    def loss_fn(self,
            label_batch: spec.Tensor,
            logits_batch: spec.Tensor,
            mask_batch: Optional[spec.Tensor] = None) -> spec.Tensor:
        
         per_example_losses = metrics.per_example_sigmoid_binary_cross_entropy(
                 logits = logits_batch, targets=label_batch)
    
         if mask_batch is not None:
             weighted_losses = per_example_losses * mask_batch
             normalization = mask_batch.sum()
    
         else:
             weighted_losses = per_example_losses
         normalization = label_batch.shape[0]
    
         return torch.sum(weighted_losses, dim=-1) / normalization

    def _eval_metric(self, 
           logits:spec.Tensor,
           targets: spec.Tensor) -> Dict[str, int]:
        # implement the metrics in question
        loss = self.loss_fn(logits, logits) 
        auc_roc = metrics.roc_auc_score(logits, targets)
        return {'auc_roc':auc_roc, 'loss': loss} 




    def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int, 
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:

       """Evaluate the model on a given dataset split, return final scalars."""
       data_rng, model_rng = prng.split(rng, 2)
       if split not in self._eval_iters:
           # These iterators repeat indefinitely.
           self._eval_iters[split] = self.build_input_queue(
                   data_rng, split, data_dir, global_batch_size = global_batch_size)

       total_metrics = {
               'loss': torch.tensor(0., device=DEVICE),
               'auc_roc': torch.tensor(0., device=DEVICE),
               }

       num_batches =  int(math.ceil(num_examples / global_batch_size))

       
       for _ in range(num_batches):
           batch = next(self._eval_iters[split])
           logits, _ = self.model_fn(
                   params,
                   batch,
                   model_state,
                   spec.ForwardPassMode.EVAL,
                   model_rng,
                   False
                   )

           batch_metrics = self._eval_metric(logits, batch['targets'])
           total_metrics = {
                   k: v + batch_metrics[k] for k, v in total_metrics.items()
           }

       if PYTORCH_DDP:
           for metric in total_metrics.values():
               dist.all_reduce(metric)
        return {k: float(v.item() / num_examples) for k, v in total_metrics.items()}



    @property 
    def step_hint(self):
        return 64_000

    
    def is_output_params(self, param_key:spec.ParameterKey) -> bool:
        raise NotImplementedError
    

  


    
