import contextlib
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
from torch import nn 
from torch.nn.parallel import DistributedDataParallel as DDP



from algorithmic_efficiency import spec

from algorithmic_efficiency.workloads.criteo1tb.criteo1tb_pytorch import DlrmSmall




PYTORCH_DDP = 'LOCAL_RANK' in os.environ 
RANK = int(os.environ['LOCAL_RANK']) if PYTORCH_DDP else 0
DEVICE = torch.device(f'cuda :{RANK}' torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


_NUM_DENSE_FEATURES = 13
_VOCAB_SIZES = [1024 * 128] * 26 

class Criteo1TbDlrmSmallPytorchWorkload(spec.Workload):
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
            raise ValueError("This should not happen, workload.init_model_fn() should be called 
          before workload.param_shapes!")
        return self._param_shapes

    @property
    def model_params_types(self):
        """The shapes of the parameters in the workload model."""
        if self._param_types is None:
          self._param_types = param_utils.pytorch_param_types(self._param_shapes)
        return self._param_types



    def _build_dataset(self, 
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str, 
                        global_batch_size: int, 
                        ):
    
    
        del data_rng
        is_train = split == "train"
    
    
    def build_input_queue(self, 
                          data_rng: jax.random.PRNGKey, 
                          split: str, 
                          data_dir: str,
                          global_batch_size: int,
                          num_batches: Optional[int] = None,
                          repeat_final_dataset: bool = False):
    
        ds = input_pipeline.get_criteo1tb_dataset(
                split = split, 
                data_dir = data_dir,
                is_training=(split == 'train'),
                global_batch_size = global_batch_size,
                vocab_sizes = _VOCAB_SIZES,
                num_batches = num_batches,
                repeat_final_dataset = repeat_final_dataset)
        
        #  run the iterator from here: https://pytorch.org/docs/master/data.html#torch.utils.data.IterableDataset
        for batch in iter(ds):
            batch = jax.tree_map(lambda x: x._numpy(), batch)
            yield batch


        
    def init_model_fn(self, rng: spec,RandomState) -> spec.ModelInitState:
        torch.random.manual_seed(rng[0])
        np.random.seed(rng[0])
        np.set_printoptions(precision=4)
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
                #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
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
            if update_batch_norm:
                raise ValueError(
                        "Batch norm statistics cannot be updated during evaluation.")
            model.eval()

        if mode == spec.ForwardPassMode.TRAIN:
            model.train()
            
        contexts = {
                spec.ForwardPassMode.EVAL: torch.no_grad,
                spec.ForwardPassMode.TRAIN: contextlib.nullcontext 
        }

        with contexts[mode]():
            logits_batch = model(augmented_and_preprocessed_input_batch['inputs'])

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
    

    @property 
    def step_hint(self):
        return 64_000

    

  


    
