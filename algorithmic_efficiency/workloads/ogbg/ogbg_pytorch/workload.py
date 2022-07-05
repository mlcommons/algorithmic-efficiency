"""OGBG workload implemented in PyTorch."""
import contextlib
import os
from typing import Dict, Tuple

import jax
import jax.tree_util as tree
from jraph import GraphsTuple
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg import metrics
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.models import GNN
from algorithmic_efficiency.workloads.ogbg.workload import BaseOgbgWorkload

USE_PYTORCH_DDP = 'LOCAL_RANK' in os.environ
RANK = int(os.environ['LOCAL_RANK']) if USE_PYTORCH_DDP else 0
DEVICE = torch.device(f'cuda:{RANK}' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count()


def _pytorch_map(input_dict: Dict) -> Dict:
  if USE_PYTORCH_DDP:
    return tree.tree_map(
        lambda array: torch.as_tensor(array[RANK], device=DEVICE), input_dict)
  return tree.tree_map(
      lambda a: torch.as_tensor(a, device=DEVICE).view(-1, a.shape[-1])
      if len(a.shape) == 3 else torch.as_tensor(a, device=DEVICE).view(-1),
      input_dict)


class OgbgWorkload(BaseOgbgWorkload):

  def build_input_queue(self,
                        data_rng: jax.random.PRNGKey,
                        split: str,
                        data_dir: str,
                        global_batch_size: int):
    data_rng = data_rng.astype('uint32')
    dataset_iter = super().build_input_queue(data_rng,
                                             split,
                                             data_dir,
                                             global_batch_size)
    for batch in dataset_iter:
      graph = batch.pop('inputs')
      targets = batch.pop('targets')
      weights = batch.pop('weights')
      if USE_PYTORCH_DDP:
        targets = torch.as_tensor(targets[RANK], device=DEVICE)
        weights = torch.as_tensor(
            weights[RANK], device=DEVICE, dtype=torch.bool)
      else:
        targets = torch.as_tensor(
            targets, device=DEVICE).view(-1, targets.shape[-1])
        weights = torch.as_tensor(
            weights, device=DEVICE,
            dtype=torch.bool).view(-1, weights.shape[-1])

      batch['inputs'] = GraphsTuple(
          nodes=_pytorch_map(graph.nodes),
          edges=_pytorch_map(graph.edges),
          receivers=_pytorch_map(graph.receivers),
          senders=_pytorch_map(graph.senders),
          globals=_pytorch_map(graph.globals),
          n_node=_pytorch_map(graph.n_node),
          n_edge=_pytorch_map(graph.n_edge))
      batch['targets'] = targets
      batch['weights'] = weights

      yield batch

  @property
  def model_params_types(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = GNN(self._num_outputs)
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
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
    """Get predicted logits from the network for input graphs."""
    del rng
    del update_batch_norm  # No BN in the GNN model.
    if model_state is not None:
      raise ValueError(
          f'Expected model_state to be None, received {model_state}.')
    model = params

    if mode == spec.ForwardPassMode.TRAIN:
      model.train()
    elif mode == spec.ForwardPassMode.EVAL:
      model.eval()

    contexts = {
        spec.ForwardPassMode.EVAL: torch.no_grad,
        spec.ForwardPassMode.TRAIN: contextlib.nullcontext
    }

    with contexts[mode]():
      logits = model(augmented_and_preprocessed_input_batch['inputs'])

    return logits, None

  def _binary_cross_entropy_with_mask(self,
                                      labels: torch.Tensor,
                                      logits: torch.Tensor,
                                      mask: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss for logits, with masked elements."""
    if not (logits.shape == labels.shape == mask.shape):  # pylint: disable=superfluous-parens
      raise ValueError(
          f'Shape mismatch between logits ({logits.shape}), targets '
          f'({labels.shape}), and weights ({mask.shape}).')
    if len(logits.shape) != 2:
      raise ValueError(f'Rank of logits ({logits.shape}) must be 2.')

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = torch.where(mask, labels, -1)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = torch.where(positive_logits, logits, 0)
    abs_logits = torch.where(positive_logits, logits, -logits)
    return relu_logits - (logits * labels) + (
        torch.log(1 + torch.exp(-abs_logits)))

  def _eval_metric(self, labels, logits, masks):
    per_example_losses = self.loss_fn(labels, logits, masks)
    loss = torch.where(masks, per_example_losses, 0).sum() / masks.sum()
    return metrics.EvalMetrics.single_from_model_output(
        loss=loss.cpu().numpy(),
        logits=logits.cpu().numpy(),
        labels=labels.cpu().numpy(),
        mask=masks.cpu().numpy())
