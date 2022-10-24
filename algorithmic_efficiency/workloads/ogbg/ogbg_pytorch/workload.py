"""OGBG workload implemented in PyTorch."""
import contextlib
from typing import Any, Callable, Dict, Optional, Tuple

import jax
from jraph import GraphsTuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import pytorch_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.ogbg import metrics
from algorithmic_efficiency.workloads.ogbg.ogbg_pytorch.models import GNN
from algorithmic_efficiency.workloads.ogbg.workload import BaseOgbgWorkload

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_utils.pytorch_setup()


def _pytorch_map(inputs: Any) -> Any:
  if USE_PYTORCH_DDP:
    return jax.tree_map(lambda a: torch.as_tensor(a, device=DEVICE), inputs)
  return jax.tree_map(
      lambda a: torch.as_tensor(a, device=DEVICE).view(-1, a.shape[-1])
      if len(a.shape) == 3 else torch.as_tensor(a, device=DEVICE).view(-1),
      inputs)


def _shard(inputs: Any) -> Any:
  if not USE_PYTORCH_DDP:
    return inputs
  return jax.tree_map(lambda tensor: tensor[RANK], inputs)


def _graph_map(function: Callable, graph: GraphsTuple) -> GraphsTuple:
  return GraphsTuple(
      nodes=function(graph.nodes),
      edges=function(graph.edges),
      receivers=function(graph.receivers),
      senders=function(graph.senders),
      globals=function(graph.globals),
      n_node=function(graph.n_node),
      n_edge=function(graph.n_edge))


class OgbgWorkload(BaseOgbgWorkload):

  def _build_input_queue(self,
                         data_rng: jax.random.PRNGKey,
                         split: str,
                         data_dir: str,
                         global_batch_size: int):
    # TODO: Check where the + 1 comes from.
    per_device_batch_size = int(global_batch_size / N_GPUS) + 1

    # Only create and iterate over tf input pipeline in one Python process to
    # avoid creating too many threads.
    if RANK == 0:
      data_rng = data_rng.astype('uint32')
      dataset_iter = super()._build_input_queue(data_rng,
                                                split,
                                                data_dir,
                                                global_batch_size)

    while True:
      if RANK == 0:
        batch = next(dataset_iter)  # pylint: disable=stop-iteration-return
        graph = _graph_map(_pytorch_map, batch['inputs'])
        targets = torch.as_tensor(batch['targets'], device=DEVICE)
        weights = torch.as_tensor(
            batch['weights'], dtype=torch.bool, device=DEVICE)
        # Send batch to other devices when using DDP.
        if USE_PYTORCH_DDP:
          dist.broadcast_object_list([graph], src=0, device=DEVICE)
          # During eval, the batch size of the remainder might be different.
          if split != 'train':
            per_device_batch_size = torch.tensor(
                len(targets[0]), dtype=torch.int32, device=DEVICE)
            dist.broadcast(per_device_batch_size, src=0)
          dist.broadcast(targets, src=0)
          targets = targets[0]
          dist.broadcast(weights, src=0)
          weights = weights[0]
        else:
          targets = targets.view(-1, targets.shape[-1])
          weights = weights.view(-1, weights.shape[-1])
      else:
        graph = [None]
        dist.broadcast_object_list(graph, src=0, device=DEVICE)
        graph = graph[0]
        # During eval, the batch size of the remainder might be different.
        if split != 'train':
          per_device_batch_size = torch.empty((1,),
                                              dtype=torch.int32,
                                              device=DEVICE)
          dist.broadcast(per_device_batch_size, src=0)
        targets = torch.empty(
            (N_GPUS, per_device_batch_size, self._num_outputs), device=DEVICE)
        dist.broadcast(targets, src=0)
        targets = targets[RANK]
        weights = torch.empty(
            (N_GPUS, per_device_batch_size, self._num_outputs),
            dtype=torch.bool,
            device=DEVICE)
        dist.broadcast(weights, src=0)
        weights = weights[RANK]

      batch = {
          'inputs': _graph_map(_shard, graph),
          'targets': targets,
          'weights': weights,
      }

      yield batch

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """aux_dropout_rate is unused."""
    del aux_dropout_rate
    torch.random.manual_seed(rng[0])
    model = GNN(num_outputs=self._num_outputs, dropout_rate=dropout_rate)
    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['decoder.weight', 'decoder.bias']

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

  def _binary_cross_entropy_with_mask(
      self,
      labels: torch.Tensor,
      logits: torch.Tensor,
      mask: torch.Tensor,
      label_smoothing: float = 0.0) -> torch.Tensor:
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

    # Apply label_smoothing.
    num_classes = labels.shape[-1]
    smoothed_labels = ((1.0 - label_smoothing) * labels +
                       label_smoothing / num_classes)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = torch.where(positive_logits, logits, 0)
    abs_logits = torch.where(positive_logits, logits, -logits)
    return relu_logits - (logits * smoothed_labels) + (
        torch.log(1 + torch.exp(-abs_logits)))

  def _eval_metric(self, labels, logits, masks):
    per_example_losses = self.loss_fn(labels, logits, masks)
    loss = torch.where(masks, per_example_losses, 0).sum() / masks.sum()
    return metrics.EvalMetrics.single_from_model_output(
        loss=loss.cpu().numpy(),
        logits=logits.cpu().numpy(),
        labels=labels.cpu().numpy(),
        mask=masks.cpu().numpy())
