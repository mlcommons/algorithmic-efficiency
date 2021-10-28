"""OGB workload implemented in Jax."""

from typing import Tuple
import sklearn.metrics

import jax
import jax.numpy as jnp
from flax import linen as nn

import spec
import input_pipeline
import models
from workloads.ogb.workload import OGB


class OGBWorkload(OGB):

  def __init__(self):
    self._eval_ds = None
    self._param_shapes = None
    self._init_graphs = None
    self._mask = None
    self._model = models.GraphNet(
        latent_size=256,
        num_mlp_layers=2,
        message_passing_steps=5,
        output_globals_size=128,
        dropout_rate=0.1,
        skip_connections=True,
        layer_norm=True,
        use_edge_model=True,
        deterministic=True)

  def _normalize(self, image):
    pass

  def _build_dataset(self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size):
    datasets = input_pipeline.get_datasets(
      batch_size,
      add_virtual_node=False,
      add_undirected_edges=True,
      add_self_loops=True)
    if self._init_graphs is None:
      self._init_graphs = next(datasets['train'].as_numpy_iterator())
    return datasets[split]

  def build_input_queue(
      self,
      data_rng: jax.random.PRNGKey,
      split: str,
      data_dir: str,
      batch_size: int):
    return iter(self._build_dataset(data_rng, split, data_dir, batch_size))

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  def model_params_types(self):
    pass

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(
      self,
      selected_raw_input_batch: spec.Tensor,
      selected_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor,
      rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(
      self,
      raw_input_batch: spec.Tensor,
      raw_label_batch: spec.Tensor,
      train_mean: spec.Tensor,
      train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch, raw_label_batch

  def _replace_globals(self, graphs: jraph.Graphssklearn) -> jraph.Graphssklearn:
    """Replaces the globals attribute with a constant feature for each graph."""
    return graphs._replace(globals=jnp.ones([graphs.n_node.shape[0], 1]))

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    if self._init_graphs is None:
      raise ValueError(
          'This should not happen, workload.build_input_queue() should be '
          'called before workload.init_model_fn()!'
      )
    rng, init_rng = jax.random.split(rng)
    init_graphs = self._replace_globals(self._init_graphs)
    params = jax.jit(self._model.init)(init_rng, init_graphs)
    self._param_shapes = jax.tree_map(
      lambda x: spec.ShapeTuple(x.shape),
      params)
    return params, None

  # Keep this separate from the loss function in order to support optimizers
  # that use the logits.
  def output_activation_fn(
      self,
      logits_batch: spec.Tensor,
      loss_type: spec.LossType) -> spec.Tensor:
    pass

  def _get_valid_mask(labels: jnp.ndarray,
                      graphs: jraph.GraphsTuple) -> jnp.ndarray:
    """Gets the binary mask indicating only valid labels and graphs."""
    # We have to ignore all NaN values - which indicate labels for which
    # the current graphs have no label.
    labels_mask = ~jnp.isnan(labels)

    # Since we have extra 'dummy' graphs in our batch due to padding, we want
    # to mask out any loss associated with the dummy graphs.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    graph_mask = jraph.get_graph_padding_mask(graphs)

    # Combine the mask over labels with the mask over graphs.
    return labels_mask & graph_mask[:, None]

  def model_fn(
      self,
      params: spec.ParameterContainer,
      input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    """Get predicted logits from the network for input graphs."""
    # Extract labels.
    labels = input_batch.globals
    # Replace the global feature for graph classification.
    graphs = self._replace_globals(input_batch)

    # Get predicted logits
    variables = {'params': params, **model_state}
    train = mode == spec.ForwardPassMode.TRAIN
    pred_graphs = self._model.apply(
        variables,
        graphs,
        train=train,
        rngs=rng)
    logits = pred_graphs.globals

    # Get the mask for valid labels and graphs.
    self._mask = self._get_valid_mask(labels, graphs)

    return logits, None

  def _binary_cross_entropy_with_mask(
      self,
      logits: jnp.ndarray,
      labels: jnp.ndarray,
      mask: jnp.ndarray) -> jnp.ndarray:
    """Binary cross entropy loss for logits, with masked elements."""
    assert logits.shape == labels.shape == mask.shape
    assert len(logits.shape) == 2

    # To prevent propagation of NaNs during grad().
    # We mask over the loss for invalid targets later.
    labels = jnp.where(mask, labels, -1)

    # Numerically stable implementation of BCE loss.
    # This mimics TensorFlow's tf.nn.sigmoid_cross_entropy_with_logits().
    positive_logits = (logits >= 0)
    relu_logits = jnp.where(positive_logits, logits, 0)
    abs_logits = jnp.where(positive_logits, logits, -logits)
    return relu_logits - (logits * labels) + (
        jnp.log(1 + jnp.exp(-abs_logits)))

  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(
      self,
      label_batch: spec.Tensor,
      logits_batch: spec.Tensor) -> spec.Tensor:  # differentiable
    if self._mask is None:
      raise ValueError(
          'This should not happen, workload.model_fn() should be '
          'called before workload.loss_fn()!'
      )
    loss = self._binary_cross_entropy_with_mask(
        logits=logits_batch, labels=label_batch, mask=self._mask)
    mean_loss = jnp.sum(jnp.where(mask, loss, 0)) / jnp.sum(mask)
    return mean_loss

  def _compute_mean_average_precision(self, labels, logits):
    """Computes the mean average precision (mAP) over different tasks."""
    # Matches the official OGB evaluation scheme for mean average precision.
    assert logits.shape == labels.shape == self._mask.shape
    assert len(logits.shape) == 2

    probs = jax.nn.sigmoid(logits)
    num_tasks = labels.shape[1]
    average_precisions = np.full(num_tasks, np.nan)

    for task in range(num_tasks):
      # AP is only defined when there is at least one negative data
      # and at least one positive data.
      if np.sum(labels[:, task] == 0) > 0 and np.sum(labels[:, task] == 1) > 0:
        is_labeled = self._mask[:, task]
        average_precisions[task] = sklearn.metrics.average_precision_score(
            labels[is_labeled, task], probs[is_labeled, task])

    # When all APs are NaNs, return NaN. This avoids raising a RuntimeWarning.
    if np.isnan(average_precisions).all():
      return np.nan
    return np.nanmean(average_precisions)

  def _eval_metric(self, labels, logits):
    """Return the accuracy, average precision, and loss as a dict."""
    preds = (logits > 0)
    accuracy = np.nanmean((preds == labels).astype(jnp.float32))
    average_precision = self._compute_mean_average_precision(labels, logits)
    loss = self.loss_fn(labels, logits)
    metrics = {
        'accuracy': accuracy,
        'average_precision': average_precision,
        'loss': loss,
    }
    return metrics
