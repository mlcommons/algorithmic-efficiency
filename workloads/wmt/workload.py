"""WMT workload implemented in Jax."""

from typing import Tuple

from . import config
from . import input_pipeline
from . import models
from . import train
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import spec
import tensorflow as tf


class WMTWorkload(spec.Workload):
  def __init__(self):
    self._eval_ds = None
    self._train_ds = None
    self._sp_tokenizer = None
    self._train_config = models.TransformerConfig(
        vocab_size=config.config.vocab_size,
        output_vocab_size=config.config.vocab_size,
        share_embeddings=config.onfig.share_embeddings,
        logits_via_embedding=config.config.logits_via_embedding,
        dtype=jnp.bfloat16 if config.config.use_bfloat16 else jnp.float32,
        emb_dim=config.config.emb_dim,
        num_heads=config.config.num_heads,
        num_layers=config.config.num_layers,
        qkv_dim=config.config.qkv_dim,
        mlp_dim=config.config.mlp_dim,
        max_len=max(config.config.max_target_length,
                    config.config.max_eval_target_length),
        dropout_rate=config.config.dropout_rate,
        attention_dropout_rate=config.config.attention_dropout_rate,
        deterministic=False,
        decode=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
    self._eval_config = self._train_config.replace(deterministic=True)

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result["accuracy"] > 0.5

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        batch_size: int):
    tf.io.gfile.makedirs(config.config.workdir)
    self._train_ds, self._eval_ds, _, self._sp_tokenizer = input_pipeline.get_wmt_datasets(
        batch_size=jax.local_device_count() * batch_size,
        config=config.config,
        reverse_translation=config.config.reverse_translation,
        vocab_path=config.config.vocab_path)
    return iter(self._train_ds)

  @property
  def param_shapes(self):
    init_params, _ = self.init_model_fn(jax.random.PRNGKey(0))
    return jax.tree_map(lambda x: spec.ShapeTuple(x.shape), init_params)

  @property
  def loss_type(self):
    return spec.LossType.SOFTMAX_CROSS_ENTROPY

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  def model_params_types(self):
    pass

  @property
  def max_allowed_runtime_sec(self):
    return 60

  @property
  def eval_period_time_sec(self):
    return 10

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del rng
    return (selected_raw_input_batch, selected_label_batch)

  def preprocess_for_eval(self, raw_input_batch: spec.Tensor,
                          raw_label_batch: spec.Tensor, train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return (raw_input_batch, raw_label_batch)

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    rng, init_rng = jax.random.split(rng)
    input_shape = (config.per_device_batch_size, config.max_target_length)
    target_shape = (config.per_device_batch_size, config.max_target_length)

    initial_variables = jax.jit(models.Transformer(self._eval_config).init)(
        init_rng, jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables["params"]

    return initial_params, None

  def model_fn(
      self, params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxillaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxillaryState]:
    del model_state
    del rng
    del update_batch_norm

    model_config = self._train_config if mode == spec.ForwardPassMode.TRAIN else self._eval_config
    inputs, targets = augmented_and_preprocessed_input_batch[
        "inputs"], augmented_and_preprocessed_input_batch["targets"]
    logits_batch = models.Transformer(model_config).apply({"params": params},
                                                          inputs, targets)

    return logits_batch, None

  # LossFn = Callable[Tuple[spec.Tensor, spec.Tensor], spec.Tensor]
  # Does NOT apply regularization, which is left to the submitter to do in
  # `update_params`.
  def loss_fn(self, label_batch: spec.Tensor, logits_batch: spec.Tensor,
              loss_type: spec.LossType) -> spec.Tensor:  # differentiable
    del loss_type
    weights = jnp.where(label_batch > 0, 1, 0).astype(jnp.float32)
    loss, weight_sum = train.compute_weighted_cross_entropy(
        logits_batch, label_batch, weights, config.config.label_smoothing)
    mean_loss = loss / weight_sum
    return mean_loss

  def eval_model(self, params: spec.ParameterTree,
                 model_state: spec.ModelAuxillaryState, rng: spec.RandomState):
    """Run a full evaluation of the model."""
    _, model_rng = jax.random.split(rng, 2)

    eval_iter = iter(self._eval_ds)
    eval_metrics = []
    for batch in eval_iter:
      batch = jax.tree_map(lambda x: x._numpy(), batch)  # pylint: disable=protected-access
      batch = common_utils.shard(batch)
      inputs, targets = batch["inputs"], batch["targets"]
      logits, _ = self.model_fn(
          params,
          inputs,
          model_state,
          spec.ForwardPassMode.EVAL,
          model_rng,
          update_batch_norm=False)

      weights = jnp.where(targets > 0, 1.0, 0.0)

      metrics = train.compute_metrics(logits, targets, weights)
      eval_metrics.append(metrics)

    eval_metrics = common_utils.get_metrics(eval_metrics)
    eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
    eval_denominator = eval_metrics_sums.pop("denominator")
    eval_summary = jax.tree_map(
        lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
        eval_metrics_sums)
    return eval_summary

