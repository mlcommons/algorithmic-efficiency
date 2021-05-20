"""WMT workload implemented in Jax."""
import functools
from typing import Tuple

from . import config
from . import decode
from . import input_pipeline
from . import models
from . import train
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import spec
import tensorflow as tf


class WMTWorkload(spec.Workload):

  def __init__(self):
    self._eval_ds = None
    self._train_ds = None
    self._predict_ds = None
    self._encoder = None
    self._vocab_size = None
    self.train_config = None
    self.eval_config = None
    self.predict_config = None
    self.p_eval_step = None
    self.p_init_cache = None
    self.p_pred_step = None

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result["bleu"] > 20

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        data_dir: str, batch_size: int):
    tf.io.gfile.makedirs(config.config.workdir)
    self._train_ds, self._eval_ds, self._predict_ds, self._encoder = input_pipeline.get_wmt_datasets(
        batch_size=jax.local_device_count() * batch_size,
        config=config.config,
        reverse_translation=config.config.reverse_translation,
        vocab_path=config.config.vocab_path)
    self._vocab_size = int(self._encoder.vocab_size())
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
    return 80000

  @property
  def eval_period_time_sec(self):
    return 800

  def _decode_tokens(self, toks):
    valid_toks = toks[:np.argmax(toks == decode.EOS_ID) + 1].astype(np.int32)
    return self._encoder.detokenize(valid_toks).numpy().decode("utf-8")

  # Return whether or not a key in spec.ParameterTree is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                           train_mean: spec.Tensor, train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
    del rng
    return selected_raw_input_batch

  def preprocess_for_eval(self, raw_input_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    self.train_config = models.TransformerConfig(
        vocab_size=self._vocab_size,
        output_vocab_size=self._vocab_size,
        share_embeddings=config.config.share_embeddings,
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
    self.eval_config = self.train_config.replace(deterministic=True)
    self.predict_config = self.train_config.replace(
        deterministic=True, decode=True)
    self.p_eval_step = jax.pmap(
        functools.partial(train.eval_step, config=self.eval_config),
        axis_name="batch")
    self.p_init_cache = jax.pmap(
        functools.partial(
            train.initialize_cache,
            max_decode_len=config.config.max_predict_length,
            config=self.predict_config),
        axis_name="batch")
    self.p_pred_step = jax.pmap(
        functools.partial(
            train.predict_step,
            config=self.predict_config,
            beam_size=config.config.beam_size),
        axis_name="batch",
        static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

    rng, init_rng = jax.random.split(rng)
    input_shape = (config.config.per_device_batch_size,
                   config.config.max_target_length)
    target_shape = (config.config.per_device_batch_size,
                    config.config.max_target_length)

    initial_variables = jax.jit(models.Transformer(self.eval_config).init)(
        init_rng, jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables["params"]

    return initial_params, None

  def model_fn(
      self, params: spec.ParameterTree,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState, mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    model_config = self.train_config if mode == spec.ForwardPassMode.TRAIN else self.eval_config
    inputs, targets = augmented_and_preprocessed_input_batch[
        "inputs"], augmented_and_preprocessed_input_batch["targets"]
    logits_batch = models.Transformer(model_config).apply({"params": params},
                                                          inputs, targets)

    return logits_batch, None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # Dense (not one-hot) labels.
      logits_batch: spec.Tensor) -> spec.Tensor:

    weights = jnp.where(label_batch > 0, 1.0, 0.0)
    metrics = train.compute_metrics(logits_batch, label_batch, weights)

    return metrics

  def eval_model(self, params: spec.ParameterTree,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState):
    """Run a full evaluation of the model."""

    eval_results = train.evaluate(
        p_eval_step=self.p_eval_step,
        target=params,
        eval_ds=self._eval_ds,
        num_eval_steps=config.config.num_eval_steps)

    _, bleu_score = train.translate_and_calculate_bleu(
        p_pred_step=self.p_pred_step,
        p_init_cache=self.p_init_cache,
        target=params,
        predict_ds=self._predict_ds,
        decode_tokens=self._decode_tokens,
        max_predict_length=config.config.max_predict_length)

    eval_results["bleu"] = bleu_score

    return eval_results

