"""WMT workload implemented in Jax."""
import functools
import types
from typing import Tuple

from . import decode
from . import input_pipeline
from . import models
from . import train
from flax import linen as nn
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import spec
import tensorflow as tf

CONFIG = types.SimpleNamespace(
    vocab_path="./wmt_256/sentencepiece_model",
    vocab_size=32000,
    max_corpus_chars=10**7,
    dataset_name="wmt17_translate/de-en",
    eval_split="test",
    reverse_translation=True,
    beam_size=4,
    num_eval_steps=20,
    num_predict_steps=-1,
    learning_rate=0.0625,
    warmup_steps=1000,
    label_smoothing=0.1,
    weight_decay=0.0,
    max_target_length=256,
    max_eval_target_length=256,
    max_predict_length=256,
    share_embeddings=True,
    logits_via_embedding=True,
    num_layers=6,
    qkv_dim=1024,
    emb_dim=1024,
    mlp_dim=4096,
    num_heads=16,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    use_bfloat16=True,
    workdir="./wmt_256",
    per_device_batch_size=16,
    eval_dataset_name="wmt14_translate/de-en")


class WMTWorkload(spec.Workload):
  """A WMT workload."""

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
    self.config = CONFIG

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result["bleu"] > 25

  def build_input_queue(self, data_rng: jax.random.PRNGKey, split: str,
                        data_dir: str, batch_size: int):
    tf.io.gfile.makedirs(self.config.workdir)
    self._train_ds, self._eval_ds, self._predict_ds, self._encoder = input_pipeline.get_wmt_datasets(
        batch_size=jax.local_device_count() * batch_size,
        config=self.config,
        reverse_translation=self.config.reverse_translation,
        vocab_path=self.config.vocab_path)
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

  # Return whether or not a key in spec.ParameterContainer is the output layer
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
        share_embeddings=self.config.share_embeddings,
        logits_via_embedding=self.config.logits_via_embedding,
        dtype=jnp.bfloat16 if self.config.use_bfloat16 else jnp.float32,
        emb_dim=self.config.emb_dim,
        num_heads=self.config.num_heads,
        num_layers=self.config.num_layers,
        qkv_dim=self.config.qkv_dim,
        mlp_dim=self.config.mlp_dim,
        max_len=max(self.config.max_target_length,
                    self.config.max_eval_target_length),
        dropout_rate=self.config.dropout_rate,
        attention_dropout_rate=self.config.attention_dropout_rate,
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
            max_decode_len=self.config.max_predict_length,
            config=self.predict_config),
        axis_name="batch")
    self.p_pred_step = jax.pmap(
        functools.partial(
            train.predict_step,
            config=self.predict_config,
            beam_size=self.config.beam_size),
        axis_name="batch",
        static_broadcasted_argnums=(3, 4))  # eos token, max_length are constant

    rng, init_rng = jax.random.split(rng)
    input_shape = (self.config.per_device_batch_size,
                   self.config.max_target_length)
    target_shape = (self.config.per_device_batch_size,
                    self.config.max_target_length)

    initial_variables = jax.jit(models.Transformer(self.eval_config).init)(
        init_rng, jnp.ones(input_shape, jnp.float32),
        jnp.ones(target_shape, jnp.float32))

    initial_params = initial_variables["params"]

    return initial_params, None

  def model_fn(
      self, params: spec.ParameterContainer,
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
    vocab_size = logits_batch.shape[-1]
    confidence = 1.0 - self.config.label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) +
        (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
    soft_targets = common_utils.onehot(
        label_batch, vocab_size, on_value=confidence, off_value=low_confidence)

    loss = -jnp.sum(soft_targets * nn.log_softmax(logits_batch), axis=-1)
    loss = loss - normalizing_constant

    loss = loss * weights

    return loss

  def eval_model(self, params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState, rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""
    del data_dir

    eval_results = train.evaluate(
        p_eval_step=self.p_eval_step,
        target=params,
        eval_ds=self._eval_ds,
        num_eval_steps=self.config.num_eval_steps)

    _, bleu_score = train.translate_and_calculate_bleu(
        p_pred_step=self.p_pred_step,
        p_init_cache=self.p_init_cache,
        target=params,
        predict_ds=self._predict_ds,
        decode_tokens=self._decode_tokens,
        max_predict_length=self.config.max_predict_length)

    eval_results["bleu"] = bleu_score

    return eval_results

