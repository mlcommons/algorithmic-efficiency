import itertools
import math
from typing import Dict, Optional, Tuple

from absl import flags
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import jax_utils
import numpy as np
import functools 
import jax.lax as lax

import optax

from algorithmic_efficiency import random_utils as prng
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech_conformer import metrics
from algorithmic_efficiency.workloads.librispeech_conformer import workload
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_jax import models
from algorithmic_efficiency import param_utils

FLAGS = flags.FLAGS

class LibriSpeechConformerWorkload(workload.BaseLibrispeechWorkload):
    def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
        model_cls = getattr(models, 'Conformer')
        model = model_cls(models.ConformerConfig())
        self._model = model
        input_shape = [(320000,), (320000,)]
        fake_input_batch = [np.zeros((2, *x), jnp.float32) for x in input_shape]

        model_init_fn = jax.jit(functools.partial(model.init, train=False))

        params_rng, dropout_rng = jax.random.split(rng, 2)
        variables = model_init_fn({'params' : params_rng, 'dropout' : dropout_rng}, *fake_input_batch)

        model_state, params = variables.pop('params')

        self._param_shapes = jax.tree_map(lambda x: spec.ShapeTuple(x.shape),
                                        params)
        model_state = jax_utils.replicate(model_state)
        params = jax_utils.replicate(params)
        return params, model_state


    def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: Dict[str, spec.Tensor],
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        variables = {'params': params, **model_state}
        
        train = (mode == spec.ForwardPassMode.TRAIN)

        if train:
            (logits, logit_paddings), new_model_state = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                train,
                rngs=rng, 
                mutable=['batch_stats'])
            return (logits, logit_paddings), new_model_state
        else:
            logits, logit_paddings = self._model.apply(
                variables,
                augmented_and_preprocessed_input_batch['inputs'],
                augmented_and_preprocessed_input_batch['input_paddings'],
                train,
                mutable=False)
            return logits, logit_paddings

    @property
    def model_params_types(self):
        if self._param_shapes is None:
            raise ValueError(
                'This should not happen, workload.init_model_fn() should be called '
                'before workload.param_shapes!')
        if self._param_types is None:
            self._param_types = param_utils.jax_param_types(
                self._param_shapes.unfreeze())
        return self._param_types

    def ctc_loss(self, logits, logit_paddings, labels, label_paddings, blank_id=0):
        return optax.ctc_loss(logits, logit_paddings, labels, label_paddings,
            blank_id)

    def _compute_metrics(self, logits, logit_paddings, labels, label_paddings):
        loss = jnp.mean(self.loss_fn(logits, logit_paddings, labels, label_paddings))
        metrics = {
            'ctc_loss': loss,
        }
        metrics = lax.psum(metrics, axis_name='batch')
        return metrics

    @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
    def _eval_model(
        self,
        params: spec.ParameterContainer,
        batch: Dict[str, spec.Tensor],
        model_state: spec.ModelAuxiliaryState,
        rng: spec.RandomState) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        logits, logit_paddings = self.model_fn(
            params,
            batch,
            model_state,
            spec.ForwardPassMode.EVAL,
            rng,
            update_batch_norm=False)
        return self._compute_metrics(logits, logit_paddings, batch['targets'], batch['target_paddings'])

    def _eval_model_on_split(self,
        split: str,
        num_examples: int,
        global_batch_size: int,
        params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        rng: spec.RandomState,
        data_dir: str):
        data_rng, model_rng = jax.random.split(rng, 2)
        # Sync batch statistics across replicas before evaluating.
        model_state = self.sync_batch_stats(model_state)
        num_batches = int(math.ceil(num_examples / global_batch_size))
        # We already repeat the dataset indefinitely in tf.data.
        if split not in self._eval_iters:
            self._eval_iters[split] = self.build_input_queue(
                data_rng,
                split=split,
                global_batch_size=global_batch_size,
                data_dir=data_dir,
                cache=True,
                repeat_final_dataset=True,
                num_batches=num_batches)

        eval_metrics = {}
        
        for _ in range(num_batches):
            batch = next(self._eval_iters[split])
            # We already average these metrics across devices inside _compute_metrics.
            synced_metrics = self._eval_model(params, batch, model_state, model_rng)
            for metric_name, metric_value in synced_metrics.items():
                if metric_name not in eval_metrics:
                    eval_metrics[metric_name] = 0.0
                    eval_metrics[metric_name] += metric_value

        eval_metrics = jax.tree_map(lambda x: float(x[0] / num_examples),
                                    eval_metrics)
        return eval_metrics

    def sync_batch_stats(self, model_state):
        # An axis_name is passed to pmap which can then be used by pmean.
        # In this case each device has its own version of the batch statistics and
        # we average them.
        avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
        new_model_state = model_state.copy(
            {'batch_stats': avg_fn(model_state['batch_stats'])})
        return new_model_state