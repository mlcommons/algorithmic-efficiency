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
        self.metrics_bundle = metrics.get_metrics_bundle()
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

    def loss_fn(self, logits, logit_paddings, targets, target_paddings):  # differentiable
        logprobs = nn.log_softmax(logits)
        per_seq_loss = self.ctc_loss(logprobs, logit_paddings, targets,
                                    target_paddings)
        normalizer = jnp.sum(1 - target_paddings)

        normalized_loss = jnp.sum(per_seq_loss) / jnp.maximum(normalizer, 1)
        return normalized_loss

    def ctc_loss(self, logits, logit_paddings, labels, label_paddings, blank_id=0):
        return optax.ctc_loss(logits, logit_paddings, labels, label_paddings,
            blank_id)


    @functools.partial(
      jax.pmap,
      axis_name='batch',
      in_axes=(None, 0, 0, 0, None),
      static_broadcasted_argnums=(0,))
    def eval_step_pmapped(
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
        normalized_loss = self.loss_fn(logits, logit_paddings, batch['targets'], batch['target_paddings'])
        return self.metrics_bundle.gather_from_model_output(normalized_loss=normalized_loss)        
    
    def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str) -> Dict[str, float]:
        """Run a full evaluation of the model."""
        if model_state is not None:
            # Sync batch statistics across replicas before evaluating.
            model_state = self.sync_batch_stats(model_state)

        num_batches = int(math.ceil(num_examples / global_batch_size))
        if split not in self._eval_iters:
            # These iterators will repeat indefinitely.
            self._eval_iters[split] = self.build_input_queue(
                rng,
                split,
                data_dir,
                global_batch_size,
                num_batches,
                repeat_final_dataset=True)

        metrics_report = None
        for _ in range(num_batches):
            eval_batch = next(self._eval_iters[split])
            computed_metrics = self.eval_step_pmapped(params, eval_batch, model_state, rng).unreplicate()
            
            if metrics_report is None:
                metrics_report = computed_metrics
            else:
                # `merge` aggregates the metrics across batches.
                metrics_report = metrics_report.merge(computed_metrics)

        return metrics_report.compute()

    def sync_batch_stats(self, model_state):
        # An axis_name is passed to pmap which can then be used by pmean.
        # In this case each device has its own version of the batch statistics and
        # we average them.
        avg_fn = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')
        new_model_state = model_state.copy(
            {'batch_stats': avg_fn(model_state['batch_stats'])})
        return new_model_state