import functools
import itertools
import os
import typing
from typing import Tuple

import Levenshtein
import ctcdecode
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.data
from flax import jax_utils

import spec
from . import ctc_loss, input_pipeline, models


@functools.partial(jax.jit, backend="cpu")
def to_cpu(inp):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(inp, cpu_device)


class LibriSpeechWorkload(spec.Workload):
    """A LibriSpeech workload."""

    def __init__(self):
        self._train_loader = None
        self._valid_loader = None
        self._valid_loader_len: typing.Optional[int] = None
        self._model = models.CNNLSTM()
        self._label_dict = {
            "_": 0,
            " ": 1,
            "'": 2,
            "A": 3,
            "B": 4,
            "C": 5,
            "D": 6,
            "E": 7,
            "F": 8,
            "G": 9,
            "H": 10,
            "I": 11,
            "J": 12,
            "K": 13,
            "L": 14,
            "M": 15,
            "N": 16,
            "O": 17,
            "P": 18,
            "Q": 19,
            "R": 20,
            "S": 21,
            "T": 22,
            "U": 23,
            "V": 24,
            "W": 25,
            "X": 26,
            "Y": 27,
            "Z": 28,
        }
        self._rev_label_dict = {v: k for k, v in self._label_dict.items()}
        self._decoder = ctcdecode.CTCBeamDecoder(
            labels=[str(c) for c in self._rev_label_dict], beam_width=1)
        self._loss = ctc_loss.ctc_loss

    def has_reached_goal(self, eval_result: float) -> bool:
        return eval_result < self.target_value

    def build_input_queue(self, data_rng, split: str, data_dir: str,
                          batch_size: int):
        torch.manual_seed(data_rng[0])
        train_set = input_pipeline.LibriSpeechDataset(
            os.path.join(data_dir, "features_train-clean-100.csv"))
        valid_set = input_pipeline.LibriSpeechDataset(
            os.path.join(data_dir, "features_test-clean.csv"))

        train_collate_fn = train_set.pad_collate

        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=False,
            collate_fn=train_collate_fn)

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=batch_size,
            num_workers=2,
            drop_last=True,
            pin_memory=False,
            collate_fn=train_collate_fn)
        self._valid_loader = iter(itertools.cycle(valid_loader))  # itertools.cycle is required for it to run on tpu
        self._valid_loader_len = len(valid_loader)

        return iter(itertools.cycle(self._train_loader))

    @property
    def param_shapes(self):
        if self._param_shapes is None:
            raise ValueError(
                'This should not happen, workload.init_model_fn() should be called '
                'before workload.param_shapes!')
        return self._param_shapes

    @property
    def target_value(self):
        return 0.1

    @property
    def loss_type(self):
        return spec.LossType.CTC_LOSS

    @property
    def num_train_examples(self):
        return 28539

    @property
    def num_eval_examples(self):
        return 2620

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
        return 800000  # TODO: Reduce this to 80000 once debugging is complete

    @property
    def eval_period_time_sec(self):
        return 800

    # Return whether or not a key in spec.ParameterContainer is the output layer
    # parameters.
    def is_output_params(self, param_key: spec.ParameterKey) -> bool:
        pass

    def preprocess_for_train(self, selected_raw_input_batch: spec.Tensor,
                             selected_label_batch: spec.Tensor,
                             train_mean: spec.Tensor, train_stddev: spec.Tensor,
                             rng: spec.RandomState) -> spec.Tensor:
        del train_mean
        del train_stddev
        del rng
        return selected_raw_input_batch, selected_label_batch

    def preprocess_for_eval(
            self,
            raw_input_batch: spec.Tensor,
            train_mean: spec.Tensor,
            train_stddev: spec.Tensor) -> spec.Tensor:
        del train_mean
        del train_stddev
        return raw_input_batch

    def initialized(self, key, model):
        init_val = [jnp.ones((1, 1, 161, 2453), jnp.float32), jnp.array([2087])]
        variables = model.init(jax.random.PRNGKey(key[0] ^ key[1]), *init_val, training=True)
        params = variables["params"]
        model_state = variables["batch_stats"]
        return params, model_state

    _InitState = Tuple[spec.ParameterContainer, spec.ModelAuxiliaryState]

    def init_model_fn(self, rng: spec.RandomState) -> _InitState:
        params, model_state = self.initialized(rng, self._model)
        self._param_shapes = jax.tree_map(
            lambda x: spec.ShapeTuple(x.shape),
            params)
        model_state = jax_utils.replicate(model_state)
        params = jax_utils.replicate(params)
        return params, model_state

    # Keep this separate from the loss function in order to support optimizers
    # that use the logits.
    def output_activation_fn(
            self,
            logits_batch: spec.Tensor,
            loss_type: spec.LossType) -> spec.Tensor:
        """Return the final activations of the model."""
        pass

    def model_fn(
            self,
            params: spec.ParameterContainer,
            augmented_and_preprocessed_input_batch: spec.Tensor,
            model_state: spec.ModelAuxiliaryState,
            mode: spec.ForwardPassMode,
            rng: spec.RandomState,
            update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
        variables = {'params': params, **model_state}
        train = mode == spec.ForwardPassMode.TRAIN
        features, input_lengths = augmented_and_preprocessed_input_batch
        if update_batch_norm:
            (log_y, output_lengths), new_model_state = self._model.apply(
                variables, features, input_lengths, training=train, mutable=['batch_stats'])
            return (log_y, output_lengths), new_model_state
        else:
            log_y, output_lengths = self._model.apply(
                variables, features, input_lengths, training=train)
            return (log_y, output_lengths), None

    def loss_fn(
            self,
            label_batch: spec.Tensor,
            logits_batch: spec.Tensor) -> spec.Tensor:
        log_y, _ = logits_batch
        label_batch, labelspaddings, logprobspaddings = label_batch
        loss, _ = self._loss(log_y, logprobspaddings, label_batch, labelspaddings)

        return jnp.mean(loss)

    def _eval_model_fn(self, in_len, out_len, params, batch, state, rng):
        _ = in_len
        _ = out_len
        logits, _ = self.model_fn(
            params,
            batch["input"],
            state,
            spec.ForwardPassMode.EVAL,
            rng,
            update_batch_norm=False)
        return logits

    eval_model_fn = jax.pmap(_eval_model_fn, axis_name='batch', in_axes=(None, None, None, 0, 0, 0, None),
                             static_broadcasted_argnums=(0, 1, 2))

    def eval_model(
            self,
            params: spec.ParameterContainer,
            model_state: spec.ModelAuxiliaryState,
            rng: spec.RandomState,
            data_dir: str):
        """Run a full evaluation of the model."""
        # sync batch statistics across replicas
        total_error = 0.0
        total_length = 0.0

        num_devices = jax.local_device_count()
        for step_idx, (_, features, transcripts, input_lengths, transcripts_padding, in_len,
                       out_len) in enumerate(self._valid_loader):
            if step_idx >= self._valid_loader_len:
                break
            features = jnp.expand_dims(features.transpose(0, 2, 1), axis=1)
            reshaped_features = jnp.reshape(
                features,
                (num_devices, features.shape[0] // num_devices, *features.shape[1:]))
            reshaped_input_lengths = jnp.reshape(
                input_lengths,
                (num_devices, input_lengths.shape[0] // num_devices, *input_lengths.shape[1:]))
            batch = {
                'input': (reshaped_features, reshaped_input_lengths)
            }
            log_y, _ = self.eval_model_fn(in_len, out_len, params, batch, model_state, rng)
            log_y = to_cpu(log_y)
            log_y = jnp.reshape(log_y, (-1, *log_y.shape[2:]))
            log_y = torch.tensor(np.asarray(log_y), device='cpu')
            input_lengths = torch.tensor(np.asarray(input_lengths), device='cpu')
            out, _, _, seq_lens = self._decoder.decode(
                torch.exp(log_y), input_lengths)
            for hyp, trn, length in zip(out, transcripts,
                                        seq_lens):  # iterate batch
                best_hyp = hyp[0, :length[0]]
                hh = "".join([self._rev_label_dict[i.item()] for i in best_hyp])
                t = np.asarray(trn).tolist()
                t = [ll for ll in t if ll != 0]
                tlength = len(t)
                tt = "".join([self._rev_label_dict[i] for i in t])
                error = Levenshtein.distance(tt, hh)
                total_error += error
                total_length += tlength

        return total_error / total_length
