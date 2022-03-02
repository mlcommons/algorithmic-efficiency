"""LibriSpeech workload implemented in Pytorch."""

import itertools
import os
from typing import Tuple

import ctcdecode
import Levenshtein
import torch

from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech.librispeech_pytorch import \
    input_pipeline
from algorithmic_efficiency.workloads.librispeech.librispeech_pytorch import \
    models


class LibriSpeechWorkload(spec.Workload):
  """A LibriSpeech workload."""

  def __init__(self):
    self._train_loader = None
    self._valid_loader = None
    self._device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    self._loss = torch.nn.CTCLoss(blank=0, reduction="none")
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

  def has_reached_goal(self, eval_result: float) -> bool:
    return eval_result < self.target_value

  def build_input_queue(self,
                        data_rng,
                        split: str,
                        data_dir: str,
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
        num_workers=2,
        pin_memory=True,
        collate_fn=train_collate_fn)

    self._valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_collate_fn)

    return iter(itertools.cycle(self._train_loader))

  @property
  def param_shapes(self):
    pass

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
    return 80000

  @property
  def eval_period_time_sec(self):
    return 800

  # Return whether or not a key in spec.ParameterContainer is the output layer
  # parameters.
  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    pass

  def preprocess_for_train(self,
                           selected_raw_input_batch: spec.Tensor,
                           selected_label_batch: spec.Tensor,
                           train_mean: spec.Tensor,
                           train_stddev: spec.Tensor,
                           rng: spec.RandomState) -> spec.Tensor:
    del train_mean
    del train_stddev
    del rng
    return selected_raw_input_batch, selected_label_batch

  def preprocess_for_eval(self,
                          raw_input_batch: spec.Tensor,
                          train_mean: spec.Tensor,
                          train_stddev: spec.Tensor) -> spec.Tensor:
    del train_mean
    del train_stddev
    return raw_input_batch

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model = models.CNNLSTM()
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(self._device)
    return model, None

  def model_fn(
      self,
      params: spec.ParameterContainer,
      augmented_and_preprocessed_input_batch: spec.Tensor,
      model_state: spec.ModelAuxiliaryState,
      mode: spec.ForwardPassMode,
      rng: spec.RandomState,
      update_batch_norm: bool) -> Tuple[spec.Tensor, spec.ModelAuxiliaryState]:
    del model_state
    del rng
    del update_batch_norm

    params.train(mode == spec.ForwardPassMode.TRAIN)
    (features, transcripts,
     input_lengths) = augmented_and_preprocessed_input_batch
    log_y, output_lengths = params(features, input_lengths, transcripts)

    return (log_y.transpose(0, 1), output_lengths), None

  def loss_fn(
      self,
      label_batch: spec.Tensor,  # transcripts
      logits_batch: spec.Tensor) -> spec.Tensor:

    log_y, output_lengths = logits_batch
    target_lengths = torch.IntTensor([len(y[y != 0]) for y in label_batch])

    loss = self._loss(log_y, label_batch, output_lengths, target_lengths) / (
        target_lengths.float().to(self._device))

    return loss

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  def eval_model(self,
                 params: spec.ParameterContainer,
                 model_state: spec.ModelAuxiliaryState,
                 rng: spec.RandomState,
                 data_dir: str):
    """Run a full evaluation of the model."""

    params.eval()
    total_error = 0.0
    total_length = 0.0
    with torch.no_grad():
      for (_, features, transcripts, input_lengths) in self._valid_loader:
        features = features.float().to(self._device)
        features = features.transpose(1, 2).unsqueeze(1)
        transcripts = transcripts.long().to(self._device)
        input_lengths = input_lengths.int()

        log_y, _ = params(features, input_lengths, transcripts)

        out, _, _, seq_lens = self._decoder.decode(
            torch.exp(log_y).detach().cpu(), input_lengths)
        for hyp, trn, length in zip(out, transcripts,
                                    seq_lens):  # iterate batch
          best_hyp = hyp[0, :length[0]]
          hh = "".join([self._rev_label_dict[i.item()] for i in best_hyp])
          t = trn.detach().cpu().tolist()
          t = [ll for ll in t if ll != 0]
          tlength = len(t)
          tt = "".join([self._rev_label_dict[i] for i in t])
          error = Levenshtein.distance(tt, hh)
          total_error += error
          total_length += tlength

    return total_error / total_length
