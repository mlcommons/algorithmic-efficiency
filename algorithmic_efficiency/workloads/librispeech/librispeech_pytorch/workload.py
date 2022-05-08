"""LibriSpeech workload implemented in Pytorch."""
import itertools
import math
import os
from typing import Tuple

import ctcdecode
import Levenshtein
import torch
import torch.utils.data as data_utils

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.workloads.librispeech.librispeech_pytorch import \
    input_pipeline
from algorithmic_efficiency.workloads.librispeech.librispeech_pytorch import \
    models

device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")

class LibriSpeechWorkload(spec.Workload):
  """A LibriSpeech workload."""

  def __init__(self):
    self._param_shapes = None
    self._param_types = None
    self._eval_iters = {}
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
                        global_batch_size: int):
    torch.manual_seed(data_rng[0])
    is_train = split in ['train', 'eval_train']
    if is_train:
      # DO NOT SUBMIT make sure we load in all train files (not just 100)
      filename = "features_train-clean-100.csv"
    elif split == 'validation':
      filename = "features_dev-clean.csv"
    elif split == 'test':
      filename = "features_test-clean.csv"
    else:
      raise ValueError('Received unsupported dataset split "{}".'.format(split))

    ds = input_pipeline.LibriSpeechDataset(
        os.path.join(data_dir, filename))
    if split == 'eval_train':
      ds, _ = data_utils.random_split(
          ds,
          [self.num_eval_train_examples,
           len(ds) - self.num_eval_train_examples],
          generator=torch.Generator().manual_seed(int(data_rng[1])))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=global_batch_size,
        shuffle=is_train,
        num_workers=2,
        pin_memory=True,
        collate_fn=ds.pad_collate)
    return iter(loader)

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
  def num_eval_train_examples(self):
    return 1000

  @property
  def num_validation_examples(self):
    return 2620

  @property
  def num_test_examples(self):
    return None

  @property
  def train_mean(self):
    return 0.0

  @property
  def train_stddev(self):
    return 1.0

  @property
  def param_shapes(self):
    if self._param_shapes is None:
      raise ValueError(
          'This should not happen, workload.init_model_fn() should be called '
          'before workload.param_shapes!')
    return self._param_shapes

  @property
  def model_params_types(self):
    """The shapes of the parameters in the workload model."""
    if self._param_types is None:
      self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    return self._param_types

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

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    model = models.CNNLSTM()
    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
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
    features = augmented_and_preprocessed_input_batch['features']
    transcripts = augmented_and_preprocessed_input_batch['transcripts']
    input_lengths = augmented_and_preprocessed_input_batch['input_lengths']
    features = features.float().to(device)
    features = features.transpose(1, 2).unsqueeze(1)
    transcripts = transcripts.long().to(device)
    input_lengths = input_lengths.long().to(device)

    params.train(mode == spec.ForwardPassMode.TRAIN)
    log_y, output_lengths = params(features, input_lengths, transcripts)

    return (log_y.transpose(0, 1), output_lengths), None

  def loss_fn(
      self,
      batch: spec.Tensor,  # transcripts
      logits_batch: spec.Tensor) -> spec.Tensor:
    label_batch = batch['transcripts']

    log_y, output_lengths = logits_batch
    target_lengths = torch.IntTensor([len(y[y != 0]) for y in label_batch])

    loss = self._loss(log_y, label_batch, output_lengths, target_lengths) / (
        target_lengths.float().to(device))

    return loss

  def output_activation_fn(self,
                           logits_batch: spec.Tensor,
                           loss_type: spec.LossType) -> spec.Tensor:
    """Return the final activations of the model."""
    pass

  def _eval_model_on_split(self,
                           split: str,
                           num_examples: int,
                           global_batch_size: int,
                           params: spec.ParameterContainer,
                           model_state: spec.ModelAuxiliaryState,
                           rng: spec.RandomState,
                           data_dir: str):
    del model_state
    if split not in self._eval_iters:
      data_loader = self.build_input_queue(
          rng,
          split,
          data_dir,
          global_batch_size)
      # Note that this saves the entire dataset split in memory.
      self._eval_iters[split] = itertools.cycle(data_loader)
    num_batches = int(math.ceil(num_examples / global_batch_size))
    params.eval()
    total_error = 0.0
    total_length = 0.0
    with torch.no_grad():
      for (bi, batch) in enumerate(self._eval_iters[split]):
        if bi > num_batches:
          break
        features = batch['features'].float().to(device)
        features = features.transpose(1, 2).unsqueeze(1)
        transcripts = batch['transcripts'].long().to(device)
        input_lengths = batch['input_lengths'].int()

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

    wer = total_error / total_length
    return {'word_error_rate': wer}
