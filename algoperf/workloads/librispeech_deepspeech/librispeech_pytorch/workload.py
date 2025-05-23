from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from algoperf import param_utils
from algoperf import spec
from algoperf.pytorch_utils import pytorch_setup
from algoperf.workloads.librispeech_conformer.librispeech_pytorch.models import \
    initialize
from algoperf.workloads.librispeech_conformer.librispeech_pytorch.workload import \
    LibriSpeechConformerWorkload
from algoperf.workloads.librispeech_deepspeech.librispeech_pytorch.models import \
    DeepspeechConfig
from algoperf.workloads.librispeech_deepspeech.librispeech_pytorch.models import \
    DeepspeechEncoderDecoder

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

MAX_INPUT_LENGTH = 320000


class LibriSpeechDeepSpeechWorkload(LibriSpeechConformerWorkload):

  def init_model_fn(
      self,
      rng: spec.RandomState,
      dropout_rate: Optional[float] = None,
      aux_dropout_rate: Optional[float] = None) -> spec.ModelInitState:
    """Deepspeech model init function.

    Here we use dropout_rate as feed_forward_dropout_rate, and aux_dropout_rate
    as input_dropout_rate.
    """
    torch.random.manual_seed(rng[0])
    model = DeepspeechEncoderDecoder(
        DeepspeechConfig(
            feed_forward_dropout_rate=dropout_rate,
            use_specaug=self.use_specaug,
            input_dropout_rate=aux_dropout_rate,
            use_tanh=self.use_tanh,
            enable_residual_connections=self.enable_residual_connections,
            enable_decoder_layer_norm=self.enable_decoder_layer_norm,
            layernorm_everywhere=self.layernorm_everywhere,
            freq_mask_count=self.freq_mask_count,
            time_mask_count=self.time_mask_count)).eval()
    self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
    # Run model once to initialize lazy layers.
    t = MAX_INPUT_LENGTH
    wave = torch.randn((2, t))
    pad = torch.zeros_like(wave)
    _ = model(wave, pad)
    initialize(model)

    self._param_shapes = param_utils.pytorch_param_shapes(model)
    self._param_types = param_utils.pytorch_param_types(self._param_shapes)
    model.to(DEVICE)
    self.requires_sync_before_eval = False
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None

  def is_output_params(self, param_key: spec.ParameterKey) -> bool:
    return param_key in ['lin.weight', 'lin.bias']

  @property
  def validation_target_value(self) -> float:
    return 0.119936

  @property
  def test_target_value(self) -> float:
    return 0.074143

  @property
  def step_hint(self) -> int:
    """Approx. steps the baseline can do in the allowed runtime budget."""
    return 38_400

  @property
  def max_allowed_runtime_sec(self) -> int:
    return 44_405  # ~12.3 hours

  @property
  def use_tanh(self) -> bool:
    return False

  @property
  def enable_residual_connections(self) -> bool:
    return True

  @property
  def enable_decoder_layer_norm(self) -> bool:
    return True

  @property
  def layernorm_everywhere(self) -> bool:
    return False

  @property
  def freq_mask_count(self) -> int:
    return 2

  @property
  def time_mask_count(self) -> int:
    return 10


class LibriSpeechDeepSpeechTanhWorkload(LibriSpeechDeepSpeechWorkload):

  @property
  def use_tanh(self) -> bool:
    return True

  @property
  def validation_target_value(self) -> float:
    return 0.150883

  @property
  def test_target_value(self) -> float:
    return 0.098613


class LibriSpeechDeepSpeechNoResNetWorkload(LibriSpeechDeepSpeechWorkload):

  @property
  def enable_residual_connections(self) -> bool:
    return False

  @property
  def validation_target_value(self) -> float:
    return 0.131564

  @property
  def test_target_value(self) -> float:
    return 0.079297


class LibriSpeechDeepSpeechNormAndSpecAugWorkload(LibriSpeechDeepSpeechWorkload
                                                 ):

  @property
  def eval_batch_size(self) -> int:
    return 128

  @property
  def enable_decoder_layer_norm(self) -> bool:
    return False

  @property
  def layernorm_everywhere(self) -> bool:
    return True

  @property
  def freq_mask_count(self) -> int:
    return 4

  @property
  def time_mask_count(self) -> int:
    return 15

  @property
  def validation_target_value(self) -> float:
    return 0.14342

  @property
  def test_target_value(self) -> float:
    return 0.090976
