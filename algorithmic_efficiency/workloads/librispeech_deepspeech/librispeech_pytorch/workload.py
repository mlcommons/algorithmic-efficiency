from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from algorithmic_efficiency import param_utils
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.model import \
    initialize
from algorithmic_efficiency.workloads.librispeech_conformer.librispeech_pytorch.workload import \
    LibriSpeechConformerWorkload
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.model import \
    DeepspeechConfig
from algorithmic_efficiency.workloads.librispeech_deepspeech.librispeech_pytorch.model import \
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
            input_dropout_rate=aux_dropout_rate)).eval()
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
