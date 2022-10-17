import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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

  def init_model_fn(self, rng: spec.RandomState) -> spec.ModelInitState:
    torch.random.manual_seed(rng[0])
    model = DeepspeechEncoderDecoder(DeepspeechConfig())
    self._model = model
    self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction='none')
    # Run model once to initialize lazy layers
    t = MAX_INPUT_LENGTH
    wave = torch.randn((2, t))
    pad = torch.zeros_like(wave)
    _ = model(wave, pad)
    initialize(model)

    self._param_shapes = {
        k: spec.ShapeTuple(v.shape) for k, v in model.named_parameters()
    }
    model.to(DEVICE)
    if N_GPUS > 1:
      if USE_PYTORCH_DDP:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[RANK], output_device=RANK)
      else:
        model = torch.nn.DataParallel(model)
    return model, None
