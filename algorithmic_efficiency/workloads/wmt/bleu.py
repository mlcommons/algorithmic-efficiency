from typing import Optional, Sequence

from sacrebleu import BLEU
from sacrebleu.metrics import BLEUScore
import torch
import torch.distributed as dist

from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, _, DEVICE, _ = pytorch_setup()


# Modified (added sync for PyTorch DDP) from
# github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/base.py.
def corpus_bleu(hypotheses: Sequence[str],
                references: Sequence[Sequence[str]],
                smooth_method: str = 'exp',
                smooth_value: Optional[float] = None,
                force: bool = False,
                lowercase: bool = False,
                tokenize: Optional[str] = BLEU.TOKENIZER_DEFAULT,
                use_effective_order: bool = False) -> BLEUScore:
  """Computes BLEU for a corpus against a single (or multiple) reference(s).
    This is the main CLI entry point for computing BLEU between a system output
    and a reference sentence.
    :param hypotheses: A sequence of hypothesis strings.
    :param references: A sequence of reference documents with document being
        defined as a sequence of reference strings.
    :param smooth_method:
        The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
    :param smooth_value: The smoothing value for `floor` and `add-k` methods.
        `None` falls back to default value.
    :param force: Ignore data that looks already tokenized
    :param lowercase: Lowercase the data
    :param tokenize: The tokenizer to use
    :param use_effective_order:
        Don't take into account n-gram orders without any match.
    :return: a `BLEUScore` object
    """
  metric = BLEU(
      lowercase=lowercase,
      force=force,
      tokenize=tokenize,
      smooth_method=smooth_method,
      smooth_value=smooth_value,
      effective_order=use_effective_order)
  metric._check_corpus_score_args(hypotheses, references)

  # Collect corpus stats
  stats = metric._extract_corpus_statistics(hypotheses, references)

  if USE_PYTORCH_DDP:
    stats = torch.as_tensor(stats, device=DEVICE)
    dist.all_reduce(stats)
    stats = stats.cpu().numpy().tolist()

  # Compute the actual system score
  actual_score = metric._aggregate_and_compute(stats)

  return actual_score
