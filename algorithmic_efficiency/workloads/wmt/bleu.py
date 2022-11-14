from typing import Optional, Sequence

from sacrebleu import BLEU
from sacrebleu.metrics import BLEUScore
import torch
import torch.distributed as dist

from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, _, DEVICE, N_GPUS = pytorch_setup()


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

  # When using PyTorch DDP, get stats from all processes and concatenate them.
  if USE_PYTORCH_DDP:
    length = torch.tensor(len(stats), dtype=torch.int64, device=DEVICE)
    all_lengths = [torch.empty_like(length) for _ in range(N_GPUS)]
    dist.all_gather(all_lengths, length)
    max_length = max(all_lengths)
    stats = torch.as_tensor(stats, dtype=torch.int64, device=DEVICE)
    # If the evaluation dataset cannot be split into N_GPUS equally sized
    # subsets, we have to pad (with zeros) to be able to use all_gather.
    if length < max_length:
      padding_size = max_length - length
      padding = stats.new_zeros((padding_size, *stats.shape[1:]))
      stats = torch.cat((stats, padding))
    all_stats = [torch.empty_like(stats) for _ in range(N_GPUS)]
    dist.all_gather(all_stats, stats)
    # Remove padding before concatenating the stats from all processes.
    stats = torch.cat(
        [all_stats[i][:length] for i, length in enumerate(all_lengths)])
    stats = stats.cpu().numpy().tolist()

  # Compute the actual system score
  actual_score = metric._aggregate_and_compute(stats)

  return actual_score
