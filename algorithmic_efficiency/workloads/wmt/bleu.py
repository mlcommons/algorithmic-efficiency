from itertools import zip_longest
from typing import Sequence

from absl import logging
import sacrebleu
import torch
import torch.distributed as dist

from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, _, DEVICE, N_GPUS = pytorch_setup()


# Modified (added sync for PyTorch DDP) from
# https://github.com/mjpost/sacrebleu/blob/v1.3.1/sacrebleu.py.
# Assumes that sacrebleu==1.3.1 is installed.
def corpus_bleu(sys_stream: Sequence[str],
                ref_streams: Sequence[str],
                smooth_method: str = 'exp',
                smooth_value: float = 0.0,
                force: bool = False,
                lowercase: bool = False,
                tokenize: str = '13a',
                use_effective_order: bool = False) -> sacrebleu.BLEU:
  """Produces BLEU scores along with its sufficient statistics from a source
  against one or more references.
      :param sys_stream: The system stream (a sequence of segments).
      :param ref_streams: A list of one or more reference streams
                          (each a sequence of segments).
      :param smooth: The smoothing method to use.
      :param smooth_value: For 'floor' smoothing, the floor to use.
      :param force: Ignore data that looks already tokenized.
      :param lowercase: Lowercase the data.
      :param tokenize: The tokenizer to use.
      :return: A BLEU object containing everything you'd want.
  """

  # Add some robustness to the input arguments.
  if isinstance(sys_stream, str):
    sys_stream = [sys_stream]
  if isinstance(ref_streams, str):
    ref_streams = [[ref_streams]]

  sys_len = 0
  ref_len = 0

  correct = [0 for _ in range(sacrebleu.NGRAM_ORDER)]
  total = [0 for _ in range(sacrebleu.NGRAM_ORDER)]

  # Look for already-tokenized sentences.
  tokenized_count = 0

  fhs = [sys_stream] + ref_streams
  for lines in zip_longest(*fhs):
    if None in lines:
      raise EOFError('Source and reference streams have different lengths!')

    if lowercase:
      lines = [x.lower() for x in lines]

    if not (force or tokenize == 'none') and lines[0].rstrip().endswith(' .'):
      tokenized_count += 1

      if tokenized_count == 100:
        logging.warning(
            'That\'s 100 lines that end in a tokenized period (\'.\')')
        logging.warning('It looks like you forgot to detokenize your test '
                        'data, which may hurt your score.')
        logging.warning('If you insist your data is detokenized, '
                        'or don\'t care, you can suppress this message with '
                        '\'--force\'.')

    output, *refs = [sacrebleu.TOKENIZERS[tokenize](x.rstrip()) for x in lines]

    ref_ngrams, _, closest_len = sacrebleu.ref_stats(output, refs)

    sys_len += len(output.split())
    ref_len += closest_len

    sys_ngrams = sacrebleu.extract_ngrams(output)
    for ngram, sys_ngram in sys_ngrams.items():
      n = len(ngram.split())
      correct[n - 1] += min(sys_ngram, ref_ngrams.get(ngram, 0))
      total[n - 1] += sys_ngram

  # When using PyTorch DDP, get stats from all processes and sum them.
  if USE_PYTORCH_DDP:
    # Sum `sys_len` and `ref_len` integers from all processes.
    sys_len = torch.tensor(sys_len, dtype=torch.int64, device=DEVICE)
    dist.all_reduce(sys_len)
    sys_len = sys_len.item()
    ref_len = torch.tensor(ref_len, dtype=torch.int64, device=DEVICE)
    dist.all_reduce(ref_len)
    ref_len = ref_len.item()
    # Sum `correct` and `total` sequences from all processes.
    correct = torch.tensor(correct, dtype=torch.int64, device=DEVICE)
    dist.all_reduce(correct)
    correct = correct.cpu().numpy().tolist()
    total = torch.tensor(total, dtype=torch.int64, device=DEVICE)
    dist.all_reduce(total)
    total = total.cpu().numpy().tolist()

  return sacrebleu.compute_bleu(
      correct,
      total,
      sys_len,
      ref_len,
      smooth_method=smooth_method,
      smooth_value=smooth_value,
      use_effective_order=use_effective_order)
