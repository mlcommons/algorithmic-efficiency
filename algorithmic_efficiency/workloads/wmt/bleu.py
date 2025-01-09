"""
Removing the dependency on sacrebleu, we reimplement the BLEU score computation
in this file.
Reference:
https://github.com/mjpost/sacrebleu/blob/v1.3.1/sacrebleu.py.
"""

from collections import Counter
from collections import namedtuple
from itertools import zip_longest
import logging
import math
import re
import sys
from typing import List, Sequence
import unicodedata

from absl import logging
import torch
import torch.distributed as dist

from algorithmic_efficiency.pytorch_utils import pytorch_setup

USE_PYTORCH_DDP, _, DEVICE, N_GPUS = pytorch_setup()

NGRAM_ORDER = 4
# The default floor value to use with `--smooth floor`
SMOOTH_VALUE_DEFAULT = 0.0


def my_log(num):
  """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """

  if num == 0.0:
    return -9999999999
  return math.log(num)


def tokenize_13a(line):
  """
    Tokenizes an input line using a relatively minimal tokenization that is 
    however equivalent to mteval-v13a, used by WMT.

    :param line: a segment to tokenize
    :return: the tokenized line
    """

  norm = line

  # language-independent part:
  norm = norm.replace('<skipped>', '')
  norm = norm.replace('-\n', '')
  norm = norm.replace('\n', ' ')
  norm = norm.replace('&quot;', '"')
  norm = norm.replace('&amp;', '&')
  norm = norm.replace('&lt;', '<')
  norm = norm.replace('&gt;', '>')

  # language-dependent part (assuming Western languages):
  norm = " {} ".format(norm)
  norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
  norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ',
                norm)  # tokenize period and comma unless preceded by a digit
  norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2',
                norm)  # tokenize period and comma unless followed by a digit
  norm = re.sub(r'([0-9])(-)', '\\1 \\2 ',
                norm)  # tokenize dash when preceded by a digit
  norm = re.sub(r'\s+', ' ', norm)  # one space only between words
  norm = re.sub(r'^\s+', '', norm)  # no leading space
  norm = re.sub(r'\s+$', '', norm)  # no trailing space

  return norm


class UnicodeRegex:
  """Ad-hoc hack to recognize all punctuation and symbols.

    without depending on https://pypi.python.org/pypi/regex/."""

  @staticmethod
  def _property_chars(prefix):
    return ''.join(
        chr(x)
        for x in range(sys.maxunicode)
        if unicodedata.category(chr(x)).startswith(prefix))

  punctuation = _property_chars('P')
  nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
  punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
  symbol_re = re.compile('([' + _property_chars('S') + '])')


def tokenize_v14_international(string):
  r"""Tokenize a string following the official BLEU implementation.

    See 
    https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a number (e.g., a year) followed by a dot at the end of sentence 
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.
    The error is not present in the non-international version,
    which uses,
    `$norm_text = " $norm_text "` (or `norm = " {} ".format(norm)` in Python).

    :param string: the input string
    :return: a list of tokens
    """
  string = UnicodeRegex.nondigit_punct_re.sub(r'\1 \2 ', string)
  string = UnicodeRegex.punct_nondigit_re.sub(r' \1 \2', string)
  string = UnicodeRegex.symbol_re.sub(r' \1 ', string)
  return string.strip()


def tokenize_zh(sentence):
  """MIT License
    Copyright (c) 2017 - Shujian Huang <huangsj@nju.edu.cn>

    Permission is hereby granted, free of charge, to any person obtaining 
    a copy of this software and associated documentation files 
    (the "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish, 
    distribute, sublicense, and/or sell copies of the Software, and to 
    permit persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included 
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE 
    USE OR OTHER DEALINGS IN THE SOFTWARE.

    The tokenization of Chinese text in this script contains two steps: 
    separate each Chinese characters (by utf-8 encoding); 
    tokenize the non Chinese part (following the mteval script).
    Author: Shujian Huang huangsj@nju.edu.cn

    :param sentence: input sentence
    :return: tokenized sentence
    """

  def is_chinese_char(uchar):
    """
      :param uchar: input char in unicode
      :return: whether the input char is a Chinese character.
      """
    if "\u3400" <= uchar <= "\u4db5":
      return True
    elif "\u4e00" <= uchar <= "\u9fa5":
      return True
    elif "\u9fa6" <= uchar <= "\u9fbb":
      return True
    elif "\uf900" <= uchar <= "\ufa2d":
      return True
    elif "\ufa30" <= uchar <= "\ufa6a":
      return True
    elif "\ufa70" <= uchar <= "\ufad9":
      return True
    elif "\u20000" <= uchar <= "\u2a6d6":
      return True
    elif "\u2f800" <= uchar <= "\u2fa1d":
      return True
    elif "\uff00" <= uchar <= "\uffef":
      return True
    elif "\u2e80" <= uchar <= "\u2eff":
      return True
    elif "\u3000" <= uchar <= "\u303f":
      return True
    elif "\u31c0" <= uchar <= "\u31ef":
      return True
    elif "\u2f00" <= uchar <= "\u2fdf":
      return True
    elif "\u2ff0" <= uchar <= "\u2fff":
      return True
    elif "\u3100" <= uchar <= "\u312f":
      return True
    elif "\u31a0" <= uchar <= "\u31bf":
      return True
    elif "\ufe10" <= uchar <= "\ufe1f":
      return True
    elif "\ufe30" <= uchar <= "\ufe4f":
      return True
    elif "\u2600" <= uchar <= "\u26ff":
      return True
    elif "\u2700" <= uchar <= "\u27bf":
      return True
    elif "\u3200" <= uchar <= "\u32ff":
      return True
    elif "\u3300" <= uchar <= "\u33ff":
      return True
    return False

  sentence = sentence.strip()
  sentence_in_chars = ""
  for char in sentence:
    if is_chinese_char(char):
      sentence_in_chars += " "
      sentence_in_chars += char
      sentence_in_chars += " "
    else:
      sentence_in_chars += char
  sentence = sentence_in_chars

  # TODO: the code above could probably be replaced with the following line:
  # import regex
  # sentence = regex.sub(r'(\p{Han})', r' \1 ', sentence)

  # tokenize punctuation
  sentence = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sentence)

  # tokenize period and comma unless preceded by a digit
  sentence = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sentence)

  # tokenize period and comma unless followed by a digit
  sentence = re.sub(r'([\.,])([^0-9])', r' \1 \2', sentence)

  # tokenize dash when preceded by a digit
  sentence = re.sub(r'([0-9])(-)', r'\1 \2 ', sentence)

  # one space only between words
  sentence = re.sub(r'\s+', r' ', sentence)

  # no leading or trailing spaces
  sentence = sentence.strip()

  return sentence


TOKENIZERS = {
    '13a': tokenize_13a,
    'intl': tokenize_v14_international,
    'zh': tokenize_zh,
    'none': lambda x: x,
}
DEFAULT_TOKENIZER = '13a'


def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
  """Extracts all the ngrams (1 <= n <= NGRAM_ORDER) from a sequence of tokens.

    :param line: a segment containing a sequence of words
    :param max_order: collect n-grams from 1<=n<=max
    :return: a dictionary containing ngrams and counts
    """

  ngrams = Counter()
  tokens = line.split()
  for n in range(min_order, max_order + 1):
    for i in range(0, len(tokens) - n + 1):
      ngram = ' '.join(tokens[i:i + n])
      ngrams[ngram] += 1

  return ngrams


def ref_stats(output, refs):
  ngrams = Counter()
  closest_diff = None
  closest_len = None
  for ref in refs:
    tokens = ref.split()
    reflen = len(tokens)
    diff = abs(len(output.split()) - reflen)
    if closest_diff is None or diff < closest_diff:
      closest_diff = diff
      closest_len = reflen
    elif diff == closest_diff:
      if reflen < closest_len:
        closest_len = reflen

    ngrams_ref = extract_ngrams(ref)
    for ngram in ngrams_ref:
      ngrams[ngram] = max(ngrams[ngram], ngrams_ref[ngram])

  return ngrams, closest_diff, closest_len


BLEU = namedtuple('BLE',
                  'score, counts, totals, precisions, bp, sys_len, ref_len')


def compute_bleu(correct: List[int],
                 total: List[int],
                 sys_len: int,
                 ref_len: int,
                 smooth_method='none',
                 smooth_value=SMOOTH_VALUE_DEFAULT,
                 use_effective_order=False) -> BLEU:
  """Computes BLEU score from its sufficient statistics. Adds smoothing.

    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques 
    for Sentence-Level BLEU", Boxing Chen and Colin Cherry, 
    WMT 2014: http://aclweb.org/anthology/W14-3346)

    - exp: NIST smoothing method (Method 3)
    - floor: Method 1
    - add-k: Method 2 (generalizing Lin and Och, 2004)
    - none: do nothing.

    :param correct: List of counts of correct ngrams, 1 <= n <= NGRAM_ORDER
    :param total: List of counts of total ngrams, 1 <= n <= NGRAM_ORDER
    :param sys_len: The cumulative system length
    :param ref_len: The cumulative reference length
    :param smooth: The smoothing method to use
    :param smooth_value: The smoothing value added, if smooth is 'floor'
    :param use_effective_order: Use effective order.
    :return: A BLEU object with the score (100-based) and other statistics.
    """

  precisions = [0 for x in range(NGRAM_ORDER)]

  smooth_mteval = 1.
  effective_order = NGRAM_ORDER
  for n in range(NGRAM_ORDER):
    if smooth_method == 'add-k' and n > 1:
      correct[n] += smooth_value
      total[n] += smooth_value
    if total[n] == 0:
      break

    if use_effective_order:
      effective_order = n + 1

    if correct[n] == 0:
      if smooth_method == 'exp':
        smooth_mteval *= 2
        precisions[n] = 100. / (smooth_mteval * total[n])
      elif smooth_method == 'floor':
        precisions[n] = 100. * smooth_value / total[n]
    else:
      precisions[n] = 100. * correct[n] / total[n]

  # If the system guesses no i-grams, 1 <= i <= NGRAM_ORDER, the BLEU
  # score is 0 (technically undefined). This is a problem for sentence-level
  # BLEU or a corpus of short sentences, where systems will get no credit
  # if sentence lengths fall under the NGRAM_ORDER threshold. This fix scales
  # NGRAM_ORDER to the observed maximum order.
  # It is only available through the API and off by default

  brevity_penalty = 1.0
  if sys_len < ref_len:
    brevity_penalty = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

  bleu = brevity_penalty * math.exp(
      sum(map(my_log, precisions[:effective_order])) / effective_order)

  return BLEU._make(
      [bleu, correct, total, precisions, brevity_penalty, sys_len, ref_len])


def corpus_bleu(sys_stream: Sequence[str],
                ref_streams: Sequence[str],
                smooth_method: str = 'exp',
                smooth_value: float = 0.0,
                force: bool = False,
                lowercase: bool = False,
                tokenize: str = '13a',
                use_effective_order: bool = False) -> BLEU:
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
      :return: A BLEU object containing everything yo'd want.
  """

  # Add some robustness to the input arguments.
  if isinstance(sys_stream, str):
    sys_stream = [sys_stream]
  if isinstance(ref_streams, str):
    ref_streams = [[ref_streams]]

  sys_len = 0
  ref_len = 0

  correct = [0 for _ in range(NGRAM_ORDER)]
  total = [0 for _ in range(NGRAM_ORDER)]

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

    output, *refs = [TOKENIZERS[tokenize](x.rstrip()) for x in lines]

    ref_ngrams, _, closest_len = ref_stats(output, refs)

    sys_len += len(output.split())
    ref_len += closest_len

    sys_ngrams = extract_ngrams(output)
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

  return compute_bleu(
      correct,
      total,
      sys_len,
      ref_len,
      smooth_method=smooth_method,
      smooth_value=smooth_value,
      use_effective_order=use_effective_order)
