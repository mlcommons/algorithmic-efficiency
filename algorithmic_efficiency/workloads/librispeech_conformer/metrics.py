"""Utilities to compute eval and loss metrics."""
from clu import metrics
import flax
import numpy as np
import tensorflow as tf
import tensorflow_text as tftxt

gfile = tf.io.gfile


def average_ctc_loss():
  """Returns a clu.Metric that computes average CTC loss.

  This metric takes padding into account.
  """

  @flax.struct.dataclass
  class _Metric(metrics.Metric):
    """Applies `fun` and computes the average."""
    total: np.float32
    weight: np.float32

    @classmethod
    def from_model_output(cls, normalized_loss, **_):
      return cls(total=normalized_loss, weight=1.0)

    def merge(self, other):
      return type(self)(
          total=self.total + other.total, weight=self.weight + other.weight)

    def compute(self):
      return self.total / self.weight

  return _Metric


def edit_distance(source, target):
  """Computes edit distance between source string and target string.

  This function assumes words are seperated by a single space.

  Args:
    source: source string.
    target: target string.

  Returns:
    Edit distance between source string and target string.
  """
  source = source.split()
  target = target.split()

  num_source_words = len(source)
  num_target_words = len(target)

  distance = np.zeros((num_source_words + 1, num_target_words + 1))

  for i in range(num_source_words + 1):
    for j in range(num_target_words + 1):
      # If first string is empty, only option is to
      # insert all words of second string
      if i == 0:
        distance[i][j] = j  # Min. operations = j

      # If second string is empty, only option is to
      # remove all characters of second string
      elif j == 0:
        distance[i][j] = i  # Min. operations = i

      # If last characters are same, ignore last char
      # and recur for remaining string
      elif source[i - 1] == target[j - 1]:
        distance[i][j] = distance[i - 1][j - 1]

      # If last character are different, consider all
      # possibilities and find minimum
      else:
        distance[i][j] = 1 + min(
            distance[i][j - 1],  # Insert
            distance[i - 1][j],  # Remove
            distance[i - 1][j - 1])  # Replace

  return distance[num_source_words][num_target_words]


def compute_wer(decoded, decoded_paddings, targets, target_paddings, tokenizer):  # pylint: disable=line-too-long
  """Computes WER."""
  word_errors = 0.0
  num_words = 0.0

  decoded_lengths = np.sum(decoded_paddings == 0.0, axis=-1)
  target_lengths = np.sum(target_paddings == 0.0, axis=-1)

  batch_size = targets.shape[0]

  for i in range(batch_size):
    decoded_length = decoded_lengths[i]
    target_length = target_lengths[i]

    decoded_i = decoded[i][:decoded_length]
    target_i = targets[i][:target_length]

    decoded_i = str(tokenizer.detokenize(decoded_i.astype(np.int32)))
    target_i = str(tokenizer.detokenize(target_i.astype(np.int32)))

    target_num_words = len(target_i.split(' '))

    word_errors += edit_distance(decoded_i, target_i)
    num_words += target_num_words

  return word_errors, num_words


def load_tokenizer(model_path: str,
                   add_bos: bool = False,
                   add_eos: bool = True,
                   reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def wer(tokenizer_vocab_path):
  """Computes WER."""
  tokenizer = load_tokenizer(tokenizer_vocab_path)

  @flax.struct.dataclass
  class WER(
      metrics.CollectingMetric.from_outputs(
          ('decoded', 'decoded_paddings', 'targets', 'target_paddings'))):
    """Computes the mean average precision for a binary classifier on CPU."""

    def compute(self):
      values = super().compute()
      # Ensure the arrays are numpy and not jax.numpy.
      values = {k: np.array(v) for k, v in values.items()}

      word_errors, num_words = compute_wer(values['decoded'],
                                           values['decoded_paddings'],
                                           values['targets'].astype(np.int32),
                                           values['target_paddings'], tokenizer)

      return word_errors / num_words

  return WER


def get_metrics_bundle(tokenizer_vocab_path):
  return metrics.Collection.create(
      ctc_loss=average_ctc_loss(), wer=wer(tokenizer_vocab_path))
