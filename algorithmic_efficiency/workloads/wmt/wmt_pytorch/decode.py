"""Fast decoding routines for inference from a trained model.

PyTorch port of https://github.com/google/flax/tree/main/examples/wmt.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import torch
import torch.nn.functional as F

from algorithmic_efficiency.pytorch_utils import pytorch_setup

DEVICE = pytorch_setup()[2]

# Constants
# We assume the default End-of-Sentence token id is 2 (SentencePiece).
EOS_ID = 2
# "Effective negative infinity" constant for masking in beam search.
NEG_INF = torch.tensor(-1.0e7, device=DEVICE)


def brevity_penalty(alpha: float, length: Union[int,
                                                torch.Tensor]) -> torch.Tensor:
  """Brevity penalty function for beam search penalizing short sequences.

  Args:
    alpha: float: brevity-penalty scaling parameter.
    length: int or scalar int tensor: length of considered sequence.

  Returns:
    Brevity penalty score as jax scalar.
  """
  if not isinstance(length, torch.Tensor):
    length = torch.tensor(length, device=DEVICE)
  return torch.pow((5.0 + length) / 6.0, alpha)


# Beam handling utility functions:


def add_beam_dim(x: torch.Tensor, beam_size: int) -> torch.Tensor:
  """Creates new beam dimension in non-scalar array and tiles into it."""
  if x.dim() < 2:  # ignore scalars (e.g. cache index)
    return x
  x = x.unsqueeze(dim=1)
  tile_dims = [1] * x.dim()
  tile_dims[1] = beam_size
  return torch.tile(x, tile_dims)


def flatten_beam_dim(x: torch.Tensor) -> torch.Tensor:
  """Flattens the first two dimensions of a non-scalar array."""
  if x.dim() < 2:  # ignore scalars (e.g. cache index)
    return x
  return x.view(-1, *x.shape[2:])


def unflatten_beam_dim(x: torch.Tensor, batch_size: int,
                       beam_size: int) -> torch.Tensor:
  """Unflattens the first, flat batch*beam dimension of a non-scalar tensor."""
  if x.dim() < 2:  # ignore scalars (e.g. cache index)
    return x
  assert batch_size * beam_size == x.shape[0]
  return x.view(batch_size, beam_size, *x.shape[1:])


def flat_batch_beam_expand(x: torch.Tensor, beam_size: int) -> torch.Tensor:
  """Expands the each batch item by beam_size in batch_dimension."""
  return flatten_beam_dim(add_beam_dim(x, beam_size))


def gather_beams(nested: Dict[str, Any],
                 beam_indices: torch.Tensor,
                 batch_size: int,
                 new_beam_size: int) -> Dict[str, Any]:
  """Gathers the beam slices indexed by beam_indices into new beam tensor.

  Args:
    nested: Dict of (dicts of) tensors.
    beam_indices: tensor of beam_indices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ beam dimension.

  Returns:
    New dict with new beam tensors.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  batch_indices = torch.reshape(
      torch.div(
          torch.arange(batch_size * new_beam_size, device=DEVICE),
          new_beam_size,
          rounding_mode='floor'), (batch_size, new_beam_size))

  def gather_fn(x):
    if x.dim() < 2:  # ignore scalars (e.g. cache index)
      return x
    return x[batch_indices, beam_indices]

  return jax.tree_map(gather_fn, nested)


def gather_topk_beams(nested: Dict[str, Any],
                      score_or_log_prob: torch.Tensor,
                      batch_size: int,
                      new_beam_size: int) -> Dict[str, Any]:
  """Gathers the top-k beam slices given by score_or_log_prob array.

  Args:
    nested: Dict of (dicts of) tensors.
    score_or_log_prob: [batch_size, old_beam_size] tensor of values to sort by
      for top-k selection of beam slices.
    batch_size: int: size of batch.
    new_beam_size: int: size of _new_ top-k selected beam dimension

  Returns:
    New dict with new beam tensors containing top k new_beam_size slices.
    [batch_size, old_beam_size, ...] --> [batch_size, new_beam_size, ...]
  """
  _, topk_indices = torch.topk(score_or_log_prob, k=new_beam_size)
  topk_indices = torch.flip(topk_indices, (1,))
  return gather_beams(nested, topk_indices, batch_size, new_beam_size)


# Beam search state:


@dataclass
class BeamState:
  """Holds beam search state data."""
  # The position of the decoding loop in the length dimension.
  cur_index: torch.Tensor  # scalar int32: current decoded length index.
  # The active sequence log probabilities and finished sequence scores.
  live_logprobs: torch.Tensor  # float32: [batch_size, beam_size]
  finished_scores: torch.Tensor  # float32: [batch_size, beam_size]
  # The current active-beam-searching and finished sequences.
  live_seqs: torch.Tensor  # int32: [batch_size, beam_size, max_decode_len]
  finished_seqs: torch.Tensor  # int32: [batch_size, beam_size, max_decode_len]
  # Records which of the 'finished_seqs' is occupied and not a filler slot.
  finished_flags: torch.Tensor  # bool: [batch_size, beam_size]
  # The current state of the autoregressive decoding caches.
  cache: Dict[str, Any]  # Any dict (of dicts), with torch.Tensors as leafs.


def beam_init(batch_size: int,
              beam_size: int,
              max_decode_len: int,
              cache: Dict[str, Any]) -> BeamState:
  """Initializes the beam search state data structure."""
  cur_index0 = torch.tensor(0, device=DEVICE)
  live_logprobs0 = torch.tile(
      torch.tensor([0.0] + [NEG_INF] * (beam_size - 1), device=DEVICE),
      [batch_size, 1])
  finished_scores0 = (
      torch.ones((batch_size, beam_size), device=DEVICE) * NEG_INF)
  live_seqs0 = torch.zeros((batch_size, beam_size, max_decode_len),
                           dtype=torch.int32,
                           device=DEVICE)
  finished_seqs0 = torch.zeros((batch_size, beam_size, max_decode_len),
                               dtype=torch.int32,
                               device=DEVICE)
  finished_flags0 = torch.zeros((batch_size, beam_size),
                                dtype=torch.bool,
                                device=DEVICE)
  # add beam dimension to attention cache pytree elements
  beam_cache0 = jax.tree_map(lambda x: add_beam_dim(x, beam_size), cache)
  return BeamState(
      cur_index=cur_index0,
      live_logprobs=live_logprobs0,
      finished_scores=finished_scores0,
      live_seqs=live_seqs0,
      finished_seqs=finished_seqs0,
      finished_flags=finished_flags0,
      cache=beam_cache0)


# Beam search routine:


def beam_search(
    inputs: torch.Tensor,
    cache: Optional[Dict[str, Any]],
    tokens_to_logits: Callable,
    beam_size: int = 4,
    alpha: float = 0.6,
    eos_id: int = EOS_ID,
    max_decode_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
  """Beam search for transformer machine translation.

  Args:
    inputs: torch.Tensor: [batch_size, length] int32 sequence of tokens.
    cache: Dict of (dicts of) tensors.
    tokens_to_logits: fast autoregressive decoder function taking single token
      slices and cache and returning next-token logits and updated cache.
    beam_size: int: number of beams to use in beam search.
    alpha: float: scaling factor for brevity penalty.
    eos_id: int: id of end-of-sentence token for target vocabulary.
    max_decode_len: int: maximum length of decoded translations.

  Returns:
     Tuple of:
       [batch_size, beam_size, max_decode_len] top-scoring sequences
       [batch_size, beam_size] beam-search scores.
  """
  # We liberally annotate shape information for clarity below.

  batch_size = inputs.shape[0]
  if max_decode_len is None:
    max_decode_len = inputs.shape[1]
  end_marker = torch.tensor(eos_id, device=DEVICE)

  # initialize beam search state
  beam_search_init_state = beam_init(batch_size,
                                     beam_size,
                                     max_decode_len,
                                     cache)

  def beam_search_loop_cond_fn(state: BeamState) -> bool:
    """Beam search loop termination condition."""
    # Have we reached max decoding length?
    not_at_end = (state.cur_index < max_decode_len - 1)

    # Is no further progress in the beam search possible?
    # Get the best possible scores from alive sequences.
    min_brevity_penalty = brevity_penalty(alpha, max_decode_len)
    best_live_scores = state.live_logprobs[:, -1:] / min_brevity_penalty
    # Get the worst scores from finished sequences.
    worst_finished_scores, _ = torch.min(
        state.finished_scores, dim=1, keepdim=True)
    # Mask out scores from slots without any actual finished sequences.
    worst_finished_scores = torch.where(state.finished_flags,
                                        worst_finished_scores,
                                        NEG_INF)
    # If no best possible live score is better than current worst finished
    # scores, the search cannot improve the finished set further.
    search_terminated = torch.all(worst_finished_scores > best_live_scores)

    # If we're not at the max decode length, and the search hasn't terminated,
    # continue looping.
    return not_at_end & (~search_terminated)

  def beam_search_loop_body_fn(state: BeamState) -> BeamState:
    """Beam search loop state update function."""
    # Collect the current position slice along length to feed the fast
    # autoregressive decoder model.  Flatten the beam dimension into batch
    # dimension for feeding into the model.
    # --> [batch * beam, 1]
    cur_index = state.cur_index
    flat_ids = flatten_beam_dim(
        state.live_seqs[:batch_size, :beam_size, cur_index:cur_index + 1])
    # Flatten beam dimension into batch to be compatible with model.
    # {[batch, beam, ...], ...} --> {[batch * beam, ...], ...}
    flat_cache = jax.tree_map(flatten_beam_dim, state.cache)

    # Call fast-decoder model on current tokens to get next-position logits.
    # --> [batch * beam, vocab]
    flat_logits, new_flat_cache = tokens_to_logits(flat_ids, flat_cache)

    # unflatten beam dimension
    # [batch * beam, vocab] --> [batch, beam, vocab]
    logits = unflatten_beam_dim(flat_logits, batch_size, beam_size)
    # Unflatten beam dimension in attention cache arrays
    # {[batch * beam, ...], ...} --> {[batch, beam, ...], ...}
    new_cache = jax.tree_map(
        lambda x: unflatten_beam_dim(x, batch_size, beam_size), new_flat_cache)

    # Gather log probabilities from logits
    candidate_log_probs = F.log_softmax(logits, dim=-1)
    # Add new logprobs to existing prefix logprobs.
    # --> [batch, beam, vocab]
    log_probs = candidate_log_probs + state.live_logprobs.unsqueeze(dim=2)

    # We'll need the vocab size, gather it from the log probability dimension.
    vocab_size = log_probs.shape[2]

    # Each item in batch has beam_size * vocab_size candidate sequences.
    # For each item, get the top 2*k candidates with the highest log-
    # probabilities. We gather the top 2*K beams here so that even if the best
    # K sequences reach EOS simultaneously, we have another K sequences
    # remaining to continue the live beam search.
    beams_to_keep = 2 * beam_size
    # Flatten beam and vocab dimensions.
    flat_log_probs = log_probs.view(batch_size, beam_size * vocab_size)
    # Gather the top 2*K scores from _all_ beams.
    # --> [batch, 2*beams], [batch, 2*beams]
    topk_log_probs, topk_indices = torch.topk(flat_log_probs, k=beams_to_keep)
    # Recover the beam index by floor division.
    topk_beam_indices = torch.div(
        topk_indices, vocab_size, rounding_mode='floor')
    # Gather 2*k top beams.
    # --> [batch, 2*beams, length]
    topk_seq = gather_beams(state.live_seqs,
                            topk_beam_indices,
                            batch_size,
                            beams_to_keep)

    # Append the most probable 2*K token IDs to the top 2*K sequences
    # Recover token id by modulo division and expand Id array for broadcasting.
    # --> [batch, 2*beams, 1]
    topk_ids = torch.unsqueeze(topk_indices % vocab_size, dim=2)
    # Update sequences for the 2*K top-k new sequences.
    # --> [batch, 2*beams, length]
    topk_seq[:, :, cur_index + 1:] = topk_ids
    # Update LIVE (in-progress) sequences:
    # Did any of these sequences reach an end marker?
    # --> [batch, 2*beams]
    newly_finished = (topk_seq[:, :, cur_index + 1] == end_marker)
    # To prevent these newly finished sequences from being added to the LIVE
    # set of active beam search sequences, set their log probs to a very large
    # negative value.
    new_log_probs = topk_log_probs + newly_finished * NEG_INF
    # Determine the top k beam indices (from top 2*k beams) from log probs.
    # --> [batch, beams]
    _, new_topk_indices = torch.topk(new_log_probs, k=beam_size)
    new_topk_indices = torch.flip(new_topk_indices, (1,))
    # Gather the top k beams (from top 2*k beams).
    # --> [batch, beams, length], [batch, beams]
    top_alive_seq, top_alive_log_probs = gather_beams([topk_seq, new_log_probs],
                                                      new_topk_indices,
                                                      batch_size, beam_size)

    # Determine the top k beam indices from the original set of all beams.
    # --> [batch, beams]
    top_alive_indices = gather_beams(topk_beam_indices,
                                     new_topk_indices,
                                     batch_size,
                                     beam_size)
    # With these, gather the top k beam-associated caches.
    # --> {[batch, beams, ...], ...}
    top_alive_cache = gather_beams(new_cache,
                                   top_alive_indices,
                                   batch_size,
                                   beam_size)

    # Update FINISHED (reached end of sentence) sequences:
    # Calculate new seq scores from log probabilities.
    new_scores = topk_log_probs / brevity_penalty(alpha, cur_index + 1)
    # Mask out the still unfinished sequences by adding large negative value.
    # --> [batch, 2*beams]
    new_scores += (~newly_finished) * NEG_INF

    # Combine sequences, scores, and flags along the beam dimension and compare
    # new finished sequence scores to existing finished scores and select the
    # best from the new set of beams.
    finished_seqs = torch.cat(  # --> [batch, 3*beams, length]
        [state.finished_seqs, topk_seq], dim=1)
    finished_scores = torch.cat(  # --> [batch, 3*beams]
        [state.finished_scores, new_scores], dim=1)
    finished_flags = torch.cat(  # --> [batch, 3*beams]
        [state.finished_flags, newly_finished], dim=1)
    # --> [batch, beams, length], [batch, beams], [batch, beams]
    top_finished_seq, top_finished_scores, top_finished_flags = (
        gather_topk_beams([finished_seqs, finished_scores, finished_flags],
                          finished_scores, batch_size, beam_size))

    return BeamState(
        cur_index=cur_index + 1,
        live_logprobs=top_alive_log_probs,
        finished_scores=top_finished_scores,
        live_seqs=top_alive_seq,
        finished_seqs=top_finished_seq,
        finished_flags=top_finished_flags,
        cache=top_alive_cache)

  state = beam_search_init_state
  while beam_search_loop_cond_fn(state):
    state = beam_search_loop_body_fn(state)
  final_state = state

  # Account for the edge-case where there are no finished sequences for a
  # particular batch item. If so, return live sequences for that batch item.
  # --> [batch]
  none_finished = torch.any(final_state.finished_flags, dim=1)
  # --> [batch, beams, length]
  finished_seqs = torch.where(none_finished[:, None, None],
                              final_state.finished_seqs,
                              final_state.live_seqs)
  # --> [batch, beams]
  finished_scores = torch.where(none_finished[:, None],
                                final_state.finished_scores,
                                final_state.live_logprobs)

  return finished_seqs, finished_scores
