"""CTC loss.

Original code from Yotaro Kubo (yotaro@google.com).
"""

import jax
import jax.numpy as jnp
import numpy as np

_LOGEPSILON = -100000.0


def ctc_loss(logprobs: np.ndarray,
             logprobspaddings: np.ndarray,
             labels: np.ndarray,
             labelspaddings: np.ndarray,
             blank_id: int = 0):
  """Forward computation of CTC loss.

  This loss performs forward-backward algorithm over an FSA that has `N * 2`
  states where `N` is the max number of labels. States can be split into two
  groups: Phi states and emission states. a phi-state accepts repetition of phi
  (blank)-symbols and transits to emission state when the correct label is
  observed. An emission state accepts repetition of the label and transits to
  the next phi states at any time (so called epsilon-transition).

  In the comments in the code, `B` denotes the batch size, `T` denotes the time
  steps in `logprobs`, and `N` denotes the time steps in `labels`.

  Args:
    logprobs: (B, T, K)-Array containing log-probabilities of the target class.
    logprobspaddings: (B, T)-array. Padding array for `logprobs`.
    labels: (B, N)-array containing reference labels.
    labelspaddings: (B, N)-array. Paddings for `labels`. Currently `labels` must
      be right-padded, i.e. each row of labelspaddings must be repetition of
      zeroes, followed by repetition of ones. On the other hand, `logprobs` can
      have padded values at any position.
    blank_id: Id for blank token.

  Returns:
    A pair of `(per_seq_loss, aux)`.
    per_seq_loss:
      (B,)-array containing loss values for each sequence in the batch.
    aux: Dictionary containing interim variables used for computing losses.
  """
  batchsize, inputlen, unused_numclass = logprobs.shape
  batchsize_, maxlabellen = labels.shape
  assert batchsize == batchsize_
  labellens = maxlabellen - jnp.sum(labelspaddings, axis=1).astype(jnp.int32)

  logprobs_phi = logprobs[:, :, blank_id]  # [B, T]
  logprobs_phi = jnp.transpose(logprobs_phi, (1, 0))  # [T, B]

  indices = jnp.reshape(labels, (batchsize, 1, maxlabellen))  # [B, 1, N]
  # The following leads to unnecessarily large padding ops
  #   logprobs_emit = jnp.take_along_axis(logprobs,indices,axis=-1) # [B, T, N]
  # So, manually flatten and reshape before and after np.take_along_axis.
  indices = jnp.repeat(indices, inputlen, axis=1)
  logprobs = jnp.reshape(logprobs, (batchsize * inputlen, unused_numclass))
  indices = jnp.reshape(indices, (batchsize * inputlen, maxlabellen))
  logprobs_emit = jnp.take_along_axis(logprobs, indices, axis=1)  # [B*T, N]
  logprobs_emit = jnp.reshape(logprobs_emit, (batchsize, inputlen, maxlabellen))
  # ^ End workaround for issues on take_along_axis.

  logprobs_emit = jnp.transpose(logprobs_emit, (1, 0, 2))  # [T, B, N]

  logalpha_phi_init = jnp.ones((batchsize, maxlabellen)) * _LOGEPSILON  # [B, N]
  logalpha_phi_init = logalpha_phi_init.at[:, 0].set(0.0)
  logalpha_emit_init = jnp.ones(
      (batchsize, maxlabellen)) * _LOGEPSILON  # [B, N]

  def loop_body(prev, x):
    prev_phi, prev_emit = prev
    # emit-to-phi epsilon transition
    prev_phi = prev_phi.at[:, 1:].set(
        jnp.logaddexp(prev_phi[:, 1:], prev_emit[:, :-1]))

    logprob_emit, logprob_phi, pad = x

    # phi-to-emit transition
    next_emit = jnp.logaddexp(prev_phi + logprob_emit, prev_emit + logprob_emit)
    # self-loop transition
    next_phi = prev_phi + logprob_phi.reshape((batchsize, 1))

    pad = pad.reshape((batchsize, 1))
    next_emit = pad * prev_emit + (1.0 - pad) * next_emit
    next_phi = pad * prev_phi + (1.0 - pad) * next_phi

    return (next_phi, next_emit), (next_phi, next_emit)

  xs = (logprobs_emit, logprobs_phi, logprobspaddings.transpose((1, 0)))
  _, (logalpha_phi,
      logalpha_emit) = jax.lax.scan(loop_body,
                                    (logalpha_phi_init, logalpha_emit_init), xs)

  # extract per_seq_loss
  per_seq_loss = -jnp.take_along_axis(
      logalpha_emit[-1, :], labellens.reshape((batchsize, 1)) - 1, axis=1)

  return per_seq_loss, {
      'logalpha_phi': logalpha_phi,
      'logalpha_emit': logalpha_emit,
      'logprobs_phi': logprobs_phi,
      'logprobs_emit': logprobs_emit,
  }
