"""
Originally based on the plainLM codebase:
https://github.com/Niccolo-Ajroldi/plainLM
under the MIT license https://github.com/Niccolo-Ajroldi/plainLM/blob/main/LICENSE.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
  model_dim: int
  num_heads: int
  seq_len: int
  num_layers: int
  vocab_size: int
  expanded_model_dim: int
  multiple_of: int = 256
  rmsnorm_epsilon: float = 1e-6
  qknorm_epsilon: float = 1e-6
  use_residual_scaling: bool = True
  tie_embeddings: bool = True
  compute_dtype: torch.dtype = torch.bfloat16
  param_dtype: torch.dtype = torch.float32


class MLP(nn.Module):
  def __init__(
    self,
    dim: int,
    hidden_dim: int,
    multiple_of: int = 256,
    dtype: torch.dtype = torch.float32,
  ):
    super().__init__()
    hidden_dim = int(
      multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    )
    self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False, dtype=dtype)
    self.fc2 = nn.Linear(hidden_dim, dim, bias=False, dtype=dtype)
    self.glu = nn.GLU(dim=2)
    nn.init.normal_(self.fc1.weight, std=0.02)
    nn.init.normal_(self.fc2.weight, std=0.02)

  def forward(self, x):
    # x: (bsz, T, dim)
    return self.fc2(self.glu(self.fc1(x)))


def precompute_freqs_cis(
  dim: int, end: int, theta: float = 10000.0, condense_ratio: int = 1
):
  inv_freqs = 1.0 / (
    theta
    ** (
      torch.arange(0, dim, 2, dtype=torch.float32, device=torch.device('cpu'))
      / dim
    )
  )
  t = (
    torch.arange(end, dtype=torch.float32, device=inv_freqs.device)
    / condense_ratio
  )
  freqs = torch.outer(t, inv_freqs).float()
  return torch.stack(
    [torch.cos(freqs)[None, :, None, :], torch.sin(freqs)[None, :, None, :]],
    dim=4,
  )


def apply_rotary_emb_complex_like(
  q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  # Rotate query and key vectors using RoPE
  qk_r2 = torch.cat([q, k], dim=2).unflatten(dim=-1, sizes=(-1, 2)).float()
  rotated_qk_r2 = torch.stack(
    [
      qk_r2[..., 0] * freqs_cis[..., 0] - qk_r2[..., 1] * freqs_cis[..., 1],
      qk_r2[..., 1] * freqs_cis[..., 0] + qk_r2[..., 0] * freqs_cis[..., 1],
    ],
    -1,
  ).flatten(3)
  rotated_qk = rotated_qk_r2
  return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)


class Attention(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    assert cfg.model_dim % cfg.num_heads == 0
    self.dim = cfg.model_dim
    self.n_heads = cfg.num_heads
    self.head_dim = cfg.model_dim // cfg.num_heads

    self.w_qkv = nn.Linear(
      cfg.model_dim, 3 * cfg.model_dim, bias=False, dtype=cfg.param_dtype
    )
    self.w_out = nn.Linear(
      cfg.model_dim, cfg.model_dim, bias=False, dtype=cfg.param_dtype
    )
    # Split into Q, K, V sections
    wq, wk, wv = torch.chunk(self.w_qkv.weight, 3, dim=0)
    for w in [wq, wk, wv]:
      nn.init.normal_(w, std=0.02)
    nn.init.normal_(self.w_out.weight, std=0.02)

    self.eps = cfg.qknorm_epsilon  # e.g., 1e-6
    seq_len = cfg.seq_len
    attn_scale0 = math.log2(seq_len**2 - seq_len)
    self.attn_scale = nn.Parameter(
      torch.tensor(attn_scale0, dtype=cfg.param_dtype)
    )

  def forward(self, x, freqs_cis):
    bsz, seqlen, d = x.shape  # (bsz, seqlen, d)

    q, k, v = self.w_qkv(x).split(d, dim=2)  # (bsz, seqlen, d)
    q = q.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)
    k = k.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)
    v = v.view(
      bsz, seqlen, self.n_heads, self.head_dim
    )  # (bsz, seqlen, nh, h_dim)

    q, k = apply_rotary_emb_complex_like(
      q, k, freqs_cis=freqs_cis
    )  # (bsz, seqlen, nh, h_dim)

    q = q.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    k = k.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)
    v = v.transpose(1, 2)  # (bsz, nh, seqlen, h_dim)

    # Apply QK normalization
    q = q / torch.norm(q, dim=-1, keepdim=True) + self.eps
    k = k / torch.norm(k, dim=-1, keepdim=True) + self.eps
    q *= self.attn_scale

    out = F.scaled_dot_product_attention(
      q, k, v, is_causal=True, scale=1.0
    )  # (bsz, nh, seqlen, h_dim)
    out = (
      out.transpose(1, 2).contiguous().view(bsz, seqlen, d)
    )  # (bsz, seqlen, d)

    return self.w_out(out)


class Block(nn.Module):
  def __init__(self, layer_id: int, cfg: ModelConfig):
    super().__init__()
    self.attn = Attention(cfg)
    self.attn_norm = nn.RMSNorm(
      cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.param_dtype
    )
    self.mlp = MLP(
      dim=cfg.model_dim,
      hidden_dim=cfg.expanded_model_dim,
      multiple_of=cfg.multiple_of,
      dtype=cfg.param_dtype,
    )
    self.mlp_norm = nn.RMSNorm(
      cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.param_dtype
    )
    self.layer_id = layer_id

  def forward(self, x, freqs_cis):
    # x: (bsz, seqlen, dim)
    x = x + self.attn(self.attn_norm(x), freqs_cis)
    x = x + self.mlp(self.mlp_norm(x))
    return x


class Transformer(nn.Module):
  def __init__(self, cfg: ModelConfig):
    super().__init__()
    self.n_layers = cfg.num_layers
    self.cfg = cfg
    head_dim = cfg.model_dim // cfg.num_heads
    assert cfg.model_dim % cfg.num_heads == 0

    self.embed_tokens = nn.Embedding(
      cfg.vocab_size, cfg.model_dim, dtype=cfg.param_dtype
    )
    self.layers = nn.ModuleList(
      [Block(idx, cfg) for idx in range(cfg.num_layers)]
    )
    self.out_norm = nn.RMSNorm(
      cfg.model_dim, eps=cfg.rmsnorm_epsilon, dtype=cfg.param_dtype
    )
    self.lm_head = nn.Linear(
      cfg.model_dim, cfg.vocab_size, bias=False, dtype=cfg.param_dtype
    )

    # Initialize freqs_cis on CPU first (more memory efficient)
    self.register_buffer(
      'freqs_cis',
      precompute_freqs_cis(head_dim, cfg.seq_len, 500000)[0 : cfg.seq_len],
      persistent=False,
    )

    # init all weights, scale residual branches
    self.apply(self._init_weights)
    self._scale_residual_branches()

    # Move model to device (which will also move freqs_cis)
    if torch.cuda.is_available():
      self.cuda()

    if cfg.tie_embeddings:
      self.tie_weights()

  def forward(self, x, targets=None):
    # x: (bsz, seqlen)
    x = self.embed_tokens(x)  # (bsz, seqlen, dim)
    L = x.shape[1]

    # Make sure we have enough precomputed frequencies
    if L > self.freqs_cis.shape[1]:
      # Need to recompute for longer sequence
      head_dim = self.cfg.model_dim // self.cfg.num_heads
      new_freqs = precompute_freqs_cis(
        head_dim, max(L, self.cfg.seq_len), 500000
      )
      self.register_buffer(
        'freqs_cis', new_freqs[0 : max(L, self.cfg.seq_len)], persistent=False
      )
      if torch.cuda.is_available():
        self.freqs_cis = self.freqs_cis.cuda()

    # Select the frequencies for current sequence length and ensure correct device
    freqs_cis = self.freqs_cis[:, :L, :].to(x.device)

    for layer in self.layers:
      x = layer(x, freqs_cis)  # (bsz, seqlen, dim)
    out = self.lm_head(self.out_norm(x))  # (bsz, seqlen, vocab_size)

    if targets is not None:
      loss = F.cross_entropy(
        out.view(-1, out.size(-1)), targets.view(-1), ignore_index=-100
      )
      return out, loss
    return out

  def predict(self, x, k=1):
    """Generate k tokens autoregressively.

    Args:
        x: Input token sequence of shape (batch_size, seq_len)
        k: Number of tokens to predict

    Returns:
        Tuple of (input_ids, predicted_ids)
    """
    # Determine device type for autocast
    device_type = 'cuda' if x.is_cuda else 'cpu'

    with torch.autocast(device_type=device_type, dtype=self.cfg.compute_dtype):
      # Store original input
      original_input = x.clone()
      generated_input = x.clone()

      # Generate k tokens autoregressively
      for i in range(k):
        # Get logits for the entire sequence
        logits = self(generated_input)

        # Get the logits for the last token in each sequence
        next_token_logits = logits[:, -1, :]

        # Zero out the last token ID to prevent repetition
        # This is a common issue - the model gets stuck repeating the last token
        last_token_id = generated_input[:, -1]
        next_token_logits.scatter_(1, last_token_id.unsqueeze(1), float('-inf'))

        # Get the most likely token
        next_token = torch.argmax(next_token_logits, dim=-1)

        # Append the predicted token to the sequence
        next_token = next_token.unsqueeze(1)  # Add sequence dimension
        generated_input = torch.cat([generated_input, next_token], dim=1)

      # For debugging, print predictions for the first item in the batch
      print('\nPyTorch detailed prediction (first item in batch):')
      predicted_sequence = generated_input[0, -k:].tolist()
      print(f'  Predicted token IDs: {predicted_sequence}')
      for i, token_id in enumerate(predicted_sequence):
        print(f'  Step {i + 1}: Predicted token {token_id}')

      # Return all tokens, not just the last k
      return original_input, generated_input[:, -k:]

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, std=0.02)

  def _scale_residual_branches(self):
    for n, p in self.named_parameters():
      if n.endswith('fc2.weight') or n.endswith(
        'w_out.weight'
      ):  # mlp/glu output layer
        torch.nn.init.normal_(
          p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers)
        )

  def tie_weights(self):
    self.lm_head.weight = self.embed_tokens.weight

  def count_params(self, non_embedding=True):
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
      n_params -= self.embed_tokens.weight.numel()
      if (
        self.lm_head.weight is not self.embed_tokens.weight
      ):  # if no weight tying
        n_params -= self.lm_head.weight.numel()
    return n_params


def main():
  print('Initializing transformer model and running forward pass...')

  seq_length = 1024

  # Define model configuration
  config = ModelConfig(
    vocab_size=50257,  # Common vocab size for tokenizers like BPE or SentencePiece
    seq_len=seq_length,  # Maximum sequence length
    model_dim=1024,  # Embedding dimension
    expanded_model_dim=4.0,  # MLP expansion factor
    num_layers=12,  # Number of transformer layers
    num_heads=8,  # Number of attention heads
    rmsnorm_epsilon=1e-6,  # RMSNorm epsilon
    tie_embeddings=True,  # Tie embedding and output weights
  )

  # Instantiate the model
  model = Transformer(config)
  print(f'Model has {model.count_params():,} parameters.')
  for n, p in model.named_parameters():
    print(f'{n}.dtype == {p.dtype}')

  # Create some random input data
  batch_size = 2
  input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

  # Move data to the same device as the model
  if torch.cuda.is_available():
    input_ids = input_ids.cuda()

  # Run a forward pass
  print(f'Running forward pass with input shape: {input_ids.shape}')
  logits = model(input_ids)
  print(f'Output logits dtype: {logits.dtype}')
  print(f'Output logits shape: {logits.shape}')

  # Run prediction
  print('Running prediction...')
  original_input, predicted_ids = model.predict(input_ids[:, :10], k=5)
  print(f'Original input shape for prediction: {original_input.shape}')
  print(f'Predicted IDs shape: {predicted_ids.shape}')
  print(f'Predicted IDs: {predicted_ids}')


if __name__ == '__main__':
  main()
