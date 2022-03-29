import math
from typing import Any, Optional, Union, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_


# Mask making utilities ported to PyTorch from
# https://github.com/google/flax/blob/main/flax/linen/attention.py
def make_attention_mask(query_input: Tensor,
                        key_input: Tensor,
                        pairwise_fn: Callable[..., Any] = torch.mul,
                        dtype: torch.dtype = torch.float32) -> Tensor:
  """Mask-making helper for attention weights.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton
      axes for, none by default
    dtype: mask return dtype

  Returns:
    A `[batch..., len_q, len_kv]` shaped attention mask.
  """
  mask = pairwise_fn(query_input.unsqueeze(-1), key_input.unsqueeze(-2))
  return mask.to(dtype)


def make_causal_mask(x: Tensor,
                     device: str = 'cuda:0',
                     dtype: torch.dtype = torch.float32) -> Tensor:
  """Make a causal mask for self-attention.

  Args:
    x: input array of shape `[batch..., len]`
    dtype: mask return dtype

  Returns:
    A `[batch..., len, len]` shaped causal attention mask.
  """
  idxs = torch.broadcast_to(
      torch.arange(x.shape[-1], dtype=torch.int32, device=device), x.shape)
  return make_attention_mask(idxs, idxs, torch.greater_equal, dtype=dtype)


def make_src_mask(src, inputs_segmentation, nhead):
  '''Utility for creating source mask and adjust it for PyTorch Transformer API.'''
  src_mask = make_attention_mask(src > 0, src > 0)
  # Add segmentation block-diagonal attention mask if using segmented data.
  if inputs_segmentation is not None:
    src_mask = torch.logical_and(
        src_mask,
        make_attention_mask(
            inputs_segmentation,
            inputs_segmentation,
            torch.eq))
  # Flip values and ensure numerical stability.
  src_mask = torch.repeat_interleave(
      torch.logical_not(src_mask), repeats=nhead, dim=0)
  new_src_mask = torch.zeros_like(src_mask, dtype=torch.float32)
  new_src_mask.masked_fill_(src_mask, -1e10)
  return new_src_mask


def make_tgt_and_memory_mask(tgt, src, inputs_segmentation,
                             targets_segmentation, decode, nhead):
  '''Utility for creating target and memory mask and adjust them for PyTorch Transformer API.'''
  if not decode:
    tgt_mask = torch.logical_and(
        make_attention_mask(tgt > 0, tgt > 0),
        make_causal_mask(tgt, device=tgt.device))
    memory_mask = make_attention_mask(tgt > 0, src > 0)
  else:
    tgt_mask = None
    memory_mask = make_attention_mask(
        torch.ones_like(tgt) > 0, src > 0)
  # Add segmentation block-diagonal attention masks if using segmented data.
  if inputs_segmentation is not None:
    tgt_mask = torch.logical_and(
        tgt_mask,
        make_attention_mask(
            targets_segmentation,
            targets_segmentation,
            torch.eq))
    memory_mask = torch.logical_and(
        memory_mask,
        make_attention_mask(
            targets_segmentation,
            inputs_segmentation,
            torch.eq))
  # Flip values and ensure numerical stability.
  memory_mask = torch.repeat_interleave(
      torch.logical_not(memory_mask), repeats=nhead, dim=0)
  new_memory_mask = torch.zeros_like(memory_mask, dtype=torch.float32)
  new_memory_mask.masked_fill_(memory_mask, -1e10)
  if tgt_mask is not None:
    tgt_mask = torch.repeat_interleave(
        torch.logical_not(tgt_mask), repeats=nhead, dim=0)
    new_tgt_mask = torch.zeros_like(tgt_mask, dtype=torch.float32)
    new_tgt_mask.masked_fill_(tgt_mask, -1e10)
    tgt_mask = new_tgt_mask
  return tgt_mask, new_memory_mask


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  pad_widths = tuple(t for tup in reversed(pad_widths) for t in tup)
  padded = F.pad(
      x, pad_widths, mode='constant')
  return padded[:, :-1]


class Transformer(nn.Module):
  '''
  Transformer architecture based on the model from the WMT Jax workload.
  '''

  def __init__(self,
               ntoken: int,
               d_model: int,
               nhead: int,
               d_hid: int,
               nlayers: int,
               dropout: float = 0.1,
               layer_norm_eps: float = 1e-6):
    super().__init__()
    self.d_model = d_model
    self.nhead = nhead
    self.pos_encoder = PositionalEncoding(d_model, dropout)
    self.shared_embedding = nn.Embedding(ntoken, d_model)

    encoder_layers = TransformerEncoderLayer(
        d_model,
        nhead,
        d_hid,
        dropout,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=True)
    encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    self.encoder = nn.TransformerEncoder(encoder_layers, nlayers, encoder_norm)

    decoder_layers = TransformerDecoderLayer(
        d_model,
        nhead,
        d_hid,
        dropout,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=True)
    decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    self.decoder = nn.TransformerDecoder(decoder_layers, nlayers, decoder_norm)

    self._reset_parameters()

  def _reset_parameters(self):
    """Initiate parameters in the transformer model."""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        xavier_uniform_(module.weight)
        if module.bias is not None:
          normal_(module.bias, std=1e-6)

  def encode(self,
             src: Tensor,
             inputs_positions: Optional[Tensor] = None,
             inputs_segmentation: Optional[Tensor] = None) -> Tensor:
    src = src.to(torch.int)
    src_mask = make_src_mask(src, inputs_segmentation, self.nhead)
    src = self.shared_embedding(src)
    src = self.pos_encoder(src, inputs_positions)
    memory = self.encoder(src, mask=src_mask)
    return memory

  def decode(self, 
             tgt: Tensor,
             memory: Tensor,
             src: Tensor,  # just for calculating the padding mask
             targets_positions: Optional[Tensor] = None,
             inputs_segmentation: Optional[Tensor] = None,
             targets_segmentation: Optional[Tensor] = None,
             decode: bool = False) -> Tensor:
    tgt = tgt.to(torch.int)
    tgt_mask, memory_mask = make_tgt_and_memory_mask(
        tgt, src, inputs_segmentation, targets_segmentation,
        decode, self.nhead)
    if not decode:
      tgt = shift_right(tgt)
    tgt = self.shared_embedding(tgt)
    tgt = self.pos_encoder(tgt, targets_positions)
    output = self.decoder(
        tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
    normalize = math.sqrt(output.shape[-1])
    output = torch.matmul(output, self.shared_embedding.weight.T) / normalize
    return output

  def forward(self,
              src: Tensor,
              tgt: Tensor,
              inputs_positions: Optional[Tensor] = None,
              targets_positions: Optional[Tensor] = None,
              inputs_segmentation: Optional[Tensor] = None,
              targets_segmentation: Optional[Tensor] = None,
              decode: bool = False) -> Tensor:
    """
    Args:
      src: Tensor, shape [batch_size, seq_len]
      tgt: Tensor, shape [batch_size, seq_len]
      inputs_positions: Optional[Tensor], shape [batch_size, seq_len],
      targets_positions: Optional[Tensor], shape [batch_size, seq_len],
      inputs_segmentation: Optional[Tensor], shape [batch_size, seq_len],
      targets_segmentation: Optional[Tensor], shape [batch_size, seq_len],
      decode: bool

    Returns:
      output Tensor of shape [batch_size, seq_len, ntoken]
    """
    if src.size(0) != tgt.size(0):
      raise RuntimeError("the batch number of src and tgt must be equal")
    memory = self.encode(
        src,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)
    output = self.decode(
        tgt,
        memory,
        src,  # just for calculating the padding mask
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        decode=decode)
    return output


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    scale_factor = -math.log(10000.0) / (d_model // 2 - 1)
    div_term = torch.exp(torch.arange(d_model // 2) *  scale_factor)
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, :d_model // 2] = torch.sin(position * div_term)
    pe[0, :, d_model // 2:2 * (d_model // 2)] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor, inputs_positions: Tensor) -> Tensor:
    """
    Args:
      x: Tensor, shape [batch_size, seq_len, embedding_dim]
    """
    if inputs_positions is not None:
      x = x + self.pe[0, inputs_positions, :]
    else:
      x = x + self.pe[:, :x.size(1), :]
    return self.dropout(x)


# TransformerEncoderLayer and TransformerDecoderLayer are taken from
# https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py,
# only difference is using custom MultiheadAttention modules without bias and
# '_qkv_same_embed_dim' always set to 'False'.
class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                         norm_first=norm_first, device=device, dtype=dtype)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=False, **factory_kwargs)


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectivaly. Otherwise it's done after.
            Default: ``False`` (after).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first,
                         norm_first=norm_first, device=device, dtype=dtype)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=False, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=False, **factory_kwargs)


# Only difference to standard PyTorch class is that 'self._qkv_same_embed_dim' is always set to 'False'.
class MultiheadAttention(nn.MultiheadAttention):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super().__init__(embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
                         add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first,
                         device=device, dtype=dtype)
        # This is set to 'True' for kdim == vdim == embed_dim in the standard PyTorch class.
        self._qkv_same_embed_dim = False

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = nn.Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = nn.Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)

        self._reset_parameters()
