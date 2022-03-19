'''
Transformer architecture based on https://pytorch.org/tutorials/beginner/transformer_tutorial.html
and https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html.
'''

import math
from typing import Optional, Union, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn.init import normal_
from torch.nn.init import xavier_uniform_


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  pad_widths = tuple(t for tup in reversed(pad_widths) for t in tup)
  padded = F.pad(
      x, pad_widths, mode='constant')
  return padded[:, :-1]


class Transformer(nn.Module):

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
             inputs_segmentation: Optional[Tensor] = None,
             src_mask: Optional[Tensor] = None) -> Tensor:
    src = src.to(torch.int)
    src_key_padding_mask = src == 0
    src = self.shared_embedding(src)
    src = self.pos_encoder(src, inputs_positions)
    memory = self.encoder(
        src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    return memory

  def decode(self, 
             tgt: Tensor,
             memory: Tensor,
             src: Tensor,  # just for calculating the padding mask
             targets_positions: Optional[Tensor] = None,
             inputs_segmentation: Optional[Tensor] = None,
             targets_segmentation: Optional[Tensor] = None,
             tgt_mask: Optional[Tensor] = None,
             memory_mask: Optional[Tensor] = None,
             decode: bool = False) -> Tensor:
    tgt = tgt.to(torch.int)
    if not decode:
      tgt_key_padding_mask = tgt == 0
      tgt = shift_right(tgt)
      if tgt_mask is None:
        sz = tgt.shape[-1]
        tgt_mask = torch.triu(
            torch.full((sz, sz), float('-inf'), device=tgt.device), diagonal=1)
    else:
      tgt_key_padding_mask = None
      tgt_mask = None
    tgt = self.shared_embedding(tgt)
    tgt = self.pos_encoder(tgt, targets_positions)
    memory_key_padding_mask = src == 0
    output = self.decoder(
        tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask)
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
              src_mask: Optional[Tensor] = None,
              tgt_mask: Optional[Tensor] = None,
              memory_mask: Optional[Tensor] = None,
              decode: bool = False) -> Tensor:
    """
    Args:
      src: Tensor, shape [batch_size, seq_len]
      tgt: Tensor, shape [batch_size, seq_len]
      src_mask: Tensor, shape [seq_len, seq_len]
      tgt_mask: Tensor, shape [seq_len, seq_len]
      memory_mask: Tensor, shape [seq_len, seq_len]
      decode: bool

    Returns:
      output Tensor of shape [batch_size, seq_len, ntoken]
    """
    if src.size(0) != tgt.size(0):
      raise RuntimeError("the batch number of src and tgt must be equal")
    memory = self.encode(
        src,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
        src_mask=src_mask)
    output = self.decode(
        tgt,
        memory,
        src,  # just for calculating the padding mask
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        decode=decode)
    return output


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
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


# TransformerEncoderLayer and TransformerDecoderLayer are taken from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
# only difference is not using bias parameters for MultiheadAttention modules
class TransformerEncoderLayer(nn.Module):
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
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=False, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerDecoderLayer(nn.Module):
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
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=False, **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=False, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
