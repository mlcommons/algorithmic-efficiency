"""DeepSpeech Models modified from https://github.com/lsari/librispeech_100/blob/main/models.py."""

import collections
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

supported_rnns = {"lstm": nn.LSTM, "rnn": nn.RNN, "gru": nn.GRU}


class SequenceWise(nn.Module):

  def __init__(self, module):
    """Collapses input of dim T*N*H to (T*N)*H, and applies to a module.

    Allows handling of variable sequence lengths and minibatch sizes.
    Args:
      module: Module to apply input to.
    """
    super(SequenceWise, self).__init__()
    self.module = module

  def forward(self, x):
    t, n = x.size(0), x.size(1)
    x = x.view(t * n, -1)
    x = self.module(x)
    x = x.view(t, n, -1)
    return x


class MaskConv(nn.Module):

  def __init__(self, seq_module):
    """Adds padding to the output of the module based on the given lengths.

    This is to ensure that the results of the model do not change when batch
    sizes change during inference. Input needs to be in the shape of (BxCxDxT)
    Args:
      seq_module: The sequential module containing the conv stack.
    """
    super(MaskConv, self).__init__()
    self.seq_module = seq_module

  def forward(self, x, lengths):
    """Forward pass.

    Args:
      x: The input of size BxCxDxT
      lengths: The actual length of each sequence in the batch

    Returns:
      Masked output from the module
    """
    for module in self.seq_module:
      x = module(x)
      mask = torch.BoolTensor(x.size()).fill_(0)
      if x.is_cuda:
        mask = mask.cuda()
      for i, length in enumerate(lengths):
        length = length.item()
        if (mask[i].size(2) - length) > 0:
          mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
      x = x.masked_fill(mask, 0)
    return x, lengths


class InferenceBatchSoftmax(nn.Module):

  def forward(self, x):
    if not self.training:
      return F.softmax(x, dim=-1)
    else:
      return x


class BatchRNN(nn.Module):

  def __init__(self,
               input_size,
               hidden_size,
               rnn_type=nn.LSTM,
               batch_norm=True):
    super(BatchRNN, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.batch_norm = SequenceWise(
        nn.BatchNorm1d(input_size)) if batch_norm else None
    self.rnn = rnn_type(
        input_size=input_size,
        hidden_size=hidden_size,
        bidirectional=True,
        bias=True,
        batch_first=True)
    self.num_directions = 2

  def flatten_parameters(self):
    self.rnn.flatten_parameters()

  def forward(self, x, output_lengths):
    self.flatten_parameters()
    if self.batch_norm is not None:
      x = self.batch_norm(x)
    x = x.transpose(0, 1)
    total_length = x.size(1)
    x = nn.utils.rnn.pack_padded_sequence(
        x, output_lengths.cpu(), batch_first=True)
    x, _ = self.rnn(x)
    x, _ = nn.utils.rnn.pad_packed_sequence(
        x, batch_first=True, total_length=total_length)
    x = x.transpose(0, 1)
    x = x.view(x.size(0), x.size(1), 2,
               -1).sum(2).view(x.size(0), x.size(1),
                               -1)  # (TxNxH*2) -> (TxNxH) by sum
    return x


class Lookahead(nn.Module):
  """Ref: Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks.

  input shape - sequence, batch, feature - TxNxH
  output shape - same as input
  """

  def __init__(self, n_features, context):
    super(Lookahead, self).__init__()
    assert context > 0
    self.context = context
    self.n_features = n_features
    self.pad = (0, self.context - 1)
    self.conv = nn.Conv1d(
        self.n_features,
        self.n_features,
        kernel_size=self.context,
        stride=1,
        groups=self.n_features,
        padding=0,
        bias=None)

  def forward(self, x):
    x = x.transpose(0, 1).transpose(1, 2)
    x = F.pad(x, pad=self.pad, value=0)
    x = self.conv(x)
    x = x.transpose(1, 2).transpose(0, 1).contiguous()
    return x


class CNNLSTM(nn.Module):

  def __init__(self):
    super(CNNLSTM, self).__init__()

    self.num_classes = 29
    self.hidden_size = 768
    self.hidden_layers = 5
    self.rnn_type = "lstm"
    self.context = 20

    self.conv = MaskConv(
        nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)))

    rnn_input_size = 161
    rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
    rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
    rnn_input_size *= 32

    rnns = []
    rnn = BatchRNN(
        input_size=rnn_input_size,
        hidden_size=self.hidden_size,
        rnn_type=supported_rnns[self.rnn_type],
        batch_norm=False)
    rnns.append(("0", rnn))
    for x in range(self.hidden_layers - 1):
      rnn = BatchRNN(
          input_size=self.hidden_size,
          hidden_size=self.hidden_size,
          rnn_type=supported_rnns[self.rnn_type])
      rnns.append(("%d" % (x + 1), rnn))
    self.rnns = nn.Sequential(collections.OrderedDict(rnns))
    self.lookahead = None

    fully_connected = nn.Sequential(
        nn.BatchNorm1d(self.hidden_size),
        nn.Linear(self.hidden_size, self.num_classes, bias=False))
    self.fc = nn.Sequential(SequenceWise(fully_connected),)

    self.inference_softmax = InferenceBatchSoftmax()
    self.pad_token = 0

  def get_seq_lens(self, input_length):
    """Get a 1D tensor or variable containing the size sequences that will be output by the network.

    Args:
      input_length: 1D Tensor

    Returns:
      1D Tensor scaled by model
    """
    seq_len = input_length
    for m in self.conv.modules():
      if isinstance(m, nn.modules.conv.Conv2d):
        seq_len = torch.div(
            seq_len + 2 * m.padding[1] - m.dilation[1] *
            (m.kernel_size[1] - 1) - 1,
            m.stride[1],
            rounding_mode="trunc") + 1
    return seq_len.int()

  def forward(self, x, lengths, transcripts):
    lengths = lengths.int()
    output_lengths = self.get_seq_lens(lengths)
    x, _ = self.conv(x, output_lengths)

    sizes = x.size()
    x = x.view(sizes[0], sizes[1] * sizes[2],
               sizes[3])  # Collapse feature dimension
    x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

    for rnn in self.rnns:
      x = rnn(x, output_lengths)

    x = self.fc(x)
    log_probs = x.log_softmax(dim=-1).transpose(0, 1)

    return log_probs, output_lengths

  def eval_loss(self, dataset, loss_fn, device):
    self.eval()

    total_loss = 0.0
    total_count = 0.0
    for (_, features, transcripts, input_lengths) in dataset:
      features = features.float().to(device)
      features = features.transpose(1, 2).unsqueeze(1)
      transcripts = transcripts.long().to(device)
      input_lengths = input_lengths.long().to(device)

      log_y, output_lengths = self(features, input_lengths, transcripts)
      target_lengths = torch.IntTensor([len(y[y != 0]) for y in transcripts])
      batch_loss = loss_fn(
          log_y.transpose(0, 1), transcripts, output_lengths, target_lengths)

      total_loss += torch.sum(batch_loss).data
      total_count += features.size(0)

    return total_loss / total_count

