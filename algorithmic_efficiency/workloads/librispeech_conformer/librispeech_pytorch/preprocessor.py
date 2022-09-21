"""
This is a pytorch implementation mirroring:
https://github.com/google/init2winit/blob/master/init2winit/model_lib/librispeech_preprocessor.py
"""
from collections import namedtuple
import math
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# mel spectrum constants.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

LIBRISPEECH_MEAN_VECTOR = [
    -7.6047816276550293,
    -7.1206226348876953,
    -6.8864245414733887,
    -6.8705768585205078,
    -6.9667720794677734,
    -7.1084094047546387,
    -6.9528026580810547,
    -6.783994197845459,
    -6.6195521354675293,
    -6.4876265525817871,
    -6.4120659828186035,
    -6.394047737121582,
    -6.4244871139526367,
    -6.3993711471557617,
    -6.5158271789550781,
    -6.7137999534606934,
    -6.8476877212524414,
    -6.9885001182556152,
    -6.9221386909484863,
    -7.146148681640625,
    -7.2040400505065918,
    -7.0537552833557129,
    -7.3140382766723633,
    -7.1223249435424805,
    -7.30251407623291,
    -7.1212143898010254,
    -7.2425732612609863,
    -7.1730537414550781,
    -7.0979413986206055,
    -7.088747501373291,
    -6.9849910736083984,
    -6.8787732124328613,
    -6.7602753639221191,
    -6.6300945281982422,
    -6.5145769119262695,
    -6.4245057106018066,
    -6.356513500213623,
    -6.31787633895874,
    -6.2660770416259766,
    -6.2468328475952148,
    -6.2821526527404785,
    -6.1908388137817383,
    -6.2484354972839355,
    -6.1472640037536621,
    -6.0924725532531738,
    -6.0171003341674805,
    -5.9250402450561523,
    -5.8535833358764648,
    -5.8209109306335449,
    -5.8118929862976074,
    -5.80783748626709,
    -5.7714629173278809,
    -5.7453732490539551,
    -5.7705655097961426,
    -5.7765641212463379,
    -5.7831673622131348,
    -5.7954087257385254,
    -5.7994823455810547,
    -5.8023476600646973,
    -5.8047118186950684,
    -5.8168182373046875,
    -5.8844799995422363,
    -5.9727106094360352,
    -6.0444660186767578,
    -6.1284866333007812,
    -6.2257585525512695,
    -6.3157496452331543,
    -6.39061164855957,
    -6.4928598403930664,
    -6.5498456954956055,
    -6.6054320335388184,
    -6.6508378982543945,
    -6.66917610168457,
    -6.6726889610290527,
    -6.684234619140625,
    -6.6974577903747559,
    -6.75471830368042,
    -6.7949142456054688,
    -6.8634209632873535,
    -6.94186544418335
]
LIBRISPEECH_STD_VECTOR = [
    3.4353282451629639,
    3.5962932109832764,
    3.7012472152709961,
    3.7369205951690674,
    3.7535104751586914,
    3.693629264831543,
    3.6922497749328613,
    3.7641522884368896,
    3.8419716358184814,
    3.8999848365783691,
    3.9294240474700928,
    3.9317409992218018,
    3.9139585494995117,
    3.9031598567962646,
    3.8691999912261963,
    3.8155081272125244,
    3.7644970417022705,
    3.7099106311798096,
    3.6965086460113525,
    3.6003766059875488,
    3.5493226051330566,
    3.5465121269226074,
    3.45003604888916,
    3.4712812900543213,
    3.4084610939025879,
    3.4408135414123535,
    3.4104881286621094,
    3.4217638969421387,
    3.4312851428985596,
    3.4199209213256836,
    3.4305806159973145,
    3.4382665157318115,
    3.4580366611480713,
    3.4817991256713867,
    3.4958710670471191,
    3.5036792755126953,
    3.5047574043273926,
    3.4988734722137451,
    3.493056058883667,
    3.4822943210601807,
    3.459430456161499,
    3.4612770080566406,
    3.4559063911437988,
    3.4755423069000244,
    3.4971549510955811,
    3.5326557159423828,
    3.5705199241638184,
    3.5920312404632568,
    3.596907377243042,
    3.5913500785827637,
    3.5865931510925293,
    3.5826809406280518,
    3.5837743282318115,
    3.5895791053771973,
    3.5819313526153564,
    3.5837869644165039,
    3.5861184597015381,
    3.5889589786529541,
    3.592214822769165,
    3.5939455032348633,
    3.5856630802154541,
    3.5884113311767578,
    3.5921022891998291,
    3.5870490074157715,
    3.5806570053100586,
    3.5731067657470703,
    3.5617532730102539,
    3.54980731010437,
    3.5527374744415283,
    3.5475366115570068,
    3.5387849807739258,
    3.5256178379058838,
    3.5031836032867432,
    3.4922726154327393,
    3.4879646301269531,
    3.4725594520568848,
    3.4558389186859131,
    3.4351828098297119,
    3.4284293651580811,
    3.4299170970916748
]

PreprocessorConfig = namedtuple(
    'PreprocessorConfig',
    field_names=[
        'sample_rate',
        'frame_size_ms',
        'frame_step_ms',
        'compute_energy',
        'window_fn',
        'output_log_floor',
        'pad_end',
        'preemph',
        'preemph_htk_flavor',
        'noise_scale',
        'num_bins',
        'lower_edge_hertz',
        'upper_edge_hertz',
        'fft_overdrive',
        'output_floor'
    ])


def _hertz_to_mel(frequencies_hertz):
  """Convert hertz to mel."""
  log_fn = math.log if type(frequencies_hertz) in [type(0.0), type(0)
                                                  ] else torch.log
  return _MEL_HIGH_FREQUENCY_Q * log_fn(1.0 + (frequencies_hertz /
                                               _MEL_BREAK_FREQUENCY_HERTZ))


def _pad_end_length(num_timesteps, frame_step, frame_size):
  """Returns how many sample needed to be padded for pad_end feature."""
  # The number of frames that can be extracted from the signal.
  num_frames = int(np.ceil(num_timesteps / frame_step))
  # Signal length required for computing `num_frames` frames.
  padded_length = frame_step * (num_frames - 1) + frame_size
  return padded_length - num_timesteps


def frame(x,
          frame_length: int,
          frame_step: int,
          pad_end: bool = False,
          pad_value: Union[int, float] = 0.0):
  """Slides a window and extract values.
      This function extracts `x[:, n:n+frame_length, :]` with sliding `n` with
      stride of `frame_step`, and returns an array `y` with the shape
      `(batch_size, num_frames, frame_length, num_channels)`. Unlike the
      counterpart in Tensorflow (`tf.signal.frame`), this function currently does
      not take `axis` argument, and the input tensor `x` is expected to have a
      shape of `(batch_size, timesteps, channels)`.
      Args:
        x: An input array with `(batch_size, timesteps, channels)`-shape.
        frame_length: The frame length.
        frame_step: The frame hop size.
        pad_end: If True, the end of signal is padded so the window can continue
          sliding while the starting point of the window is in the valid range.
        pad_value: A scalar used as a padding value when `pad_end` is True.
      Returns:
        A tensor with shape `(*, num_frames, frame_length, num_channels)`.
      """
  num_timesteps = x.shape[1]

  if pad_end:
    num_extends = _pad_end_length(num_timesteps, frame_step, frame_length)
    x = F.pad(x, (0, 0, 0, num_extends), mode='constant', value=pad_value)

  x = x.unfold(dimension=1, size=frame_length, step=frame_step)
  return x.permute(0, 1, 3, 2)


def linear_to_mel_weight_matrix(num_mel_bins: int = 20,
                                num_spectrogram_bins: int = 129,
                                sample_rate: Union[int, float] = 8000,
                                lower_edge_hertz: Union[int, float] = 125.0,
                                upper_edge_hertz: Union[int, float] = 3800.0,
                                dtype: Any = torch.float32,
                                device='cpu'):
  r"""Pytorch-port of `tf.signal.linear_to_mel_weight_matrix`.
      Args:
        num_mel_bins: Python int. How many bands in the resulting mel spectrum.
        num_spectrogram_bins: An integer `Tensor`. How many bins there are in the
          source spectrogram data, which is understood to be `fft_size // 2 + 1`,
          i.e. the spectrogram only contains the nonredundant FFT bins.
        sample_rate: An integer or float `Tensor`. Samples per second of the input
         signal used to create the spectrogram. Used to figure out the frequencies
         corresponding to each spectrogram bin, which dictates how they are mapped
         into the mel scale.
        lower_edge_hertz: Python float. Lower bound on the frequencies to be
          included in the mel spectrum. This corresponds to the lower edge of the
          lowest triangular band.
        upper_edge_hertz: Python float. The desired top edge of the highest
          frequency band.
        dtype: The `DType` of the result matrix. Must be a floating point type.
      Returns:
        An array of shape `[num_spectrogram_bins, num_mel_bins]`.
      Raises:
        ValueError: If `num_mel_bins`/`num_spectrogram_bins`/`sample_rate` are not
         positive, `lower_edge_hertz` is negative, frequency edges are incorrectly
         ordered, `upper_edge_hertz` is larger than the Nyquist frequency.
      [mel]: https://en.wikipedia.org/wiki/Mel_scale
      """

  # Input validator from tensorflow/python/ops/signal/mel_ops.py#L71
  if num_mel_bins <= 0:
    raise ValueError('num_mel_bins must be positive. Got: %s' % num_mel_bins)
  if lower_edge_hertz < 0.0:
    raise ValueError('lower_edge_hertz must be non-negative. Got: %s' %
                     lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError('lower_edge_hertz %.1f >= upper_edge_hertz %.1f' %
                     (lower_edge_hertz, upper_edge_hertz))
  if sample_rate <= 0.0:
    raise ValueError('sample_rate must be positive. Got: %s' % sample_rate)
  if upper_edge_hertz > sample_rate / 2:
    raise ValueError('upper_edge_hertz must not be larger than the Nyquist '
                     'frequency (sample_rate / 2). Got %s for sample_rate: %s' %
                     (upper_edge_hertz, sample_rate))

  # HTK excludes the spectrogram DC bin.
  bands_to_zero = 1
  nyquist_hertz = sample_rate / 2.0
  linear_frequencies = torch.linspace(
      0.0, nyquist_hertz, num_spectrogram_bins, dtype=dtype,
      device=device)[bands_to_zero:]
  spectrogram_bins_mel = _hertz_to_mel(linear_frequencies)[:, None]

  # Compute num_mel_bins triples of (lower_edge, center, upper_edge). The
  # center of each band is the lower and upper edge of the adjacent bands.
  # Accordingly, we divide [lower_edge_hertz, upper_edge_hertz] into
  # num_mel_bins + 2 pieces.
  edges = torch.linspace(
      _hertz_to_mel(lower_edge_hertz),
      _hertz_to_mel(upper_edge_hertz),
      num_mel_bins + 2,
      dtype=dtype,
      device=device)

  # Split the triples up and reshape them into [1, num_mel_bins] tensors.
  lower_edge_mel = edges[:-2][None, :]
  center_mel = edges[1:-1][None, :]
  upper_edge_mel = edges[2:][None, :]

  # Calculate lower and upper slopes for every spectrogram bin.
  # Line segments are linear in the mel domain, not Hertz.
  lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
      center_mel - lower_edge_mel)
  upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
      upper_edge_mel - center_mel)

  # Intersect the line segments with each other and zero.
  mel_weights_matrix = torch.minimum(lower_slopes, upper_slopes).clamp(
      min=0.0, max=None)

  # Re-add the zeroed lower bins we sliced out above.
  return F.pad(mel_weights_matrix, (0, 0, bands_to_zero, 0))


def _hanning_greco(win_support, frame_size, dtype, device='cpu'):
  """Add a greco-style hanning window to the graph.
      Note that the Hanning window in Wikipedia is not the same as the Hanning
      window in Greco.  The Greco3 Hanning window at 0 is NOT 0, as the wikipedia
      page would indicate. Talkin's explanation was that it was like wasting two
      samples to have the values at the edge of the window to be 0.0 exactly.
      Args:
        win_support: Number of samples for non-zero support in the window
        frame_size: Total size of the window (frame_size >= win_support)
        dtype: TF data type
      Returns:
        Tensor of size frame_size with the window to apply.
      """
  if frame_size < win_support:
    raise ValueError(
        'Provided frame_size = {} is lower than win_support = {}'.format(
            frame_size, win_support))

  arg = torch.pi * 2.0 / (win_support)
  hann = 0.5 - (0.5 * torch.cos(
      arg * (torch.arange(win_support, dtype=dtype, device=device) + 0.5)))
  zero_size = frame_size - win_support
  return F.pad(hann, (0, zero_size))


def _next_pow_of_two(x: Union[int, float]) -> int:
  return int(2**np.ceil(np.log2(x)))


class SpectrogramFrontend(nn.Module):
  """Layer to convert input audio signals from time domain to frequency domain.
      """

  def __init__(self,
               config: PreprocessorConfig = None,
               input_scale_factor: float = 1.0,
               output_log: bool = False,
               dtype=torch.float32,
               device='cpu'):
    super().__init__()

    self.config = config
    self.input_scale_factor = input_scale_factor
    self.output_log = output_log

    p = self.config
    self._frame_step = int(round(p.sample_rate * p.frame_step_ms / 1000.0))
    self._frame_size = int(round(
        p.sample_rate * p.frame_size_ms / 1000.0)) + 1  # +1 for the preemph

    # TF-version has maximum of 512, but it's not always necessary
    self.fft_size = _next_pow_of_two(self._frame_size)

    if p.window_fn is None:
      self._window_fn = None
    elif p.window_fn.upper() == 'HANNING':

      def _hanning_window(frame_size, dtype):
        # Preparing 1-point longer window to follow TF's definition
        if frame_size % 2 == 0:
          # simulate periodic=True in tf.signal.hann_window
          return torch.hann_window(
              window_length=frame_size,
              periodic=True,
              dtype=dtype,
              device=device)
        else:
          return torch.hann_window(
              window_length=frame_size,
              periodic=False,
              dtype=dtype,
              device=device)

      self._window_fn = _hanning_window
    elif p.window_fn.upper() == 'HANNING_GRECO':
      # Greco-compatible hanning window
      def f(frame_size, dtype):
        return _hanning_greco(
            self._frame_size - 1, frame_size, dtype, device=device)

      self._window_fn = f
    else:
      raise ValueError('Illegal value %r for window_fn param' % p.window_fn)
    if self._window_fn is None:
      self.window = None
    else:
      window = self._window_fn(self._frame_size - 1, dtype)
      window = window.reshape((1, 1, self._frame_size - 1, 1))
      self.register_buffer('window', window)

  def _apply_preemphasis(self, framed_signal):
    p = self.config
    if p.preemph_htk_flavor:
      return torch.cat([
          framed_signal[:, :, :1, :] * (1. - p.preemph),
          (framed_signal[:, :, 1:-1, :] -
           p.preemph * framed_signal[:, :, :-2, :])
      ],
                       dim=2)
    else:
      return (framed_signal[:, :, 1:, :] -
              p.preemph * framed_signal[:, :, :-1, :])

  def fprop_paddings(self, input_paddings):
    p = self.config
    if p.pad_end:
      num_extends = _pad_end_length(input_paddings.shape[1],
                                    self._frame_step,
                                    self._frame_size)
      input_paddings = F.pad(input_paddings, (0, num_extends), value=1.0)
    x = input_paddings.unfold(
        dimension=1, size=self._frame_size, step=self._frame_step)
    return x.min(dim=2)[0]

  def forward(self, inputs, input_paddings):
    p = self.config

    # Expand to have a channel axis
    if inputs.ndim == 2:
      inputs = inputs[:, :, None]
    output_paddings = None
    if input_paddings is not None:
      inputs = inputs * (1.0 - input_paddings[:, :, None])
      output_paddings = self.fprop_paddings(input_paddings)
    else:
      output_paddings = None

    pcm_audio_chunk = inputs * self.input_scale_factor

    framed_signal = frame(
        pcm_audio_chunk, self._frame_size, self._frame_step, pad_end=p.pad_end)

    if p.preemph != 0.0:
      preemphasized = self._apply_preemphasis(framed_signal)
    else:
      preemphasized = framed_signal[..., :-1, :]

    if p.noise_scale > 0.0:
      noise_signal = torch.randn_like(preemphasized) * p.noise_scale
    else:
      noise_signal = torch.zeros_like(preemphasized)

    windowed_signal = preemphasized + noise_signal
    # Window here
    if self.window is not None:
      windowed_signal *= self.window
    spectrum = torch.fft.rfft(windowed_signal, self.fft_size, dim=2)

    spectrum = torch.abs(spectrum)
    if p.compute_energy:
      spectrum = spectrum**2.0
    outputs = spectrum
    if self.output_log:
      outputs = torch.log(torch.maximum(outputs, p.output_log_floor))
    return outputs, output_paddings


class MelFilterbankFrontend(nn.Module):
  """Layer to compute log mel spectograms from input audio signals.
      """

  def __init__(self,
               config: PreprocessorConfig = None,
               use_divide_stream: bool = True,
               per_bin_mean: Optional[float] = None,
               per_bin_stddev: Optional[float] = None,
               device='cpu'):
    super().__init__()

    self.config = config
    self.use_divide_stream = use_divide_stream
    self.per_bin_mean = per_bin_mean
    self.per_bin_stddev = per_bin_stddev
    p = self.config

    input_scale_factor = 2**-15 if self.use_divide_stream else 1.0
    self.stft = SpectrogramFrontend(
        p, input_scale_factor=input_scale_factor, output_log=False)

    if self.per_bin_mean is None:
      per_bin_mean = [0.0] * p.num_bins
    else:
      per_bin_mean = self.per_bin_mean

    if self.per_bin_stddev is None:
      per_bin_stddev = [1.0] * p.num_bins
    else:
      per_bin_stddev = self.per_bin_stddev

    self.register_buffer('_normalizer_mean',
                         torch.FloatTensor(per_bin_mean)[None, None, :, None])
    self.register_buffer('_normalizer_stddev',
                         torch.FloatTensor(per_bin_stddev)[None, None, :, None])

  def forward(self, inputs, input_paddings):
    p = self.config

    spect, spect_paddings = self.stft(inputs, input_paddings)

    mel_weights = linear_to_mel_weight_matrix(
        num_mel_bins=p.num_bins,
        num_spectrogram_bins=spect.shape[2],
        sample_rate=p.sample_rate,
        lower_edge_hertz=p.lower_edge_hertz,
        upper_edge_hertz=p.upper_edge_hertz,
        device=spect.device)

    mel_spectrogram = torch.einsum('fn,btfc->btnc', mel_weights, spect)
    logmel_spectrogram = torch.log(
        mel_spectrogram.clamp(min=p.output_floor, max=None))

    normalized_logmel_spectrogram = (
        (logmel_spectrogram - self._normalizer_mean) / self._normalizer_stddev)

    normalized_logmel_spectrogram = torch.squeeze(normalized_logmel_spectrogram,
                                                  -1)
    return normalized_logmel_spectrogram, spect_paddings
