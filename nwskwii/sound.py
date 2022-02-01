import numpy as np

class FeatureExtractor():
  """
  Attributes
  -------------

  Methods
  -------
  ComputeFBANK : メルフィルタバンクの計算
  ComputeMFCC : MFCCの計算
  """
  def __init__(self,
    sample_frequency=16000,
    frame_length=25,
    frame_shift=10,
    num_mel_bins=23,
    num_ceps=13,
    lifter_coef=22,
    low_frequency=20,
    high_frequency=8000,
    dither=1.0):
    """
    Parameters
    ---------------
    sample_frequency (int) : default=16000
      [Hz]
    frame_length (int) : default=25
      [ms]
    frame_shift (int) : default=10
      [ms]
    num_mel_bins (int) : default=23
      FBANK特徴の次元数
    num_ceps (int) : default=13
      MFCC特徴の次元数
    lifter_coef (int) : default=22
      リフタリング処理のパラメータ
    low_frequency : default=20
      [Hz]
    high_frequency : default=8000
      [Hz]
    dither : default=1
      ディザリングの強さ
    """

    self.sample_frequency = sample_frequency
    self.frame_size = int(sample_frequency * frame_length * 0.001)
    self.frame_shift = int(sample_frequency * frame_shift * 0.001)
    self.num_mel_bins = num_mel_bins
    self.num_ceps = num_ceps
    self.lifter_coef = lifter_coef
    self.low_frequency = low_frequency
    self.high_frequency = high_frequency
    self.dither = dither

    # calc fft size
    self.fft_size = 1
    while self.fft_size < self.frame_size:
      self.fft_size *= 2

    self.mel_filter_bank = self._MakeMelFilterBank()

    self.dct_matrix = self._MakeDCTMatrix()

    self.lifter = self._MakeLifter()

  def Herz2Mel(self, herz):
    return (1127.0 * np.log(1.0 + herz / 700))

  def _MakeMelFilterBank(self):
    mel_high_freq = self.Herz2Mel(self.high_frequency)
    mel_low_freq = self.Herz2Mel(self.low_frequency)

    mel_points = np.linspace(
      mel_low_freq, mel_high_freq, self.num_mel_bins*2)

    dim_spectrum = int(self.fft_size / 2) + 1

    mel_filter_bank = np.zeros((self.num_mel_bins, dim_spectrum))

    for m in range(self.num_mel_bins):
      left_mel = mel_points[m]
      center_mel = mel_points[m+1]
      right_mel = mel_points[m+2]

      for n in range(dim_spectrum):
        
        freq = 1.0 * n * self.sample_frequency /2 / dim_spectrum

        mel = self.Herz2Mel(freq)

        if left_mel <= mel <= right_mel:
          if mel <= center_mel:
            weight = (mel - left_mel) / (center_mel - left_mel)
          else:
            weight = (right_mel - mel) / (right_mel - center_mel)

          mel_filter_bank[m][n] = weight

    return mel_filter_bank

  def _MakeDCTMatrix(self):
    N = self.num_mel_bins

    dct_matrix = np.zeros((self.num_ceps, self.num_mel_bins))
    for k in range(self.num_ceps):
      if k == 0:
        dct_matrix[k] = np.ones(self.num_mel_bins) * 1.0 / np.sqrt(N)
      else:
        dct_matrix[k] = \
          np.sqrt(2/N) * np.cos(((2.0*np.arange(N)+1)*k*np.pi) / (2*N))

    return dct_matrix

  def _MakeLifter(self):
    Q = self.lifter_coef
    I = np.arange(self.num_ceps)
    lifter = 1.0 + 0.5 * Q * np.sin(np.pi * I / Q)
    return lifter

  def _ExtractWindow(self, waveform, start_idx):
    
    window = waveform[start_idx:start_idx+self.frame_size].copy()

    # dithering
    if self.dither > 0:
      window = window + np.random.rand(self.frame_size) \
        * (2*self.dither) - self.dither

    # dc cut
    window = window - np.mean(window)

    power = np.sum(window ** 2)

    if power < 1E-10:
      power = 1E-10

    log_power = np.log(power)

    # 高域強調
    window = np.convolve(window, np.array([1.0, -0.97]), mode='same')
    window[0] -= 0.97*window[0]

    window *= np.hamming(self.frame_size)

    return window, log_power

  def ComputeFBANK(self, waveform):
    """
    Parameters
    ----------
    waveform (list[N]) :
      波形データ

    Returns
    -------
    メルフィルタバンク特徴
    対数パワー値
    """
    num_samples = np.size(waveform)
    num_frames = (num_samples - self.frame_size) // self.frame_shift + 1

    fbank_features = np.zeros((num_frames, self.num_mel_bins))

    log_power = np.zeros(num_frames)

    for frame in range(num_frames):
      start_idx = frame * self.frame_shift

      window, log_pow = self._ExtractWindow(waveform, start_idx)

      spectrum = np.fft.fft(window, n=self.fft_size)

      spectrum = spectrum[:int(self.fft_size/2)+1]

      spectrum = np.abs(spectrum) ** 2

      fbank = np.dot(spectrum, self.mel_filter_bank.T)

      fbank[fbank<0.1] = 0.1

      fbank_features[frame] = np.log(fbank)

      log_power[frame] = log_pow

    return fbank_features, log_power

  def ComputeMFCC(self, waveform):
    """
    Parameters
    ----------
    waveform (list[N]) :

    Returns
    -------
    MFCC特徴
    """

    fbank, log_power = self.ComputeFBANK(waveform)

    mfcc = np.dot(fbank, self.dct_matrix.T)

    mfcc *= self.lifter

    mfcc[:,0] = log_power

    return mfcc


def fbank(
  waveform,
  sample_frequency=16000,
  frame_length=25,
  frame_shift=10,
  num_mel_bins=23,
  lifter_coef=22,
  low_frequency=20,
  high_frequency=8000,
  dither=1.0):
  """
  Parameters
  ---------------
  sample_frequency (int) : default=16000
    [Hz]
  frame_length (int) : default=25
    [ms]
  frame_shift (int) : default=10
    [ms]
  num_mel_bins (int) : default=23
    FBANK特徴の次元数
  lifter_coef (int) : default=22
    リフタリング処理のパラメータ
  low_frequency : default=20
    [Hz]
  high_frequency : default=8000
    [Hz]
  dither : default=1
    ディザリングの強さ

  Returns
  -------
  fbank ( list[frame_size, num_mel_bins] )
  log_power ( list[frame_size] )
  """
  fe = FeatureExtractor(
    sample_frequency=sample_frequency,
    frame_length=frame_length,
    frame_shift=frame_shift,
    num_mel_bins=num_mel_bins,
    lifter_coef=lifter_coef,
    low_frequency=low_frequency,
    high_frequency=high_frequency,
    dither=dither)

  fbank, log_power = fe.ComputeFBANK(waveform)

  return fbank, log_power


def mfcc(
  waveform,
  sample_frequency=16000,
  frame_length=25,
  frame_shift=10,
  num_mel_bins=23,
  num_ceps=13,
  lifter_coef=22,
  low_frequency=20,
  high_frequency=8000,
  dither=1.0):
  """
  Parameters
  ---------------
  sample_frequency (int) : default=16000
    [Hz]
  frame_length (int) : default=25
    [ms]
  frame_shift (int) : default=10
    [ms]
  num_mel_bins (int) : default=23
    FBANK特徴の次元数
  num_ceps (int) : default=13
    MFCC特徴の次元数
  lifter_coef (int) : default=22
    リフタリング処理のパラメータ
  low_frequency : default=20
    [Hz]
  high_frequency : default=8000
    [Hz]
  dither : default=1
    ディザリングの強さ

  Returns
  -------
  mfcc ( list[frame_size, num_ceps] )
  """
  fe = FeatureExtractor(
    sample_frequency=sample_frequency,
    frame_length=frame_length,
    frame_shift=frame_shift,
    num_mel_bins=num_mel_bins,
    num_ceps=num_ceps,
    lifter_coef=lifter_coef,
    low_frequency=low_frequency,
    high_frequency=high_frequency,
    dither=dither)

  mfcc = fe.ComputeMFCC(waveform)

  return mfcc


# test code
if __name__ == "__main__":
  x = np.arange(16000*3)
  y = np.sin(x)

  feat_fbank, p = fbank(y)
  feat_mfcc = mfcc(y)
  print(feat_fbank, feat_mfcc)






      
      


