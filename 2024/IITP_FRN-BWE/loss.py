import librosa
import pytorch_lightning as pl
import torch
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, apply_reduction, SpectralConvergenceLoss, STFTMagnitudeLoss

from config import CONFIG


class STFTLossDDP(STFTLoss):
    def __init__(self,
                 fft_size=1024,
                 hop_size=256,
                 win_length=1024,
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 eps=1e-8,
                 output="loss",
                 reduction="mean",
                 device=None):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.w_sc = w_sc
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.w_phs = w_phs
        self.sample_rate = sample_rate
        self.scale = scale
        self.n_bins = n_bins
        self.scale_invariance = scale_invariance
        self.eps = eps
        self.output = output
        self.reduction = reduction
        self.device = device

        self.spectralconv = SpectralConvergenceLoss()
        self.logstft = STFTMagnitudeLoss(log=True, reduction=reduction)
        self.linstft = STFTMagnitudeLoss(log=False, reduction=reduction)

        # setup mel filterbank
        if self.scale == "mel":
            assert (sample_rate is not None)  # Must set sample rate to use mel scale
            assert (n_bins <= fft_size)  # Must be more FFT bins than Mel bins
            fb = librosa.filters.mel(sample_rate, fft_size, n_mels=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)
        elif self.scale == "chroma":
            assert (sample_rate is not None)  # Must set sample rate to use chroma scale
            assert (n_bins <= fft_size)  # Must be more FFT bins than chroma bins
            fb = librosa.filters.chroma(sample_rate, fft_size, n_chroma=n_bins)
            self.fb = torch.tensor(fb).unsqueeze(0)

        if scale is not None and device is not None:
            self.fb = self.fb.to(self.device)  # move filterbank to device

    def compressed_loss(self, x, y, alpha=None):
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        if alpha is not None:
            x_mag = x_mag ** alpha
            y_mag = y_mag ** alpha

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb.to(x_mag.device), x_mag)
            y_mag = torch.matmul(self.fb.to(y_mag.device), y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0

        # combine loss terms
        loss = (self.w_sc * sc_loss) + (self.w_log_mag * mag_loss) + (self.w_lin_mag * lin_loss)
        loss = apply_reduction(loss, reduction=self.reduction)
        return loss

    def forward(self, x, y):
        return self.compressed_loss(x, y, 0.3)


class MRSTFTLossDDP(MultiResolutionSTFTLoss):
    def __init__(self,
                 fft_sizes=(1024, 2048, 512),
                 hop_sizes=(120, 240, 50),
                 win_lengths=(600, 1200, 240),
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 **kwargs):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossDDP(fs,
                                             ss,
                                             wl,
                                             window,
                                             w_sc,
                                             w_log_mag,
                                             w_lin_mag,
                                             w_phs,
                                             sample_rate,
                                             scale,
                                             n_bins,
                                             scale_invariance,
                                             **kwargs)]



class Loss(pl.LightningModule):
    def __init__(self):
        super(Loss, self).__init__()
        self.stft_loss = MRSTFTLossDDP(sample_rate=CONFIG.DATA.sr, device="cpu", w_log_mag=0.0, w_lin_mag=1.0)
        self.window = torch.sqrt(torch.hann_window(CONFIG.DATA.window_size))

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        wave_x = torch.istft(torch.view_as_complex(x.contiguous()), CONFIG.DATA.window_size, CONFIG.DATA.stride,
                             window=self.window.to(x.device))
        wave_y = torch.istft(torch.view_as_complex(y.contiguous()), CONFIG.DATA.window_size, CONFIG.DATA.stride,
                             window=self.window.to(y.device))
        loss = self.stft_loss(wave_x, wave_y)
        return loss

from torch import nn
from torch.nn import functional as F
import librosa


class STFTLossDDP_custom(STFTLoss):

    def forward(self, x, y):
        # compute the magnitude and phase spectra of input and target
        self.window = self.window.to(x.device)
        x_mag, x_phs = self.stft(x.view(-1, x.size(-1)))
        y_mag, y_phs = self.stft(y.view(-1, y.size(-1)))

        # apply relevant transforms
        if self.scale is not None:
            x_mag = torch.matmul(self.fb.to(x_mag.device), x_mag)
            y_mag = torch.matmul(self.fb.to(y_mag.device), y_mag)

        # normalize scales
        if self.scale_invariance:
            alpha = (x_mag * y_mag).sum([-2, -1]) / ((y_mag ** 2).sum([-2, -1]))
            y_mag = y_mag * alpha.unsqueeze(-1)

        # compute loss terms
        sc_loss = self.spectralconv(x_mag, y_mag) if self.w_sc else 0.0
        mag_loss = self.logstft(x_mag, y_mag) if self.w_log_mag else 0.0
        lin_loss = self.linstft(x_mag, y_mag) if self.w_lin_mag else 0.0

        # combine loss terms
        loss = (self.w_sc * sc_loss) + (self.w_log_mag * mag_loss) + (self.w_lin_mag * lin_loss)
        loss = apply_reduction(loss, reduction=self.reduction)

        if self.output == "loss":
            return loss
        elif self.output == "full":
            return loss, sc_loss, mag_loss


class MRSTFTLossDDP_custom(MultiResolutionSTFTLoss):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window",
                 w_sc=1.0,
                 w_log_mag=1.0,
                 w_lin_mag=0.0,
                 w_phs=0.0,
                 sample_rate=None,
                 scale=None,
                 n_bins=None,
                 scale_invariance=False,
                 **kwargs):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)  # must define all
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLossDDP_custom(fs,
                                             ss,
                                             wl,
                                             window,
                                             w_sc,
                                             w_log_mag,
                                             w_lin_mag,
                                             w_phs,
                                             sample_rate,
                                             scale,
                                             n_bins,
                                             scale_invariance,
                                             **kwargs)]
            

class TorchSTFT(nn.Module):  # pylint: disable=abstract-method
    """Some of the audio processing funtions using Torch for faster batch processing.

    Args:

        n_fft (int):
            FFT window size for STFT.

        hop_length (int):
            number of frames between STFT columns.

        win_length (int, optional):
            STFT window length.

        pad_wav (bool, optional):
            If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.

        window (str, optional):
            The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        n_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        use_mel (bool, optional):
            If True compute the melspectrograms otherwise. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.

        spec_gain (float, optional):
            gain applied when converting amplitude to DB. Defaults to 1.0.

        power (float, optional):
            Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.

        use_htk (bool, optional):
            Use HTK formula in mel filter instead of Slaney.

        mel_norm (None, 'slaney', or number, optional):
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        pad_wav=False,
        window="hann_window",
        sample_rate=None,
        mel_fmin=0,
        mel_fmax=None,
        n_mels=80,
        use_mel=False,
        do_amp_to_db=False,
        spec_gain=1.0,
        power=None,
        use_htk=False,
        mel_norm="slaney",
        normalized=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.mel_norm = mel_norm
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.mel_basis = None
        self.normalized = normalized
        if use_mel:
            self._build_mel_basis()

    def __call__(self, x):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        # B x D x T x 2
        o = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel:
            S = torch.matmul(self.mel_basis.to(x), S)
        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)
        return S

    def _build_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    @staticmethod
    def _amp_to_db(x, spec_gain=1.0):
        return torch.log(torch.clamp(x, min=1e-5) * spec_gain)

    @staticmethod
    def _db_to_amp(x, spec_gain=1.0):
        return torch.exp(x) / spec_gain
    

class STFTLoss_custom(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc
    
class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_ffts=(1024, 2048, 512), 
                 hop_lengths=(120, 240, 50), 
                 win_lengths=(600, 1200, 240)):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss_custom(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc
    

class MultiScaleSubbandSTFTLoss(MultiScaleSTFTLoss):
    """Multiscale STFT loss for multi band model outputs.
    From MultiBand-MelGAN paper https://arxiv.org/abs/2005.05106"""

    # pylint: disable=no-self-use
    def forward(self, y_hat, y):
        y_hat = y_hat.view(-1, 1, y_hat.shape[2])
        y = y.view(-1, 1, y.shape[2])
        return super().forward(y_hat.squeeze(1), y.squeeze(1))
