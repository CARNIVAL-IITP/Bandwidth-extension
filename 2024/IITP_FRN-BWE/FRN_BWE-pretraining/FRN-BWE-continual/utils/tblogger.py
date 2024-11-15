from os import path
import torch
import torch.nn as nn

import librosa as rosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from librosa.display import waveshow
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only



matplotlib.use('Agg')

class STFTMag(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)

    # x: [B,T] or [T]
    @torch.no_grad()
    def forward(self, x):
        stft = torch.stft(x.cpu(),
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )  # return_complex=False)  #[B, F, TT,2]
        mag = torch.norm(stft, p=2, dim=-1)  # [B, F, TT]
        return mag


class TensorBoardLoggerExpanded(TensorBoardLogger):
    def __init__(self):
        super().__init__(save_dir='lightning_logs', default_hp_metric=False, name='')
        self.sr = 16000
        self.stftmag = STFTMag()

    def fig2np(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def plot_spectrogram_to_numpy(self, y, y_low, y_recon, step):
        name_list = ['y', 'y_low', 'y_recon']
        fig = plt.figure(figsize=(9, 15))
        fig.suptitle(f'Epoch_{step}')
        for i, yy in enumerate([y, y_low, y_recon]):
            ax = plt.subplot(3, 1, i + 1)
            ax.set_title(name_list[i])
            plt.imshow(rosa.amplitude_to_db(self.stftmag(yy).numpy(),
                                            ref=np.max, top_db=80.),
                       # vmin = -20,
                       vmax=0.,
                       aspect='auto',
                       origin='lower',
                       interpolation='none')
            # plt.colorbar()
            plt.xlabel('Frames')
            plt.ylabel('Channels')
            plt.tight_layout()

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data

    @rank_zero_only
    def log_spectrogram(self, y, y_low, y_recon, epoch):
        y, y_low, y_recon = y.detach().cpu(), y_low.detach().cpu(), y_recon.detach().cpu()
        spec_img = self.plot_spectrogram_to_numpy(y, y_low, y_recon, epoch)
        self.experiment.add_image(path.join(self.save_dir, 'result'),
                                  spec_img,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return
    
    def plot_waveform_to_numpy(self, y, y_low, y_recon, step):
        name_list = ['y', 'y_low', 'y_recon']
        fig = plt.figure(figsize=(9, 15))
        fig.suptitle(f'Epoch_{step}')
        for i, yy in enumerate([y, y_low, y_recon]):
            ax = plt.subplot(3, 1, i + 1)
            # plt.subplot()
            ax.set_title(name_list[i])
            waveshow(yy.numpy(), self.sr)

        fig.canvas.draw()
        data = self.fig2np(fig)

        plt.close()
        return data
    
    def waveform(self, y, y_low, y_recon, epoch):
        y, y_low, y_recon = y.detach().cpu(), y_low.detach().cpu(), y_recon.detach().cpu()
        spec_img = self.plot_waveform_to_numpy(y, y_low, y_recon, epoch)
        self.experiment.add_image(path.join(self.save_dir, 'result'),
                                  spec_img,
                                  epoch,
                                  dataformats='HWC')
        self.experiment.flush()
        return

