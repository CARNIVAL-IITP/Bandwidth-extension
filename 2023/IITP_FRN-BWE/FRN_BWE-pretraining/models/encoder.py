import os

import librosa
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from PLCMOS.plc_mos import PLCMOSEstimator
from config import CONFIG
from loss import Loss
from models.blocks import Encoder, Predictor
from utils.utils import visualize, LSD, compute_metrics
from dataset import TrainDataset
from natsort import natsorted
from os import makedirs

import librosa
import pytorch_lightning as pl
import torch
from einops.layers.torch import Rearrange
from torch import nn
from loss import Loss

def load_txt(target_root, txt_list):
    target = []
    with open(txt_list) as f:
        for line in f:
            target.append(os.path.join(target_root, line.strip('\n')))
    target = list(set(target))
    target = natsorted(target)
    return target

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLPBlock(nn.Module):

    def __init__(self, dim, mlp_dim, dropout=0., init_values=1e-4):
        super().__init__()

        self.pre_affine = Aff(dim)
        self.inter = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1,
                             bidirectional=False, batch_first=True)
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, x, state=None):
        x = self.pre_affine(x)
        if state is None:
            inter, _ = self.inter(x)
        else:
            inter, state = self.inter(x, (state[0], state[1]))
        x = x + self.gamma_1 * inter
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        if state is None:
            return x
        state = torch.stack(state, 0)
        return x, state


# class Encoder(nn.Module):
class Encoder(pl.LightningModule):
    def __init__(self, train_dataset = None, val_dataset = None, in_dim=320, dim=384, depth=4, mlp_dim=768):
        super(Encoder).__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c f t -> b t (c f)'),
            nn.Linear(in_dim, dim),
            nn.GELU()
        )

        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPBlock(self.dim, mlp_dim, dropout=0.15))

        self.affine = nn.Sequential(
            Aff(self.dim),
            nn.Linear(dim, in_dim),
            Rearrange('b t (c f) -> b c f t', c=2),
        )

    def forward(self, x_in, states=None):
        x = self.to_patch_embedding(x_in)
        if states is not None:
            out_states = []
        for i, mlp_block in enumerate(self.mlp_blocks):
            if states is None:
                x = mlp_block(x)
            else:
                x, state = mlp_block(x, states[i])
                out_states.append(state)
        x = self.affine(x)
        x = x + x_in
        if states is None:
            return x
        else:
            return x, torch.stack(out_states, 0)
    
    def forward_onnx(self, x, prev_mag, predictor_state=None, mlp_state=None):
        prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
        feat, mlp_state = self.encoder(x, mlp_state)

        feat = torch.cat((feat, prev_mag), 1)
        feat = self.joiner(feat)
        prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)
        feat = feat + x
        return feat, prev_mag, predictor_state, mlp_state

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, #,collate_fn=TrainDataset.collate_fn, 
                          pin_memory = True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, #, collate_fn=TrainDataset.collate_fn, 
                          pin_memory = True, persistent_workers=True)

    def training_step(self, batch, batch_idx):
        x_in, y = batch
        f_0 = x_in[:, :, 0:1, :]
        x = x_in[:, :, 1:, :]

        x = self(x)
        x = torch.cat([f_0, x], dim=2)

        loss = self.loss(x, y)
        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        f_0 = x[:, :, 0:1, :]
        x_in = x[:, :, 1:, :]

        pred = self(x_in)
        pred = torch.cat([f_0, pred], dim=2)

        loss = self.loss(pred, y)
        self.window = self.window.to(pred.device)
        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
        pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window)
        y = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())
        y = torch.istft(y, self.window_size, self.hop_size, window=self.window)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        # if batch_idx == 0:
        #     i = torch.randint(0, x.shape[0], (1,)).item()
        #     x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        #     x = torch.istft(x[i], self.window_size, self.hop_size, window=self.window)

        #     self.trainer.logger.log_spectrogram(y[i], x, pred[i], self.current_epoch)
        #     self.trainer.logger.log_audio(y[i], x, pred[i], self.current_epoch)


    def test_step(self, test_batch, batch_idx):
        inp, tar, inp_wav, tar_wav = test_batch
        inp_wav = inp_wav.squeeze()
        tar_wav = tar_wav.squeeze()
        f_0 = inp[:, :, 0:1, :]
        x = inp[:, :, 1:, :]
        pred = self(x)
        pred = torch.cat([f_0, pred], dim=2)
        pred = torch.istft(pred.squeeze(0).permute(1, 2, 0), self.window_size, self.hop_size,
                           window=self.window.to(pred.device), return_complex = False)

        tar_wav = tar_wav.cpu().numpy()
        inp_wav = inp_wav.cpu().numpy()
        pred = pred.detach().cpu().numpy()

        data_dir = CONFIG.DATA.data_dir
        name = CONFIG.DATA.dataset
        target_root = data_dir[name]['root']
        clean_txt_list = data_dir[name]['test']
        clean_data_list = load_txt(target_root,clean_txt_list)
 
        ret = compute_metrics(tar_wav, pred)

        current_path = os.path.abspath(os.getcwd())
        out_path = current_path + '/output/' + CONFIG.DATA.dataset + '/version_' + str(self.version) + '/' 
        makedirs(out_path, exist_ok = True)
        out_path1 = current_path + '/output/' + CONFIG.DATA.dataset + '/hr/'  
        makedirs(out_path1, exist_ok = True)
        out_path2 = current_path + '/output/' + CONFIG.DATA.dataset + '/lr/'  
        makedirs(out_path2, exist_ok = True)

        head, tail = os.path.split(clean_data_list[batch_idx])
        tail = tail.replace('_mic1.flac', '.wav')
        os.replace('mic1.flac')
        tail = tail.split('.wav')[0]

        if self.save:
            sf.write(os.path.join(out_path, tail), pred, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            sf.write(os.path.join(out_path2, tail), tar_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            sf.write(os.path.join(out_path1, tail), inp_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')

        metrics = {

            'STOI': ret[0],
            'ESTOI': ret[1],
            'SNR': ret[2],
            'LSD': ret[3],
            'LSD-H': ret[4],
            'LSD-L': ret[5],
            'PESQ': ret[6],
            'SI-SDR': ret[7],
        }
        self.log_dict(metrics)

        return metrics
    
    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        f_0 = batch[:, :, 0:1, :]
        x = batch[:, :, 1:, :]
        pred = self(x)
        pred = torch.cat([f_0, pred], dim=2)
        pred = torch.istft(pred.squeeze(0).permute(1, 2, 0), self.window_size, self.hop_size,
                           window=self.window.to(pred.device))
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
