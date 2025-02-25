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
from models.blocks import Encoder, Predictor, Predictor_traj_lstm
from utils.utils import visualize, LSD, compute_metrics
from dataset import TrainDataset
from models.blocks_PLUS import Encoder_PLUS, Predictor_PLUS, RI_Predictor
from natsort import natsorted
from os import makedirs

from loss import MRSTFTLossDDP_custom, MultiScaleSubbandSTFTLoss
from torchsubband import SubbandDSP


# plcmos = PLCMOSEstimator()

def load_txt(target_root, txt_list):
    target = []
    with open(txt_list) as f:
        for line in f:
            target.append(os.path.join(target_root, line.strip('\n')))
    target = list(set(target))
    target = natsorted(target)
    return target


class FRN_Subband(pl.LightningModule):
    def __init__(self, train_dataset = None, val_dataset = None, 
                 pred_ckpt_path = None, version = None, save = None):
                #  pred_ckpt_path='lightning_logs/predictor/checkpoints/predictor.ckpt', version = None, save = None):
        super(FRN_Subband, self).__init__()

        self.window_size = CONFIG.DATA.window_size 
        self.enc_lstm_type = CONFIG.MODEL.enc_lstm_tpye
        self.enc_layers = CONFIG.MODEL.enc_layers 
        self.enc_in_dim = CONFIG.MODEL.enc_in_dim 
        self.enc_dim = CONFIG.MODEL.enc_dim
        self.pred_dim = CONFIG.MODEL.pred_dim 
        self.pred_layers = CONFIG.MODEL.pred_layers 
        self.pred_lstm_type = CONFIG.MODEL.pred_lstm_tpye
        self.enc_state = CONFIG.MODEL.enc_state
        self.pred_state = CONFIG.MODEL.pred_state

        self.hop_size = self.window_size // 2
        self.learning_rate = CONFIG.TRAIN.lr
        self.save = save
        self.version = version
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.stoi = STOI(16000)
        self.pesq = PESQ(16000, 'wb')
        self.loss = Loss()
        self.window = torch.sqrt(torch.hann_window(self.window_size))

        self.hparams['batch_size'] = CONFIG.TRAIN.batch_size
        self.hparams['window_size'] = self.window_size
        self.hparams['enc_layers'] = self.enc_layers
        self.hparams['enc_in_dim'] = self.enc_in_dim
        self.hparams['enc_dim'] = self.enc_dim
        self.hparams['pred_dim'] = self.pred_dim
        self.hparams['pred_layers'] = self.pred_layers
        self.hparams['enc_lstm_type'] = self.enc_lstm_type
        self.hparams['pred_lstm_type'] = self.pred_lstm_type
        self.hparams['mode_name'] = CONFIG.MODEL.model_name
        self.hparams['task'] = CONFIG.TASK.task
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])

        if CONFIG.TRAIN.subband.subband_training == True:
            self.subband = SubbandDSP(subband=CONFIG.TRAIN.subband.subband)
            self.stft_loss = MRSTFTLossDDP_custom(n_bins=64, sample_rate=CONFIG.DATA.sr, device="cpu", scale='mel')
            # self.stft_loss = Loss()
            self.subband_stft_loss = MultiScaleSubbandSTFTLoss()
            self.subband_weigh_stft_loss = CONFIG.TRAIN.subband.weight_loss
            self.hparams['subband_training'] = CONFIG.TRAIN.subband.subband_training
            self.hparams['subband'] = CONFIG.TRAIN.subband.subband
            self.hparams['subband_weight_loss'] = CONFIG.TRAIN.subband.weight_loss


        if pred_ckpt_path is not None:
            self.predictor = Predictor.load_from_checkpoint(pred_ckpt_path)
        else:
            self.predictor = Predictor(window_size=self.window_size, 
                                       lstm_dim=self.pred_dim,
                                        lstm_layers=self.pred_layers, 
                                        pred_lstm_type=self.pred_lstm_type,
                                        state = self.pred_state)
            # self.predictor = Predictor(window_size=self.window_size, lstm_dim=self.pred_dim,
            #                            lstm_layers=self.pred_layers)
        self.joiner = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=(9, 1), stride=1, padding=(4, 0), padding_mode='reflect',
                      groups=3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 2, kernel_size=1, stride=1, padding=0, groups=2),
        )
        self.encoder = Encoder(in_dim=self.window_size, 
                               dim=self.enc_in_dim, 
                               depth=self.enc_layers,
                               mlp_dim=self.enc_dim, 
                               enc_lstm_type=self.enc_lstm_type,
                               state = self.enc_state)
        self.predictor.eval()
        self.joiner.eval()

    def forward(self, x):
        """
        Input: real-imaginary; shape (B, F, T, 2); F = hop_size + 1
        Output: real-imaginary
        """

        B, C, F, T = x.shape

        x = x.permute(3, 0, 1, 2).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device)
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim), device=x.device)
        mlp_state = torch.zeros((self.encoder.depth, 2, 1, B, self.encoder.dim), device=x.device)
        result = []
        for step in x:
            if  not self.enc_state:
                feat = self.encoder(step)
            else:
                feat, mlp_state = self.encoder(step, mlp_state)
            if  not self.pred_state:
                prev_mag = self.predictor(prev_mag)
            else:
                prev_mag, predictor_state = self.predictor(prev_mag, predictor_state)
            feat = torch.cat((feat, prev_mag), 1)
            feat = self.joiner(feat)
            feat = feat + step
            result.append(feat)
            prev_mag = torch.linalg.norm(feat, dim=1, ord=1, keepdims=True)  # compute magnitude
        output = torch.cat(result, -1)
        return output

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
                          num_workers=CONFIG.TRAIN.workers,  
                          pin_memory = True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, 
                          pin_memory = True, persistent_workers=True)

    def forward_loss(self, x, y):
        if self.hparams.loss_type == 1:
            loss = self.loss(x, y)
        elif self.hparams.loss_type ==2:
            loss = self.stft_loss(x, y) + self.loss(x, y) * CONFIG.TRAIN.mse_weight
        elif self.hparams.loss_type ==3:

            x_sub = self.subband.wav_to_sub(x)
            y_sub = self.subband.wav_to_sub(y)
   
            subband_loss_weight = CONFIG.TRAIN.subband.weight_loss
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(x_sub, y_sub)
            subband_loss = subband_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc)
            
            time_loss = self.loss(x, y) * CONFIG.TRAIN.mse_weight
            stft_loss = self.stft_loss(x, y) * CONFIG.TRAIN.stft_weight

            self.log('train_time_loss', stft_loss, logger=True)
            self.log('train_freq_loss', time_loss, logger=True)
            self.log('train_subband_loss', subband_loss, logger=True)

            loss = time_loss + stft_loss + subband_loss

        elif self.hparams.regularizer == 'L2' :
            l2_reg = torch.tensor(0.)#.to(device=device)
            for param in self.parameters():
                new_param = param.to(l2_reg)
                l2_reg += torch.norm(new_param)
            loss = self.freq_loss(x, y) + self.time_loss(x, y) \
                * CONFIG.TRAIN.mse_weight + self.hparams.lambda_reg * l2_reg
        else:
            l1_norm = sum([p.abs().sum() for p in self.parameters()])
            loss = self.freq_loss(x, y) + self.time_loss(x, y) \
                * CONFIG.TRAIN.mse_weight + self.hparams.lambda_reg * l1_norm
        return loss
    
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

        if batch_idx == 0:
            i = torch.randint(0, x.shape[0], (1,)).item()
            x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
            x = torch.istft(x[i], self.window_size, self.hop_size, window=self.window)

            self.trainer.logger.log_spectrogram(y[i], x, pred[i], self.current_epoch)
            self.trainer.logger.log_audio(y[i], x, pred[i], self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        inp, tar, inp_wav, tar_wav = test_batch
        inp_wav = inp_wav.squeeze()
        tar_wav = tar_wav.squeeze()
        f_0 = inp[:, :, 0:1, :]
        x = inp[:, :, 1:, :]
        pred = self(x)
        pred = torch.cat([f_0, pred], dim=2)
        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()).squeeze(0)
        pred = torch.istft(pred, self.window_size, self.hop_size,
                           window=self.window.to(pred.device))

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
        # out_path1 = current_path + '/output/' + CONFIG.DATA.dataset + '/hr/'  
        # makedirs(out_path1, exist_ok = True)
        # out_path2 = current_path + '/output/' + CONFIG.DATA.dataset + '/lr/'  
        # makedirs(out_path2, exist_ok = True)

        head, tail = os.path.split(clean_data_list[batch_idx])

        if self.save:
            sf.write(os.path.join(out_path, tail), pred, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path2, tail), tar_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path1, tail), inp_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')

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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                  patience=CONFIG.TRAIN.patience,
                                                                  factor=CONFIG.TRAIN.factor, 
                                                                  verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]


class OnnxWrapper(pl.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        batch_size = 1
        pred_states = torch.zeros((2, 1, batch_size, model.predictor.lstm_dim))
        mlp_states = torch.zeros((model.encoder.depth, 2, 1, batch_size, model.encoder.dim))
        mag = torch.zeros((batch_size, 1, model.hop_size, 1))
        x = torch.randn(batch_size, model.hop_size + 1, 2)
        self.sample = (x, mag, pred_states, mlp_states)
        self.input_names = ['input', 'mag_in_cached_', 'pred_state_in_cached_', 'mlp_state_in_cached_']
        self.output_names = ['output', 'mag_out_cached_', 'pred_state_out_cached_', 'mlp_state_out_cached_']

    def forward(self, x, prev_mag, predictor_state=None, mlp_state=None):
        x = x.permute(0, 2, 1).unsqueeze(-1)
        f_0 = x[:, :, 0:1, :]
        x = x[:, :, 1:, :]

        output, prev_mag, predictor_state, mlp_state = self.model.forward_onnx(x, prev_mag, predictor_state, mlp_state)
        output = torch.cat([f_0, output], dim=2)
        output = output.squeeze(-1).permute(0, 2, 1)
        return output, prev_mag, predictor_state, mlp_state
