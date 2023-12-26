import os

import librosa
import pytorch_lightning as pl
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI

from config_continual import CONFIG
from loss import Loss
# from models.blocks import Encoder, Predictor, Predictor_traj_lstm
from models.blocks_continual import Encoder, Predictor, Predictor_traj_lstm
from models.blocks_PLUS import Encoder_PLUS, Predictor_PLUS, RI_Predictor
from natsort import natsorted
from utils.utils import visualize, LSD, compute_metrics
from os import makedirs


def load_txt(target_root, txt_list):
    target = []
    with open(txt_list) as f:
        for line in f:
            target.append(os.path.join(target_root, line.strip('\n')))
    target = list(set(target))
    target = natsorted(target)
    return target


class ContinualFRN(pl.LightningModule):
    def __init__(self, train_dataset = None, val_dataset = None, 
                 pred_ckpt_path = None, version = None, save = None):
        super(ContinualFRN, self).__init__()

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
        self.hparams['ewc_mode'] = CONFIG.TRAIN.pretraining.ewc_mode
        self.hparams['regularizer'] = CONFIG.TRAIN.pretraining.regularizer
        self.hparams['reg_lambda'] = CONFIG.TRAIN.pretraining.lambda_reg
        self.hparams['ema_decay'] = CONFIG.TRAIN.pretraining.ema_decay
        self.hparams['lr0'] = CONFIG.TRAIN.pretraining.lr0
        self.hparams['num_prior_training'] = CONFIG.TRAIN.pretraining.num_prior_training
        self.save_hyperparameters(ignore=['train_dataset', 'val_dataset'])

        self.index_count = 0
        self.index_size = 0

        self.experiment_id = 0
        self.experiment_name = ''
        self.index_mode = False
        self.forgetting_mode = False
        self.ewc_mode = None
        self.fisher_matrix = {}
        self.ema_mode = None
        self._error_loading_ema = False
        self.ema_decay = CONFIG.TRAIN.pretraining.ema_decay
        self.regularizer_mode = None

        self.param_list = [self.named_parameters] 
        self.learning_rate = CONFIG.TRAIN.lr
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.data_size = len(train_dataset)

        if pred_ckpt_path is not None:
            self.predictor = Predictor.load_from_checkpoint(pred_ckpt_path)
        else:
            self.predictor = Predictor(window_size=self.window_size, 
                                       lstm_dim=self.pred_dim,
                                        lstm_layers=self.pred_layers, 
                                        pred_lstm_type=self.pred_lstm_type,
                                        state = self.pred_state)

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
        # self.predictor.eval()
        # self.joiner.eval()

    def forward(self, x):
        """
        Input: real-imaginary; shape (B, F, T, 2); F = hop_size + 1
        Output: real-imaginary
        """

        B, C, F, T = x.shape

        x = x.permute(3, 0, 1, 2).unsqueeze(-1)
        prev_mag = torch.zeros((B, 1, F, 1), device=x.device).clone()
        predictor_state = torch.zeros((2, self.predictor.lstm_layers, B, self.predictor.lstm_dim), device=x.device).clone()
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
        loss = self.loss(x, y)
        if CONFIG.TRAIN.pretraining.regularizer_mode:
            if self.hparams.regularizer == 'L2' :
                l2_norm = torch.tensor(0.)
                for param in self.parameters():
                    new_param = param.to(l2_norm)
                    l2_norm += torch.norm(new_param)
                loss += self.hparams.reg_lambda * l2_norm
            elif self.hparams.regularizer == 'L1':
                l1_norm = sum([p.abs().sum() for p in self.parameters()])
                loss += self.hparams.reg_lambda * l1_norm
        return loss
    
    def training_step(self, batch, batch_idx):
        loss_step = self.shared_step(batch = batch, batch_idx = batch_idx)
        self.log('train_loss', loss_step, logger=True)
        if self.ewc_mode:
            self.ewc.apply_penalty(self, loss_step)
        return loss_step

    def shared_step(self, batch, batch_idx, log=True):
        if log:
            self.log('experiment_id', float(self.experiment_id))
            self.log('global_step', float(self.global_step))

        x_in, y = batch
        f_0 = x_in[:, :, 0:1, :]
        x = x_in[:, :, 1:, :]

        x = self(x)
        x = torch.cat([f_0, x], dim=2)
        loss = self.forward_loss(x, y)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        f_0 = x[:, :, 0:1, :]
        x_in = x[:, :, 1:, :]

        pred = self(x_in)
        pred = torch.cat([f_0, pred], dim=2)

        loss = self.forward_loss(pred, y)
        self.window = self.window.to(pred.device)
        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous())
        pred = torch.istft(pred, self.window_size, self.hop_size, window=self.window)
        y = torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())
        y = torch.istft(y, self.window_size, self.hop_size, window=self.window)

        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

    def _ewc_step(self, batch, batch_idx):
        with torch.enable_grad():
            self.zero_grad()
            ewc_loss = self.shared_step(batch, batch_idx, False)
            ewc_loss.requires_grad_(True)
            ewc_loss.retain_grad()
            ewc_loss.backward()
            for n, p in self.named_parameters():
                if n in self.fisher_matrix and p.grad is not None:
                    self.fisher_matrix[n].data += p.grad.data.to('cpu').clone().pow(2)
        return ewc_loss
    

    # def test_step(self, batch, batch_idx) -> None:
    #     if self.ewc_mode:
    #         self._ewc_step(batch, batch_idx)

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
            # sf.write(os.path.join(out_path1, tail), tar_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')
            # sf.write(os.path.join(out_path2, tail), inp_wav, samplerate=CONFIG.DATA.sr, subtype='PCM_16')

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
        pred = torch.cat([f_0, x], dim=2)

        pred = torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()).squeeze(0)
        pred = torch.istft(pred, self.window_size, self.hop_size,
                           window=self.window.to(pred.device))
        print("input window size: ", self.window_size)
        print("input stride: ", self.hop_size)
        print("sampling rate:: ", CONFIG.DATA.sr)

        # 1프레임 지연시간 = delay + stride = window/sampling rate + stride / sampling rate
        print("1 프레임 지연시간 = (window size - stride + stride) / sampling rate (미래 프레임을 보지 않기 때문에)")
        print("1 프레임 지연시간: {x} ms"  .format(x = ((self.window_size + self.hop_size - self.hop_size) / CONFIG.DATA.sr *1000)))
        exit()
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

    def on_after_backward(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

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
