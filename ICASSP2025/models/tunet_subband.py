import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from performer_pytorch import Performer
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import CONFIG


from dataset import CustomDataset
from loss import MRSTFTLossDDP

from pytorch_lightning.utilities import rank_zero_info
from tqdm import tqdm
from torch import autograd

from torch.linalg import norm

from torchsubband import SubbandDSP

from loss import MultiScaleSubbandSTFTLoss

class TFiLM(nn.Module):
    def __init__(self, block_size, input_dim, **kwargs):
        super(TFiLM, self).__init__(**kwargs)
        self.block_size = block_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.block_size)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)

    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then
            runs LSTM to generate normalization weights.
        """
        x_in_down = self.max_pool(x_in).permute([0, 2, 1])
        x_rnn, _ = self.lstm(x_in_down)
        return x_rnn.permute([0, 2, 1])

    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective blocks.
        """
        # channel first
        n_blocks = x_in.shape[2] // self.block_size
        n_filters = x_in.shape[1]

        # reshape input into blocks
        x_norm = torch.reshape(x_norm, shape=(-1, n_filters, n_blocks, 1))
        x_in = torch.reshape(x_in, shape=(-1, n_filters, n_blocks, self.block_size))

        # multiply
        x_out = x_norm * x_in

        # return to original shape
        x_out = torch.reshape(x_out, shape=(-1, n_filters, n_blocks * self.block_size))

        return x_out

    def forward(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        assert x.shape[2] % self.block_size == 0, 'Number of steps must be a \
                                                   multiple of the block size.'

        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x


class Encoder(nn.Module):
    def __init__(self, max_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Encoder, self).__init__()
        self.tfilm = tfilm

        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) // 2 for i in range(n_layers)]

        if self.tfilm:
            b_size = max_len // (n_blocks * strides[0])
            self.tfilm_d = TFiLM(block_size=b_size, input_dim=out_channels[0])
            b_size //= strides[1]
            self.tfilm_d1 = TFiLM(block_size=b_size, input_dim=out_channels[1])

        self.downconv = nn.Conv1d(in_channels=1, out_channels=out_channels[0], kernel_size=kernel_sizes[0],
                                  stride=strides[0], padding=paddings[0], padding_mode='replicate')
        self.downconv1 = nn.Conv1d(in_channels=out_channels[0], out_channels=out_channels[1],
                                   kernel_size=kernel_sizes[1],
                                   stride=strides[1], padding=paddings[1], padding_mode='replicate')
        self.downconv2 = nn.Conv1d(in_channels=out_channels[1], out_channels=out_channels[2],
                                   kernel_size=kernel_sizes[2],
                                   stride=strides[2], padding=paddings[2], padding_mode='replicate')
        self.out_len = max_len // (strides[0] * strides[1] * strides[2])

    def forward(self, x):
        x1 = F.leaky_relu(self.downconv(x), 0.2)  # 2048
        if self.tfilm:
            x1 = self.tfilm_d(x1)
        x2 = F.leaky_relu(self.downconv1(x1), 0.2)  # 1024
        if self.tfilm:
            x2 = self.tfilm_d1(x2)
        x3 = F.leaky_relu(self.downconv2(x2), 0.2)  # 512
        return [x1, x2, x3]


class Decoder(nn.Module):
    def __init__(self, in_len, kernel_sizes, strides, out_channels, tfilm, n_blocks):
        super(Decoder, self).__init__()
        self.tfilm = tfilm
        n_layers = len(strides)
        paddings = [(kernel_sizes[i] - strides[i]) // 2 for i in range(n_layers)]

        if self.tfilm:
            in_len *= strides[2]
            self.tfilm_u1 = TFiLM(block_size=in_len // n_blocks, input_dim=out_channels[1])
            in_len *= strides[1]
            self.tfilm_u = TFiLM(block_size=in_len // n_blocks, input_dim=out_channels[0])

        self.convt3 = nn.ConvTranspose1d(in_channels=out_channels[2], out_channels=out_channels[1], stride=strides[2],
                                         kernel_size=kernel_sizes[2], padding=paddings[2])
        self.convt2 = nn.ConvTranspose1d(in_channels=out_channels[1], out_channels=out_channels[0], stride=strides[1],
                                         kernel_size=kernel_sizes[1], padding=paddings[1])
        self.convt1 = nn.ConvTranspose1d(in_channels=out_channels[0], out_channels=1, stride=strides[0],
                                         kernel_size=kernel_sizes[0], padding=paddings[0])
        self.dropout = nn.Dropout(0.0)

    def forward(self, x_list):
        x, x1, x2, bottle_neck = x_list
        x_dec = self.dropout(F.leaky_relu(self.convt3(bottle_neck), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u1(x_dec)
        x_dec = x2 + x_dec

        x_dec = self.dropout(F.leaky_relu(self.convt2(x_dec), 0.2))
        if self.tfilm:
            x_dec = self.tfilm_u(x_dec)
        x_dec = x1 + x_dec
        x_dec = x + torch.tanh(self.convt1(x_dec))
        return x_dec



class BaseModel(pl.LightningModule):
    def __init__(self, train_dataset=None, val_dataset=None):
        super(BaseModel, self).__init__()
        self.hparams['Task'] = CONFIG.TASK.task
        self.hparams['donwsampling'] = CONFIG.TASK.downsampling
        self.hparams['LPF_order'] = CONFIG.TASK.orders
        self.hparams['LPF_ripples'] = CONFIG.TASK.ripples
        self.hparams['dataset'] = CONFIG.DATA.dataset
        self.hparams['patch_stride'] = CONFIG.DATA.stride
        self.hparams['patch_window'] = CONFIG.DATA.window_size
        self.hparams['sr'] = CONFIG.DATA.sr
        self.hparams['ratio'] = CONFIG.DATA.ratio
        self.hparams['epoch'] = CONFIG.TRAIN.epochs
        self.hparams['loss_type'] = CONFIG.TRAIN.loss_type
        self.hparams['weight_time_loss'] = CONFIG.TRAIN.mse_weight
        self.hparams['weight_stft_loss'] = CONFIG.TRAIN.stft_weight_loss
        self.hparams['max_len'] = CONFIG.DATA.window_size
        self.learning_rate = CONFIG.TRAIN.lr
        self.hparams['batch_size'] = CONFIG.TRAIN.batch_size
        self.hparams['optimizer'] = CONFIG.TRAIN.optimizer
        self.hparams['momentum'] = CONFIG.TRAIN.momentum
        self.save_hyperparameters(ignore=["train_dataset", "val_dataset"])

        if CONFIG.TRAIN.pretraining.pretrained == True: 
            self.hparams['pretrained'] = CONFIG.TRAIN.pretraining.pretrained
            self.hparams['number_prior_training'] = CONFIG.TRAIN.pretraining.num_prior_training
            self.hparams['pretrained_checkpoint'] = CONFIG.TRAIN.pretraining.pretrained_checkpoint
            self.hparams['strategy'] = CONFIG.TRAIN.pretraining.strategy
            self.hparams['regularizer'] = CONFIG.TRAIN.pretraining.regularizer
            self.hparams['regularizer_weight'] = CONFIG.TRAIN.pretraining.regularizer_weight

        if CONFIG.TRAIN.subband.subband_training == True:
            self.subband = SubbandDSP(subband=CONFIG.TRAIN.subband.subband)
            self.subband_stft_loss = MultiScaleSubbandSTFTLoss()
            self.subband_weigh_stft_loss = CONFIG.TRAIN.subband.weight_loss
            self.hparams['subband_training'] = CONFIG.TRAIN.subband.subband_training
            self.hparams['subband'] = CONFIG.TRAIN.subband.subband
            self.hparams['subband_weight_loss'] = CONFIG.TRAIN.subband.weight_loss

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        # self.train_sisdr = SI_SDR()
        # self.valid_sisdr = SI_SDR()
        # self.valid_stoi = STOI(fs = CONFIG.DATA.sr, extended = False)
        # self.valid_estoi = STOI(fs = CONFIG.DATA.sr, extended = True)
        self.time_loss = nn.MSELoss()
        self.freq_loss = MRSTFTLossDDP(n_bins=64, sample_rate=CONFIG.DATA.sr, device="cpu", scale='mel')
        
        self.save_hyperparameters(ignore=["train_dataset", "val_dataset"])
        # self.ema = ExponentialMovingAverage(self.parameters(), decay=0.995)

    def forward(self, x):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, collate_fn=CustomDataset.collate_fn, 
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.batch_size,
                          num_workers=CONFIG.TRAIN.workers, collate_fn=CustomDataset.collate_fn, 
                          persistent_workers=True)

    def forward_loss(self, x, y):
        if self.hparams.loss_type == 1:
            loss = self.time_loss(x, y)
        elif self.hparams.loss_type ==2:
            loss = self.freq_loss(x, y) + self.time_loss(x, y) * CONFIG.TRAIN.mse_weight
        elif self.hparams.loss_type ==3:

            x_sub = self.subband.wav_to_sub(x)
            y_sub = self.subband.wav_to_sub(y)
   
            subband_loss_weight = CONFIG.TRAIN.subband.weight_loss
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(x_sub, y_sub)
            subband_loss = subband_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc)

            stft_loss = self.freq_loss(x, y) + CONFIG.TRAIN.stft_weight_loss
            time_loss = self.time_loss(x, y) * CONFIG.TRAIN.mse_weight

            self.log('train_time_loss', stft_loss, logger=True)
            self.log('train_freq_loss', time_loss, logger=True)
            self.log('train_subband_loss', subband_loss, logger=True)

            loss = time_loss + stft_loss + subband_loss

        elif self.hparams.loss_type == 4:

            time_loss = self.time_loss(x, y) * CONFIG.TRAIN.mse_weight
            # stft_loss = self.freq_loss(x, y) + CONFIG.TRAIN.stft_weight_loss

            x_sub = self.subband.wav_to_sub(x)
            y_sub = self.subband.wav_to_sub(y)
            subband_loss_weight = CONFIG.TRAIN.subband.weight_loss
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(x_sub, y_sub)
            subband_loss = subband_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc)

            self.log('train_time_loss', time_loss, logger=True)
            # self.log('train_stft_loss', stft_loss, logger=True)
            self.log('train_subband_loss', subband_loss, logger=True)
            
            loss = time_loss + subband_loss

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
        x = self(x_in)
        loss = self.forward_loss(x, y)
        self.log('train_loss', loss, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_in, y = val_batch
        x = self(x_in)
        loss = self.forward_loss(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0:
            i = torch.randint(0, x_in.shape[0], (1,)).item()
            lr, hr, recon = torch.squeeze(x_in[i]), torch.squeeze(y[i]), torch.squeeze(x[i])
            self.trainer.logger.log_spectrogram(hr, lr, recon, self.current_epoch)
            # self.trainer.logger.waveform(hr, lr, recon, self.current_epoch)

    def configure_optimizers(self):
        if CONFIG.TRAIN.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate) 
        elif CONFIG.TRAIN.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif CONFIG.TRAIN.optimizer == 'adamax':
            optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=CONFIG.TRAIN.patience, factor=CONFIG.TRAIN.factor,
                                         verbose=True)

        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]
    

class TUNet_subband(BaseModel):
    def __init__(self, train_dataset, val_dataset):
        super(TUNet_subband, self).__init__(train_dataset, val_dataset)
        self.hparams['model_name'] = CONFIG.MODEL.model_name

        self.hparams['out_channels'] = CONFIG.MODEL.out_channels
        self.hparams['kernel_sizes'] = CONFIG.MODEL.kernel_sizes
        self.hparams['bottleneck_type'] = CONFIG.MODEL.bottleneck_type
        self.hparams['strides'] = CONFIG.MODEL.strides
        self.hparams['tfilm'] = CONFIG.MODEL.tfilm
        self.hparams['n_blocks'] = CONFIG.MODEL.n_blocks
        self.save_hyperparameters(ignore=["train_dataset", "val_dataset"])

        self.encoder = Encoder(max_len=self.hparams.max_len,
                               kernel_sizes=self.hparams.kernel_sizes,
                               strides=self.hparams.strides,
                               out_channels=self.hparams.out_channels,
                               tfilm=self.hparams.tfilm,
                               n_blocks=self.hparams.n_blocks)
        bottleneck_size = self.hparams.max_len // np.array(self.hparams.strides).prod()

        if self.hparams.bottleneck_type == 'performer':
            self.bottleneck = Performer(dim=self.hparams.out_channels[2], depth=CONFIG.MODEL.TRANSFORMER.depth,
                                        heads=CONFIG.MODEL.TRANSFORMER.heads, causal=False,
                                        dim_head=CONFIG.MODEL.TRANSFORMER.dim_head, local_window_size=bottleneck_size)
        elif self.hparams.bottleneck_type == 'lstm':
            self.bottleneck = nn.LSTM(input_size=self.hparams.out_channels[2], hidden_size=self.hparams.out_channels[2],
                                      num_layers=CONFIG.MODEL.TRANSFORMER.depth, batch_first=True)

        self.decoder = Decoder(in_len=self.encoder.out_len,
                               kernel_sizes=self.hparams.kernel_sizes,
                               strides=self.hparams.strides,
                               out_channels=self.hparams.out_channels,
                               tfilm=self.hparams.tfilm,
                               n_blocks=self.hparams.n_blocks)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        if self.hparams.bottleneck_type is not None:
            x3 = x3.permute([0, 2, 1])
            if self.hparams.bottleneck_type == 'performer':
                bottle_neck = self.bottleneck(x3)
            elif self.hparams.bottleneck_type == 'lstm':
                bottle_neck = self.bottleneck(x3)[0].clone()
            else:
                bottle_neck = self.bottleneck(inputs_embeds=x3)[0]
            bottle_neck += x3
            bottle_neck = bottle_neck.permute([0, 2, 1])
        else:
            bottle_neck = x3
        x_dec = self.decoder([x, x1, x2, bottle_neck])
        return x_dec
