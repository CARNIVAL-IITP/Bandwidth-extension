
import numpy as np
import torch
import torch.nn as nn

from torch import nn
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from models.transformer import TransformerBlock


class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x


def SubPixel1d(tensor, r): #(b,r,w)
    ps = nn.PixelShuffle(r)
    tensor = torch.unsqueeze(tensor, -1) #(b,r,w,1)
    tensor = ps(tensor)
    #print(tensor.shape) #(b,1,w*r,r)
    tensor = torch.mean(tensor, -1)
    #print(tensor.shape) #(b,1,w*r)
    return tensor


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


class AFiLM(nn.Module):
    def __init__(self, block_size, input_dim, max_len, **kwargs):
        super(AFiLM, self).__init__(**kwargs)
        self.block_size = block_size
        self.max_pool = nn.MaxPool1d(kernel_size=self.block_size)
        # self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True)
        # self.transformer = nn.Transformer()
        # self.maximum_position_encoding = max_len // 128 #block_size
        self.max_len = max_len
        # print(self.max_len)
        # exit()
        self.transformer = TransformerBlock(num_layers = 4, embed_dim= input_dim, 
                                            num_heads=8, ff_dim=2048, 
                                            maximum_position_encoding=self.max_len // 128)

    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then
            runs LSTM to generate normalization weights.
        """
        x_in_down = self.max_pool(x_in).permute([0, 2, 1])
        # if self.afilm == True:
        #     x_in_down = self.transformer
        # x_transformer = TransformerBlock(num_layers=4, embed_dim=n_filters, 
        #                             num_heads=8, ff_dim=2048, maximum_position_encoding=max_len)
        # x_rnn, _ = self.lstm(x_in_down)
        x_in_down = self.max_pool(x_in).permute([0,2,1])#pool_size=n_block, padding='valid'))(x_in)
        x_transformer = self.transformer(x_in_down)
        return x_transformer.permute([0, 2, 1])

    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective  blocks.
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

