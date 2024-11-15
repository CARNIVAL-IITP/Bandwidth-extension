import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf
import torch
from numpy.random import default_rng
from pydtmc import MarkovChain
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config_folder.config_FRN_NB_BWE import CONFIG
from utils.utils import decimate, frame

np.random.seed(0)
rng = default_rng()


def load_audio(
        path,
        sample_rate: int = 16000,
        chunk_len=None,
):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        audio_len = f.frames

        if chunk_len is not None and chunk_len < audio_len:
            start_index = torch.randint(0, audio_len - chunk_len, (1,))[0]

            frames = f._prepare_read(start_index, start_index + chunk_len, -1)
            audio = f.read(frames, always_2d=True, dtype="float32")

        else:
            audio = f.read(always_2d=True, dtype="float32")

    if sr != sample_rate:
        audio = librosa.resample(np.squeeze(audio), orig_sr = sr, target_sr = sample_rate)[:, np.newaxis]
    return audio.T

def mask_input(sig):
    sig = np.reshape(sig, (-1, CONFIG.TASK.mask_chunk))
    mask = np.ones(len(sig))
    mask[:int(CONFIG.TASK.mask_ratio * len(mask))] = 0
    np.random.shuffle(mask)
    sig *= mask[:, np.newaxis]
    sig = np.reshape(sig, -1)
    return sig

def pad(sig, length):
    if sig.shape[1] < length:
        pad_len = length - sig.shape[1]
        sig = torch.hstack((sig, torch.zeros((sig.shape[0], pad_len))))

    else:
        start = random.randint(0, sig.shape[1] - length)
        sig = sig[:, start:start + length]
    return sig

    
class MaskGenerator:
    def __init__(self, is_train=True, probs=((0.9, 0.1), (0.5, 0.1), (0.5, 0.5))):
        '''
            is_train: if True, mask generator for training otherwise for evaluation
            probs: a list of transition probability (p_N, p_L) for Markov Chain. Only allow 1 tuple if 'is_train=False'
        '''
        self.is_train = is_train
        self.probs = probs
        self.mcs = []
        if self.is_train:
            for prob in probs:
                self.mcs.append(MarkovChain([[prob[0], 1 - prob[0]], [1 - prob[1], prob[1]]], ['1', '0']))
        else:
            assert len(probs) == 1
            prob = self.probs[0]
            self.mcs.append(MarkovChain([[prob[0], 1 - prob[0]], [1 - prob[1], prob[1]]], ['1', '0']))

    def gen_mask(self, length, seed=0):
        if self.is_train:
            mc = random.choice(self.mcs)
        else:
            mc = self.mcs[0]
        mask = mc.walk(length - 1, seed=seed)
        mask = np.array(list(map(int, mask)))
        return mask


class TestLoader(Dataset):
    def __init__(self):
        dataset_name = CONFIG.DATA.dataset
        self.mask = CONFIG.DATA.EVAL.masking
        self.task = CONFIG.TASK.task
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']
        txt_list = CONFIG.DATA.data_dir[dataset_name]['test']
        self.data_list = self.load_txt(txt_list)
        self.downsampling = CONFIG.TASK.downsampling
        self.down_rate = CONFIG.DATA.ratio
        if self.mask == 'real':
            trace_txt = glob.glob(os.path.join(CONFIG.DATA.EVAL.trace_path, '*.txt'))
            trace_txt.sort()
            self.trace_list = [1 - np.array(list(map(int, open(txt, 'r').read().strip('\n').split('\n')))) for txt in
                               trace_txt]
        else:
            self.mask_generator = MaskGenerator(is_train=True, probs=CONFIG.DATA.EVAL.transition_probs)

        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.window_size = CONFIG.DATA.window_size
        self.audio_chunk_len = CONFIG.DATA.audio_chunk_len
        self.p_size = CONFIG.DATA.EVAL.packet_size  # 20ms
        self.hann = torch.sqrt(torch.hann_window(self.window_size))

    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target

    def lowpass(self, sig):
        low_sr = self.sr // self.down_rate
        if self.downsampling == 'augment':
            n = random.choice(CONFIG.TASK.orders)
            ripple = random.choice(CONFIG.TASK.ripples)
            sig = decimate(sig, self.down_rate, n=n, ripple=ripple)
            sig = librosa.resample(sig, low_sr, self.sr)
        elif self.downsampling == 'cheby':
            sig = decimate(sig, self.down_rate)
            sig = librosa.resample(sig, orig_sr = low_sr, target_sr = self.sr)
        else:
            sig = librosa.resample(sig, self.sr, low_sr, res_type=self.downsampling)
            sig = librosa.resample(sig, low_sr, self.sr)
        return sig
    
    def __getitem__(self, index):
        target = load_audio(self.data_list[index], sample_rate=self.sr)
        target = target[:, :(target.shape[1] // self.p_size) * self.p_size]

        if self.task == 'PLC':
            sig = np.reshape(target, (-1, self.p_size)).copy()
            if self.mask == 'real':
                mask = self.trace_list[index % len(self.trace_list)]
                mask = np.repeat(mask, np.ceil(len(sig) / len(mask)), 0)[:len(sig)][:, np.newaxis]
            else:
                mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
            sig *= mask
            sig = torch.tensor(sig).reshape(-1)

        if self.task == 'HB-BWE':
            sig = self.lowpass(target)
            if len(target) != len(sig):
                sig = pad(sig, len(target))

        target = torch.tensor(target).squeeze(0)
        sig = torch.tensor(sig).reshape(-1)
        sig_wav = sig.clone()
        target_wav = target.clone()

        target = torch.stft(target, self.window_size, self.stride, window=self.hann,
                            return_complex=True)
        target = torch.view_as_real(target).permute(2, 0, 1).float()
        sig = torch.stft(sig, self.window_size, self.stride, window=self.hann, 
                         return_complex=True)
        sig = torch.view_as_real(sig).permute(2, 0, 1).float()
        return sig, target, sig_wav, target_wav


class BlindTestLoader(Dataset):
    def __init__(self, test_dir):
        self.data_list = glob.glob(os.path.join(test_dir, '*.wav'))
        self.sr = CONFIG.DATA.sr
        self.stride = CONFIG.DATA.stride
        self.chunk_len = CONFIG.DATA.window_size
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr)
        sig = torch.from_numpy(sig).squeeze(0)
        sig = torch.stft(sig, self.chunk_len, self.stride, window=self.hann, 
                         return_complex=True)
        sig = torch.view_as_real(sig).permute(2, 0, 1).float()
        return sig


class TrainDataset_FRN_NB_BWE(Dataset):

    def __init__(self, mode='train'):
        dataset_name = CONFIG.DATA.dataset
        self.target_root = CONFIG.DATA.data_dir[dataset_name]['root']

        txt_list = CONFIG.DATA.data_dir[dataset_name]['train']
        self.data_list = self.load_txt(txt_list)
        # txt_list = data_dir[name]['train']
        if dataset_name == 'sitec-rir-each':
            if mode == 'train':
                txt_list = CONFIG.DATA.data_dir[dataset_name]['train']
                self.data_list = self.load_txt(txt_list)
                self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

            elif mode == 'val':
                txt_list = CONFIG.DATA.data_dir[dataset_name]['val']
                self.data_list = self.load_txt(txt_list)
                _, self.data_list = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        else:
            if mode == 'train':
                self.data_list, _ = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)
            elif mode == 'val':
                _, self.data_list = train_test_split(self.data_list, test_size=CONFIG.TRAIN.val_split, random_state=0)

        self.task = CONFIG.TASK.task
        self.p_sizes = CONFIG.DATA.TRAIN.packet_sizes
        self.mode = mode
        self.sr = CONFIG.DATA.sr
        self.window = CONFIG.DATA.audio_chunk_len
        self.stride = CONFIG.DATA.stride
        self.chunk_len = CONFIG.DATA.window_size
        self.hann = torch.sqrt(torch.hann_window(self.chunk_len))
        self.mask_generator = MaskGenerator(is_train=True, probs=CONFIG.DATA.TRAIN.transition_probs)
        self.downsampling = CONFIG.TASK.downsampling
        self.down_rate = CONFIG.DATA.ratio

    
    def __len__(self):
        return len(self.data_list)

    def load_txt(self, txt_list):
        target = []
        with open(txt_list) as f:
            for line in f:
                target.append(os.path.join(self.target_root, line.strip('\n')))
        target = list(set(target))
        target.sort()
        return target

    def fetch_audio(self, index):
        sig = load_audio(self.data_list[index], sample_rate=self.sr, chunk_len=self.window)
        while sig.shape[1] < self.window:
            idx = torch.randint(0, len(self.data_list), (1,))[0]
            pad_len = self.window - sig.shape[1]
            if pad_len < 0.02 * self.sr:
                padding = np.zeros((1, pad_len), dtype=float)
            else:
                padding = load_audio(self.data_list[idx], sample_rate=self.sr, chunk_len=pad_len)
            sig = np.hstack((sig, padding))
        return sig

    def lowpass(self, sig):
        low_sr = self.sr // self.down_rate
        if self.downsampling == 'augment':
            n = random.choice(CONFIG.TASK.orders)
            ripple = random.choice(CONFIG.TASK.ripples)
            sig = decimate(sig, self.down_rate, n=n, ripple=ripple)
            sig = librosa.resample(sig, low_sr, self.sr)
        elif self.downsampling == 'cheby':
            sig = decimate(sig, self.down_rate)
            sig = librosa.resample(sig, orig_sr = low_sr, target_sr = self.sr)
        else:
            sig = librosa.resample(sig, orig_sr = self.sr, target_sr = low_sr, res_type=self.downsampling)
            sig = librosa.resample(sig, orig_sr = low_sr, target_sr = self.sr)
        return sig

    def __getitem__(self, index):
        sig = self.fetch_audio(index)
        # print('0',sig.shape) #0 (1, 40960)

        sig = sig.reshape(-1).astype(float)
        # print('1', sig.shape) # 1 (40960,)
        target = torch.tensor(sig.copy())
        # p_size = random.choice(self.p_sizes)

        # sig = np.reshape(sig, (-1, p_size))
        # print('2', sig.shape) #2 (512, 80)
        if self.task == 'PLC':
            mask = self.mask_generator.gen_mask(len(sig), seed=index)[:, np.newaxis]
            sig *= mask
            sig = np.reshape(sig, -1) # add
            sig = torch.tensor(sig.copy())
        # print('3', sig.shape, target.shape) # 3 torch.Size([40960]) torch.Size([40960])

        if self.task == 'HB-BWE':
            low_sig = self.lowpass(sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            low_sig = torch.tensor(low_sig.copy())

        if self.task == 'MSM-clean':
            low_sig = mask_input(target)
            low_sig = torch.tensor(low_sig.copy())

        if self.task == 'MSM-noisy':
            low_sig = self.lowpass(sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))            
            target = torch.tensor(low_sig.copy())
            low_sig = mask_input(low_sig)
            low_sig = torch.tensor(low_sig.copy())
        
        if self.task == 'NB-NAE':
            low_sig = self.lowpass(sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            target = torch.tensor(low_sig.copy())

        if self.task == 'HB-NAE':
            low_sig = sig
            target = sig

        if self.task == 'NB-BWE':
            low_sig = self.lowpass(sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            target = torch.tensor(low_sig.copy())
            low_sig = self.lowpass(low_sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            low_sig = torch.tensor(low_sig.copy())
        
        if self.task == 'NB-BWE+MSM':
            low_sig = self.lowpass(sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            target = torch.tensor(low_sig.copy())
            low_sig = self.lowpass(low_sig)
            if len(target) != len(low_sig):
                low_sig = pad(low_sig, len(target))
            low_sig = mask_input(low_sig)
            low_sig = torch.tensor(low_sig.copy())

        target = torch.stft(target, self.chunk_len, self.stride, window=self.hann,
                            return_complex=True)
        target = torch.view_as_real(target).permute(2, 0, 1).float()
        low_sig = torch.stft(low_sig, self.chunk_len, self.stride, window=self.hann, 
                             return_complex=True)
        low_sig = torch.view_as_real(low_sig).permute(2, 0, 1).float()

        return low_sig, target