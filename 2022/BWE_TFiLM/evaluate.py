import os, librosa
import numpy as np
import soundfile as sf

from pystoi import stoi
from tqdm.auto import tqdm
from natsort import natsorted

input_clean_path = os.path.abspath(os.getcwd()) + '/output/clean/'
input_enhanced_path = os.path.abspath(os.getcwd()) + '/output/enhanced/'

txt_name = os.path.abspath(os.getcwd()) + '/result/model_name.txt'

def SNR(x, ref):
    ref_pow = (ref**2).mean().mean() + np.finfo('float32').eps
    dif_pow = ((x - ref)**2).mean().mean() + np.finfo('float32').eps
    snr_val = 10 * np.log10(ref_pow / dif_pow)
    return snr_val

def SI_SDR(target, preds):
    EPS = 1e-8
    alpha = (np.sum(preds * target, axis=-1, keepdims=True) + EPS) / (np.sum(target ** 2, axis=-1, keepdims=True) + EPS)
    target_scaled = alpha * target
    noise = target_scaled - preds
    si_sdr_value = (np.sum(target_scaled ** 2, axis=-1) + EPS) / (np.sum(noise ** 2, axis=-1) + EPS)
    si_sdr_value = 10 * np.log10(si_sdr_value)
    return si_sdr_value

def get_power(x, nfft):
    S = librosa.stft(x, nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8) 
    return S

def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=2048)
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high

def compute_metrics(x_hr, pred_audio, fs):
    py_stoi = stoi(x_hr, pred_audio, fs, extended=False)
    estoi = stoi(x_hr, pred_audio, fs, extended=True)
    snr = SNR(x_hr, pred_audio)
    lsd, lsd_high = LSD(x_hr, pred_audio)
    sisdr = SI_SDR(x_hr, pred_audio)
    return np.array([py_stoi, estoi, snr, lsd, lsd_high, sisdr])

def evaluate_dataset(input_clean_path, input_enhanced_path):
    results = []

    hr_files = os.listdir(input_clean_path)
    hr_files = natsorted(hr_files)
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(input_clean_path + hr_file)

    lr_files = os.listdir(input_enhanced_path)
    lr_files = natsorted(lr_files)
    lr_file_list = []
    for lr_file in lr_files:
        lr_file_list.append(input_enhanced_path + lr_file)
 
    for i in tqdm(range(len(lr_file_list))):

        x_hr, fs = sf.read(hr_file_list[i])
        pred, fs = sf.read(lr_file_list[i])

        ret = compute_metrics(x_hr, pred, fs)
        results.append(ret)
    
    results = np.array(results)
    
    return np.vstack((results.mean(0), results.std(0))).T

res = evaluate_dataset(input_clean_path, input_enhanced_path)
print("Evaluate-- STOI: {} ESTOI: {} SNR: {} LSD: {} LSD-HF: {} SI-SDR: {}".format(res[0], res[1], res[2], res[3], res[4], res[5]))
file = open(txt_name, "w")
f = "{0:<16} {1:<16}"
file.write(f.format("Mean", "Std")+"\n")
file.write("---------------------------------\n")
file.write(str(res))
file.write("\n")
file.write("---------------------------------\n")
metric = "STOI, ESTOI, SNR, LSD, LSD-HF, SI-SDR"
file.write(metric)
file.close()
print(txt_name)