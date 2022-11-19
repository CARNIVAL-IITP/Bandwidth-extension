import glob
import os
import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pystoi import stoi
from scipy import signal
from tqdm import tqdm

fs = 44100
IR_filepath = './impulse_output/anechoic/fabric/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf80/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf94/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf99/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/dental/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf-ad/5_10_44.1khz/'
# IR_filepath = './impulse_output/anechoic/nomask/5_10_44.1khz/'


# IR_filepath = './impulse_output/anechoic/fabric/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf80/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf94/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf99/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/dental/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/kf-ad/10_5_44.1khz/'
# IR_filepath = './impulse_output/anechoic/nomask/10_5_44.1khz/'

# fs = 16000
# IR_filepath = './impulse_output/anechoic/fabric/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/kf80/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/kf94/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/kf99/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/dental/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/kf-ad/5_10_16khz/'
# IR_filepath = './impulse_output/anechoic/nomask/5_10_16khz/'

 
# IR_filepath = './impulse_output/anechoic/fabric/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/kf80/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/kf94/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/kf99/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/dental/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/kf-ad/10_5_16khz/'
# IR_filepath = './impulse_output/anechoic/nomask/10_5_16khz/'


audio_filepath = './sample_audio_input/'
wav_audio_name = 'p347_001.wav'

output_filepath = './output/'

wav_IR_name = '1_estimated_mir_fabric_10_5_44100.wav'
wav_output_name = 'temp_output_fabric_10_5_44100_normalized.wav'


impulse_file = IR_filepath + wav_IR_name
audio_file = audio_filepath + wav_audio_name
output_file = output_filepath + wav_output_name

def zero_padding(length, data):
    zeros = np.zeros(length)
    zeros[:len(data)] = data
    data = zeros
    return data

def read_wave(filepath):
    data, fs = sf.read(filepath, dtype='float32')
    return data

def write_wav(convolution, filepath):
    sf.write(filepath, convolution, samplerate=16000)

def normalize(data):
    high, low = abs(max(data)), abs(min(data))
    return data / max(high, low)


def convolution(original_audio, impulse_file):

    ir = read_wave(impulse_file)[:1000,]
    plt.plot(ir)
    plt.show()
    plt.close()

    convolution = normalize(signal.convolve(original_audio, ir, mode='same'))
    plt.plot(convolution)
    plt.show()
    plt.close()

    return original_audio, convolution


if __name__ == '__main__':

    files_0 = os.listdir(IR_filepath)
    files_wav = [file_0 for file_0 in files_0 if file_0.endswith(".wav")]
    files_wav.sort()

    for file in tqdm(files_wav):

        original_audio = read_wave(audio_file)
        plt.plot(original_audio)
        plt.show()
        plt.close()

        audio, convoluted_audio = convolution(original_audio, IR_filepath + file)

        write_wav(convolution, output_file)
        py_stoi = stoi(original_audio, convolution, fs)
        print(output_file)
        print("STOI is :", py_stoi)