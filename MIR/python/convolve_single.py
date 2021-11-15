import numpy as np
import glob
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from pystoi import stoi

# ## Window filepath
# IR_filepath = 'C:/Users/User/Desktop/MIR/temp/1013/Mouth_simulator/16khz_sinesweep/masked/kf-ad/processed/'
# audio_filepath = 'C:/Users/User/Desktop/MIR/'
# output_filepath = 'C:/Users/User/Desktop/MIR/output/'

# wav_IR_name = 'impulse.wav'
# wav_audio_name = 'test1.wav'
# wav_output_name = 'output_masked_kf-ad1.wav'

# ## Linux filepath
IR_filepath = '/home/donghyun/Project/IITP/MIR/input/1013/Mouth_simulator/16khz_sinesweep/masked/kf-ad_3d/processed/'
audio_filepath = '/home/donghyun/Project/IITP/MIR/sample_audio_input/'
output_filepath = '/home/donghyun/Project/IITP/MIR/output/1013/masked/'

wav_IR_name = 'impulse.wav'
wav_audio_name = 'set200002_clncut.wav'
wav_output_name = 'output_kf-80_sitec_1000.wav'

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
    # print(data.shape, fs)
    return data

def write_wav(convolution, filepath):
    sf.write(filepath, convolution, samplerate=16000)

def normalize(data):
    high, low = abs(max(data)), abs(min(data))
    return data / max(high, low)

def convolution(audio_file, impulse_file, output_file):

    audio = read_wave(audio_file)
    # plt.plot(audio)
    # plt.show()
    # plt.close()
    # print(len(audio))

    ir = read_wave(impulse_file)[:1000,]
    # plt.plot(ir)
    # plt.show()
    # plt.close()

    # print(len(ir))

    # if len(audio) > len(ir):
    #     ir = zero_padding(len(audio), audio)

    # else:k
    #     audio = zero_padding(len(ir), ir)
    # plt.plot(ir)
    # plt.show()
    # plt.close()
    # print(len(audio), len(ir))

    convolution = normalize(signal.convolve(audio, ir, mode='same'))
    # plt.plot(convolution)
    # plt.show()

    # write_wav(convolution, output_file)
    STOI = stoi(audio, convolution, 16000, extended=False)
    print("STOI is :", STOI)
    np.savetxt("stoi_kf-ad_3d.txt", [STOI])

if __name__ == '__main__':
    convolution(audio_file, impulse_file, output_file)
