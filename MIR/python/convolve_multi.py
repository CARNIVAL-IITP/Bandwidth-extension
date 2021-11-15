import numpy as np
import glob, os
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
from pystoi import stoi

input_filepath = '/home/donghyun/Progress/Project/IITP/MIR/Sitec/Dict01/clncut/female/fcb1jkh00s200/'
output_filepath = '/home/donghyun/Progress/Project/IITP/MIR/output/sitec/dental/'

# ## Linux filepath
IR_filepath = '/home/donghyun/Progress/Project/IITP/MIR/input/1013/Mouth_simulator/16khz_sinesweep/masked/dental/processed/'
# audio_filepath = '/home/donghyun/Progress/Project/IITP/MIR/audio/'
# output_filepath = '/home/donghyun/Progress/Project/IITP/MIR/output/1013/masked/'

wav_IR_name = 'impulse.wav'
# wav_audio_name = 'set200002_clncut.wav'
# wav_output_name = 'output_dental_sitec_1000.wav'

impulse_file = IR_filepath + wav_IR_name
# audio_file = audio_filepath + wav_audio_name
# output_file = output_filepath + wav_output_name

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

def convolution(audio_file, impulse_file):

    audio = read_wave(audio_file)
    # plt.plot(audio)
    # plt.show()
    # plt.close()

    ir = read_wave(impulse_file)[:1000,]
    # plt.plot(ir)
    # plt.show()
    # plt.close()

    convolution = normalize(signal.convolve(audio, ir, mode='same'))
    return convolution
    # plt.plot(convolution)
    # plt.show()

    # print("STOI is :", stoi(audio, convolution, 16000, extended=False))

if __name__ == '__main__':
    for filename in glob.glob(os.path.join(input_filepath, '*.wav')):
        convolve= convolution(audio_file=filename, impulse_file=impulse_file)
        head, tail = os.path.split(filename)
        print(tail)
        outputfile = output_filepath + 'convolved_dental_' + tail
        write_wav(convolve,outputfile)
