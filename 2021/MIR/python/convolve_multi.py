import os
import random
from math import remainder
from os import makedirs

import numpy as np
import soundfile as sf
from natsort import natsorted
from pystoi import stoi
from tqdm.auto import tqdm


input_filepath = ''
output_filepath = ''

makedirs(output_filepath, exist_ok = True)

IR_filepath = './impulse_output/anechoic/using/10_5_44100/'


wav_IR_name1 = 'kf80.wav'
wav_IR_name2 = 'kf94.wav'
wav_IR_name3 = 'kf99.wav'
wav_IR_name4 = 'dental.wav'
wav_IR_name5 = 'fabric.wav'

impulse_file1 = IR_filepath + wav_IR_name1
impulse_file2 = IR_filepath + wav_IR_name2
impulse_file3 = IR_filepath + wav_IR_name3
impulse_file4 = IR_filepath + wav_IR_name4
impulse_file5 = IR_filepath + wav_IR_name5

train_impulse_files = [impulse_file2, impulse_file3, impulse_file4, impulse_file5]
test_impulse_files = [impulse_file1, impulse_file2, impulse_file3, impulse_file4, impulse_file5]


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

def convolution(audio_file, impulse_file):

    original_audio = read_wave(audio_file)

    ir = random.choice(impulse_file) #mixed
    ir = impulse_file4 #dental

    head, tail = os.path.split(ir)
    impulse = tail.split('.wav')
    tail_name = impulse[0]

    ir = read_wave(ir)[:1000,]

    return convolution, tail_name
    

if __name__ == '__main__':
    hr_files = os.listdir(input_filepath)

    hr_files = natsorted(hr_files)
    # hr_files.sort()
    
    hr_file_list = []
    for hr_file in hr_files:
        hr_file_list.append(input_filepath + hr_file)


    for i in tqdm(range(len(hr_file_list))):
        convolve, tail_name = convolution(hr_file_list[i],impulse_file=test_impulse_files)
        # convolve, tail_name = convolution(audio_file=filename, impulse_file=test_impulse_files)

        head, tail = os.path.split(hr_file_list[i])
        tail = tail.split('.wav')[0]

        outputfile = output_filepath + str(i) + '_'+ tail + '_' + tail_name + '.wav'
        # outputfile = output_filepath + str(i) + '_'+ tail + '.wav'
        write_wav(convolve,outputfile)

        # hr = read_wave(hr_file_list[i])
        # write_wav(hr,outputfile)
