from scipy.io import wavfile
from scipy import signal
import numpy as np
import os, glob

def audio_read(dir):
    sr, audio = wavfile.read(dir)
    return sr, audio

# Input direcotry
original_dir = '/home/donghyun/Progress/Project/IITP/BWE/BWE_DB/TIMIT_dataset_16kHz_indexed/Test_speech/' 

if (os.path.isdir(original_dir) == False):
    os.listdir(original_dir)

print("Original filepath is: ", original_dir)
all_files = glob.glob(original_dir +"*.wav")
len_files = len(all_files)
# print("Length of all_files list is", len(all_files))
# exit()

output_dir = '/home/donghyun/Progress/Project/IITP/BWE/BWE_DB/TIMIT_dataset_16kHz_indexed/Test_speech/LPF/'

if (os.path.isdir(output_dir) == False):
    os.makedirs(output_dir)

print("Output file path is ", output_dir)
#exit()

sr = 16000 #samplerate
filecounter = 0

def main():
    for filecounter in range(0,len_files):# number of files in directory
        sr, audio = audio_read(all_files[filecounter]) #read audio
        outfiles = all_files[filecounter].replace(original_dir, output_dir) #make output filename same as original filename at output dir
        LPF = signal.firwin(101, cutoff = 0.5) #Low pass filter
        new_audio = signal.lfilter(LPF, [1.0], audio)
        wavfile.write(outfiles, sr, new_audio.astype(np.int16)) #write audio

if __name__ == "__main__":
    main()