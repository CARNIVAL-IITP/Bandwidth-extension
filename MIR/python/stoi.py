from pystoi import stoi
import soundfile as sf
import numpy as np
import glob, os


input_clean_path = '/home/donghyun/Progress/Project/IITP/BWE/BWE_DB/TIMIT_dataset_16kHz_indexed/Test_speech/'
input_enhanced_path = '/home/donghyun/Progress/Project/IITP/BWE/BWE_DB/TIMIT_dataset_16kHz_indexed/Test_speech/LPF/'

stoi_list = []
input_clean = glob.glob(input_clean_path +"*.wav")
input_enhanced = glob.glob(input_enhanced_path +"*.wav")
# print(input_clean[1].shape)
# print(input_enhanced[1].shape)
i = 0

len_files = len(input_clean)
print(len_files)
fs = 16000

# exit()

if __name__ == '__main__':
    for i in range(0, len_files):
        clean, fs = sf.read(input_clean[i])
        enhanced , fs = sf.read(input_enhanced[i])
        STOI = stoi(clean, enhanced, fs_sig = fs, extended = False)
        stoi_list.append(STOI)
        print("STOI: ", STOI)

# print(stoi_list)
avg_stoi = np.mean(stoi_list)
print("Average STOI is: ", avg_stoi)
np.savetxt("stoi_LPF.txt", [avg_stoi])

