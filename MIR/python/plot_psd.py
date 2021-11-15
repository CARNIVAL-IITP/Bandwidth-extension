import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import filtfilt

input_original_path = '/home/donghyun/Project/IITP/MIR/sample_audio_input/'
input_masked_path = '/home/donghyun/Project/IITP/MIR/output/1013/masked/'

wav_original_name = 'set200002_clncut.wav'
wav_masked1_name = 'output_kf99_sitec_1000.wav'
wav_masked2_name = 'output_dental_sitec_1000.wav'

wav_original = input_original_path + wav_original_name
wav_masked1 = input_masked_path + wav_masked1_name
wav_masked2 = input_masked_path + wav_masked2_name

# fig = plt.figure(constrained_layout=True)

# s1_w = fig.add_subplot(1,3,1)
# y, sr = librosa.load(wav_original, sr=16000)
# plt.psd(y, 150, sr)
# plt.xlabel('Frequency (Hz)')
# plt.title('Original')

# s1 = fig.add_subplot(1, 3, 2)
# y, sr = librosa.load(wav_masked1, sr=16000)
# plt.psd(y, 150, sr)
# plt.xlabel('Frequency (Hz)')
# plt.title('Masked_Dental')

# s1 = fig.add_subplot(1, 3, 3)
# y, sr = librosa.load(wav_masked2, sr=16000)
# plt.psd(y, 150, sr,pad_to=512 ,noverlap=75)
# plt.xlabel('Frequency (Hz)')
# plt.title('Masked_fabric')

# plt.tight_layout()
# plt.show()
fig = plt.figure()

y_label = [-150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30]
x_ticks = np.arange(0,9000,1000)
yrange = (-150, -30)

s1_w = fig.add_subplot(1,3,1)
y, sr = librosa.load(wav_original, sr=16000)
y*=3.5
plt.psd(y, 150, sr)
plt.xticks(x_ticks)
plt.yticks(y_label, labels = y_label)
plt.grid(True)
plt.ylim((yrange))
plt.xlabel('Frequency (Hz)')
plt.title('Original')

s1 = fig.add_subplot(1, 3, 2)
y, sr = librosa.load(wav_masked1, sr=16000)
plt.psd(y, 150, sr)
plt.xticks(x_ticks)
plt.yticks(y_label, labels = y_label)
plt.grid(True)
plt.ylim((yrange))
plt.xlabel('Frequency (Hz)')
plt.title('KF-99 Mask')

s1 = fig.add_subplot(1, 3, 3)
y, sr = librosa.load(wav_masked2, sr=16000)
plt.psd(y, 150, sr,pad_to=512 ,noverlap=75)
plt.xticks(x_ticks)
plt.yticks(y_label, labels = y_label)
plt.grid(True)
plt.ylim((yrange))
plt.xlabel('Frequency (Hz)')
plt.title('Dental Mask')

# plt.savefig('dental_fabric_psd.jpg')
plt.tight_layout()
plt.show()
