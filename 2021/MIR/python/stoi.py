from pystoi import stoi
import soundfile as sf


original_audio = './sample_audio_input/set200002_clncut.wav'
degraded_audio = './sample_audio_input/degraded_sample.wav'

clean, fs = sf.read(original_audio)
noisy,fs = sf.read(degraded_audio)

STOI = stoi(clean, noisy, 16000, extended=False)
print("STOI is :", STOI)