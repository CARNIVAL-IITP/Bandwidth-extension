import tensorflow as tf
import librosa

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal.windows import hann as hanning


def generate_sr_sample(model, crop_length, in_dir_lr, save_path):

    window = hanning(crop_length)
    x_lr, fs = librosa.load(in_dir_lr, sr=None)

    length = x_lr.shape[0]

    batches = int((length - crop_length / 2) / (crop_length / 2))

    x_lr = x_lr[0: int(batches * crop_length / 2 + crop_length / 2)]

    for i in range(batches):
        x_lr_ = x_lr[int(i * crop_length / 2): int((i * crop_length / 2) + crop_length)]
        x_in = np.expand_dims(np.expand_dims(x_lr_, axis=-1), axis=0)
        x_in = tf.convert_to_tensor(x_in, dtype=tf.float32)
        pred = model(x_in)
        pred = pred.numpy()
        pred = np.squeeze(np.squeeze(pred))

        if i == 0:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            pred_audio = pred[0: int(crop_length / 2)]
        else:
            pred_audio_frame = pred * window
            pred_audio_font = pred_audio_frame[0: int(crop_length / 2)]
            pred_overlap = pred_audio_font + pred_audio_end

            pred_audio = np.concatenate((pred_audio, pred_overlap), axis=0)

            pred_audio_end = pred_audio_frame[int(crop_length / 2):]

            if i == batches - 1:
                pred_audio = np.concatenate((pred_audio, pred[int(crop_length / 2):]), axis=0)

    ## Post proecessing with notch filter
    # 4 kHz notch filter
    b, a = signal.iirnotch(4000, 30, fs)
    pred_audio = signal.lfilter(b, a, pred_audio)
     # 2 kHz notch filter
    d, c = signal.iirnotch(2000, 30, fs)
    pred_audio = signal.lfilter(d, c, pred_audio)

    ## Save pred audio as .wav file
    sf.write(save_path, pred_audio, samplerate=fs)