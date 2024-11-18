import argparse
import glob
import os

import librosa
import numpy as np
import onnx
import onnxruntime
import soundfile as sf
import torch
import tqdm
import time

from config_TUNet import CONFIG

# parser = argparse.ArgumentParser()

# parser.add_argument('--onnx_path', default=None,
#                     help='path to onnx')
# args = parser.parse_args()

if __name__ == '__main__':
    # input_audio, _ = librosa.load('test_samples/input.wav', sr=16000)
# input_audio, _ = librosa.load('test_samples/SA1_2.wav', sr=16000)

    # onnx_model = onnx.load(path)

    # path = args.onnx_path
    path = '/home/dh2/Research/TUNet/TUNet-bwe-pretraining/lightning_logs/version_149/checkpoints/tunet.onnx'
    window = CONFIG.DATA.window_size
    stride = CONFIG.DATA.stride
    onnx_model = onnx.load(path)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 8
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(path, options)
    input_names = [x.name for x in session.get_inputs()]
    output_names = [x.name for x in session.get_outputs()]
    # print(input_names)
    # print(output_names)

    audio_files = glob.glob(os.path.join(CONFIG.TEST.in_dir, '*.wav'))
    # print(audio_files)
    # exit()
    hann = torch.sqrt(torch.hann_window(window))
    os.makedirs(CONFIG.TEST.out_dir, exist_ok=True)
    for file in tqdm.tqdm(audio_files, total=len(audio_files)):
        sig, _ = librosa.load(file, sr=16000)
        sig = torch.tensor(sig)
        # print(sig.shape)
        # print(sig)
        # exit()

        re_im = torch.stft(sig, window, stride, window=hann, return_complex=False).permute(1, 0, 2).unsqueeze(
    1).numpy().astype(np.float32)

        inputs = {input_names[i]: np.zeros([d.dim_value for d in _input.type.tensor_type.shape.dim],
                                           dtype=np.float32)
                  for i, _input in enumerate(onnx_model.graph.input)
                  }

        # start = time.perf_counter()
        # pred = self(x)
        # end = time.perf_counter()

        # inference_time = end-start

        elapsed = 0
        output_audio = []
        for t in range(re_im.shape[-1]):
            inputs[input_names[0]] = re_im[t]
            start = time.perf_counter()
            out, prev_mag, predictor_state, mlp_state = session.run(output_names, inputs)
            end = time.perf_counter()
            inputs[input_names[1]] = prev_mag
            inputs[input_names[2]] = predictor_state
            inputs[input_names[3]] = mlp_state
            output_audio.append(out)
            inference_time = end-start
        # print(inference_time)
            elapsed = elapsed + inference_time
        # print(output_audio)
        # print(output_audio.shape)
        # exit()
        output_audio = torch.tensor(np.concatenate(output_audio, 0))
        output_audio = output_audio.permute(1, 0, 2).contiguous()
        output_audio = torch.view_as_complex(output_audio)
        output_audio = torch.istft(output_audio, window, stride, window=hann)

        duration = librosa.get_duration(y=output_audio, sr=16000)
        # RTF = np.divide(elasped, duration)
        print(f"Total time to compute to enhance: {inference_time} (s)")
        print(f"Total duration of audio samples: {duration} (s)")
        print(f"RTF: {inference_time/duration}")
        # print(output_audio)
        # print(output_audio.shape)
        exit()
        sf.write(os.path.join(CONFIG.TEST.out_dir, os.path.basename(file)), output_audio, samplerate=16000,
                 subtype='PCM_16')
