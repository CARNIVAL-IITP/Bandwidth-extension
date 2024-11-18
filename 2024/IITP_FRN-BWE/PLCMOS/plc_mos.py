import math
import os

import librosa
import numpy as np
# import onnxruntime as ort
from numpy.fft import rfft
from numpy.lib.stride_tricks import as_strided


class PLCMOSEstimator():
    def __init__(self, model_version=1):
        """
        Initialize a PLC-MOS model of a given version. There are currently three models available, v0 (intrusive)
        and v1 (both non-intrusive and intrusive available). The default is to use the v1 models.
        """

        self.model_version = model_version
        model_paths = [
            # v0 model:
            [("models/plcmos_v0.onnx", 999999999999), (None, 0)],

            # v1 models:
            [("models/plcmos_v1_intrusive.onnx", 768),
             ("models/plcmos_v1_nonintrusive.onnx", 999999999999)],
        ]
        self.sessions = []
        self.max_lens = []
        options = ort.SessionOptions()
        options.intra_op_num_threads = 8
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for path, max_len in model_paths[model_version]:
            if not path is None:
                file_dir = os.path.dirname(os.path.realpath(__file__))
                self.sessions.append(ort.InferenceSession(
                    os.path.join(file_dir, path), options))
                self.max_lens.append(max_len)
            else:
                self.sessions.append(None)
                self.max_lens.append(0)

    def logpow_dns(self, sig, floor=-30.):
        """
        Compute log power of complex spectrum.

        Floor any -`np.inf` value to (nonzero minimum + `floor`) dB.
        If all values are 0s, floor all values to -80 dB.
        """
        log10e = np.log10(np.e)
        pspec = sig.real ** 2 + sig.imag ** 2
        zeros = pspec == 0
        logp = np.empty_like(pspec)
        if np.any(~zeros):
            logp[~zeros] = np.log(pspec[~zeros])
            logp[zeros] = np.log(pspec[~zeros].min()) + floor / 10 / log10e
        else:
            logp.fill(-80 / 10 / log10e)

        return logp

    def hop2hsize(self, wind, hop):
        """
        Convert hop fraction to integer size if necessary.
        """
        if hop >= 1:
            assert type(hop) == int, "Hop size must be integer!"
            return hop
        else:
            assert 0 < hop < 1, "Hop fraction has to be in range (0,1)!"
            return int(len(wind) * hop)

    def stana(self, sig, sr, wind, hop, synth=False, center=False):
        """
        Short term analysis by windowing
        """
        ssize = len(sig)
        fsize = len(wind)
        hsize = self.hop2hsize(wind, hop)
        if synth:
            sstart = hsize - fsize  # int(-fsize * (1-hfrac))
        elif center:
            sstart = -int(len(wind) / 2)  # odd window centered at exactly n=0
        else:
            sstart = 0
        send = ssize

        nframe = math.ceil((send - sstart) / hsize)

        # Calculate zero-padding sizes
        zpleft = -sstart
        zpright = (nframe - 1) * hsize + fsize - zpleft - ssize
        if zpleft > 0 or zpright > 0:
            sigpad = np.zeros(ssize + zpleft + zpright, dtype=sig.dtype)
            sigpad[zpleft:len(sigpad) - zpright] = sig
        else:
            sigpad = sig

        return as_strided(sigpad, shape=(nframe, fsize),
                          strides=(sig.itemsize * hsize, sig.itemsize)) * wind

    def stft(self, sig, sr, wind, hop, nfft):
        """
        Compute STFT: window + rfft
        """
        frames = self.stana(sig, sr, wind, hop, synth=True)
        return rfft(frames, n=nfft)

    def stft_transform(self, audio, dft_size=512, hop_fraction=0.5, sr=16000):
        """
        Compute STFT parameters, then compute STFT
        """
        window = np.hamming(dft_size + 1)
        window = window[:-1]
        amp = np.abs(self.stft(audio, sr, window, hop_fraction, dft_size))
        feat = self.logpow_dns(amp, floor=-120.)
        return feat / 20.

    def run(self, audio_degraded, audio_clean=None, combined=False):
        """
        Run the PLCMOS model and return the MOS for the given audio. If a clean audio file is passed and the
        selected model version has an intrusive version, that version will be used, otherwise, the nonintrusive
        model will be used. If combined is set to true (default), the mean of intrusive and nonintrusive models
        results will be returned, when both are available

        For intrusive models, the clean reference should be the unprocessed audio file the degraded audio is
        based on. It is not required to be aligned with the degraded audio.

        Audio data should be 16kHz, mono, [-1, 1] range.
        """
        audio_features_degraded = np.float32(self.stft_transform(audio_degraded))[
            np.newaxis, np.newaxis, ...]
        assert len(
            audio_features_degraded) <= self.max_lens[0], "Maximum input length exceeded"

        if audio_clean is None:
            combined = False

        mos = 0

        session = self.sessions[0]
        assert not session is None, "Intrusive model not available for this model version."
        audio_features_clean = np.float32(self.stft_transform(audio_clean))[
            np.newaxis, np.newaxis, ...]
        assert len(
            audio_features_clean) <= self.max_lens[0], "Maximum input length exceeded"
        onnx_inputs = {"degraded_audio": audio_features_degraded,
                       "clean_audio": audio_features_clean}
        mos = float(session.run(None, onnx_inputs)[0])

        session = self.sessions[1]
        assert not session is None, "Nonintrusive model not available for this model version."
        onnx_inputs = {"degraded_audio": audio_features_degraded}
        mos_2 = float(session.run(None, onnx_inputs)[0])
        mos = [mos, mos_2]
        return mos
