"""Phase 3: log Mel spectrogram with tf.signal only (no librosa).

The browser does the same steps with tf.signal in TensorFlow.js (src/ml/features.js):

    1 second, 16 kHz, float in [-1, 1]            shape [16000]
    tf.signal.stft(frame 1024, step 256, hann)    shape [59, 513]
    power = |stft|^2
    mel = power x mel matrix (64 bins)            shape [59, 64]
    log(mel + 1e-6)
    per sample normalisation: (x - mean) / (std + 1e-6)
    add channel                                    shape [59, 64, 1]

Run this file to export public/model/mel.json and run the shape test:
    ml\\.venv\\Scripts\\python.exe ml\\features.py
"""
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

N_BINS = C.N_FFT // 2 + 1  # 513

MEL_MATRIX = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=C.N_MELS,
    num_spectrogram_bins=N_BINS,
    sample_rate=C.SAMPLE_RATE,
    lower_edge_hertz=C.MEL_FMIN,
    upper_edge_hertz=C.MEL_FMAX,
)  # [513, 64] float32


def log_mel_raw(waveform):
    """waveform: float32 [..., 16000] in [-1, 1]. Returns log Mel [..., 59, 64] before normalisation."""
    stft = tf.signal.stft(waveform, frame_length=C.N_FFT, frame_step=C.STFT_HOP,
                          fft_length=C.N_FFT, window_fn=tf.signal.hann_window, pad_end=False)
    power = tf.square(tf.abs(stft))
    mel = tf.matmul(power, MEL_MATRIX)
    return tf.math.log(mel + C.LOG_EPS)


def normalise(log_mel):
    """Per sample normalisation over the last two axes (time, mel)."""
    mean = tf.reduce_mean(log_mel, axis=[-2, -1], keepdims=True)
    std = tf.math.reduce_std(log_mel, axis=[-2, -1], keepdims=True)
    return (log_mel - mean) / (std + C.NORM_EPS)


def log_mel(waveform):
    """Full feature: float32 [..., 16000] -> [..., 59, 64, 1]."""
    return normalise(log_mel_raw(waveform))[..., tf.newaxis]


def int16_to_float(x):
    return tf.cast(x, tf.float32) / 32768.0


def features_numpy(x_int16, batch=512):
    """int16 [N, 16000] -> float32 numpy [N, 59, 64, 1]."""
    out = []
    for i in range(0, len(x_int16), batch):
        out.append(log_mel(int16_to_float(x_int16[i:i + batch])).numpy())
    return np.concatenate(out) if out else np.zeros((0, C.N_FRAMES, C.N_MELS, 1), np.float32)


def export_mel_json(path=None):
    path = path or os.path.join(C.PUBLIC_MODEL_DIR, "mel.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "description": "Feature settings for the PulseTrack crash sound CNN. Written by ml/features.py.",
        "sampleRate": C.SAMPLE_RATE,
        "windowSamples": C.WINDOW_SAMPLES,
        "hopSamples": C.HOP_SAMPLES,
        "frameLength": C.N_FFT,
        "fftLength": C.N_FFT,
        "frameStep": C.STFT_HOP,
        "window": "hann (periodic), tf.signal.hann_window",
        "padEnd": False,
        "numFrames": C.N_FRAMES,
        "numSpectrogramBins": N_BINS,
        "numMelBins": C.N_MELS,
        "melLowerEdgeHertz": C.MEL_FMIN,
        "melUpperEdgeHertz": C.MEL_FMAX,
        "spectrum": "power (magnitude squared)",
        "logEpsilon": C.LOG_EPS,
        "normalisation": "per sample: (x - mean) / (std + normEpsilon) over all frames and mel bins",
        "normEpsilon": C.NORM_EPS,
        "inputShape": [C.N_FRAMES, C.N_MELS, 1],
        "classes": C.CLASSES,
        "melMatrixShape": [N_BINS, C.N_MELS],
        "melMatrix": [[float(v) for v in row] for row in MEL_MATRIX.numpy()],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def _shape_test():
    rng = np.random.default_rng(0)
    one = rng.uniform(-0.5, 0.5, size=(C.WINDOW_SAMPLES,)).astype(np.float32)
    f1 = log_mel(one)
    assert f1.shape == (C.N_FRAMES, C.N_MELS, 1), f1.shape
    batch = rng.uniform(-0.5, 0.5, size=(3, C.WINDOW_SAMPLES)).astype(np.float32)
    fb = log_mel(batch)
    assert fb.shape == (3, C.N_FRAMES, C.N_MELS, 1), fb.shape
    assert MEL_MATRIX.shape == (N_BINS, C.N_MELS), MEL_MATRIX.shape
    assert np.all(np.isfinite(fb.numpy()))
    m = fb.numpy().reshape(3, -1)
    assert np.allclose(m.mean(axis=1), 0, atol=1e-3) and np.allclose(m.std(axis=1), 1, atol=1e-2)
    silent = log_mel(np.zeros((C.WINDOW_SAMPLES,), np.float32))
    assert np.all(np.isfinite(silent.numpy())), "silence must not give NaN"
    # batch and single must agree
    assert np.allclose(log_mel(batch[0]).numpy(), fb.numpy()[0], atol=1e-4)
    print(f"shape test OK: one window -> {tuple(f1.shape)}, batch -> {tuple(fb.shape)}, mel matrix {tuple(MEL_MATRIX.shape)}")


if __name__ == "__main__":
    _shape_test()
    print("mel.json written to", export_mel_json())
