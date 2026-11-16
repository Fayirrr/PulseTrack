"""Shared settings for the PulseTrack crash sound CNN.

Every script in ml/ imports from here so that the numbers cannot drift apart.
The same numbers are exported to public/model/mel.json for the browser.
"""
import os

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(ML_DIR)
DATA_DIR = os.path.join(ML_DIR, "data")
MIVIA_DIR = os.path.join(DATA_DIR, "MIVIA_ROAD_DB1")
SYNTH_DIR = os.path.join(DATA_DIR, "SYNTHETIC_ROAD")
ESC50_DIR = os.path.join(DATA_DIR, "ESC50")
WINDOWS_NPZ = os.path.join(DATA_DIR, "windows.npz")
RESULTS_DIR = os.path.join(ML_DIR, "results")
MODELS_DIR = os.path.join(ML_DIR, "models")
EXPORT_DIR = os.path.join(ML_DIR, "export")
PUBLIC_MODEL_DIR = os.path.join(REPO_DIR, "public", "model")

# Audio
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)   # 16000
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)         # 8000
MIN_EVENT_OVERLAP = 0.5   # window is an event window if MORE than this fraction overlaps an event

# Log Mel spectrogram (tf.signal on the Python side, tf.signal in TF.js on the browser side)
N_FFT = 1024          # frame length and FFT length
STFT_HOP = 256
N_MELS = 64
MEL_FMIN = 50.0
MEL_FMAX = 8000.0
LOG_EPS = 1e-6
NORM_EPS = 1e-6
N_FRAMES = 1 + (WINDOW_SAMPLES - N_FFT) // STFT_HOP   # 59

# Classes. Index order is the model output order. Do not reorder after training.
CLASSES = ["background", "crash", "skid", "loud_not_crash"]
CRASH = CLASSES.index("crash")

ESC50_LOUD_CLASSES = [
    "door_wood_knock", "door_wood_creaks", "glass_breaking", "fireworks", "car_horn",
    "siren", "clapping", "dog", "thunderstorm", "can_opening",
]
