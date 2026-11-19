"""Phase 7: run the model on your own phone recordings to see the domain shift.

Run:
    ml\\.venv\\Scripts\\python.exe ml\\phone_test.py path\\to\\folder_with_recordings

Accepts wav, mp3, m4a, webm, ogg, flac, aac, 3gp. wav and flac are read directly.
The other formats need ffmpeg on the PATH (install: winget install Gyan.FFmpeg).

For each file the audio is resampled to 16 kHz mono, cut into 1 second windows with a 0.5 second
hop (same as training) and every window goes through the same tf.signal features and the model.

Columns
    predicted      class with the highest mean probability over the file
    confidence     that mean probability
    max crash p    highest crash probability of any single window
    app trigger    yes if crash probability was above 0.7 for 2 windows in a row (the app rule)

Tip: name files like crash_01.m4a, skid_02.wav, background_03.mp3, loud_04.webm.
If a file name starts with crash / skid / background / loud the script also prints how many
predictions matched.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402
from prepare_data import load_audio_16k  # noqa: E402

DIRECT = {".wav", ".flac"}
VIA_FFMPEG = {".mp3", ".m4a", ".webm", ".ogg", ".aac", ".3gp", ".mp4", ".opus"}
PREFIX = {"crash": "crash", "skid": "skid", "background": "background", "bg": "background", "loud": "loud_not_crash"}


def read_any(path, ffmpeg):
    ext = os.path.splitext(path)[1].lower()
    if ext in DIRECT:
        return load_audio_16k(path)[0]
    if not ffmpeg:
        raise RuntimeError("needs ffmpeg (not found on PATH)")
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "a.wav")
        r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", str(C.SAMPLE_RATE), tmp],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("ffmpeg failed: " + r.stderr.strip()[:120])
        return load_audio_16k(tmp)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--model", default=os.path.join(C.MODELS_DIR, "crash_cnn.h5"))
    ap.add_argument("--csv", default=os.path.join(C.RESULTS_DIR, "phone_test.csv"))
    args = ap.parse_args()

    meta_path = os.path.join(C.MODELS_DIR, "model_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if meta.get("synthetic"):
        print("!" * 78)
        print("WARNING: this model is the PLACEHOLDER trained on SYNTHETIC audio. Its predictions")
        print("on real recordings mean nothing. Train on the real MIVIA data first.")
        print("!" * 78)
    model = tf.keras.models.load_model(args.model, compile=False)
    ffmpeg = shutil.which("ffmpeg")
    files = sorted(f for f in os.listdir(args.folder)
                   if os.path.splitext(f)[1].lower() in DIRECT | VIA_FFMPEG)
    if not files:
        print("No audio files found in", args.folder)
        return 1

    rows, hits, labelled = [], 0, 0
    print(f"model: {meta.get('modelVersion', os.path.basename(args.model))}   ffmpeg: {ffmpeg or 'NOT FOUND'}\n")
    print(f"{'file':34s} {'sec':>6s} {'win':>4s}  {'predicted':15s} {'conf':>5s}  {'max crash p':>11s}  app trigger")
    for fn in files:
        try:
            audio = read_any(os.path.join(args.folder, fn), ffmpeg)
        except Exception as e:  # noqa: BLE001
            print(f"{fn[:34]:34s} skipped: {e}")
            continue
        if len(audio) < C.WINDOW_SAMPLES:
            audio = np.pad(audio, (0, C.WINDOW_SAMPLES - len(audio)))
        n = 1 + (len(audio) - C.WINDOW_SAMPLES) // C.HOP_SAMPLES
        wins = np.stack([audio[i * C.HOP_SAMPLES:i * C.HOP_SAMPLES + C.WINDOW_SAMPLES] for i in range(n)])
        probs = model.predict(F.log_mel(tf.constant(wins, tf.float32)).numpy(), batch_size=256, verbose=0)
        mean = probs.mean(axis=0)
        top = int(mean.argmax())
        crash = probs[:, C.CRASH]
        above = crash > 0.7
        trigger = bool(np.any(above[:-1] & above[1:])) if n > 1 else False
        expected = next((v for k, v in PREFIX.items() if fn.lower().startswith(k)), None)
        if expected:
            labelled += 1
            hits += int(C.CLASSES[top] == expected)
        rows.append([fn, f"{len(audio) / C.SAMPLE_RATE:.1f}", n, C.CLASSES[top], f"{mean[top]:.3f}",
                     f"{crash.max():.3f}", "yes" if trigger else "no", expected or ""])
        print(f"{fn[:34]:34s} {len(audio) / C.SAMPLE_RATE:6.1f} {n:4d}  {C.CLASSES[top]:15s} {mean[top]:5.2f}  "
              f"{crash.max():11.2f}  {'YES' if trigger else 'no'}")
    if labelled:
        print(f"\nfiles with a class in the name: {labelled}, predicted class matched: {hits}")
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "seconds", "windows", "predicted", "confidence", "max_crash_prob", "app_trigger", "expected"])
        w.writerows(rows)
    print("saved", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
