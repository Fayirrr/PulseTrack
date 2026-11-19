"""Create a small SYNTHETIC road audio dataset.

!!! SYNTHETIC DATA. NOT REAL CRASHES. NOT FOR RESULTS. !!!

This exists only so the whole pipeline (windows, features, training, export, browser)
can be tested end to end while the real MIVIA Road Audio Events dataset is not available.
The "crash" and "skid" sounds here are made from noise bursts and sine tones with numpy.
Scores measured on this data say nothing about real crash detection and must never be
reported as results.

Layout written to ml/data/SYNTHETIC_ROAD (same idea as MIVIA: about one minute clips,
32 kHz, 16 bit mono, events mixed into a road like background, XML annotations, 4 folds):

    SYNTHETIC_ROAD/
        SYNTHETIC_README.txt
        fold1/ synth_01.xml  sounds/synth_01.wav
        fold2/ ...

Run:  ml\\.venv\\Scripts\\python.exe ml\\make_synthetic.py
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

SR = 32000
CLIP_SECONDS = 60


def _bandpass(x, lo, hi, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def _lowpass(x, hi, order=4):
    sos = signal.butter(order, hi, btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def make_background(rng):
    n = SR * CLIP_SECONDS
    t = np.arange(n) / SR
    rumble = _lowpass(rng.standard_normal(n), rng.uniform(200, 600))
    rumble /= np.max(np.abs(rumble)) + 1e-9
    hiss = _bandpass(rng.standard_normal(n), 300, rng.uniform(3000, 7000))
    hiss /= np.max(np.abs(hiss)) + 1e-9
    bg = 0.6 * rumble + rng.uniform(0.05, 0.25) * hiss
    # passing vehicles: band limited noise with a slow swell
    for _ in range(rng.integers(3, 8)):
        centre = rng.uniform(0, CLIP_SECONDS)
        width = rng.uniform(1.0, 4.0)
        env = np.exp(-0.5 * ((t - centre) / width) ** 2)
        lo = rng.uniform(80, 300)
        car = _bandpass(rng.standard_normal(n), lo, lo + rng.uniform(500, 2500))
        car /= np.max(np.abs(car)) + 1e-9
        hum = np.sin(2 * np.pi * rng.uniform(70, 140) * t)
        bg += env * (rng.uniform(0.3, 1.0) * car + rng.uniform(0.0, 0.3) * hum)
    bg /= np.max(np.abs(bg)) + 1e-9
    return bg * rng.uniform(0.05, 0.2)


def make_crash(rng):
    dur = rng.uniform(0.8, 2.5)
    n = int(dur * SR)
    t = np.arange(n) / SR
    burst = rng.standard_normal(n) * np.exp(-t / rng.uniform(0.12, 0.5))
    burst = _lowpass(burst, rng.uniform(3000, 9000))
    thump = np.sin(2 * np.pi * rng.uniform(50, 120) * t) * np.exp(-t / rng.uniform(0.08, 0.2))
    out = burst / (np.max(np.abs(burst)) + 1e-9) + rng.uniform(0.3, 0.9) * thump
    for _ in range(rng.integers(3, 9)):   # metal like ringing
        f = rng.uniform(400, 6000)
        start = rng.uniform(0, dur * 0.5)
        ring = np.sin(2 * np.pi * f * t) * np.exp(-np.maximum(t - start, 0) / rng.uniform(0.03, 0.25))
        ring[t < start] = 0
        out += rng.uniform(0.05, 0.3) * ring
    for _ in range(rng.integers(5, 25)):  # debris clicks
        i = rng.integers(0, n - 200)
        out[i:i + 200] += rng.uniform(0.1, 0.6) * rng.standard_normal(200) * np.hanning(200)
    return out / (np.max(np.abs(out)) + 1e-9)


def make_skid(rng):
    dur = rng.uniform(1.5, 4.0)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(900, 2200)
    drift = np.linspace(1.0, rng.uniform(0.75, 1.05), n)
    vib = 1.0 + rng.uniform(0.005, 0.03) * np.sin(2 * np.pi * rng.uniform(4, 12) * t)
    phase = 2 * np.pi * np.cumsum(f0 * drift * vib) / SR
    out = np.sin(phase) + rng.uniform(0.2, 0.6) * np.sin(2 * phase) + rng.uniform(0.0, 0.3) * np.sin(3 * phase)
    noise = _bandpass(rng.standard_normal(n), f0 * 0.6, min(f0 * 3.0, SR / 2 - 500))
    out += rng.uniform(0.3, 1.0) * noise / (np.max(np.abs(noise)) + 1e-9)
    edge = max(int(rng.uniform(0.05, 0.3) * SR), 1)
    env = np.ones(n)
    env[:edge] = np.linspace(0, 1, edge)
    env[-edge:] = np.linspace(1, 0, edge)
    out *= env
    return out / (np.max(np.abs(out)) + 1e-9)


def make_clip(rng, events_per_clip):
    bg = make_background(rng)
    audio = bg.copy()
    bg_rms = np.sqrt(np.mean(bg ** 2)) + 1e-9
    events = []
    cursor = rng.uniform(1.0, 4.0)
    for k in range(events_per_clip):
        is_crash = (k % 2 == 0)
        ev = make_crash(rng) if is_crash else make_skid(rng)
        dur = len(ev) / SR
        if cursor + dur > CLIP_SECONDS - 1.0:
            break
        snr_db = rng.uniform(5, 25)
        ev_rms = np.sqrt(np.mean(ev ** 2)) + 1e-9
        ev = ev * (bg_rms * 10 ** (snr_db / 20) / ev_rms)
        i = int(cursor * SR)
        audio[i:i + len(ev)] += ev
        events.append((3 if is_crash else 2, "car_crash" if is_crash else "tire_skidding", cursor, cursor + dur))
        cursor += dur + rng.uniform(2.0, 6.0)
    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio / peak * 0.99
    return audio.astype(np.float32), events


def write_xml(path, wav_name, events):
    root = ET.Element("root")
    note = ET.SubElement(root, "NOTE")
    note.text = "SYNTHETIC DATA made by ml/make_synthetic.py. Not real crashes."
    evs = ET.SubElement(root, "events")
    for cid, cname, s, e in events:
        item = ET.SubElement(evs, "item")
        ET.SubElement(item, "CLASS_ID").text = str(cid)
        ET.SubElement(item, "CLASS_NAME").text = cname
        ET.SubElement(item, "PATHNAME").text = wav_name
        ET.SubElement(item, "STARTSECOND").text = f"{s:.4f}"
        ET.SubElement(item, "ENDSECOND").text = f"{e:.4f}"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    ap = argparse.ArgumentParser(description="Create SYNTHETIC road audio (pipeline test only)")
    ap.add_argument("--clips-per-fold", type=int, default=6)
    ap.add_argument("--events-per-clip", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=C.SYNTH_DIR)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "SYNTHETIC_README.txt"), "w", encoding="utf-8") as f:
        f.write("SYNTHETIC DATA. Made by ml/make_synthetic.py from noise and sine tones.\n"
                "These are NOT real car crashes or tire skids. Use only to test the code.\n"
                "Never report scores measured on this data as results.\n")
    n_events = {"car_crash": 0, "tire_skidding": 0}
    n = 0
    for fold in range(1, 5):
        fdir = os.path.join(args.out, f"fold{fold}")
        os.makedirs(os.path.join(fdir, "sounds"), exist_ok=True)
        for _ in range(args.clips_per_fold):
            n += 1
            stem = f"synth_{n:02d}"
            audio, events = make_clip(rng, args.events_per_clip)
            sf.write(os.path.join(fdir, "sounds", stem + ".wav"), audio, SR, subtype="PCM_16")
            write_xml(os.path.join(fdir, stem + ".xml"), stem + ".wav", events)
            for _, cname, _, _ in events:
                n_events[cname] += 1
    print(f"SYNTHETIC dataset written to {args.out}")
    print(f"clips: {n}, events: {n_events}")


if __name__ == "__main__":
    main()
