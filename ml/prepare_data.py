r"""Phase 1 + 2: build the labelled set from open datasets and cut it into 1 second windows.

Run:  ml\.venv\Scripts\python.exe ml\prepare_data.py

Classes and sources
    crash           Freesound clips (ml/data/freesound_crash, from ml/freesound_download.py)
    skid            Freesound clips (ml/data/freesound_skid)
    background      UrbanSound8K engine_idling, street_music, air_conditioner, drilling,
                    children_playing, jackhammer
    loud_not_crash  ESC-50 door_wood_knock, glass_breaking, fireworks, car_horn, siren, dog,
                    clapping, can_opening, thunderstorm + UrbanSound8K car_horn, siren, gun_shot

Steps
* every clip is resampled to 16 kHz mono (scipy resample_poly), clips under 1 s are padded
* 1 second windows, 0.5 second hop
* crash / skid / loud_not_crash: windows quieter than 10 percent (RMS) of the loudest window of
  the same clip are dropped (silent padding), then at most N loudest windows per clip are kept
* background: at most N windows per clip, picked at random with a fixed seed
* 4 folds by SOURCE, never by window. All windows of one clip stay in one fold. Clips that share
  a Freesound id (ESC-50 and UrbanSound8K both come from Freesound) or a Freesound uploader are
  joined into one group, and a whole group goes to one fold. Groups are placed greedily so each
  fold gets about a quarter of every class.
* saves ml/data/windows.npz and prints counts per class per fold
"""
import argparse
import collections
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

SEED = 42


def load_audio_16k(path):
    """Any wav/flac/ogg soundfile can read (ffmpeg as fallback) -> float32 mono 16 kHz."""
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001  (odd formats in UrbanSound8K)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "a.wav")
            subprocess.run([ffmpeg, "-v", "error", "-y", "-i", path, "-c:a", "pcm_s16le", tmp], check=True)
            audio, sr = sf.read(tmp, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        g = gcd(int(sr), C.SAMPLE_RATE)
        audio = resample_poly(audio, C.SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return audio, sr


def to_int16(x):
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def cut_windows(audio):
    if len(audio) < C.WINDOW_SAMPLES:
        audio = np.pad(audio, (0, C.WINDOW_SAMPLES - len(audio)))
    n = 1 + (len(audio) - C.WINDOW_SAMPLES) // C.HOP_SAMPLES
    return np.stack([audio[i * C.HOP_SAMPLES:i * C.HOP_SAMPLES + C.WINDOW_SAMPLES] for i in range(n)])


def select_windows(wins, label, rng, stats):
    cap = C.MAX_WINDOWS_PER_CLIP[label]
    if label == "background":
        idx = np.arange(len(wins))
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        return wins[idx]
    rms = np.sqrt(np.mean(wins.astype(np.float64) ** 2, axis=1))
    if rms.max() <= 0:
        stats["clips_silent"] += 1
        return wins[:0]
    keep = np.where(rms >= C.EVENT_RMS_FRACTION * rms.max())[0]
    stats["windows_dropped_quiet"] += len(wins) - len(keep)
    if len(keep) > cap:
        stats["windows_dropped_cap"] += len(keep) - cap
        keep = np.sort(keep[np.argsort(rms[keep])[::-1][:cap]])
    return wins[keep]


# --------------------------------------------------------------------------- clip lists
def freesound_clips():
    clips = []
    if not os.path.exists(C.FREESOUND_CSV):
        return clips
    with open(C.FREESOUND_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = C.FREESOUND_CRASH_DIR if row["class"] == "crash" else C.FREESOUND_SKID_DIR
            path = os.path.join(d, row["file"])
            if os.path.exists(path):
                clips.append({"path": path, "label": row["class"], "source": f"freesound_{row['class']}",
                              "keys": [f"fs:{row['id']}", f"user:{row['author'].lower()}"],
                              "clip": f"freesound:{row['id']}"})
    return clips


def find_file(root, name):
    for dp, _, files in os.walk(root):
        if name in files:
            return os.path.join(dp, name)
    return None


def urbansound_clips():
    meta_path = find_file(C.URBANSOUND_DIR, "UrbanSound8K.csv")
    if not meta_path:
        print("UrbanSound8K not found, skipping")
        return []
    import pandas as pd
    meta = pd.read_csv(meta_path).rename(columns={"class": "cls"})
    # official zip: metadata/UrbanSound8K.csv + audio/foldN; Kaggle copy: UrbanSound8K.csv next to foldN
    here = os.path.dirname(meta_path)
    candidates = [os.path.join(here, "audio"), os.path.join(os.path.dirname(here), "audio"), here]
    audio_root = next((d for d in candidates if os.path.isdir(os.path.join(d, "fold1"))), here)
    clips = []
    for r in meta.itertuples():
        if r.cls in C.US8K_BACKGROUND_CLASSES:
            label = "background"
        elif r.cls in C.US8K_LOUD_CLASSES:
            label = "loud_not_crash"
        else:
            continue
        path = os.path.join(audio_root, f"fold{r.fold}", r.slice_file_name)
        if not os.path.exists(path):
            path = find_file(audio_root, r.slice_file_name)
            if not path:
                continue
        clips.append({"path": path, "label": label, "source": f"us8k_{r.cls}",
                      "keys": [f"fs:{r.fsID}"], "clip": f"us8k:{r.slice_file_name}"})
    return clips


def esc50_clips():
    meta_path = find_file(C.ESC50_DIR, "esc50.csv")
    if not meta_path:
        print("ESC-50 not found, skipping")
        return []
    import pandas as pd
    meta = pd.read_csv(meta_path)
    base = os.path.dirname(os.path.dirname(meta_path))
    clips = []
    for r in meta[meta["category"].isin(C.ESC50_LOUD_CLASSES)].itertuples():
        clips.append({"path": os.path.join(base, "audio", r.filename), "label": "loud_not_crash",
                      "source": f"esc50_{r.category}", "keys": [f"fs:{r.src_file}"], "clip": f"esc50:{r.filename}"})
    return clips


# --------------------------------------------------------------------------- folds
def find_groups(clips):
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c in clips:
        r0 = root(c["keys"][0])
        for k in c["keys"][1:]:
            parent[root(k)] = r0
    return [root(c["keys"][0]) for c in clips]


def assign_folds(groups, labels):
    """Greedy: biggest group first, put it where the folds are most behind their per class share."""
    n_cls = len(C.CLASSES)
    size = collections.defaultdict(lambda: np.zeros(n_cls))
    for g, y in zip(groups, labels):
        size[g][y] += 1
    total = np.maximum(sum(size.values()), 1)
    fold_cnt = np.zeros((4, n_cls))
    fold_of = {}
    for g in sorted(size, key=lambda k: (-size[k].sum(), k)):
        v = size[g]
        cost = [((fold_cnt[f] + v) / total * (v > 0)).sum() for f in range(4)]
        f = int(np.argmin(cost))
        fold_of[g] = f + 1
        fold_cnt[f] += v
    return np.array([fold_of[g] for g in groups], dtype=np.int8)


def counts_table(y, folds):
    lines = ["| class | fold 1 | fold 2 | fold 3 | fold 4 | total |", "|---|---|---|---|---|---|"]
    for ci, name in enumerate(C.CLASSES):
        row = [int(np.sum((y == ci) & (folds == f))) for f in (1, 2, 3, 4)]
        share = " / ".join(f"{100 * v / max(sum(row), 1):.0f}%" for v in row)
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} (split {share}) |")
    row = [int(np.sum(folds == f)) for f in (1, 2, 3, 4)]
    lines.append("| all | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-crash", action="store_true",
                    help="build the file even if there are no crash/skid clips (code test only)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    clips = freesound_clips() + urbansound_clips() + esc50_clips()
    per_label = collections.Counter(c["label"] for c in clips)
    print("clips found:", dict(per_label))
    if (per_label["crash"] == 0 or per_label["skid"] == 0) and not args.allow_no_crash:
        raise SystemExit("No Freesound crash or skid clips. Run ml/freesound_download.py first.")

    X, y, src, clip_ids, clip_keys = [], [], [], [], []
    stats = collections.Counter()
    src_clips = collections.Counter()
    src_seconds = collections.defaultdict(float)
    src_rates = collections.defaultdict(collections.Counter)
    for i, c in enumerate(clips):
        try:
            audio, sr = load_audio_16k(c["path"])
        except Exception as e:  # noqa: BLE001
            stats[f"unreadable_{c['source']}"] += 1
            print(f"  cannot read {c['path']}: {type(e).__name__}")
            continue
        src_clips[c["source"]] += 1
        src_seconds[c["source"]] += len(audio) / C.SAMPLE_RATE
        src_rates[c["source"]][int(sr)] += 1
        wins = select_windows(cut_windows(audio), c["label"], rng, stats)
        for w in wins:
            X.append(to_int16(w))
            y.append(C.CLASSES.index(c["label"]))
            src.append(c["source"])
            clip_ids.append(c["clip"])
            clip_keys.append(i)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(clips)} clips")

    groups_per_clip = find_groups(clips)
    group = np.array([groups_per_clip[k] for k in clip_keys])
    X = np.stack(X)
    y = np.array(y, dtype=np.int8)
    fold = assign_folds(group, y)
    src = np.array(src)
    clip_ids = np.array(clip_ids)

    # leakage check: every clip and every group must sit in exactly one fold
    for name, arr in (("clip", clip_ids), ("group", group)):
        bad = [k for k in np.unique(arr) if len(np.unique(fold[arr == k])) > 1]
        assert not bad, f"{name} split across folds: {bad[:5]}"
    print("leakage check passed: no clip and no source group is in more than one fold")

    meta = {"road_dataset": "open_freesound_us8k_esc50", "synthetic": False, "classes": C.CLASSES,
            "sample_rate": C.SAMPLE_RATE, "window_samples": C.WINDOW_SAMPLES, "hop_samples": C.HOP_SAMPLES}
    np.savez_compressed(C.WINDOWS_NPZ, X=X, y=y, fold=fold, group=group, source=src, clip=clip_ids,
                        meta=json.dumps(meta))

    table = counts_table(y, fold)
    lines = ["| source | clips read | total seconds | original sample rates (Hz: clips) | windows kept |",
             "|---|---|---|---|---|"]
    for s in sorted(src_clips):
        rates = ", ".join(f"{k}: {v}" for k, v in sorted(src_rates[s].items()))
        lines.append(f"| {s} | {src_clips[s]} | {src_seconds[s]:.1f} | {rates} | {int(np.sum(src == s))} |")
    src_table = "\n".join(lines)
    print("\nstats:", dict(stats))
    print(f"source groups: {len(np.unique(group))}")
    print("\nPer source:\n" + src_table)
    print("\nWindows per class per fold:\n" + table)
    print(f"\nsaved {C.WINDOWS_NPZ}  X={X.shape} {X.dtype}")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "window_counts.md"), "w", encoding="utf-8") as f:
        f.write("Printed by ml/prepare_data.py\n\n## Per source\n\n" + src_table +
                "\n\n## Windows per class per fold\n\n" + table +
                f"\n\nOther counts: {dict(stats)}. Source groups: {len(np.unique(group))}.\n")


if __name__ == "__main__":
    main()
