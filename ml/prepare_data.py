"""Phase 2: cut all audio into 1 second windows and save ml/data/windows.npz.

Road data (MIVIA, or the SYNTHETIC stand in) gives classes background / crash / skid.
ESC-50 gives the 4th class loud_not_crash.

Run:
    ml\\.venv\\Scripts\\python.exe ml\\prepare_data.py                  (auto: MIVIA if present, else SYNTHETIC)
    ml\\.venv\\Scripts\\python.exe ml\\prepare_data.py --dataset mivia
    ml\\.venv\\Scripts\\python.exe ml\\prepare_data.py --dataset synthetic

What it does
* resample every file to 16 kHz mono (scipy resample_poly)
* 1 second windows, 0.5 second hop
* a window is crash or skid if MORE than 50 percent of it overlaps an annotated event
* a window with no event overlap at all is background
* a window that touches an event but 50 percent or less is dropped (see DECISIONS.md)
* the fold number of the source file is kept for every window
* ESC-50 clips go to loud_not_crash; ESC folds 1..4 keep their number, fold 5 is spread over
  groups 1..4 by original Freesound clip id, so one original clip is never in two groups

About the MIVIA annotation format: the real files could not be downloaded when this was written
(the site needs registration). The parser reads <events><item> entries with CLASS_ID, CLASS_NAME,
STARTSECOND and ENDSECOND. That layout was seen in a public third party extraction script, not in
the real files, so the script prints what it finds and stops if anything is unexpected.
"""
import argparse
import collections
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

# CLASS_ID fallback, used only when CLASS_NAME is missing. 3 = crash comes from a third party
# script (github.com/the6thsense/Mivia-Event-Data-road-Data-Extraction-script), NOT verified on
# the real files. Check the output of inspect_dataset.py before trusting it.
CLASS_ID_FALLBACK = {"3": "crash", "2": "skid"}


def load_audio_16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        g = gcd(sr, C.SAMPLE_RATE)
        audio = resample_poly(audio, C.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio, sr


def to_int16(x):
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def class_from_item(d, unknown):
    name = (d.get("CLASS_NAME") or "").lower()
    if "crash" in name:
        return "crash"
    if "skid" in name or "tire" in name or "tyre" in name:
        return "skid"
    cid = (d.get("CLASS_ID") or "").strip()
    if not name and cid in CLASS_ID_FALLBACK:
        return CLASS_ID_FALLBACK[cid]
    unknown[(d.get("CLASS_ID"), d.get("CLASS_NAME"))] += 1
    return None


def parse_events(xml_path, unknown):
    events = []
    for item in ET.parse(xml_path).iter("item"):
        d = {c.tag.upper(): (c.text or "").strip() for c in item}
        cls = class_from_item(d, unknown)
        if cls is None:
            continue
        events.append((float(d["STARTSECOND"]), float(d["ENDSECOND"]), cls))
    return events


def fold_from_path(rel_path, fold_regex):
    m = re.search(fold_regex, rel_path.replace("\\", "/"), flags=re.IGNORECASE)
    if not m:
        return None
    g = m.group(1)
    if g.isdigit():
        return int(g)
    return "abcd".index(g.lower()) + 1


def road_windows(root, fold_regex, tag):
    """Windows from a MIVIA style folder. Returns lists X, y, fold, group plus a stats dict."""
    wav_index = collections.defaultdict(list)
    xmls = []
    for dp, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dp, fn)
            if fn.lower().endswith(".wav"):
                wav_index[os.path.basename(dp) + "/" + fn].append(p)
            elif fn.lower().endswith(".xml"):
                xmls.append(p)
    all_wavs = [p for ps in wav_index.values() for p in ps]
    if not xmls or not all_wavs:
        raise SystemExit(f"No xml/wav files under {root}. Download the dataset first (see MORNING_SUMMARY.md).")

    X, y, folds, groups = [], [], [], []
    unknown = collections.Counter()
    stats = collections.Counter()
    no_fold = []
    for xml_path in sorted(xmls):
        stem = os.path.splitext(os.path.basename(xml_path))[0]
        xdir = os.path.dirname(xml_path)
        # audio for this annotation: same folder or its "sounds" subfolder, file name starts with the xml stem
        cands = [p for p in all_wavs
                 if os.path.dirname(p) in (xdir, os.path.join(xdir, "sounds"))
                 and re.match(re.escape(stem) + r"(\D.*)?\.wav$", os.path.basename(p), flags=re.IGNORECASE)]
        if not cands:
            stats["xml_without_audio"] += 1
            continue
        events = parse_events(xml_path, unknown)
        stats["events_crash"] += sum(1 for e in events if e[2] == "crash")
        stats["events_skid"] += sum(1 for e in events if e[2] == "skid")
        for wav in sorted(cands):
            rel = os.path.relpath(wav, root)
            fold = fold_from_path(rel, fold_regex)
            if fold is None:
                no_fold.append(rel)
                continue
            audio, _ = load_audio_16k(wav)
            stats["files"] += 1
            n_win = 1 + (len(audio) - C.WINDOW_SAMPLES) // C.HOP_SAMPLES if len(audio) >= C.WINDOW_SAMPLES else 0
            for i in range(n_win):
                s = i * C.HOP_SECONDS
                e = s + C.WINDOW_SECONDS
                best_cls, best_ov = None, 0.0
                for es, ee, cls in events:
                    ov = max(0.0, min(e, ee) - max(s, es)) / C.WINDOW_SECONDS
                    if ov > best_ov:
                        best_cls, best_ov = cls, ov
                if best_ov > C.MIN_EVENT_OVERLAP:
                    label = best_cls
                elif best_ov == 0.0:
                    label = "background"
                else:
                    stats["dropped_partial_overlap"] += 1
                    continue
                a = i * C.HOP_SAMPLES
                X.append(to_int16(audio[a:a + C.WINDOW_SAMPLES]))
                y.append(C.CLASSES.index(label))
                folds.append(fold)
                groups.append(f"{tag}:{fold}:{stem}")
    if unknown:
        raise SystemExit(f"Unknown event classes in the XML (CLASS_ID, CLASS_NAME): {dict(unknown)}. "
                         "Edit class_from_item / CLASS_ID_FALLBACK in prepare_data.py after checking the files.")
    if no_fold:
        raise SystemExit("Could not read a fold number from these paths (first 5): "
                         f"{no_fold[:5]}. Pass --fold-regex with one capture group that gives 1..4 or A..D.")
    return X, y, folds, groups, stats


def esc50_windows(esc_dir):
    """Windows for loud_not_crash from ESC-50. Silent padding windows are dropped."""
    import pandas as pd
    base = None
    for dp, dirs, files in os.walk(esc_dir):
        if "esc50.csv" in files and os.path.basename(dp) == "meta":
            base = os.path.dirname(dp)
            break
    if base is None:
        print("ESC-50 not found, skipping the loud_not_crash class.")
        return [], [], [], [], collections.Counter()
    meta = pd.read_csv(os.path.join(base, "meta", "esc50.csv"))
    meta = meta[meta["category"].isin(C.ESC50_LOUD_CLASSES)]
    X, y, folds, groups = [], [], [], []
    stats = collections.Counter()
    label = C.CLASSES.index("loud_not_crash")
    for row in meta.itertuples():
        esc_fold = int(row.fold)
        src = str(row.src_file)
        # folds 1..4 stay; fold 5 is spread by original clip id so takes of one clip stay together
        if esc_fold <= 4:
            fold = esc_fold
        else:
            digits = re.sub(r"\D", "", src)
            fold = (int(digits) if digits else sum(map(ord, src))) % 4 + 1
        audio, _ = load_audio_16k(os.path.join(base, "audio", row.filename))
        stats["files"] += 1
        stats[f"clips_{row.category}"] += 1
        wins = []
        n_win = 1 + (len(audio) - C.WINDOW_SAMPLES) // C.HOP_SAMPLES if len(audio) >= C.WINDOW_SAMPLES else 0
        for i in range(n_win):
            a = i * C.HOP_SAMPLES
            wins.append(audio[a:a + C.WINDOW_SAMPLES])
        if not wins:
            continue
        rms = np.array([np.sqrt(np.mean(w ** 2)) for w in wins])
        # ESC-50 clips are padded with silence. Keep a window only if it is at least
        # 10 percent as loud (RMS) as the loudest window of the same clip.
        keep = rms >= 0.1 * rms.max() if rms.max() > 0 else np.zeros(len(wins), bool)
        stats["dropped_quiet"] += int((~keep).sum())
        for w, k in zip(wins, keep):
            if k:
                X.append(to_int16(w))
                y.append(label)
                folds.append(fold)
                groups.append(f"esc50:{esc_fold}:{src}")
    return X, y, folds, groups, stats


def counts_table(y, folds):
    lines = ["| class | fold 1 | fold 2 | fold 3 | fold 4 | total |", "|---|---|---|---|---|---|"]
    for ci, name in enumerate(C.CLASSES):
        row = [int(np.sum((y == ci) & (folds == f))) for f in (1, 2, 3, 4)]
        lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    row = [int(np.sum(folds == f)) for f in (1, 2, 3, 4)]
    lines.append("| all | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["auto", "mivia", "synthetic"], default="auto")
    ap.add_argument("--fold-regex", default=r"fold[_\- ]?(\d)",
                    help="regex with one group giving the fold (1..4 or A..D) from the relative wav path")
    ap.add_argument("--no-esc50", action="store_true")
    args = ap.parse_args()

    def has_wavs(d):
        return os.path.isdir(d) and any(f.lower().endswith(".wav") for _, _, fs in os.walk(d) for f in fs)

    dataset = args.dataset
    if dataset == "auto":
        dataset = "mivia" if has_wavs(C.MIVIA_DIR) else "synthetic"
    synthetic = dataset == "synthetic"
    root = C.SYNTH_DIR if synthetic else C.MIVIA_DIR
    if synthetic:
        print("=" * 70)
        print("USING SYNTHETIC ROAD DATA. Pipeline test only. Scores are NOT results.")
        print("=" * 70)

    X, y, folds, groups, st = road_windows(root, args.fold_regex, "synthetic" if synthetic else "mivia")
    print(f"road data ({dataset}):", dict(st))
    if not args.no_esc50:
        X2, y2, f2, g2, st2 = esc50_windows(C.ESC50_DIR)
        print("esc50:", dict(st2))
        X += X2; y += y2; folds += f2; groups += g2

    X = np.stack(X)
    y = np.array(y, dtype=np.int8)
    folds = np.array(folds, dtype=np.int8)
    groups = np.array(groups)
    if not set(np.unique(folds)) <= {1, 2, 3, 4}:
        raise SystemExit(f"Folds must be 1..4, found {np.unique(folds)}")
    meta = {"road_dataset": dataset, "synthetic": synthetic, "classes": C.CLASSES,
            "sample_rate": C.SAMPLE_RATE, "window_samples": C.WINDOW_SAMPLES, "hop_samples": C.HOP_SAMPLES}
    np.savez_compressed(C.WINDOWS_NPZ, X=X, y=y, fold=folds, group=groups, meta=json.dumps(meta))

    table = counts_table(y, folds)
    title = "SYNTHETIC road data + real ESC-50" if synthetic else "MIVIA road data + ESC-50"
    print(f"\nWindows per class per fold ({title}):")
    print(table)
    print(f"\nsaved {C.WINDOWS_NPZ}  X={X.shape} {X.dtype}")
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    name = "window_counts_SYNTHETIC.md" if synthetic else "window_counts.md"
    with open(os.path.join(C.RESULTS_DIR, name), "w", encoding="utf-8") as f:
        f.write(f"Windows per class per fold ({title}), printed by ml/prepare_data.py\n\n{table}\n")


if __name__ == "__main__":
    main()
