"""Phase 1: look at what is really on disk. Prints facts only, guesses nothing.

Run:  ml\\.venv\\Scripts\\python.exe ml\\inspect_dataset.py [folder]
Default folder is ml/data/MIVIA_ROAD_DB1. Also works on SYNTHETIC_ROAD and ESC50.
Paste the output into ml/DATASET.md when the real MIVIA data has been downloaded.
"""
import collections
import os
import sys
import xml.etree.ElementTree as ET

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else C.MIVIA_DIR
    if not os.path.isdir(root):
        print(f"Folder not found: {root}")
        return 1
    ext_count = collections.Counter()
    dir_count = collections.Counter()
    wavs, xmls = [], []
    for dp, _, files in os.walk(root):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            ext_count[ext] += 1
            dir_count[os.path.relpath(dp, root)] += 1
            if ext == ".wav":
                wavs.append(os.path.join(dp, fn))
            elif ext == ".xml":
                xmls.append(os.path.join(dp, fn))
    print(f"ROOT: {root}")
    print("FILES BY EXTENSION:", dict(ext_count))
    print("FILES PER FOLDER:")
    for d, n in sorted(dir_count.items()):
        print(f"  {d}: {n}")
    if not wavs and not xmls:
        print("No wav or xml files here. The dataset has not been downloaded yet.")
        return 1

    fmt = collections.Counter()
    total = 0.0
    for w in wavs:
        info = sf.info(w)
        fmt[(info.samplerate, info.channels, info.subtype)] += 1
        total += info.duration
    print("WAV FORMATS (sample rate, channels, subtype): count")
    for k, n in fmt.items():
        print(f"  {k}: {n}")
    print(f"TOTAL WAV DURATION: {total:.1f} s in {len(wavs)} files")
    print("FIRST WAV NAMES:", [os.path.relpath(w, root) for w in sorted(wavs)[:8]])

    tags = collections.Counter()
    classes = collections.Counter()
    n_items = 0
    dur = collections.defaultdict(float)
    for x in xmls:
        try:
            tree = ET.parse(x)
        except ET.ParseError as e:
            print(f"  cannot parse {x}: {e}")
            continue
        for el in tree.iter():
            tags[el.tag] += 1
        for item in tree.iter("item"):
            n_items += 1
            d = {c.tag: (c.text or "").strip() for c in item}
            key = (d.get("CLASS_ID"), d.get("CLASS_NAME"))
            classes[key] += 1
            try:
                dur[key] += float(d["ENDSECOND"]) - float(d["STARTSECOND"])
            except (KeyError, ValueError):
                pass
    print(f"XML FILES: {len(xmls)}")
    print("XML TAGS:", dict(tags))
    print(f"ANNOTATED <item> EVENTS: {n_items}")
    print("EVENTS BY (CLASS_ID, CLASS_NAME): count, total seconds")
    for k, n in sorted(classes.items(), key=str):
        print(f"  {k}: {n}, {dur[k]:.2f} s")
    if xmls:
        first = sorted(xmls)[0]
        print(f"FIRST XML FILE: {os.path.relpath(first, root)}")
        with open(first, "r", encoding="utf-8", errors="replace") as f:
            print(f.read(1500))
    return 0


if __name__ == "__main__":
    sys.exit(main())
