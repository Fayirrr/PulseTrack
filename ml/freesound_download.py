r"""Download real car crash and tire skid sounds from Freesound (https://freesound.org/docs/api/).

Run:  ml\.venv\Scripts\python.exe ml\freesound_download.py

* API key: read from freesound_key.txt in the repo root (one line). The key is never printed
  and never written to any file. Both the key file and ml/data/ are git ignored.
* Only Creative Commons 0 or Attribution licences, duration under 30 seconds.
* Downloads the HQ OGG preview of each sound and converts it to 16 bit wav with ffmpeg.
* Writes ml/data/freesound_crash/*.wav, ml/data/freesound_skid/*.wav and
  ml/data/freesound_sounds.csv (id, url, license, author, tags, ... for every file) and a copy
  of the csv without audio to ml/freesound_sources.csv, which IS committed for citation.

Labels come from the uploader's own title, tags and description plus our search words. They
are not checked by listening. Clear mistakes (crash cymbals, music, computer crashes, game
sounds) are removed with the keyword rules below.
"""
import csv
import os
import shutil
import subprocess
import sys
import time

import requests
import truststore

truststore.inject_into_ssl()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

API = "https://freesound.org/apiv2/search/text/"
KEY_FILE = os.path.join(C.REPO_DIR, "freesound_key.txt")
CRASH_DIR = os.path.join(C.DATA_DIR, "freesound_crash")
SKID_DIR = os.path.join(C.DATA_DIR, "freesound_skid")
CSV_PATH = os.path.join(C.DATA_DIR, "freesound_sounds.csv")
PUBLIC_CSV = os.path.join(C.ML_DIR, "freesound_sources.csv")

QUERIES = {
    "crash": ["car crash", "car crash skid", "car collision", "vehicle crash", "car accident", "car impact"],
    "skid": ["tire skid", "tyre skid", "tire screech", "car skid", "brake skid", "tires squeal"],
}
# order of the license filter values as Freesound names them
LICENSE_FILTER = 'license:("Creative Commons 0" OR "Attribution")'
FIELDS = "id,name,url,license,username,tags,duration,description,previews,pack,samplerate,type"
MAX_PER_CLASS = 150

# Rules applied to title + tags + description (lower case)
REJECT_ANY = ["cymbal", "drum", "music", "loop", "beat", "song", "synth", "computer", "glitch",
              "game", "8bit", "8-bit", "cartoon", "comic", "toy", "bike", "bicycle", "skateboard",
              "keyboard", "piano", "guitar", "voice", "speech", "scream", "tv", "radio"]
CRASH_NEED = ["crash", "collision", "accident", "impact", "wreck", "smash", "hit"]
CRASH_VEHICLE = ["car", "vehicle", "auto", "truck", "van"]
SKID_NEED = ["skid", "screech", "squeal", "drift", "brake", "braking"]
SKID_REJECT = ["crash", "collision", "accident", "impact", "wreck", "smash"]  # skid + crash = not a clean skid


def read_key():
    if not os.path.exists(KEY_FILE):
        raise SystemExit(f"{KEY_FILE} not found")
    with open(KEY_FILE, encoding="utf-8") as f:
        key = f.read().strip()
    if not key:
        raise SystemExit("freesound_key.txt is empty. Put the Freesound API key on one line and run again.")
    return key


def license_ok(lic):
    lic = (lic or "").lower()
    if "noncommercial" in lic or "/by-nc" in lic or "sampling" in lic:
        return False
    return "publicdomain/zero" in lic or "/licenses/by/" in lic


def license_name(lic):
    lic = (lic or "").lower()
    if "publicdomain/zero" in lic:
        return "CC0 1.0"
    if "/licenses/by/4.0" in lic:
        return "CC BY 4.0"
    if "/licenses/by/3.0" in lic:
        return "CC BY 3.0"
    return lic


def text_of(s):
    return " ".join([s.get("name") or "", " ".join(s.get("tags") or []), s.get("description") or ""]).lower()


def accept(cls, s):
    t = text_of(s)
    words = set(t.replace(",", " ").replace(".", " ").replace("-", " ").replace("_", " ").split())
    if any(w in words or (len(w) > 4 and w in t) for w in REJECT_ANY):
        return False, "reject keyword"
    if cls == "crash":
        if not any(w in t for w in CRASH_NEED):
            return False, "no crash word"
        if not any(w in words for w in CRASH_VEHICLE) and "car" not in t:
            return False, "no vehicle word"
        return True, ""
    if not any(w in t for w in SKID_NEED):
        return False, "no skid word"
    if any(w in t for w in SKID_REJECT):
        return False, "mentions crash"
    return True, ""


def search(key, query, page):
    params = {"query": query, "filter": f"{LICENSE_FILTER} duration:[0.5 TO 30]", "fields": FIELDS,
              "page_size": 150, "page": page, "token": key}
    for attempt in range(4):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 404:
            return {"results": [], "next": None}
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited (429) four times")


def to_wav(ffmpeg, src, dst):
    r = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", src, "-ac", "1", "-c:a", "pcm_s16le", dst],
                       capture_output=True, text=True)
    return r.returncode == 0


def main():
    key = read_key()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is needed to convert the previews (winget install Gyan.FFmpeg)")
    os.makedirs(CRASH_DIR, exist_ok=True)
    os.makedirs(SKID_DIR, exist_ok=True)
    tmp_dir = os.path.join(C.DATA_DIR, "_freesound_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    found = {"crash": {}, "skid": {}}
    query_of = {}
    stats = {"crash": {}, "skid": {}}
    for cls, queries in QUERIES.items():
        for q in queries:
            n_hits = 0
            page = 1
            while page <= 3:
                data = search(key, q, page)
                for s in data.get("results", []):
                    n_hits += 1
                    if not license_ok(s.get("license")):
                        stats[cls]["licence not CC0/BY"] = stats[cls].get("licence not CC0/BY", 0) + 1
                        continue
                    if float(s.get("duration") or 99) >= 30:
                        continue
                    ok, why = accept(cls, s)
                    if not ok:
                        stats[cls][why] = stats[cls].get(why, 0) + 1
                        continue
                    found[cls].setdefault(s["id"], s)
                    query_of.setdefault((cls, s["id"]), q)
                if not data.get("next"):
                    break
                page += 1
                time.sleep(1.0)
            print(f"query '{q}' ({cls}): {n_hits} hits")
            time.sleep(1.0)

    both = set(found["crash"]) & set(found["skid"])
    for i in both:  # a sound found by both kinds of query is kept as crash only
        del found["skid"][i]
    print(f"in both result lists (kept as crash only): {len(both)}")
    for cls in found:
        print(f"{cls}: {len(found[cls])} accepted after filters; rejected: {stats[cls]}")

    rows = []
    for cls, out_dir in (("crash", CRASH_DIR), ("skid", SKID_DIR)):
        items = sorted(found[cls].values(), key=lambda s: s["id"])[:MAX_PER_CLASS]
        n_ok = 0
        for s in items:
            wav = os.path.join(out_dir, f"fs{s['id']}.wav")
            if not os.path.exists(wav):
                url = (s.get("previews") or {}).get("preview-hq-ogg") or (s.get("previews") or {}).get("preview-hq-mp3")
                if not url:
                    continue
                ext = ".ogg" if url.endswith(".ogg") else ".mp3"
                tmp = os.path.join(tmp_dir, f"{s['id']}{ext}")
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                except Exception as e:  # noqa: BLE001
                    print(f"  download failed for {s['id']}: {type(e).__name__}")
                    continue
                with open(tmp, "wb") as f:
                    f.write(r.content)
                if not to_wav(ffmpeg, tmp, wav):
                    print(f"  ffmpeg failed for {s['id']}")
                    continue
                os.remove(tmp)
                time.sleep(0.3)
            n_ok += 1
            pack = s.get("pack") or ""
            rows.append({
                "class": cls, "id": s["id"], "file": os.path.basename(wav), "name": s.get("name", ""),
                "url": s.get("url", ""), "license": license_name(s.get("license")), "license_url": s.get("license", ""),
                "author": s.get("username", ""), "pack": pack, "duration_s": s.get("duration", ""),
                "original_samplerate": s.get("samplerate", ""), "original_type": s.get("type", ""),
                "search_query": query_of.get((cls, s["id"]), ""), "tags": " ".join(s.get("tags") or []),
            })
        print(f"{cls}: {n_ok} files on disk")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    cols = list(rows[0].keys()) if rows else ["class", "id"]
    for path in (CSV_PATH, PUBLIC_CSV):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    print("written", CSV_PATH, "and", PUBLIC_CSV)


if __name__ == "__main__":
    main()
