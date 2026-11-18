r"""Download a Kaggle dataset zip, streamed straight to disk, and unzip it.

Run:  ml\.venv\Scripts\python.exe ml\kaggle_download.py chrisfilo/urbansound8k ml\data\UrbanSound8K

Why not just "kaggle datasets download": on this laptop the kaggle CLI kept the whole 6 GB
download in memory, and Python's certificate bundle fails behind this machine's HTTPS
interception. This script uses the same Kaggle API endpoint, the same credentials
(%USERPROFILE%\.kaggle\kaggle.json, read in memory, never printed or written anywhere),
truststore for the Windows certificate store (verification stays ON), resumes a broken
download, and writes in 8 MB chunks.
"""
import json
import os
import sys
import time
import zipfile

import requests
import truststore

truststore.inject_into_ssl()


def credentials():
    path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d["username"], d["key"]


def download(ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, ref.split("/")[-1] + ".zip")
    part = zip_path + ".part"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    auth = credentials()
    for attempt in range(1, 21):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            # auth goes to kaggle.com only; requests drops it on the redirect to the storage host
            with requests.get(url, auth=auth, headers=headers, stream=True, timeout=120) as r:
                if r.status_code == 416:
                    break
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server ignored the range, start again
                total = have + int(r.headers.get("Content-Length", 0))
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    done, t0, last = have, time.time(), 0
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if done - last > 500 * 1024 * 1024:
                            last = done
                            print(f"  {done / 1e9:.2f} / {total / 1e9:.2f} GB "
                                  f"({(done - have) / 1e6 / max(time.time() - t0, 1):.1f} MB/s)", flush=True)
                if total and done >= total:
                    break
        except (requests.RequestException, OSError) as e:
            print(f"  attempt {attempt} interrupted: {type(e).__name__}, resuming", flush=True)
            time.sleep(min(5 * attempt, 60))
    os.replace(part, zip_path)
    print("downloaded", zip_path, os.path.getsize(zip_path), "bytes", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad:
            raise SystemExit(f"zip is damaged at {bad}")
        z.extractall(out_dir)
    os.remove(zip_path)
    print("unzipped into", out_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    download(sys.argv[1], sys.argv[2])
