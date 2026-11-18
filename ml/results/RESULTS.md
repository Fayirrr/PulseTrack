# Results: crash sound CNN vs loudness rule

## There are no real results yet

Real training was **skipped** on 2026-11-18. The crash and skid clips come from Freesound, and
`freesound_key.txt` in the repo root was empty (0 bytes) for the whole run, so nothing could be
downloaded. Without crash audio there is no positive class, so neither the CNN nor the
loudness rule can be scored for crash detection.

So this file has **no accuracy, precision, recall, F1 or false alarm numbers**. Nothing here may
be quoted in the project report.

What is ready: UrbanSound8K and ESC-50 are downloaded and windowed (17,046 windows, counts in
`ml/DATASET.md`), and every script ran on them. A 3 epoch code test of `train.py` on those
windows was written to a temporary folder and thrown away, because without a crash class its
numbers mean nothing.

The folder `ml/results/synthetic_smoke_test/` is an older code test on SYNTHETIC audio. It is
not a result either.

## How to get the real results

1. Paste the Freesound API key into `freesound_key.txt` (one line) and save the file.
2. From the repo root run:

```powershell
powershell -ExecutionPolicy Bypass -File ml\run_pipeline.ps1
```

`train.py` then overwrites this file with the real tables: mean and standard deviation over the
4 folds, CNN vs loudness rule, and the false alarm rate on `loud_not_crash` for both, per source.
