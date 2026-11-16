# Results: crash sound CNN vs loudness rule

## There are no real results yet

Real training was **skipped**. The MIVIA Road Audio Events dataset could not be downloaded
during the unattended run (the site needs a registered account, login and reCAPTCHA) and no
other open dataset with properly labelled car crash and tire skid audio was found. See
`ml/DATASET.md` and `ml/MORNING_SUMMARY.md`.

So this file has **no accuracy, precision, recall, F1 or false alarm numbers**. Nothing here may
be quoted in the project report.

## What was run instead

The complete pipeline was run on SYNTHETIC road audio (made from noise and sine tones by
`ml/make_synthetic.py`) plus real ESC-50 clips, only to prove that the code works from start to
end: windows, features, 4 fold cross validation, loudness baseline, figures, final model,
TensorFlow.js export.

The log of that run is in `ml/results/synthetic_smoke_test/SMOKE_TEST_LOG.md`. Those scores
describe how well a CNN separates computer made beeps and noise bursts. They say nothing about
real crashes and are not results.

## How to get the real results

1. Put the MIVIA dataset in `ml/data/MIVIA_ROAD_DB1` (steps in `ml/MORNING_SUMMARY.md`).
2. Run:

```powershell
ml\.venv\Scripts\python.exe ml\inspect_dataset.py
ml\.venv\Scripts\python.exe ml\prepare_data.py --dataset mivia
ml\.venv\Scripts\python.exe ml\train.py
ml\.venv\Scripts\python.exe ml\export_tfjs.py
ml\.venv\Scripts\python.exe ml\check_export.py
```

`train.py` then overwrites this file with the real tables: mean and standard deviation over the
4 folds, CNN vs loudness rule, and the false alarm rate on `loud_not_crash` for both.
