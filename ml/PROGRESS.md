# Progress log

Run started 2026-11-16 late evening. Branch `ml-crash-cnn`.

## Setup: done

* Branch `ml-crash-cnn` created from `master`.
* `py -0` showed 3.14, 3.12, 3.10, 3.9. Used Python 3.10.11, venv at `ml/.venv`.
* Failed first: pip could not verify HTTPS certificates. Fixed with pip 24.3.1 + truststore.
* Failed first: installing tensorflowjs dependencies replaced Keras 2 / NumPy 1 with Keras 3 /
  NumPy 2 and broke TensorFlow. Fixed by reinstalling pinned versions and installing
  tensorflowjs and tensorflow-hub with `--no-deps`.
* `.gitignore` now has `ml/data/`, `ml/.venv/`, `ml/models/*.keras`, `ml/models/*.h5`.
* ESLint now ignores `ml/.venv` and `ml/data`.

## Phase 0, get the data: MIVIA NOT obtained, ESC-50 obtained

* MIVIA: live site reset every connection (tried 6 URLs, twice). Internet Archive copy of the
  dataset page shows the download is behind registration + login + reCAPTCHA. Not bypassed.
* Searched Zenodo, Hugging Face, GitHub and the web for an open crash and skid dataset. Nothing
  passed the rules (see `DATASET.md`).
* ESC-50 downloaded and unzipped to `ml/data/ESC50` (2000 wav files).
* Fallback 0e: `ml/make_synthetic.py` wrote 24 SYNTHETIC one minute clips to
  `ml/data/SYNTHETIC_ROAD`. Code test only.

## Phase 1, inspect: done for what exists

* `ml/inspect_dataset.py` written and run on ESC-50 and on the synthetic folder.
* `ml/DATASET.md` written. MIVIA section holds only what the official page says and is marked
  as not verified on files.

## Phase 2, windows: done (on synthetic + ESC-50)

* `ml/prepare_data.py` works: 5028 windows saved to `ml/data/windows.npz`. Counts per class per
  fold are in `DATASET.md`.
* Not tested on real MIVIA files (none available). The MIVIA parser is based on a third party
  script and stops with a message if the files look different.

## Phase 3, features: done

* `ml/features.py` with `tf.signal` only. Shape test passed: one window gives (59, 64, 1), a
  batch of 3 gives (3, 59, 64, 1), mel matrix (513, 64).
* `public/model/mel.json` written with the mel matrix and every setting.
