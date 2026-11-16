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

## Phase 4, train and evaluate: code done, REAL TRAINING SKIPPED

* `ml/train.py` written: small CNN (79,700 parameters), 4 fold cross validation, class weights,
  augmentation (gain, time shift, background mix, SpecAugment), early stopping, loudness rule
  baseline with threshold tuned on train folds, per fold json + confusion matrix png + curves
  png, RESULTS.md writer, final model on all folds.
* Real training was skipped because there is no real crash data (Phase 0).
* What worked: a 3 epoch smoke run, then a full length run (early stopping, 23 to 51
  epochs per fold, about 20 minutes on CPU) on SYNTHETIC road audio + ESC-50. Output is in
  `ml/results/synthetic_smoke_test/` and is labelled as not a result.
* `ml/results/RESULTS.md` says that there are no real results yet and has no numbers.
* Not done: the "up to 3 improvements" step. It needs real data.

## Phase 5, export: done (placeholder model)

* Failed first: `import tensorflowjs` needs `tensorflow_decision_forests` (no Windows build) and
  `jax`. Fixed in `ml/export_tfjs.py` with stand in modules; the official
  `tensorflowjs 4.17.0 save_keras_model` converter then worked. A manual writer exists as a
  fallback and was also tested.
* `public/model/`: `model.json`, `group1-shard1of1.bin` (318,800 bytes), `mel.json`,
  `model_meta.json` (`synthetic: true`, version `crashcnn-SYNTHETIC-TEST-20261117`).
* `ml/check_export.py` + `ml/check_export.mjs`: 10 saved windows, features with tf.signal on
  both sides. Printed: overall max difference 1.192e-07 (tolerance 1e-3), 0 leaked tensors.
  EXPORT CHECK PASSED.

## Phase 6, app integration: done

* `npm install @tensorflow/tfjs` (4.22.0).
* New: `src/ml/features.js`, `src/ml/crashDetector.js`. Changed: `src/pages/VictimUI.jsx`,
  `src/pages/CommandCenter.jsx`. Gemini code not touched.
* `npm run build`: passes. TensorFlow.js is in its own chunk, loaded only on the Victim screen.
* `npm run lint`: 22 errors and 3 warnings, the same 22 and 3 as before my changes (all in old
  code: CommandCenter 2, RescueTeam 4, SafeUI 5 + 1 warning, VictimUI 11 + 2 warnings). My two
  new errors were fixed. Old lint issues were left alone as asked.
* Tested in headless Chrome with a fake microphone playing a synthetic clip, Firebase blocked:
  * placeholder model: card shows "TEST MODEL: crash 100%", "skid 100%", "background 100%"
    following the clip, one update every 0.5 s, no countdown from the CNN
  * model pretending to be real (test only override of `model_meta.json`): countdown started
    after 2 crash windows in a row, Cancel returned to monitoring
  * `model.json` blocked: card shows "AI model offline (loudness rule active)"
* `node ml/sim_detector.mjs`: detector run in Node, tensor count stayed flat.
* NOT tested: a real phone, real microphone, real Firestore write, Command Center display with
  a real alert. Speed on a real phone is unknown.

## Phase 7, phone test tool: done

* `ml/phone_test.py` tested on wav, mp3, m4a and webm files. ffmpeg is needed for everything
  except wav and flac; it is already installed on this laptop.

## Phase 8, report material: done, results sections empty

* `ml/REPORT_SECTION.md`. Sections 6 and 7 wait for real training.

## Finish

* `gh` (GitHub CLI) is not installed on this laptop, so the pull request was NOT sent. The
  branch is committed locally only. Commands are in `MORNING_SUMMARY.md`, PR text is in
  `ml/PR_BODY.md`.
