## What was added and why

Our guide wants an AI model that the team trains itself, not only a call to the Gemini API.
Also, the old audio trigger only checked loudness, so any loud sound could start an alarm.

This PR adds a small CNN (79,700 parameters) that classifies each second of microphone audio as
`crash`, `skid`, `background` or `loud_not_crash`. We trained it ourselves and it runs in the
browser with TensorFlow.js.

**Gemini triage is unchanged.** Gemini is an external API; the CNN is trained by the team and
is an extra signal, not a replacement.

* `ml/`: Python code. Data download (`kaggle_download.py`, `freesound_download.py`), 4 leak
  free folds and 1 second windows (`prepare_data.py`), log Mel features with `tf.signal`
  (`features.py`), 4 fold training plus comparison with the old loudness rule (`train.py`),
  export to TensorFlow.js (`export_tfjs.py`), Keras vs TensorFlow.js check (`check_export.py`
  + `check_export.mjs`), phone recording test (`phone_test.py`), `run_pipeline.ps1` to run all.
* `public/model/`: the trained model (`crashcnn-open_freesound_us8k_esc50-20261118`) and
  `mel.json` (feature settings shared by Python and the browser).
* `src/ml/crashDetector.js`, `src/ml/features.js`: run the model every 0.5 s on the existing
  mic stream.
* Docs: `ml/DATASET.md`, `ml/DECISIONS.md`, `ml/REPORT_SECTION.md` (report text),
  `ml/results/RESULTS.md` (all numbers), `ml/freesound_sources.csv` (every Freesound clip with
  id, link, licence, author).

## Datasets and licences

| source | used for | licence | used |
|---|---|---|---|
| Freesound (searched through its API) | crash, skid | per clip: CC0 1.0 or CC BY 3.0 / 4.0 only | 136 crash clips (65 uploaders), 54 skid clips (20 uploaders) |
| UrbanSound8K (Salamon, Jacoby, Bello, ACM MM 2014), Kaggle `chrisfilo/urbansound8k` | background (engine idling, street music, air conditioner, drilling, children playing, jackhammer); loud_not_crash (car horn, siren, gun shot) | CC BY-NC 4.0 (as listed on Kaggle) | 7732 clips |
| ESC-50 (Piczak, ACM MM 2015) | loud_not_crash (door knock, glass breaking, fireworks, car horn, siren, dog, clapping, can opening, thunderstorm) | CC BY-NC 3.0 | 360 clips |

Freesound licences: crash 98 CC0, 32 CC BY 4.0, 6 CC BY 3.0; skid 44 CC0, 8 CC BY 4.0,
2 CC BY 3.0. CC BY authors and links are in `ml/freesound_sources.csv`. No audio is in the repo
(`ml/data/` is git ignored), and neither are `kaggle.json` or `freesound_key.txt`.

Windows (1 s, 0.5 s hop, 16 kHz), 4 folds split by source so no clip, recording or Freesound
uploader is in two folds:

| class | fold 1 | fold 2 | fold 3 | fold 4 | total |
|---|---|---|---|---|---|
| background | 2955 | 2955 | 2954 | 2954 | 11818 |
| crash | 527 | 263 | 263 | 263 | 1316 |
| skid | 458 | 88 | 88 | 88 | 722 |
| loud_not_crash | 1307 | 1307 | 1307 | 1307 | 5228 |
| all | 5247 | 4613 | 4612 | 4612 | 19084 |

## Results

All numbers were printed by `ml/train.py` (full tables in `ml/results/RESULTS.md`). Mean ±
standard deviation over the 4 test folds.

| metric | CNN | loudness rule (threshold tuned on train folds) |
|---|---|---|
| accuracy, 4 classes | 0.673 ± 0.062 | not applicable |
| macro F1, 4 classes | 0.503 ± 0.024 | not applicable |
| crash vs rest accuracy | 0.887 ± 0.037 | 0.755 ± 0.069 |
| crash precision | 0.309 ± 0.070 | 0.108 ± 0.016 |
| crash recall | 0.475 ± 0.034 | 0.387 ± 0.203 |
| **crash F1** | **0.370 ± 0.058** | **0.151 ± 0.025** |
| **false alarm rate on loud_not_crash** | **0.104 ± 0.045** | **0.220 ± 0.090** |
| false alarm rate on all non crash windows | 0.084 ± 0.036 | 0.214 ± 0.099 |

CNN per class:

| class | precision | recall | F1 |
|---|---|---|---|
| background | 0.884 ± 0.048 | 0.713 ± 0.076 | 0.787 ± 0.050 |
| crash | 0.309 ± 0.070 | 0.475 ± 0.034 | 0.370 ± 0.058 |
| skid | 0.095 ± 0.039 | 0.441 ± 0.150 | 0.148 ± 0.045 |
| loud_not_crash | 0.775 ± 0.041 | 0.656 ± 0.055 | 0.707 ± 0.018 |

## CNN vs old loudness rule

* **Crash F1: 0.370 for the CNN vs 0.151 for the loudness rule.** Crash precision roughly
  triples (0.309 vs 0.108), recall is a bit higher (0.475 vs 0.387).
* **False alarms on loud sounds that are not crashes: the CNN raised a crash alarm on 10.4% of
  `loud_not_crash` windows, the loudness rule on 22.0%**, about half as many.
* Per source it is mixed. The CNN is much better on sirens, car horns, can opening, clapping,
  glass breaking and gun shots (gun shot: 0.380 vs 0.715). It is **worse** on fireworks
  (0.630 vs 0.294), dog (0.402 vs 0.277), door knock (0.252 vs 0.150) and thunderstorm
  (0.200 vs 0.092). Full table in `RESULTS.md`.
* With the app's old default threshold of 85 the loudness rule almost never fires on these
  recordings (crash recall 0.039), so the comparison uses a threshold tuned on the training
  folds, which is the fairer baseline.

## What changed in `VictimUI.jsx`

* The loudness meter is still shown.
* The audio trigger for the 15 second countdown is now: crash probability above 0.7 for 2
  windows in a row, **or** crash probability above 0.5 together with a motion spike. One
  prediction alone never starts it. The cancel countdown is unchanged.
* If the model cannot load, the card shows "AI model offline" and the old loudness rule is used.
* The Audio Pulse card shows the live class and confidence.
* The Firestore alert has three new fields: `cnnLabel`, `cnnConfidence`, `modelVersion`.
* The motion trigger, SOS button, timer, Survival Mode and beacon are not changed.

## What changed in `CommandCenter.jsx`

* One new box on each alert card, "Team CNN (audio)", with the class, confidence and model
  version, just above the Gemini assessment. The Gemini code was not touched.

## Other small changes

* `package.json`: `@tensorflow/tfjs`, loaded only on the Victim screen.
* `eslint.config.js`: ignore `ml/.venv` and `ml/data`.
* `.gitignore`: `ml/data/`, `ml/.venv/`, model checkpoints, `kaggle.json`, `freesound_key.txt`.

## How to test

```powershell
npm install
npm run build
npm run dev          # open the https address on a phone, press "Initialize Tracking Sensors"
```

The Audio Pulse card shows "AI model loading..." and then "AI: background NN%". Play a crash
sound from a speaker near the phone and watch the label. Rename `public/model/model.json` for a
moment to see "AI model offline".

Python side (Python 3.10 venv, see `ml/MORNING_SUMMARY.md`):

```powershell
ml\.venv\Scripts\python.exe ml\check_export.py       # Keras vs TensorFlow.js, must print PASSED
ml\.venv\Scripts\python.exe ml\phone_test.py "C:\path\to\recordings"
```

Checked: `npm run build` passes; `npm run lint` shows only errors that were already there; the
Keras and TensorFlow.js outputs of the trained model match (max difference 2.98e-07, limit
1e-3, no leaked tensors); the Victim screen was run in headless Chrome with a fake microphone
(live labels, countdown after 2 crash windows, cancel, "AI model offline").

## Known limits

* **Skid detection is weak** (F1 0.148, precision 0.095), mainly because of the small sample:
  only 54 skid clips (722 windows) from 20 uploaders, and 27 of the 54 clips come from one
  uploader. Because a whole uploader stays in one fold, fold 1 holds 458 of the 722 skid
  windows, so the other folds train and test on very few skids. More skid recordings are the
  first thing to add.
* **Crash detection is moderate, not reliable**: crash F1 0.370, macro F1 0.503. At the app's
  trigger level (crash probability above 0.7) crash recall is only 0.128 per window, so many
  crashes will not start the countdown from audio alone; the motion rule and the SOS button
  still work. The "up to 3 improvements" step has not been run yet.
* The CNN false alarms more than the loudness rule on fireworks, dog, door knock and
  thunderstorm sounds.
* The crash and skid clips are crowd sourced uploads, not phone recordings of real crashes, so
  real world accuracy will differ (domain shift). The classes also come from three different
  datasets, not one controlled recording setup.
* Freesound labels are the uploaders' own tags, filtered by keywords, not checked by listening.
* Not tested on a real phone yet; `ml/phone_test.py` is ready for that.
* For demos we play crash audio from a speaker. Nobody stages a real crash.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
