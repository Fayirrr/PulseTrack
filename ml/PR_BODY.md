> **Draft, please do not merge yet.** The crash and skid sounds (from Freesound) are not
> downloaded yet because the Freesound API key was missing, so the real model is **not
> trained** and there are **no results** in this PR. The CNN now in `public/model/` is a
> placeholder trained on synthetic sounds. The app shows it as "TEST MODEL" and does not let it
> start alarms. All code is finished and tested and the app builds. One command
> (`ml\run_pipeline.ps1`) does the rest once the key is in place.

## What was added and why

Our guide wants an AI model that the team trains itself, not only a call to the Gemini API.
Also, the old audio trigger only checked loudness, so any loud sound could start an alarm.

This PR adds a small CNN (79,700 parameters) that classifies each second of microphone audio as
`crash`, `skid`, `background` or `loud_not_crash`, trained by us and run in the browser.

* `ml/`: Python code. Download the data (`kaggle_download.py`, `freesound_download.py`), build
  4 folds without leakage and 1 second windows (`prepare_data.py`), log Mel features with
  `tf.signal` (`features.py`), 4 fold training and the comparison with the old loudness rule
  (`train.py`), export to TensorFlow.js (`export_tfjs.py`), a Keras vs TensorFlow.js check
  (`check_export.py` + `check_export.mjs`), a phone recording test (`phone_test.py`), and
  `run_pipeline.ps1` to run it all.
* `public/model/`: the TensorFlow.js model and `mel.json` (feature settings).
* `src/ml/crashDetector.js`, `src/ml/features.js`: run the model every 0.5 s on the existing
  mic stream.
* Docs: `ml/DATASET.md`, `ml/DECISIONS.md`, `ml/REPORT_SECTION.md` (report text).

**Gemini triage is unchanged.** The CNN is an extra, it does not replace Gemini.

## Datasets and licences

| source | used for | licence | status |
|---|---|---|---|
| Freesound (crash and skid recordings) | crash, skid | per clip, CC0 or CC BY only (filter in the script) | **not downloaded, API key missing** |
| UrbanSound8K (Salamon, Jacoby, Bello, 2014), Kaggle `chrisfilo/urbansound8k` | background (6 classes), loud_not_crash (car horn, siren, gun shot) | CC BY-NC 4.0 (as on Kaggle) | downloaded, 7732 clips used |
| ESC-50 (Piczak, 2015) | loud_not_crash (9 classes) | CC BY-NC 3.0 | downloaded, 360 clips used |

Every downloaded Freesound clip will be listed with id, link, licence and author in
`ml/freesound_sources.csv`. No audio is in the repo (`ml/data/` is git ignored), and neither
are `kaggle.json` and `freesound_key.txt`.

## Results

**None yet.** `ml/results/RESULTS.md` has no numbers on purpose: without crash audio there is
nothing to score. What is done on real data: UrbanSound8K + ESC-50 are cut into 17,046 windows
in 4 folds with exactly 25 percent of each class per fold, and the leakage check passed.

## CNN vs old loudness rule

Not measured yet. `train.py` is ready to score the old rule on the same test windows (threshold
tuned on the training folds only) and to report, in a highlighted section, how often each
method raises a crash alarm on `loud_not_crash` sounds, per source (door knock, fireworks,
glass breaking, siren, gun shot, ...). That false alarm rate is the main weak point of the
loudness rule.

## What changed in `VictimUI.jsx`

* The loudness meter is still shown.
* The audio trigger for the 15 second countdown is now: crash probability above 0.7 for 2
  windows in a row, **or** crash probability above 0.5 together with a motion spike. One
  prediction alone never starts it. The cancel countdown is unchanged.
* If the model cannot load, the card shows "AI model offline" and the old loudness rule is used.
* While the model is the synthetic placeholder, the card shows "TEST MODEL" and the old loudness
  rule stays in charge.
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

The Audio Pulse card should show "AI model loading..." and then "TEST MODEL: background NN%".

To train the real model (Python 3.10 venv, see `ml/MORNING_SUMMARY.md`): put the Freesound API
key in `freesound_key.txt`, then

```powershell
powershell -ExecutionPolicy Bypass -File ml\run_pipeline.ps1
```

Already checked: `npm run build` passes, `npm run lint` shows only the errors that were there
before, Keras and TensorFlow.js outputs match (difference about 1e-7), the Victim screen was
run in headless Chrome with a fake microphone (live labels, countdown after 2 crash windows,
cancel, "AI model offline").

## Known limits

* No real trained model and no results yet (see the top).
* The crash and skid clips will be crowd sourced uploads, not phone recordings of real crashes,
  so real world accuracy will be lower than the test fold score (domain shift).
* The classes come from three different datasets, not one controlled recording setup.
* Not tested on a real phone yet.
* For demos we play crash audio from a speaker. Nobody stages a real crash.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
