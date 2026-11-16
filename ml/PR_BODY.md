> **Draft, please do not merge yet.** The CNN in this PR is a **placeholder trained on synthetic
> (computer made) sounds**, because the real MIVIA dataset could not be downloaded yet (the site
> needs a registered account). There are **no real accuracy numbers** in this PR. All the code
> is finished and tested, and the app builds. It becomes a normal PR once the model is trained
> on the real data.

## What was added

A small CNN, trained by our team, that listens to the phone microphone on the Victim screen and
classifies each second of sound as `crash`, `skid`, `background` or `loud_not_crash`.

* `ml/`: all the Python code: cut audio into 1 second windows, log Mel spectrogram features
  (`tf.signal`), 4 fold cross validation training, comparison with the old loudness rule,
  export to TensorFlow.js, a check that Keras and TensorFlow.js give the same output, and a
  tool to test our own phone recordings.
* `public/model/`: the TensorFlow.js model and `mel.json` (feature settings).
* `src/ml/crashDetector.js` and `src/ml/features.js`: run the model in the browser every
  0.5 seconds on the existing mic stream.
* `ml/REPORT_SECTION.md`: text for the project report. `ml/DECISIONS.md`: every choice made.

## Why

Our guide wants an AI model that the team trains itself, not only a call to the Gemini API.
Also, the old audio trigger only checked loudness, so any loud sound could start an alarm.

**Gemini triage is unchanged.** The CNN is an extra, it does not replace Gemini.

## Dataset and licence

* **MIVIA Road Audio Events** (University of Salerno): 200 car crashes and 200 tire skids mixed
  into road background, 4 folds (numbers from the dataset page). Free for registered users,
  please cite Foggia, Petkov, Saggese, Strisciuglio, Vento, Pattern Recognition Letters, 2015.
  **Not downloaded yet**, so not used yet.
* **ESC-50** (K. J. Piczak, 2015), licence CC BY-NC 3.0: 10 classes of other loud sounds (door
  knock, fireworks, glass breaking, car horn, siren and more) for the `loud_not_crash` class.
* No dataset audio is in this repo (`ml/data/` is git ignored).

## Results

**None yet.** `ml/results/RESULTS.md` has no numbers on purpose. The folder
`ml/results/synthetic_smoke_test/` is only a code test on fake sounds. It is not a result and
must not go in the report.

## CNN vs old loudness rule

Not measured on real data yet. The code for the comparison is ready: `ml/train.py` scores the
old rule on the same test windows, with its threshold tuned on the training folds, and reports
the false alarm rate on `loud_not_crash` for both.

## What changed in `VictimUI.jsx`

* The loudness meter is still shown.
* The audio trigger for the 15 second countdown is now: crash probability above 0.7 for 2
  windows in a row, **or** crash probability above 0.5 together with a motion spike. One
  prediction alone never starts it. The cancel countdown is unchanged.
* If the model cannot load, the card shows "AI model offline" and the old loudness rule is used.
* While the model is the synthetic placeholder, the card shows "TEST MODEL" and the old loudness
  rule stays in charge. A fake model is not allowed to start alarms.
* The Audio Pulse card shows the live class and confidence.
* The Firestore alert has three new fields: `cnnLabel`, `cnnConfidence`, `modelVersion`.
* The motion trigger, SOS button, timer, Survival Mode and beacon are not changed.

## What changed in `CommandCenter.jsx`

* One new box on each alert card, "Team CNN (audio)", showing the class, the confidence and the
  model version, just above the Gemini assessment. Nothing in the Gemini code was touched.

## Other small changes

* `package.json`: added `@tensorflow/tfjs`. It is loaded only on the Victim screen.
* `eslint.config.js`: ignore `ml/.venv` and `ml/data`.
* `.gitignore`: `ml/data/`, `ml/.venv/`, Keras model files.

## How to test

```powershell
npm install
npm run build
npm run dev          # open the https address on a phone, press "Initialize Tracking Sensors"
```

The Audio Pulse card should show "AI model loading..." and then "TEST MODEL: background NN%".
Rename `public/model/model.json` for a moment to see "AI model offline".

Python side (needs Python 3.10 and the venv, see `ml/MORNING_SUMMARY.md`):

```powershell
ml\.venv\Scripts\python.exe ml\features.py        # shape test
ml\.venv\Scripts\python.exe ml\check_export.py    # Keras vs TensorFlow.js, must print PASSED
```

Already checked: `npm run build` passes, `npm run lint` shows only the errors that were there
before, the Keras and TensorFlow.js outputs match (difference about 1e-7), and the Victim
screen was run in headless Chrome with a fake microphone.

## Known limits

* No real trained model and no real results yet (see the top).
* Not tested on a real phone yet. Speed on slow phones is unknown.
* MIVIA was recorded with fixed roadside microphones. A phone in a car or pocket sounds
  different, so real world accuracy will be lower than the dataset score.
* Other loud sounds not in the training data can still look like a crash. A crash played from
  a speaker also counts as a crash sound. That is why the countdown and motion check stay.
* For demos we play crash audio from a speaker. Nobody stages a real crash.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
