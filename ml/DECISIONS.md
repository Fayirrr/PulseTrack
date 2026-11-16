# Decisions made during the unattended run

Every choice that was not spelled out in the task is listed here with the reason.

## Setup

1. **Python 3.10.11** (`py -3.10`). `py -0` showed 3.14, 3.12, 3.10 and 3.9. 3.11 is not
   installed; 3.10 was allowed by the task, so nothing was installed with winget.
2. **TensorFlow 2.15.1 with Keras 2.15.0**, not the newest TensorFlow. Reason: Keras 2 models
   convert cleanly to the TensorFlow.js layers format. Newer TensorFlow uses Keras 3, which the
   tensorflowjs converter handles badly.
3. **CPU training.** The laptop has an RTX 3050, but TensorFlow newer than 2.10 has no GPU
   support on native Windows. The model is small, one epoch takes about 4 to 5 seconds on CPU.
4. **pip SSL problem.** pip could not verify `pypi.org` certificates (something on this machine
   re-signs HTTPS). Fix used: a newer pip (24.3.1) with its `truststore` feature, which uses the
   Windows certificate store. Certificate checks were NOT turned off. The setting is in
   `ml/.venv/pip.ini`.
5. **tensorflowjs was installed with `--no-deps`.** Its normal dependencies pulled in Keras 3,
   NumPy 2 and a new protobuf, which broke TensorFlow 2.15, and one dependency
   (`tensorflow_decision_forests`) has no Windows build at all. `ml/export_tfjs.py` registers an
   empty stand in for that module (the Keras converter does not use it) and has its own small
   writer for the layers format as a second fallback.
6. **Final package versions** (from `pip list`): tensorflow 2.15.1, keras 2.15.0,
   tensorflowjs 4.17.0, tensorflow-hub 0.16.1, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.7.2,
   matplotlib 3.10.9, soundfile 0.14.0, pandas 2.3.3, requests 2.34.2, h5py 3.14.0,
   protobuf 4.25.9, pip 24.3.1. Node side: @tensorflow/tfjs 4.22.0, Node 22.18.0.
7. **ESLint ignores `ml/.venv` and `ml/data`** (one line in `eslint.config.js`). Without it
   `npm run lint` scanned JavaScript inside the Python packages and reported 868 problems.

## Phase 0: data

8. **MIVIA was not downloaded.** The official page sends the download to a request page that is
   "only registered users" and has a login form with reCAPTCHA. Per the task this was not
   bypassed. The live site also reset every connection during the run, so the page text was
   read from the Internet Archive. Steps for you are in `MORNING_SUMMARY.md`.
9. **No replacement dataset was accepted.** Everything found was YouTube derived, gated, had no
   crash class, or had labels that were only search keywords. Details in `DATASET.md`.
   `Titung/car-crash-audio-cc` on Hugging Face was the closest (open, CC BY 3.0) and was still
   rejected: YouTube sound effect clips, labels not annotated, no skid class.
10. **So Phase 0e applies:** a small SYNTHETIC road dataset (`ml/make_synthetic.py`) is used
    only to test the code. Real training in Phase 4 is skipped. No synthetic score is reported
    as a result anywhere.
11. **The synthetic folder copies the assumed MIVIA layout** (`foldN/name.xml`,
    `foldN/sounds/name.wav`, `<events><item>` with CLASS_ID, CLASS_NAME, STARTSECOND,
    ENDSECOND). This layout is a guess from a third party script, so `prepare_data.py` stops
    with a clear message if the real MIVIA files have unknown classes or no readable fold. The
    fold is read from the path with `--fold-regex` (default `fold(\d)`); change it once you see
    the real folder names.

## Phase 2: windows

12. **Windows that touch an event by 50 percent or less are dropped**, not called background.
    A window that is 40 percent crash is neither clean background nor a clear crash, and
    keeping it as background teaches the model that crash sound is background. Windows with
    zero overlap are background, more than 50 percent are the event class, as the task says.
13. **ESC-50 windows that are near silent are dropped** (RMS under 10 percent of the loudest
    window of the same clip). Many ESC-50 clips are a short sound padded with digital silence;
    a silent window labelled "loud_not_crash" would be a wrong label.
14. **ESC-50 fold 5** is spread over groups 1 to 4 by `(Freesound clip id mod 4) + 1`. Folds 1
    to 4 keep their number. All takes of one original clip stay in one group.
15. Windows are stored as int16 to keep the file small. Features are computed from
    `int16 / 32768`.

## Phase 3: features

16. Mel range 50 Hz to 8000 Hz. The task fixed 16 kHz, n_fft 1024, hop 256 and 64 mel bins but
    not the range. 8000 Hz is the Nyquist limit, 50 Hz cuts mains hum and DC.
17. Power spectrogram (magnitude squared), `log(x + 1e-6)`, then per sample normalisation
    `(x - mean) / (std + 1e-6)`. No padding, so one window gives 59 frames: input 59 x 64 x 1.
18. Per sample normalisation removes absolute loudness on purpose. Phone microphones have very
    different gain, so the model should look at the shape of the sound, not its level.
19. `mel.json` is written to `public/model/mel.json` (the task text) and not to
    `ml/export/mel.json` (CLAUDE.md), because the browser must be able to fetch it.

## Phase 4: training

20. Model: 4 conv blocks (16, 32, 64, 96 filters, 3x3, batch norm, ReLU, max pool after the
    first three, dropout 0.25), global average pooling, dropout 0.3, dense softmax over 4
    classes. 79,700 parameters.
21. Validation split: 15 percent of the train folds, split by source clip
    (`GroupShuffleSplit`), so windows of one clip are never in both train and validation.
22. Final model: trained on all folds for the mean "best epoch" of the 4 folds, with no
    validation set (all data is used for training).
23. Loudness baseline: an offline copy of the maths in `VictimUI.jsx` (AnalyserNode, fftSize
    2048, Blackman window, byte scale from -100 dB to -30 dB, mean over 1024 bins, times
    120/255). The loudest 2048 sample frame in the window is the score. The analyser's time
    smoothing and real microphone gain cannot be copied offline. The threshold is tuned for
    best crash F1 on the train folds only; the app default 85 is reported too.
24. **Because the data is synthetic**, `train.py` writes to `ml/results/synthetic_smoke_test/`
    and `ml/results/RESULTS.md` only says that real training was skipped. The "try at most 3
    improvements" step was not run: tuning on fake data is meaningless. The switches for it
    exist (`--width`, `--dropout`, `--no-specaug`, `--tag`).
25. A full length run (up to 60 epochs with early stopping) was done on the synthetic data, not
    only the 3 epoch smoke test, so early stopping, curves and the final model step are tested.

## Phase 5 and 6: export and app

26. **A placeholder model is shipped in `public/model/`.** It is trained on synthetic audio and
    marked `"synthetic": true` with version `crashcnn-SYNTHETIC-TEST-<date>`. It exists so the
    export check and the browser code path can run.
27. **The placeholder is not allowed to trigger alerts.** When `model_meta.json` says
    `synthetic: true` the app shows the live class with the words "TEST MODEL" and keeps the old
    loudness rule as the audio trigger. A model trained on fake sounds must not decide when an
    emergency countdown starts. After real training the flag becomes false and the CNN rule
    (above 0.7 twice in a row, or above 0.5 with a motion spike) takes over with no code change.
28. While the model is still loading the loudness rule is also active, so there is never a
    moment without an audio trigger. Loading gives up after 20 seconds and shows
    "AI model offline".
29. TensorFlow.js is loaded with a dynamic `import()` inside `crashDetector.js`, so the other
    pages do not download it.
30. Audio capture uses a `ScriptProcessorNode` on the existing mic source. It is deprecated but
    works on every mobile browser with no extra worklet file. Its output goes through a gain of
    0, so the mic is never played on the speaker.
31. Resampling in the browser is a 63 tap low pass filter plus linear interpolation
    (`src/ml/features.js`). Training used `scipy.signal.resample_poly`. These are not bit
    identical; everything after resampling is. This is noted as a limitation.
32. "Motion spike" for the combined rule means a motion reading above the motion threshold in
    the last 2 seconds. Note that the app's existing motion rule already starts the countdown on
    such a spike by itself; that old behaviour was not changed.
33. `cnnLabel`, `cnnConfidence`, `modelVersion` in the Firestore payload hold the prediction
    that was live when the countdown started (for a manual SOS: the newest prediction). The
    prediction at send time, 15 seconds later, would usually just say "background".
34. The detector is stopped when Survival Mode starts, to save battery.
35. The existing microphone request (`audio: true`) was not changed. Browsers apply echo
    cancellation, noise suppression and auto gain by default, which can soften a crash sound.
    Turning them off would also change the recording sent to the Command Center, so it is left
    as a suggestion in `MORNING_SUMMARY.md`.

## Git and PR

36. `PulseTrack_Claude_Context_Brief.pdf` is NOT committed. It contains the team's names and
    university roll numbers and the PR goes to a public GitHub repo. `CLAUDE.md` is committed.
37. `ml/export/check_windows.*` (10 audio windows used by the export check) are git ignored
    because they can contain ESC-50 audio (CC BY-NC). The check script recreates them.
