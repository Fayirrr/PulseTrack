# CLAUDE.md

## Project

PulseTrack: An AI Driven Autonomous Disaster Management System.
Final year BE AIML project, P.A. College of Engineering, Mangalore (VTU), 2026 to 2027.
Guide: Prof. Afsar Baig.

PulseTrack is a zero install PWA. A phone opens a link, the browser watches motion, mic, GPS, camera and battery, and when something looks like an emergency it shows a 15 second cancel countdown, then goes into Survival Mode (black screen, 3 kHz beacon) and sends an alert to Firebase Firestore. The Command Center shows alerts on a Leaflet map. The Rescue Terminal tracks rescue status.

Stack: React 19, Vite 8, Tailwind CSS 4, React Router 7, Leaflet, Firebase Firestore, Gemini API, Vercel.

## The task we are doing now

The professor wants a real AI model trained by the team, not only an API call.
Right now the only AI is Gemini triage in `src/pages/CommandCenter.jsx`. That stays as it is.

We are adding a team trained 2D CNN that classifies road audio from the phone mic:
crash, tire skid, background (and optionally a fourth class, loud but not a crash).

It replaces the current audio trigger in `src/pages/VictimUI.jsx` (inside `updateAudio`, around lines 185 to 196). That trigger only checks if average loudness is above a threshold (default 85), so any loud sound sets off an alarm. The CNN should tell a crash apart from other loud sounds.

## Key files

* `src/pages/VictimUI.jsx`: sensors, mic stream, loudness trigger, countdown, Survival Mode, Firestore payload
* `src/pages/CommandCenter.jsx`: Gemini triage and command map
* `src/pages/RescueTeam.jsx`: rescue workflow
* `src/pages/SafeUI.jsx`: citizen safe portal and mesh relay
* `src/firebase.js`: Firebase setup
* `PulseTrack_Claude_Context_Brief.pdf`: full brief for the ML work

## Dataset

MIVIA Road Audio Events (University of Salerno), placed in `ml/data/MIVIA_ROAD_DB1`.
400 events (200 car crashes, 200 tire skids) mixed into road background audio, about one minute clips, 32 kHz, 16 bit, split into 4 folds.
Read the actual files and XML annotations before writing any loader. Do not guess the format.
Cite: Foggia, Petkov, Saggese, Strisciuglio, Vento, Reliable detection of audio events in highly noisy environments, Pattern Recognition Letters, 2015.

Optional extra negatives: ESC 50 clips such as door knock, fireworks, glass breaking, car horn. Keep their source noted.

## ML plan

All ML code lives in `ml/`.

1. `ml/prepare_data.py`: resample to 16 kHz mono, cut 1 second windows with 0.5 second hop, label by more than 50 percent overlap with an annotated event, keep fold number, save `ml/data/windows.npz`, print counts per class per fold.
2. `ml/features.py`: log Mel spectrogram using `tf.signal` only, not librosa. Settings: 16 kHz, n_fft 1024, hop 256, 64 mel bins. Export the mel weight matrix to `ml/export/mel.json`.
3. `ml/train.py`: small 2D CNN (3 or 4 conv blocks, batch norm, dropout, under 200k params). 4 fold cross validation on the MIVIA folds. Class weights. Light augmentation (gain, time shift, background mixing). Also score the old loudness rule on the same test windows for comparison.
4. Save to `ml/results/`: accuracy, per class precision, recall, F1, macro F1, confusion matrix png, training curves png, comparison table CNN vs loudness rule.
5. Train a final model on all folds and convert it to TensorFlow.js in `public/model/`.
6. `ml/check_export.py`: confirm Keras and TF.js give the same output on the same windows.
7. `ml/phone_test.py`: run the model on our own phone recordings to measure domain shift.

## App integration

* Add `@tensorflow/tfjs`.
* `src/ml/crashDetector.js`: load `public/model` and `mel.json`, keep a rolling 1 second 16 kHz buffer from the existing mic stream, predict every 0.5 seconds using `tf.signal.stft` and the same mel matrix.
* In `VictimUI.jsx`: start the existing 15 second countdown only when crash probability is above 0.7 for 2 windows in a row, or crash audio plus a motion spike.
* Show the live class and confidence on the Audio Pulse card.
* Add the CNN result to the Firestore payload and show it in the Command Center next to the Gemini score.

## Rules

* Never invent dataset numbers, metrics or results. Only report what the scripts actually print.
* Preprocessing in the browser must match training exactly. Use `tf.signal` on both sides.
* Never trigger a real alert from one audio prediction alone. Keep the cancel countdown.
* Keep Gemini triage. The CNN is an addition, not a replacement for it.
* Do not change unrelated features. Keep changes small and explain what changed.
* Clearly separate what already works in the app from what is new or proposed.
* For demos, play crash audio from a speaker. Never stage a real crash.
* Do not commit the dataset or API keys. Add `ml/data/` to `.gitignore`.
