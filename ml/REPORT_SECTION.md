# Report section: Team trained CNN for crash sound detection

> **Read this first.** The method, the code and the app integration described here are finished
> and tested. The **results tables are empty** because the real dataset (MIVIA) could not be
> downloaded yet. Fill sections 6 and 7 from `ml/results/RESULTS.md` after training on the real
> data. Do not put any number from `ml/results/synthetic_smoke_test/` in the report.

## 1. Why this component was added

PulseTrack already uses AI in the Command Center: the Gemini API reads the telemetry and the
photo of an alert and gives a priority score from 1 to 10. Gemini is an **external API**. It
was trained by Google, not by us, and we only send it a request.

Our guide asked for an AI model that the team trains itself. So we added a small Convolutional
Neural Network (CNN) that listens to the phone microphone and classifies the sound as crash,
tire skid, background, or loud but not a crash. This CNN is **trained by the team** with our
own code (`ml/`), runs inside the browser on the phone, and needs no server.

The two AI parts do different jobs and both stay in the system:

| | Gemini triage | Crash sound CNN |
|---|---|---|
| Who trained it | Google (external API) | The PulseTrack team |
| Where it runs | Google cloud, called from the Command Center | In the browser on the victim phone |
| Input | Telemetry text and one photo | 1 second of microphone audio |
| Output | Priority score 1 to 10 and a short assessment | Class and confidence, every 0.5 seconds |
| Job | Rank alerts for the commander | Decide whether a sound is worth starting the cancel countdown |

Before this work the audio trigger was a simple loudness rule: if the average loudness was
above a threshold (default 85), the countdown started. Any loud sound, such as a door slam or
fireworks, could start it. The CNN is meant to tell a crash apart from other loud sounds.

## 2. Dataset

**Planned training data: MIVIA Road Audio Events** (University of Salerno). According to the
official dataset page it has 400 events (200 car crashes and 200 tire skids) mixed into road
background sound, in audio files of about one minute, 32 kHz, 16 bit, WAV, split into 4 folds
for cross validation. The page lists 326.38 s of crash sound, 522.5 s of skid sound and 2732 s
of background.

* Access: free for registered users of the MIVIA lab site. The page does not print a licence;
  it asks users to cite the papers below. *[Confirm the terms on the site when you download.]*
* Citations:
  * P. Foggia, N. Petkov, A. Saggese, N. Strisciuglio, M. Vento, "Reliable detection of audio
    events in highly noisy environments", Pattern Recognition Letters, 2015.
  * P. Foggia, N. Petkov, A. Saggese, N. Strisciuglio, M. Vento, "Audio surveillance of roads:
    a system for detecting anomalous sounds", IEEE Transactions on Intelligent Transportation
    Systems.

*[These numbers are quoted from the dataset page. Replace them with the counts printed by
`ml/inspect_dataset.py` once the files are on disk.]*

**Extra class: ESC-50.** MIVIA has no "other loud sound" examples, so we added a fourth class,
`loud_not_crash`, from the ESC-50 dataset. We used 10 of its 50 classes: door knock, door
creak, glass breaking, fireworks, car horn, siren, clapping, dog, thunderstorm and can opening.
That is 400 clips of 5 seconds at 44.1 kHz (40 per class).

* Licence: Creative Commons Attribution-NonCommercial (CC BY-NC 3.0).
* Citation: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", Proceedings
  of the 23rd ACM International Conference on Multimedia, 2015.

No audio from either dataset is stored in our code repository.

## 3. Preprocessing

All steps are in `ml/prepare_data.py` and `ml/features.py`.

1. Every file is converted to 16 kHz mono.
2. The audio is cut into windows of 1 second with a hop of 0.5 seconds.
3. Labels: a window is `crash` or `skid` if more than 50 percent of it lies inside an annotated
   event. A window with no event at all is `background`. Windows that touch an event by 50
   percent or less are left out because their label is unclear.
4. Every window keeps the fold number of its source file, so cross validation never mixes
   windows of one recording between train and test.
5. ESC-50 clips become `loud_not_crash`. Its 5 folds are mapped into 4 groups and each original
   clip stays in one group. Windows that are almost silent (padding) are left out.
6. Each window becomes a **log Mel spectrogram**:

| setting | value |
|---|---|
| sample rate | 16 kHz |
| FFT size / frame length | 1024 samples, Hann window |
| hop | 256 samples |
| Mel bins | 64, from 50 Hz to 8000 Hz |
| spectrum | power, then log(x + 0.000001) |
| normalisation | per window: subtract the mean, divide by the standard deviation |
| result | 59 time frames x 64 Mel bins x 1 channel |

The spectrogram is computed with `tf.signal` in Python for training and with `tf.signal` in
TensorFlow.js in the browser. Both sides read the same Mel matrix from
`public/model/mel.json`, so the phone computes the same features the model was trained on.

*[Add the real table of windows per class per fold from `ml/results/window_counts.md`.]*

## 4. Model architecture

A small 2D CNN with 4 convolution blocks (`ml/train.py`, function `build_model`).

| layer | output shape | parameters |
|---|---|---|
| Input (log Mel spectrogram) | 59 x 64 x 1 | 0 |
| Conv2D 16 filters, 3x3 | 59 x 64 x 16 | 144 |
| Batch normalisation + ReLU | 59 x 64 x 16 | 64 |
| Max pooling 2x2 + Dropout 0.25 | 29 x 32 x 16 | 0 |
| Conv2D 32 filters, 3x3 | 29 x 32 x 32 | 4,608 |
| Batch normalisation + ReLU | 29 x 32 x 32 | 128 |
| Max pooling 2x2 + Dropout 0.25 | 14 x 16 x 32 | 0 |
| Conv2D 64 filters, 3x3 | 14 x 16 x 64 | 18,432 |
| Batch normalisation + ReLU | 14 x 16 x 64 | 256 |
| Max pooling 2x2 + Dropout 0.25 | 7 x 8 x 64 | 0 |
| Conv2D 96 filters, 3x3 | 7 x 8 x 96 | 55,296 |
| Batch normalisation + ReLU + Dropout 0.25 | 7 x 8 x 96 | 384 |
| Global average pooling | 96 | 0 |
| Dropout 0.3 | 96 | 0 |
| Dense, softmax | 4 | 388 |
| **Total** | | **79,700** (79,284 trainable) |

The model file is about 0.3 MB, small enough to download and run on a phone.

## 5. Training setup

* 4 fold cross validation on the dataset folds: train on 3 folds, test on the fourth, repeat.
* 15 percent of the training folds, split by recording, is held out as a validation set.
* Optimiser Adam, learning rate 0.001, halved when the validation loss stops improving.
  Batch size 64, at most 60 epochs, early stopping after 8 epochs without improvement.
* Class weights, because background windows are far more common than crash windows.
* Augmentation on training data only: random gain (plus or minus 6 dB), random time shift (up
  to 0.1 s), mixing with a background window from the training folds, and SpecAugment (one
  time mask and one frequency mask).
* Baseline for comparison: the old loudness rule of the app, with its threshold tuned on the
  training folds only and tested on the same test windows.
* The final model for the app is trained on all folds and converted to TensorFlow.js.
* Software: Python 3.10, TensorFlow 2.15.1, trained on a laptop CPU.

## 6. Results

**Not available yet.** *[Copy the tables from `ml/results/RESULTS.md` after training on MIVIA:
accuracy, per class precision, recall and F1, macro F1 and crash recall, each as mean and
standard deviation over the 4 folds.]*

## 7. Comparison with the loudness rule

**Not available yet.** *[Copy the "CNN vs loudness rule" table from `ml/results/RESULTS.md`,
including the false alarm rate on `loud_not_crash` for both methods.]*

What can be said now without numbers: the loudness rule looks at one value, how loud the sound
is, so it cannot tell a crash from any other loud sound. The CNN looks at how the energy is
spread over frequency and time.

## 8. Figures

Created by `ml/train.py` for every fold. *[These files appear in `ml/results/` after real
training. Figures inside `ml/results/synthetic_smoke_test/` are from fake data, do not use
them.]*

| figure | file |
|---|---|
| Confusion matrix, fold 1 to 4 | `ml/results/fold1_confusion.png` ... `ml/results/fold4_confusion.png` |
| Training and validation loss and accuracy, fold 1 to 4 | `ml/results/fold1_curves.png` ... `ml/results/fold4_curves.png` |

## 9. How the model fits into the app

1. The victim page already opens the microphone. The same stream is given to
   `src/ml/crashDetector.js`.
2. The detector keeps the last 1 second of audio, converts it to 16 kHz, builds the log Mel
   spectrogram and runs the CNN every 0.5 seconds with TensorFlow.js.
3. The Audio Pulse card shows the live class and confidence.
4. The 15 second cancel countdown starts only if
   * the crash probability is above 0.7 for 2 windows in a row, or
   * the crash probability is above 0.5 and there is a motion spike above the motion threshold.

   One prediction alone never starts it, and the user can always cancel.
5. If the model cannot be loaded, the screen shows "AI model offline" and the old loudness rule
   is used.
6. The alert sent to Firebase contains `cnnLabel`, `cnnConfidence` and `modelVersion`. The
   Command Center shows this next to the Gemini assessment.

Flow: microphone → 1 second window → log Mel spectrogram → CNN → crash probability → combine
with repeat check or motion → 15 second cancel countdown → Survival Mode and alert → Command
Center (Gemini triage).

`ml/check_export.py` confirms that the Keras model and the TensorFlow.js model give the same
probabilities for the same audio (difference below 0.001).

## 10. Limitations

* **Different microphones.** MIVIA was recorded with fixed roadside surveillance microphones.
  PulseTrack uses a phone microphone, often inside a car or a pocket. The sound that reaches
  the phone is different (cabin, distance, cheap microphone, automatic gain and noise
  suppression in the browser). This is called domain shift and the scores from MIVIA will not
  carry over fully. `ml/phone_test.py` measures this on our own recordings.
* **Mixed events.** MIVIA events are mixed into background sound by the dataset authors. Real
  crashes on a real road may sound different.
* **Other loud sounds.** We added 10 kinds of loud sounds from ESC-50, but the world has many
  more (construction, a dropped metal sheet, a burst tyre, a truck tailgate). Some will still
  be mistaken for a crash.
* **A crash sound is not proof of a crash.** A crash played from a video or a speaker sounds
  like a crash. This is why the app keeps the cancel countdown and combines audio with motion.
* **Small dataset.** 200 crashes and 200 skids is small for deep learning. We use 4 fold cross
  validation, augmentation and a small model to limit overfitting.
* **Resampling.** The browser converts audio to 16 kHz with its own simple filter, which is
  not identical to the one used in training.
* **Testing.** All demos use crash audio played from a speaker. No real crash is staged.
