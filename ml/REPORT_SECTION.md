# Report section: Team trained CNN for crash sound detection

> **Read this first.** The method, the code and the app integration described here are finished
> and tested. The **results tables are empty** because the crash and skid sounds (from
> Freesound) have not been downloaded yet: the Freesound API key file was empty. Fill sections
> 2.1, 6 and 7 from `ml/freesound_sources.csv`, `ml/results/window_counts.md` and
> `ml/results/RESULTS.md` after running `ml\run_pipeline.ps1`. Do not put any number from
> `ml/results/synthetic_smoke_test/` in the report.

## 1. Why this component was added

PulseTrack already uses AI in the Command Center: the Gemini API reads the telemetry and the
photo of an alert and gives a priority score from 1 to 10. Gemini is an **external API**. It
was trained by Google, not by us, and we only send it a request.

Our guide asked for an AI model that the team trains itself. So we added a small Convolutional
Neural Network (CNN) that listens to the phone microphone and classifies the sound as crash,
tire skid, background, or loud but not a crash. This CNN is **trained by the team** with our
own code (`ml/`), runs inside the browser on the phone, and needs no server.

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

## 2. Datasets

We combine three openly available sources. No audio from them is stored in our code repository.

| class | source | what we use |
|---|---|---|
| crash | Freesound | car crash recordings, CC0 or CC BY only |
| skid | Freesound | tire skid recordings, CC0 or CC BY only |
| background | UrbanSound8K | engine idling, street music, air conditioner, drilling, children playing, jackhammer |
| loud_not_crash | ESC-50 + UrbanSound8K | door knock, glass breaking, fireworks, car horn, siren, dog, clapping, can opening, thunderstorm (ESC-50); car horn, siren, gun shot (UrbanSound8K) |

### 2.1 Freesound (crash and skid)

Freesound (freesound.org) is an open website where people upload sound recordings. We searched
it through its official API for "car crash", "car crash skid", "car collision", "vehicle
crash", "tire skid" and a few similar words. We kept only sounds under 30 seconds with a
Creative Commons 0 or Creative Commons Attribution licence, removed obvious wrong results
(crash cymbals, music, computer crashes, game sounds) with keyword rules, and downloaded the
preview audio.

*[Fill in after download: number of crash and skid clips, total duration, number of
uploaders. Every clip's id, link, licence and author is in `ml/freesound_sources.csv`. For the
CC BY clips, list the authors and links in an appendix, as the licence requires.]*

* Licence: per clip, CC0 1.0 or CC BY 3.0 / 4.0.
* Citation: F. Font, G. Roma, X. Serra, "Freesound technical demo", Proceedings of the 21st
  ACM International Conference on Multimedia, 2013.

### 2.2 UrbanSound8K (background and some loud sounds)

8732 labelled sound excerpts of 4 seconds or less from 10 urban sound classes, all taken from
field recordings on Freesound. We used 6000 excerpts for background (1000 each of engine
idling, street music, air conditioner, drilling, children playing, jackhammer) and 1732 for
loud_not_crash (429 car horn, 929 siren, 374 gun shot). They come from 960 different original
recordings and have many sample rates (8 kHz to 192 kHz), which we convert to 16 kHz.

* Licence: Creative Commons Attribution-NonCommercial 4.0 (as listed on the Kaggle copy we
  downloaded, `chrisfilo/urbansound8k`). *[Confirm on the authors' site.]*
* Citation: J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound
  Research", 22nd ACM International Conference on Multimedia, Orlando, USA, 2014.

### 2.3 ESC-50 (loud sounds that are not crashes)

2000 environmental recordings of 5 seconds in 50 classes. We used 9 classes with 40 clips
each, 360 clips: door knock, glass breaking, fireworks, car horn, siren, dog, clapping, can
opening, thunderstorm.

* Licence: Creative Commons Attribution-NonCommercial 3.0.
* Citation: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", Proceedings
  of the 23rd ACM International Conference on Multimedia, 2015, pp. 1015-1018.

## 3. Preprocessing

All steps are in `ml/prepare_data.py` and `ml/features.py`.

1. Every file is converted to 16 kHz mono. Clips shorter than 1 second are padded with silence.
2. The audio is cut into windows of 1 second with a hop of 0.5 seconds. Every window gets the
   class of its clip.
3. For crash, skid and loud sounds, windows that are almost silent (quieter than 10 percent of
   the loudest window of the same clip) are left out, because an uploaded "car crash" file
   often has silence before and after the impact.
4. To keep the classes from being too unequal, each clip gives at most 2 background windows,
   3 loud_not_crash windows, or 20 crash or skid windows.
5. **4 folds without leakage.** All windows of one clip stay in the same fold. Clips that come
   from the same original Freesound recording, or (for crash and skid) from the same uploader,
   are kept together in one fold. The folds are filled so each gets about a quarter of every
   class. The script checks this and stops if any clip appears in two folds.
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

Windows per class per fold, for the parts already downloaded (printed by `prepare_data.py`):

| class | fold 1 | fold 2 | fold 3 | fold 4 | total |
|---|---|---|---|---|---|
| background | 2955 | 2955 | 2954 | 2954 | 11818 |
| crash | *[pending]* | | | | |
| skid | *[pending]* | | | | |
| loud_not_crash | 1307 | 1307 | 1307 | 1307 | 5228 |

*[Replace with the full table from `ml/results/window_counts.md` after the Freesound download;
the numbers above change slightly when crash and skid groups are added.]*

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

* 4 fold cross validation on the folds above: train on 3 folds, test on the fourth, repeat.
* 15 percent of the training folds, split by source, is held out as a validation set.
* Optimiser Adam, learning rate 0.001, halved when the validation loss stops improving.
  Batch size 64, at most 60 epochs, early stopping after 8 epochs without improvement.
* Class weights, because background windows are far more common than crash windows.
* Augmentation on training data only: random gain (plus or minus 6 dB), random time shift (up
  to 0.1 s), mixing with a background window from the training folds, and SpecAugment (one
  time mask and one frequency mask).
* Baseline for comparison: the old loudness rule of the app, copied in Python, with its
  threshold tuned on the training folds only and tested on the same test windows.
* The final model for the app is trained on all folds and converted to TensorFlow.js.
* Software: Python 3.10, TensorFlow 2.15.1, trained on a laptop CPU.

## 6. Results

**Not available yet.** *[Copy the tables from `ml/results/RESULTS.md`: accuracy, per class
precision, recall and F1, macro F1 and crash recall, each as mean and standard deviation over
the 4 folds.]*

## 7. Comparison with the loudness rule

**Not available yet.** *[Copy the "CNN vs loudness rule" table and the "Key point: false
alarms on loud sounds that are not crashes" section from `ml/results/RESULTS.md`.]*

The point to make: the loudness rule looks at one value, how loud the sound is, so a door
knock, fireworks or a siren can look exactly like a crash to it. The `loud_not_crash` class
exists to measure this. `RESULTS.md` reports the fraction of these windows on which each method
would raise a crash alarm, for each source separately.

## 8. Figures

Created by `ml/train.py` for every fold. *[These appear in `ml/results/` after real training.
Figures in `ml/results/synthetic_smoke_test/` are from fake data, do not use them.]*

| figure | file |
|---|---|
| Confusion matrix, folds 1 to 4 | `ml/results/fold1_confusion.png` ... `ml/results/fold4_confusion.png` |
| Training and validation loss and accuracy, folds 1 to 4 | `ml/results/fold1_curves.png` ... `ml/results/fold4_curves.png` |

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

Flow: microphone → 1 second window → log Mel spectrogram → CNN → crash probability → repeat
check or motion check → 15 second cancel countdown → Survival Mode and alert → Command Center
(Gemini triage).

`ml/check_export.py` confirms that the Keras model and the TensorFlow.js model give the same
probabilities for the same audio (difference below 0.001).

## 10. Limitations

* **Not phone recordings of real crashes.** The crash and skid clips are crowd sourced uploads:
  sound effects, film foley, test recordings, made with many different microphones. They are
  not recordings from a phone inside a car during a real accident. A phone in a pocket or on a
  dashboard hears something different (cabin, distance, a small microphone, automatic gain and
  noise suppression in the browser). This is called domain shift, so the scores from the test
  folds will not fully carry over. `ml/phone_test.py` measures this on our own recordings.
* **Several sources, no single controlled setup.** The four classes come from three datasets
  recorded in different ways. The model could partly learn *which dataset* a sound comes from
  (for example its recording quality) instead of *what* the sound is. Mixing background into
  the other classes during training reduces this but does not remove it.
* **Labels come from uploaders.** Freesound labels are the uploader's own title and tags. We
  removed obvious mistakes with keyword rules but did not listen to every clip.
* **Few crash examples.** Freesound has a limited number of openly licensed crash and skid
  recordings, far fewer than background sounds. Class weights and augmentation help but the
  crash class stays small.
* **Other loud sounds.** We test on 12 kinds of loud sounds, but the world has many more
  (construction, a dropped metal sheet, a burst tyre). Some will still be mistaken for a crash.
  drilling and jackhammer are counted as background, although they are loud.
* **A crash sound is not proof of a crash.** A crash played from a video or a speaker sounds
  like a crash. This is why the app keeps the cancel countdown and combines audio with motion.
* **Resampling.** The browser converts audio to 16 kHz with its own simple filter, which is
  not identical to the one used in training.
* **Testing.** All demos use crash audio played from a speaker. No real crash is staged.
