# Datasets for the crash sound CNN

Status on 2026-11-17: **the real MIVIA road dataset is NOT on this machine.**
Everything below is split into what was checked on disk and what was only read on a web page.

## 1. MIVIA Road Audio Events (the planned training data): NOT DOWNLOADED

Why: the MIVIA site needs a registered account, login and a reCAPTCHA before it gives the
download. That was not bypassed. During the run the live site `mivia.unisa.it` also reset every
connection, so the facts below come from the Internet Archive copy of the official page
(snapshot 2026-04-17 of
`https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/`).

What the official page says (read from the page, **not verified on the files**):

* 400 events for road surveillance: tire skidding and car crashes
* 4 folds of 100 events each, made for cross validation
* recorded with an Axis P8221 Audio Module and an Axis T83 omnidirectional microphone
* 32000 Hz, 16 bit PCM, WAV files
* each fold has audio files of about 1 minute, events are mixed over road background sound,
  each file has a different background
* table on the page: Background 2732 s; Car Crashes 200 events, 326.38 s; Tire skidding
  200 events, 522.5 s
* download: "In order to download the data set click here", which leads to the
  `/datasets-request/` page: "Select the dataset you want to download (only registered users)"

Not known yet, because the files have not been seen: the folder layout, file names, number of
wav files, how folds are stored, and the exact XML annotation format. A public third party
script (github.com/the6thsense/Mivia-Event-Data-road-Data-Extraction-script) reads
`<events><item>` entries with `CLASS_ID`, `PATHNAME`, `STARTSECOND`, `ENDSECOND`, takes the audio
from a `sounds` folder next to the XML, and treats `CLASS_ID` 3 as car crash. That is a hint
only. `ml/prepare_data.py` is written for that layout and stops with a clear message if the real
files look different.

Licence: the page does not print a licence. It says the lab makes the data available to
registered users and asks users to cite the papers below. Ask the guide if this is enough for
the project report; it is an academic research dataset.

Citations requested on the page:

* P. Foggia, N. Petkov, A. Saggese, N. Strisciuglio, M. Vento, "Audio surveillance of roads:
  a system for detecting anomalous sounds", IEEE Transactions on Intelligent Transportation
  Systems.
* P. Foggia, A. Saggese, N. Strisciuglio, M. Vento, "Cascade Classifiers Trained on
  Gammatonegrams for Reliably Detecting Audio Events", IEEE AVSS 2014.

Citation given in CLAUDE.md: Foggia, Petkov, Saggese, Strisciuglio, Vento, "Reliable detection
of audio events in highly noisy environments", Pattern Recognition Letters, 2015.

After you download it, run `ml\.venv\Scripts\python.exe ml\inspect_dataset.py` and paste the
real structure into this section.

## 2. Other open crash datasets that were checked and rejected

| candidate | why it was not used |
|---|---|
| Hugging Face `Titung/car-crash-audio-cc` (46 clips, CC BY 3.0, not gated) | Clips cut from YouTube sound effect videos found by searching "car crash audio". Labels come from the search words, not from annotation, one source is titled "(No audio)", there is no skid class. Fails "labels are real, not guessed" and it is YouTube material. |
| Hugging Face `Titung/car-crash-audio-clips`, `car-crash-audio-manual`, `vehicle-audio-dataset` | Gated (manual approval) and built from YouTube videos. |
| DeepCrashzam (GitHub `leosole/DeepCrashzam-Dataset`) | Only a script that downloads YouTube videos. |
| FSD50K (Zenodo 4060432) | Checked its `vocabulary.csv` (200 classes): no car crash and no skidding class. |
| Zenodo "Car crash dataset RUSSIA 2022-2023" | Dashcam video dataset, not labelled audio. |
| AudioSet "Skidding" / "Tire squeal" | Distributed as YouTube links only. |

Result: no open, properly labelled crash and skid audio was found, so Phase 0e applies.

## 3. SYNTHETIC_ROAD (stand in, code test only)

**SYNTHETIC. Not real crashes. Scores on it are not results and are not reported as results.**

Made by `ml/make_synthetic.py` with numpy: road like noise with passing vehicle swells as
background, "crashes" from decaying noise bursts, a low thump, ringing tones and clicks, "skids"
from wobbling tones with band noise. It copies only the general idea of MIVIA (one minute clips,
32 kHz, 16 bit, events over background, XML annotations, 4 folds).

Checked on disk with `ml/inspect_dataset.py` (2026-11-16):

* `ml/data/SYNTHETIC_ROAD/fold1..fold4/`, each with 6 XML files and `sounds/` with 6 WAV files
* 24 WAV files, 32000 Hz, mono, PCM_16, 1440.0 s in total
* 220 annotated events: 119 `car_crash` (CLASS_ID 3, 200.18 s), 101 `tire_skidding`
  (CLASS_ID 2, 280.17 s)

## 4. ESC-50 (real data, used for the 4th class `loud_not_crash`)

Source: https://github.com/karolpiczak/ESC-50 (archive zip of the repo), in `ml/data/ESC50`.

Checked on disk:

* 2000 WAV files in `ESC-50-master/audio`, 44100 Hz, mono, PCM_16, 5 s each (10000.0 s total)
* `meta/esc50.csv` columns: filename, fold, target, category, esc10, src_file, take
* 5 folds of 400 clips
* used classes (40 clips each, 400 clips, 80 per ESC fold, 313 different original Freesound
  clips): door_wood_knock, door_wood_creaks, glass_breaking, fireworks, car_horn, siren,
  clapping, dog, thunderstorm, can_opening

Licence (from the LICENSE file in the repo): the dataset as a whole is under Creative Commons
Attribution-NonCommercial (CC BY-NC 3.0). Fine for a college project, not for commercial use.
The audio is not committed to git.

Citation: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", Proceedings of
the 23rd ACM International Conference on Multimedia, 2015, pp. 1015-1018.

Fold mapping 5 to 4: ESC folds 1, 2, 3, 4 keep their number. Clips of ESC fold 5 go to group
`(Freesound clip id mod 4) + 1`. ESC-50 keeps all takes of one original clip in one fold, so an
original clip is never in two groups.

## 5. Windows (output of `ml/prepare_data.py`)

1 second windows, 0.5 second hop, 16 kHz mono, int16, saved to `ml/data/windows.npz`.

### Current file: SYNTHETIC road data + real ESC-50

Printed by the script on 2026-11-16. background / crash / skid rows are SYNTHETIC.

| class | fold 1 | fold 2 | fold 3 | fold 4 | total |
|---|---|---|---|---|---|
| background | 369 | 358 | 377 | 345 | 1449 |
| crash | 99 | 108 | 101 | 99 | 407 |
| skid | 136 | 138 | 130 | 156 | 560 |
| loud_not_crash | 622 | 626 | 706 | 658 | 2612 |
| all | 1226 | 1230 | 1314 | 1258 | 5028 |

Also printed: 440 road windows dropped because they touched an event by 50 percent or less,
988 ESC-50 windows dropped as near silent padding.

### MIVIA + ESC-50

Not available yet. Run `ml\.venv\Scripts\python.exe ml\prepare_data.py --dataset mivia` after
the download and copy the table from `ml/results/window_counts.md` here.
