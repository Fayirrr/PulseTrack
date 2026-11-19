# Datasets for the crash sound CNN

Updated 2026-11-18. MIVIA was dropped (it could not be reached or registered). The plan is now
three open sources: Freesound for crash and skid, UrbanSound8K for background, ESC-50 and
UrbanSound8K for loud sounds that are not crashes.

**Status: crash and skid audio is NOT downloaded yet.** `freesound_key.txt` in the repo root was
empty (0 bytes) during the whole run, so the Freesound API could not be called. Every number
below was printed by the scripts from the files on disk.

## Classes and where they come from

| class | source | status |
|---|---|---|
| crash | Freesound, search "car crash", "car crash skid", "car collision", "vehicle crash" (+ "car accident", "car impact") | **0 clips, not downloaded (no API key)** |
| skid | Freesound, search "tire skid" (+ "tyre skid", "tire screech", "car skid", "brake skid", "tires squeal") | **0 clips, not downloaded (no API key)** |
| background | UrbanSound8K: engine_idling, street_music, air_conditioner, drilling, children_playing, jackhammer | 6000 clips, done |
| loud_not_crash | ESC-50: door_wood_knock, glass_breaking, fireworks, car_horn, siren, dog, clapping, can_opening, thunderstorm. UrbanSound8K: car_horn, siren, gun_shot | 360 + 1732 clips, done |

## 1. Freesound (crash, skid): NOT DOWNLOADED

* Script: `ml/freesound_download.py` (ready, filters tested offline, never run against the API).
* Rules in the script: only Creative Commons 0 or Attribution (CC BY 3.0 / 4.0), duration
  0.5 to 30 seconds, up to 150 per class, HQ OGG preview converted to 16 bit wav with ffmpeg.
* Keyword rules drop obvious wrong hits (crash cymbals, drums, music, computer crashes, game
  and cartoon sounds, bicycles, voices). A crash must mention crash/collision/accident/impact
  and a vehicle word; a skid must mention skid/screech/squeal/brake and must not mention a crash.
  A sound found by both crash and skid searches is kept as crash only.
* For every file it writes id, url, licence, author, pack, duration, original sample rate,
  search query and tags to `ml/data/freesound_sounds.csv` and to `ml/freesound_sources.csv`
  (committed, for citation).
* Labels on Freesound are the uploader's own title and tags. They are not checked by listening.

Licence: each clip has its own licence (CC0 or CC BY). CC BY clips need the author's name and
the clip link in the report; `ml/freesound_sources.csv` has both.

Citation for the platform: F. Font, G. Roma, X. Serra, "Freesound technical demo", Proceedings
of the 21st ACM International Conference on Multimedia, 2013.

## 2. UrbanSound8K (background, part of loud_not_crash): done

* Downloaded from Kaggle `chrisfilo/urbansound8k` with `ml/kaggle_download.py` (zip of
  6,026,232,524 bytes, streamed to disk, then unzipped into `ml/data/UrbanSound8K`).
* Layout on disk: `UrbanSound8K.csv` and `fold1` ... `fold10`.
* `UrbanSound8K.csv`: 8732 slices, columns slice_file_name, fsID, start, end, salience, fold,
  classID, class. Clips per class: air_conditioner 1000, car_horn 429, children_playing 1000,
  dog_bark 1000, drilling 1000, engine_idling 1000, gun_shot 374, jackhammer 1000, siren 929,
  street_music 1000. dog_bark is not used.
* The 7732 used slices are all readable. They come from 960 different Freesound recordings
  (fsID). Formats vary: mostly 44.1 or 48 kHz, also 96 kHz and some 8 to 24 kHz, mono and
  stereo, 16 bit, 24 bit and float. Every one is converted to 16 kHz mono.
* Licence: the Kaggle dataset page (read through the Kaggle API) gives "Attribution-NonCommercial
  4.0 International (CC BY-NC 4.0)": non commercial use with attribution, fine for a college
  project. The authors' own site (urbansounddataset.weebly.com) was not checked in this run;
  confirm the licence there before submitting the report.
* Citation: J. Salamon, C. Jacoby, J. P. Bello, "A Dataset and Taxonomy for Urban Sound
  Research", 22nd ACM International Conference on Multimedia, Orlando, USA, 2014.

## 3. ESC-50 (part of loud_not_crash): done

* From https://github.com/karolpiczak/ESC-50 (archive zip of the repo), in `ml/data/ESC50`.
* 2000 wav files, 44100 Hz, mono, PCM_16, 5 s each. `meta/esc50.csv` has filename, fold,
  target, category, esc10, src_file, take.
* Used: 9 classes x 40 clips = 360 clips.
* Licence: Creative Commons Attribution-NonCommercial (CC BY-NC 3.0), from the LICENSE file.
* Citation: K. J. Piczak, "ESC: Dataset for Environmental Sound Classification", Proceedings
  of the 23rd ACM International Conference on Multimedia, 2015, pp. 1015-1018.

## 4. Folds (no leakage)

* Folds are built by source, never by window. All windows of one clip are in one fold.
* Clips are joined into one group when they share a Freesound id (UrbanSound8K slices of one
  recording; ESC-50 takes of one clip; the same Freesound recording appearing in two datasets)
  or, for the Freesound crash and skid clips, the same uploader. A group goes to one fold.
* Groups are placed biggest first into the fold that is most behind its share of each class.
* `prepare_data.py` checks after the split that no clip and no group is in two folds. It
  printed "leakage check passed" on the current file.
* The original UrbanSound8K 10 folds and ESC-50 5 folds are not used, our 4 folds replace them.

## 5. Windows (output of `ml/prepare_data.py`, file `ml/data/windows.npz`)

1 second, 0.5 second hop, 16 kHz mono, int16. Per clip limits: background 2 windows (random,
fixed seed), loud_not_crash 3, crash 20, skid 20 (the loudest ones). Windows quieter than
10 percent of the loudest window of the same clip are dropped for crash, skid and
loud_not_crash (silent padding). Clips shorter than 1 second are padded with silence.

### Current file (2026-11-18): UrbanSound8K + ESC-50 only, no crash and no skid

Printed by the script (run with `--allow-no-crash`):

Per source:

| source | clips read | total seconds | windows kept |
|---|---|---|---|
| esc50_can_opening | 40 | 200.0 | 107 |
| esc50_car_horn | 40 | 200.0 | 114 |
| esc50_clapping | 40 | 200.0 | 120 |
| esc50_dog | 40 | 200.0 | 112 |
| esc50_door_wood_knock | 40 | 200.0 | 107 |
| esc50_fireworks | 40 | 200.0 | 119 |
| esc50_glass_breaking | 40 | 200.0 | 85 |
| esc50_siren | 40 | 200.0 | 120 |
| esc50_thunderstorm | 40 | 200.0 | 120 |
| us8k_air_conditioner | 1000 | 3994.9 | 2000 |
| us8k_car_horn | 429 | 1053.7 | 913 |
| us8k_children_playing | 1000 | 3962.1 | 1996 |
| us8k_drilling | 1000 | 3546.3 | 1904 |
| us8k_engine_idling | 1000 | 3936.0 | 1990 |
| us8k_gun_shot | 374 | 616.2 | 569 |
| us8k_jackhammer | 1000 | 3610.7 | 1928 |
| us8k_siren | 929 | 3632.7 | 2742 |
| us8k_street_music | 1000 | 4000.0 | 2000 |

Windows per class per fold:

| class | fold 1 | fold 2 | fold 3 | fold 4 | total |
|---|---|---|---|---|---|
| background | 2955 | 2955 | 2954 | 2954 | 11818 |
| crash | 0 | 0 | 0 | 0 | 0 |
| skid | 0 | 0 | 0 | 0 | 0 |
| loud_not_crash | 1307 | 1307 | 1307 | 1307 | 5228 |
| all | 4262 | 4262 | 4261 | 4261 | 17046 |

Balance achieved: 25 / 25 / 25 / 25 percent per class. 1209 source groups. 1084 windows
dropped as near silent, 5840 dropped by the per clip limit.

### With Freesound crash and skid

Not available yet. After the download, `prepare_data.py` writes the full tables to
`ml/results/window_counts.md`; copy them here.

## Old plan: MIVIA (dropped)

The MIVIA Road Audio Events page needs a registered account, login and reCAPTCHA, and the site
reset connections. It is no longer used. `ml/make_synthetic.py` (SYNTHETIC test audio made from
noise and tones) is kept only because the placeholder model in `public/model/` was trained on
it; it is not part of the dataset.
