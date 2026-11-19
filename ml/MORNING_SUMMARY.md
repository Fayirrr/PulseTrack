# Morning summary

**PR link: https://github.com/Fayirrr/PulseTrack/pull/1** (opened from the fork
MohammedZaid-AI/PulseTrack, branch `ml-crash-cnn`, after the real model was trained).
The notes below were written before training; real numbers are in `ml/results/RESULTS.md`.

(Second run, 2026-11-18. MIVIA replaced by Freesound + UrbanSound8K + ESC-50.)

## 1. The short version

* **`freesound_key.txt` was empty (0 bytes) the whole night.** It was open in your editor, so
  maybe it was never saved. Without the key, no crash or skid audio could be downloaded.
* So **the real model was not trained** and there are **no real numbers**.
  `ml/results/RESULTS.md` says so and is empty on purpose.
* **UrbanSound8K is downloaded** (6 GB from Kaggle) and **ESC-50 is on disk**. Both are cut into
  17,046 windows in 4 leak free folds (exactly 25 percent of each class per fold). Real counts
  are in `ml/DATASET.md`.
* Every script is ready. **Once the key is saved, one command does the rest** (section 3).
* The app still builds and works with the placeholder model ("TEST MODEL", cannot start alarms).

## 2. What is done

| step | state |
|---|---|
| kaggle.json / freesound_key.txt in .gitignore | done, first commit of this run |
| UrbanSound8K | done. The kaggle CLI failed (certificate check, then held 6 GB in memory), replaced by `ml/kaggle_download.py` |
| Freesound crash / skid | **not done: key file empty.** `ml/freesound_download.py` written, filters tested offline |
| ESC-50 | done (already there from the first run) |
| Phase 1 labelled set + `DATASET.md` | done for UrbanSound8K + ESC-50, crash/skid pending |
| Phase 2 windows | done for those (17,046 windows), leakage check passed |
| Phase 3 features | unchanged from first run, shape test passes |
| Phase 4 training | code updated (highlighted loud_not_crash false alarms, per source table). **Real training skipped.** "3 improvements" step not run |
| Phase 5 export | unchanged, placeholder model, check passed (max diff 1.192e-07) |
| Phase 6 app | unchanged from first run. Build passes, no new lint errors |
| Phase 7 phone test tool | unchanged, works with wav, mp3, m4a, webm |
| Phase 8 report text | rewritten for the new datasets, results sections left empty |
| PR | **not sent**, `gh` missing |

Details: `ml/PROGRESS.md`, `ml/DECISIONS.md` (points 38 to 55 are from this run).

## 3. What you must do by hand

1. **Save the Freesound key.** Get it at https://freesound.org/apiv2/apply (log in, "Create new
   API credentials"; use the long "Client secret / API key" value). Paste it as the only line
   of `freesound_key.txt` in the repo root and **save** the file (Ctrl+S).
2. **Run the whole pipeline** from `d:\Zaids Work\PulseTrack`:

```powershell
powershell -ExecutionPolicy Bypass -File ml\run_pipeline.ps1
```

   It downloads the Freesound clips, builds the windows, trains (4 folds + final model),
   exports to TensorFlow.js, checks Keras vs TensorFlow.js, and builds the app. It stops at the
   first error. Training should fit in under 2 hours on this CPU (17,046 windows now, plus the crash and skid windows).

   The same steps one by one:

```powershell
ml\.venv\Scripts\python.exe ml\freesound_download.py
ml\.venv\Scripts\python.exe ml\prepare_data.py
ml\.venv\Scripts\python.exe ml\train.py
ml\.venv\Scripts\python.exe ml\export_tfjs.py
ml\.venv\Scripts\python.exe ml\check_export.py
npm run build
```

3. **Check what came back.** `freesound_download.py` prints how many crash and skid clips passed
   the licence and keyword filters; the task asks for at least 48 of each. If there are fewer,
   note the real number in `ml/DATASET.md`. Listening to a few files in `ml\data\freesound_crash`
   is a good idea: labels are the uploaders' own tags.
4. **If macro F1 is weak**, try at most 3 variants and keep the best, for example
   `ml\.venv\Scripts\python.exe ml\train.py --tag wide --width 24 --skip-final`.
   Each writes `ml\results\variant_<tag>\`. Then train the final model with the best settings.
5. **Fill the report**: sections 2.1, 3 (table), 6 and 7 of `ml/REPORT_SECTION.md`, and copy
   the CC BY authors from `ml/freesound_sources.csv` into an appendix.
6. **Commit** (`ml/freesound_sources.csv`, `ml/results/`, `public/model/` change; data does not):
   `git add -A` then `git status` and make sure no `ml/data`, `kaggle.json` or
   `freesound_key.txt` appear, then commit.

## 4. Commands

Run the app over HTTPS (the dev server already uses a self signed certificate):

```powershell
cd "d:\Zaids Work\PulseTrack"
npm install
npm run dev
```

Vite prints a `Network: https://192.168.x.x:5173/` address. Open it on the phone (same Wi-Fi),
accept the certificate warning, press "Initialize Tracking Sensors", allow mic and camera.
There is no `.env` file in this folder; Firebase and Gemini need the `VITE_FIREBASE_...` and
`VITE_GEMINI_API_KEY` values in `.env` to send and show alerts.

Phone test (a folder of your own recordings, names like `crash_01.m4a`, `loud_door.mp3`):

```powershell
ml\.venv\Scripts\python.exe ml\phone_test.py "C:\path\to\my_recordings"
```

wav and flac work directly; mp3, m4a, webm, ogg need ffmpeg, which is already installed on this
laptop. With the placeholder model the output means nothing; use it after real training.

## 5. Send the PR (ready, just not sent)

Best after step 3 above, so the PR has real results. To send it now as a draft:

```powershell
winget install --id GitHub.cli
# close and reopen the terminal, then:
gh auth login
cd "d:\Zaids Work\PulseTrack"
gh api repos/Fayirrr/PulseTrack --jq .permissions.push
```

If that prints `true`:

```powershell
git push -u origin ml-crash-cnn
gh pr create --repo Fayirrr/PulseTrack --base main --head ml-crash-cnn --draft --title "Add team trained CNN for crash sound detection" --body-file ml/PR_BODY.md
```

If it prints `false`:

```powershell
gh repo fork Fayirrr/PulseTrack --remote
git remote -v
git push -u origin ml-crash-cnn
gh pr create --repo Fayirrr/PulseTrack --base main --head YOUR_GITHUB_USERNAME:ml-crash-cnn --draft --title "Add team trained CNN for crash sound detection" --body-file ml/PR_BODY.md
```

(After the fork command your fork is `origin` and your friend's repo is `upstream`.)
After real training, put the results table from `RESULTS.md` into `ml/PR_BODY.md`, remove the
draft note at the top, update the PR (`gh pr edit --body-file ml/PR_BODY.md`) and run
`gh pr ready`. Put the PR link at the top of this file.

## 6. Things to know

* Do not show anything from `ml/results/synthetic_smoke_test/` as a result; it is fake audio.
* There are other key files in your Downloads folder; I did not open them.
* UrbanSound8K and ESC-50 are non commercial (CC BY-NC). Fine for the project, cite both.
* `PulseTrack_Claude_Context_Brief.pdf` is still left out of git (names and roll numbers,
  public repo).
* I checked git before committing: no dataset files, no venv, no `.env`, no `kaggle.json`,
  no `freesound_key.txt` are staged or committed.
