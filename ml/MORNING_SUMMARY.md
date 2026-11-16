# Morning summary

**PR link: none. The pull request was NOT sent.** The GitHub CLI (`gh`) is not installed on
this laptop, so step 3a stopped there. Everything is committed on the local branch
`ml-crash-cnn`, nothing was pushed. The commands to send it are in section 5.

## 1. The short version

* **The real model was not trained.** MIVIA could not be downloaded: the site needs a
  registered account, login and a captcha (not bypassed), and the site was also down all night.
  No other open dataset with properly labelled crash and skid audio was found.
* **All the code is written and tested end to end** on SYNTHETIC road audio (noise and beeps
  made by a script) plus real ESC-50 clips.
* **There are no real numbers.** `ml/results/RESULTS.md` is empty on purpose. Do not use
  anything from `ml/results/synthetic_smoke_test/` in the report or in front of the professor.
* The app builds and the new code runs in Chrome. The model in `public/model/` is a
  **placeholder**. The app shows it as "TEST MODEL" and does not let it start alarms; the old
  loudness rule stays in charge until a real model is trained.

## 2. What is done

| phase | state |
|---|---|
| Setup (branch, Python 3.10 venv, packages, .gitignore) | done |
| 0 Data | MIVIA **not obtained**. ESC-50 downloaded. Synthetic stand in made. |
| 1 Inspect | done for ESC-50 and synthetic. MIVIA section of `DATASET.md` is from the web page only. |
| 2 Windows | `prepare_data.py` works (5028 windows from synthetic + ESC-50). Not tried on real MIVIA files. |
| 3 Features | done, shape test passes, `public/model/mel.json` written |
| 4 Train | code done and run on synthetic data. **Real training skipped.** "3 improvements" step not run. |
| 5 Export | done. Keras vs TensorFlow.js max difference 1.192e-07 (limit 1e-3). |
| 6 App | done. Build passes. Lint: same 22 errors + 3 warnings as before, none new. |
| 7 Phone test tool | done, tested on wav, mp3, m4a, webm |
| 8 Report text | done, results sections left empty |
| PR | **not sent**, `gh` missing |

Tested in headless Chrome with a fake microphone: live labels on the Audio Pulse card every
0.5 s, countdown after 2 crash windows in a row when the model is marked real, Cancel works,
"AI model offline" when the model file is blocked. **Not tested:** a real phone, a real
Firestore alert, the new box in the Command Center with a real alert.

More detail: `ml/PROGRESS.md` (what worked and failed), `ml/DECISIONS.md` (37 choices I made).

## 3. Data I must download

MIVIA Road Audio Events, by hand:

1. Open https://mivia.unisa.it/datasets/audio-analysis/mivia-road-audio-events-data-set/
   (if the site does not open, try later or from mobile data; it reset every connection during
   the night).
2. Click "Registrati" (Register) and make an account. Confirm by email if asked.
3. Log in, then open https://mivia.unisa.it/datasets-request/
4. In "Select the dataset you want to download" choose **MIVIA ROAD audio events data set** and
   download it. If there is no download after login, write to mivia@unisa.it from your college
   email and say it is for a final year project.
5. Read the terms shown on the site and note them in `ml/DATASET.md` (the page I could read
   shows no licence text, only the papers to cite).
6. Unzip so that the wav and xml files are somewhere under `ml\data\MIVIA_ROAD_DB1\`.

Then run, from `d:\Zaids Work\PulseTrack`:

```powershell
ml\.venv\Scripts\python.exe ml\inspect_dataset.py
ml\.venv\Scripts\python.exe ml\prepare_data.py --dataset mivia
ml\.venv\Scripts\python.exe ml\train.py
ml\.venv\Scripts\python.exe ml\export_tfjs.py
ml\.venv\Scripts\python.exe ml\check_export.py
npm run build
```

Things that may need a small fix at step 2, because I never saw the real files:

* **Fold folders.** The script looks for `fold1` .. `fold4` in the path. If the real folders
  are named differently, pass for example `--fold-regex "/([A-D])/"`. The script stops and
  shows the paths if it cannot find a fold.
* **Class ids.** If the XML has no class name, the script assumes id 3 = crash and id 2 = skid
  (from someone else's script on GitHub, not verified). Check the "EVENTS BY" lines printed by
  `inspect_dataset.py`. The script stops if it sees an id it does not know.

Training will probably take under an hour on this laptop CPU (the synthetic test run took about 20 minutes). `train.py` then writes the real
`ml/results/RESULTS.md` and figures, and the exported model is marked as real, so the CNN
trigger in the app turns on by itself. If macro F1 is weak, try for example
`ml\.venv\Scripts\python.exe ml\train.py --tag wide --width 24 --skip-final` and compare.
After that fill sections 2, 3, 6, 7 of `ml/REPORT_SECTION.md` and update the PR text.

## 4. Commands

Run the app over HTTPS (the dev server already uses a self signed certificate):

```powershell
cd "d:\Zaids Work\PulseTrack"
npm install
npm run dev
```

Vite prints a `Network: https://192.168.x.x:5173/` address. Open it on the phone (same Wi-Fi),
accept the certificate warning, press "Initialize Tracking Sensors" and allow mic and camera.
Note: there is no `.env` file in this folder. Firebase and Gemini need the `VITE_FIREBASE_...`
and `VITE_GEMINI_API_KEY` values in `.env` to send and show alerts.

Phone test (put recordings in one folder, names like `crash_01.m4a`, `loud_door.mp3`):

```powershell
ml\.venv\Scripts\python.exe ml\phone_test.py "C:\path\to\my_recordings"
```

**ffmpeg is needed** for mp3, m4a and webm (wav and flac work without it). It is already
installed on this laptop. With the placeholder model the output means nothing; use it after
real training.

Other checks:

```powershell
ml\.venv\Scripts\python.exe ml\features.py       # feature shape test
node ml\sim_detector.mjs                         # browser detector simulated in Node
```

If you ever rebuild the venv: Python 3.10, then install as described in `DECISIONS.md` points
4 to 6 (tensorflow 2.15.1, tensorflowjs with `--no-deps`).

## 5. Send the PR (ready, just not sent)

```powershell
winget install --id GitHub.cli
# close and reopen the terminal, then:
gh auth login
gh auth status
cd "d:\Zaids Work\PulseTrack"
gh api repos/Fayirrr/PulseTrack --jq .permissions.push
```

If the last command prints `true` (you can push to your friend's repo):

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

(After the fork command your fork is `origin` and your friend's repo is `upstream`; check with
`git remote -v` and push to the one that is your fork.)

It is a **draft** PR because training was skipped; the reason is at the top of
`ml/PR_BODY.md`. After real training, update the body with the results table and run
`gh pr ready`. Put the PR link at the top of this file.

## 6. Things to know before the demo

* Nothing here proves crash detection yet. Tell the professor honestly that the pipeline is
  built and the dataset access is pending, or train first.
* `PulseTrack_Claude_Context_Brief.pdf` was left out of git on purpose: it has the team's names
  and roll numbers and the repo is public. Add it yourself if you want it there.
* `package-lock.json` already had an uncommitted change before I started (vite `^8.3.5` to
  `^8.3.4`). It went into the commit together with the TensorFlow.js install.
* Suggestion, not done: the mic is opened with `audio: true`, so the browser applies echo
  cancellation, noise suppression and auto gain. These can soften a crash sound. Consider
  turning them off in `getUserMedia` and test on a phone.
* I checked git before every commit: no dataset files, no venv, no `.env`, no keys are staged
  or committed.
