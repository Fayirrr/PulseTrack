# Runs the whole ML pipeline after the data is in place.
# From the repo root:   powershell -ExecutionPolicy Bypass -File ml\run_pipeline.ps1
# Needs: freesound_key.txt with the Freesound API key (one line), UrbanSound8K and ESC-50 in ml\data.
# Stops at the first failing step.

$ErrorActionPreference = "Stop"
$py = "ml\.venv\Scripts\python.exe"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n===== $name =====" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
}

if (-not (Test-Path "ml\data\UrbanSound8K\UrbanSound8K.csv")) {
    Step "download UrbanSound8K" { & $py -I ml\kaggle_download.py chrisfilo/urbansound8k ml\data\UrbanSound8K }
}
Step "download Freesound crash and skid clips" { & $py -I ml\freesound_download.py }
Step "build labelled windows"                  { & $py -I ml\prepare_data.py }
Step "feature shape test + mel.json"           { & $py -I ml\features.py }
Step "train and evaluate (4 folds + final)"    { & $py -I ml\train.py }
Step "export to TensorFlow.js"                 { & $py -I ml\export_tfjs.py }
Step "check Keras vs TensorFlow.js"            { & $py -I ml\check_export.py }
Step "build the app"                           { npm run build }
Write-Host "`nDone. Results: ml\results\RESULTS.md" -ForegroundColor Green
