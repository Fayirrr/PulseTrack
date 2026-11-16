"""Phase 5 check, Python side: Keras and TF.js must give the same probabilities.

Run:  ml\\.venv\\Scripts\\python.exe ml\\check_export.py

1. picks 10 windows from ml/data/windows.npz (fixed seed, at least 2 per class when possible)
2. computes features with tf.signal (ml/features.py) and runs the Keras model
3. saves the windows (ml/export/check_windows.bin, int16) and the Keras probabilities
4. runs ml/check_export.mjs with Node, which computes features with tf.signal in TF.js
   (src/ml/features.js) and runs public/model
5. fails if any probability differs by more than 1e-3
"""
import json
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402

TOL = 1e-3


def main():
    d = np.load(C.WINDOWS_NPZ, allow_pickle=False)
    X, y = d["X"], d["y"]
    rng = np.random.default_rng(7)
    picks = []
    for c in range(len(C.CLASSES)):
        idx = np.where(y == c)[0]
        if len(idx):
            picks += list(rng.choice(idx, size=min(2, len(idx)), replace=False))
    rest = np.setdiff1d(np.arange(len(y)), picks)
    picks += list(rng.choice(rest, size=10 - len(picks), replace=False))
    picks = np.array(sorted(picks[:10]))
    xw = X[picks]

    model = tf.keras.models.load_model(os.path.join(C.MODELS_DIR, "crash_cnn.h5"), compile=False)
    keras_probs = model.predict(F.features_numpy(xw), verbose=0)

    os.makedirs(C.EXPORT_DIR, exist_ok=True)
    xw.astype("<i2").tofile(os.path.join(C.EXPORT_DIR, "check_windows.bin"))
    with open(os.path.join(C.EXPORT_DIR, "check_windows.json"), "w", encoding="utf-8") as f:
        json.dump({"indices": picks.tolist(), "labels": [C.CLASSES[int(v)] for v in y[picks]],
                   "kerasProbs": keras_probs.tolist()}, f)

    r = subprocess.run(["node", os.path.join(C.ML_DIR, "check_export.mjs")], cwd=C.REPO_DIR,
                       capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0 and not os.path.exists(os.path.join(C.EXPORT_DIR, "check_tfjs_probs.json")):
        print(r.stderr.strip())
        return 1
    with open(os.path.join(C.EXPORT_DIR, "check_tfjs_probs.json"), encoding="utf-8") as f:
        tfjs_probs = np.array(json.load(f)["tfjsProbs"])
    diff = np.abs(tfjs_probs - keras_probs)
    print("\nwindow  true label        keras argmax (p)          tfjs argmax (p)           max diff")
    for i in range(len(picks)):
        ka, ta = int(keras_probs[i].argmax()), int(tfjs_probs[i].argmax())
        print(f"{picks[i]:6d}  {C.CLASSES[int(y[picks[i]])]:16s}  {C.CLASSES[ka]:16s} {keras_probs[i, ka]:.4f}   "
              f"{C.CLASSES[ta]:16s} {tfjs_probs[i, ta]:.4f}   {diff[i].max():.2e}")
    print(f"\noverall max difference: {diff.max():.3e}  (tolerance {TOL})")
    ok = diff.max() <= TOL and r.returncode == 0
    print("EXPORT CHECK PASSED" if ok else "EXPORT CHECK FAILED")
    if r.returncode != 0:
        print(r.stderr.strip())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
