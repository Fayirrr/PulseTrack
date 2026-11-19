"""Phase 5: convert the final Keras model to TensorFlow.js layers format in public/model/.

Run:  ml\\.venv\\Scripts\\python.exe ml\\export_tfjs.py

Writes public/model/model.json, public/model/group1-shard1of1.bin and
public/model/model_meta.json (version, classes, synthetic flag). mel.json is written by features.py.

The tensorflowjs pip package imports tensorflow_decision_forests, which has no Windows build.
The Keras to layers converter does not use it, so an empty stand in module is registered before
the import. If the tensorflowjs converter still fails, a small writer in this file produces the
same layers format (model topology json + one float32 weights file).
"""
import json
import os
import shutil
import sys
import types

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402

try:
    import keras
    KERAS_VERSION = keras.__version__
except Exception:  # noqa: BLE001
    KERAS_VERSION = tf.__version__


def export_with_tensorflowjs(model, out_dir):
    # Stand ins for optional parts that are not needed for a Keras model and are not installed here:
    # tensorflow_decision_forests (no Windows build) and the JAX converter (needs jax + flax).
    try:
        __import__("tensorflow_decision_forests")
    except Exception:
        sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")
    try:
        __import__("jax")
    except Exception:
        fake = types.ModuleType("tensorflowjs.converters.jax_conversion")
        fake.convert_jax = None
        sys.modules["tensorflowjs.converters.jax_conversion"] = fake
    import tensorflowjs as tfjs
    tfjs.converters.save_keras_model(model, out_dir)
    return f"tensorflowjs {tfjs.__version__} save_keras_model"


def export_manual(model, out_dir):
    """Write the TF.js layers format by hand: topology from model.to_json(), weights as float32."""
    os.makedirs(out_dir, exist_ok=True)
    entries, blobs = [], []
    for layer in model.layers:
        for w, value in zip(layer.weights, layer.get_weights()):
            value = value.astype(np.float32)
            entries.append({"name": w.name.split(":")[0], "shape": list(value.shape), "dtype": "float32"})
            blobs.append(value.tobytes())
    with open(os.path.join(out_dir, "group1-shard1of1.bin"), "wb") as f:
        f.write(b"".join(blobs))
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{KERAS_VERSION}",
        "convertedBy": "ml/export_tfjs.py manual writer",
        "modelTopology": {"keras_version": KERAS_VERSION, "backend": "tensorflow",
                          "model_config": json.loads(model.to_json())},
        "weightsManifest": [{"paths": ["group1-shard1of1.bin"], "weights": entries}],
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model_json, f)
    return "manual layers format writer"


def main():
    src = os.path.join(C.MODELS_DIR, "crash_cnn.h5")
    if not os.path.exists(src):
        raise SystemExit(f"{src} not found. Run ml/train.py first.")
    model = tf.keras.models.load_model(src, compile=False)
    out = C.PUBLIC_MODEL_DIR
    os.makedirs(out, exist_ok=True)
    for old in os.listdir(out):
        if old == "model.json" or old.endswith(".bin"):
            os.remove(os.path.join(out, old))
    try:
        how = export_with_tensorflowjs(model, out)
    except Exception as e:  # noqa: BLE001
        print(f"tensorflowjs converter failed ({type(e).__name__}: {e}); using the manual writer")
        how = export_manual(model, out)
    shutil.copyfile(os.path.join(C.MODELS_DIR, "model_meta.json"), os.path.join(out, "model_meta.json"))
    F.export_mel_json()
    files = {f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out))}
    print("converted with:", how)
    print("public/model files (bytes):", files)


if __name__ == "__main__":
    main()
