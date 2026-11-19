"""Phase 4: train and evaluate the crash sound CNN with 4 fold cross validation.

Run:
    ml\\.venv\\Scripts\\python.exe ml\\train.py                 full run (4 folds + final model)
    ml\\.venv\\Scripts\\python.exe ml\\train.py --smoke         quick code test, 3 epochs
    ml\\.venv\\Scripts\\python.exe ml\\train.py --tag wide --width 24   one "improvement" variant

Outputs
* real data:      ml/results/            (RESULTS.md, metrics json, png figures)
* SYNTHETIC data: ml/results/synthetic_smoke_test/  and RESULTS.md only says that real training
                  was skipped. Synthetic scores are never written as results.
* final model:    ml/models/crash_cnn.keras, crash_cnn.h5, model_meta.json

Baseline: the old rule of VictimUI.jsx (average loudness above a threshold means alert), scored
on the same test windows with crash as the positive class. Its threshold is tuned on the train
folds only.
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import features as F  # noqa: E402

SEED = 42


# --------------------------------------------------------------------------- model
def build_model(n_classes, width=16, dropout=0.25):
    """Small 2D CNN: 4 conv blocks, batch norm, dropout, global average pooling, softmax."""
    L = tf.keras.layers
    inp = tf.keras.Input(shape=(C.N_FRAMES, C.N_MELS, 1), name="log_mel")
    x = inp
    for i, mult in enumerate([1, 2, 4, 6]):
        x = L.Conv2D(width * mult, 3, padding="same", use_bias=False, name=f"conv{i + 1}")(x)
        x = L.BatchNormalization(name=f"bn{i + 1}")(x)
        x = L.ReLU(name=f"relu{i + 1}")(x)
        if i < 3:
            x = L.MaxPooling2D(2, name=f"pool{i + 1}")(x)
        x = L.Dropout(dropout, name=f"drop{i + 1}")(x)
    x = L.GlobalAveragePooling2D(name="gap")(x)
    x = L.Dropout(0.3, name="drop_head")(x)
    out = L.Dense(n_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inp, out, name="pulsetrack_crash_cnn")


# --------------------------------------------------------------------------- augmentation
def make_train_dataset(X, y, bg_pool, n_classes, batch, specaug=True):
    """Augmentation happens on the waveform (gain, time shift, background mix), then features,
    then SpecAugment (one time mask and one frequency mask)."""
    bg_const = tf.constant(bg_pool, dtype=tf.int16) if len(bg_pool) else None

    def augment(xb, yb):
        x = F.int16_to_float(xb)
        n = tf.shape(x)[0]
        # time shift up to 0.1 s (circular)
        shift = tf.random.uniform([], -1600, 1601, dtype=tf.int32)
        x = tf.roll(x, shift, axis=1)
        # mix with a background window from the train folds, 5 to 20 dB below the sample
        if bg_const is not None:
            idx = tf.random.uniform([n], 0, tf.shape(bg_const)[0], dtype=tf.int32)
            bg = F.int16_to_float(tf.gather(bg_const, idx))
            rms_x = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-12)
            rms_b = tf.sqrt(tf.reduce_mean(tf.square(bg), axis=1, keepdims=True) + 1e-12)
            snr = tf.random.uniform([n, 1], 5.0, 20.0)
            use = tf.cast(tf.random.uniform([n, 1]) < 0.5, tf.float32)
            x = x + use * bg * (rms_x / rms_b) * tf.pow(10.0, -snr / 20.0)
        # random gain, plus or minus 6 dB
        gain = tf.pow(10.0, tf.random.uniform([n, 1], -6.0, 6.0) / 20.0)
        x = tf.clip_by_value(x * gain, -1.0, 1.0)
        feat = F.log_mel(x)
        if specaug:
            t = tf.random.uniform([n, 1, 1, 1], 0, 9, dtype=tf.int32)
            t0 = tf.random.uniform([n, 1, 1, 1], 0, C.N_FRAMES - 8, dtype=tf.int32)
            tr = tf.reshape(tf.range(C.N_FRAMES), [1, -1, 1, 1])
            f = tf.random.uniform([n, 1, 1, 1], 0, 9, dtype=tf.int32)
            f0 = tf.random.uniform([n, 1, 1, 1], 0, C.N_MELS - 8, dtype=tf.int32)
            fr = tf.reshape(tf.range(C.N_MELS), [1, 1, -1, 1])
            keep = tf.logical_not(tf.logical_or(tf.logical_and(tr >= t0, tr < t0 + t),
                                                tf.logical_and(fr >= f0, fr < f0 + f)))
            feat = feat * tf.cast(keep, tf.float32)   # 0 is the mean after normalisation
        return feat, yb

    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.int32)))
    ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True).batch(batch)
    return ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- loudness baseline
def loudness_score(x_int16):
    """Offline copy of the old trigger in VictimUI.jsx.

    The app does: AnalyserNode (fftSize 2048, Blackman window, minDecibels -100, maxDecibels -30),
    getByteFrequencyData, average of all 1024 bins, score = avg / 255 * 120, alert if score > 85.
    Here the same maths runs on each 2048 sample frame of the window and the loudest frame is the
    window score. The analyser's time smoothing (0.8) and the phone mic gain are not modelled.
    """
    x = x_int16.astype(np.float32) / 32768.0
    win = np.blackman(2048).astype(np.float32)
    scores = np.zeros(len(x), np.float32)
    starts = range(0, C.WINDOW_SAMPLES - 2048 + 1, 1024)
    for i in range(0, len(x), 1024):
        xb = x[i:i + 1024]
        best = np.zeros(len(xb), np.float32)
        for s in starts:
            mag = np.abs(np.fft.rfft(xb[:, s:s + 2048] * win, axis=1))[:, :1024] / 2048.0
            db = 20.0 * np.log10(mag + 1e-12)
            byte = np.clip(np.floor(255.0 * (db + 100.0) / 70.0), 0, 255)
            best = np.maximum(best, byte.mean(axis=1) / 255.0 * 120.0)
        scores[i:i + 1024] = best
    return scores


def tune_threshold(scores, is_crash):
    """Best F1 threshold on the train folds."""
    cands = np.unique(np.quantile(scores, np.linspace(0, 1, 400)))
    best_t, best_f1 = float(cands[0]), -1.0
    for t in cands:
        f1 = f1_score(is_crash, scores > t, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def binary_report(is_crash, pred, is_loud):
    p, r, f1, _ = precision_recall_fscore_support(is_crash, pred, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(is_crash, pred)),
        "crash_precision": float(p), "crash_recall": float(r), "crash_f1": float(f1),
        "false_alarm_rate_loud_not_crash": float(pred[is_loud].mean()) if is_loud.any() else None,
        "false_alarm_rate_all_non_crash": float(pred[~is_crash].mean()) if (~is_crash).any() else None,
    }


def balanced_weights(labels, n_classes):
    """sklearn 'balanced' weights; classes with no samples get 1.0 because Keras wants every class."""
    present = np.unique(labels)
    cw = compute_class_weight("balanced", classes=present, y=labels)
    weights = {c: 1.0 for c in range(n_classes)}
    weights.update({int(c): float(w) for c, w in zip(present, cw)})
    return weights


# --------------------------------------------------------------------------- plots
def plot_confusion(cm, names, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curves(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="validation")
    ax[0].set_title("loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="validation")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# --------------------------------------------------------------------------- main
def mean_std(vals):
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)


def fmt(ms):
    return "n/a" if ms[0] is None else f"{ms[0]:.3f} ± {ms[1]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-specaug", action="store_true")
    ap.add_argument("--tag", default="base", help="name of this run (used for improvement variants)")
    ap.add_argument("--smoke", action="store_true", help="3 epochs, only to test the code")
    ap.add_argument("--skip-final", action="store_true")
    ap.add_argument("--data", default=C.WINDOWS_NPZ, help="windows file (default ml/data/windows.npz)")
    ap.add_argument("--out", default=None, help="write results here instead of ml/results (for code tests)")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 3

    tf.keras.utils.set_random_seed(SEED)
    d = np.load(args.data, allow_pickle=False)
    X, y, fold, group = d["X"], d["y"].astype(np.int64), d["fold"], d["group"]
    meta = json.loads(str(d["meta"]))
    synthetic = bool(meta["synthetic"])
    present = sorted(set(y.tolist()))
    names = C.CLASSES
    n_classes = len(names)

    out_dir = os.path.join(C.RESULTS_DIR, "synthetic_smoke_test") if synthetic else C.RESULTS_DIR
    if args.out:
        out_dir = args.out
        args.skip_final = True  # a code test never replaces the saved model
    if args.tag != "base":
        out_dir = os.path.join(out_dir, f"variant_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    banner = "SYNTHETIC DATA, CODE TEST ONLY, NOT RESULTS" if synthetic else "real data"
    print(f"data: {meta['road_dataset']} ({banner}), windows {X.shape}, classes present {[names[i] for i in present]}")
    print("GPU:", tf.config.list_physical_devices("GPU") or "none, running on CPU")

    model = build_model(n_classes, args.width, args.dropout)
    n_params = model.count_params()
    print(f"model parameters: {n_params}")
    assert n_params < 200_000, "model must stay under 200k parameters"
    arch_lines = []
    model.summary(print_fn=arch_lines.append)
    with open(os.path.join(out_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(arch_lines))

    print("computing features and loudness scores for all windows ...")
    feats = F.features_numpy(X)
    loud = loudness_score(X)

    src_all = d["source"] if "source" in d.files else np.array(["unknown"] * len(y))
    cnn_crash_pred = np.zeros(len(y), bool)    # out of fold predictions, filled fold by fold
    loud_crash_pred = np.zeros(len(y), bool)
    fold_metrics, best_epochs = [], []
    for k in (1, 2, 3, 4):
        test = fold == k
        trainval = np.where(~test)[0]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED + k)
        tr_rel, va_rel = next(gss.split(trainval, y[trainval], group[trainval]))
        tr, va = trainval[tr_rel], trainval[va_rel]
        class_weight = balanced_weights(y[tr], n_classes)
        bg = X[tr][y[tr] == names.index("background")]
        if len(bg) > 2000:
            bg = bg[np.random.default_rng(SEED).choice(len(bg), 2000, replace=False)]

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + k)
        model = build_model(n_classes, args.width, args.dropout)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        ds = make_train_dataset(X[tr], y[tr], bg, n_classes, args.batch, not args.no_specaug)
        cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)]
        h = model.fit(ds, validation_data=(feats[va], y[va]), epochs=args.epochs,
                      class_weight=class_weight, callbacks=cbs, verbose=2)
        hist = {k2: [float(v) for v in vs] for k2, vs in h.history.items()}
        best_epochs.append(int(np.argmin(hist["val_loss"])) + 1)

        probs = model.predict(feats[test], batch_size=256, verbose=0)
        pred = probs.argmax(axis=1)
        yt = y[test]
        labels = list(range(n_classes))
        p, r, f1, sup = precision_recall_fscore_support(yt, pred, labels=labels, zero_division=0)
        cm = confusion_matrix(yt, pred, labels=labels)
        present_test = [i for i in labels if sup[i] > 0]
        is_crash = yt == C.CRASH
        is_loud = yt == names.index("loud_not_crash")

        thr = tune_threshold(loud[trainval], y[trainval] == C.CRASH)
        cnn_crash_pred[test] = pred == C.CRASH
        loud_crash_pred[test] = loud[test] > thr
        m = {
            "fold": k, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(test.sum()),
            "epochs_run": len(hist["loss"]), "best_epoch": best_epochs[-1],
            "cnn": {
                "accuracy": float(accuracy_score(yt, pred)),
                "macro_f1": float(np.mean([f1[i] for i in present_test])),
                "crash_recall": float(r[C.CRASH]),
                "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                         "f1": float(f1[i]), "support": int(sup[i])} for i in labels},
                "confusion_matrix": cm.tolist(),
                "crash_vs_rest_argmax": binary_report(is_crash, pred == C.CRASH, is_loud),
                "crash_vs_rest_p_above_0.7": binary_report(is_crash, probs[:, C.CRASH] > 0.7, is_loud),
            },
            "loudness_rule": {
                "tuned_threshold": thr,
                "tuned": binary_report(is_crash, loud[test] > thr, is_loud),
                "app_default_threshold_85": binary_report(is_crash, loud[test] > 85.0, is_loud),
            },
        }
        fold_metrics.append(m)
        with open(os.path.join(out_dir, f"fold{k}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({**m, "history": hist}, f, indent=1)
        suffix = " (SYNTHETIC, code test)" if synthetic else ""
        plot_confusion(cm, names, os.path.join(out_dir, f"fold{k}_confusion.png"), f"Fold {k} confusion matrix{suffix}")
        plot_curves(hist, os.path.join(out_dir, f"fold{k}_curves.png"), f"Fold {k} training curves{suffix}")
        print(f"[fold {k}] CNN acc {m['cnn']['accuracy']:.4f} macroF1 {m['cnn']['macro_f1']:.4f} "
              f"crash recall {m['cnn']['crash_recall']:.4f} | loudness rule (thr {thr:.1f}) "
              f"crash F1 {m['loudness_rule']['tuned']['crash_f1']:.4f}")

    # ---------------- summary over folds
    def col(fn):
        return mean_std([fn(m) for m in fold_metrics])

    rows = [
        ("accuracy (4 classes)", col(lambda m: m["cnn"]["accuracy"]), None),
        ("macro F1 (4 classes)", col(lambda m: m["cnn"]["macro_f1"]), None),
        ("crash vs rest accuracy", col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["accuracy"]),
         col(lambda m: m["loudness_rule"]["tuned"]["accuracy"])),
        ("crash precision", col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["crash_precision"]),
         col(lambda m: m["loudness_rule"]["tuned"]["crash_precision"])),
        ("crash recall", col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["crash_recall"]),
         col(lambda m: m["loudness_rule"]["tuned"]["crash_recall"])),
        ("crash F1", col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["crash_f1"]),
         col(lambda m: m["loudness_rule"]["tuned"]["crash_f1"])),
        ("false alarm rate on loud_not_crash",
         col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["false_alarm_rate_loud_not_crash"]),
         col(lambda m: m["loudness_rule"]["tuned"]["false_alarm_rate_loud_not_crash"])),
        ("false alarm rate on all non crash windows",
         col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["false_alarm_rate_all_non_crash"]),
         col(lambda m: m["loudness_rule"]["tuned"]["false_alarm_rate_all_non_crash"])),
    ]
    md = []
    title = "SMOKE TEST ON SYNTHETIC DATA. THESE NUMBERS ARE NOT RESULTS." if synthetic else "Results: crash sound CNN vs loudness rule"
    md.append(f"# {title}\n")
    if synthetic:
        md.append("The road audio here was made by `ml/make_synthetic.py` from noise and sine tones. "
                  "This file only proves that the training and evaluation code runs. "
                  "Do not copy these numbers into the report.\n")
    md.append(f"Run tag: `{args.tag}`. Road data: `{meta['road_dataset']}`. Windows: {len(X)}. "
              f"Model parameters: {n_params}. Max epochs: {args.epochs}. Date: {datetime.date.today().isoformat()}.\n")
    md.append("All numbers below were printed by `ml/train.py`. Mean ± standard deviation over the 4 test folds.\n")
    md.append("## CNN vs loudness rule\n")
    md.append("| metric | CNN (argmax) | loudness rule (threshold tuned on train folds) |\n|---|---|---|")
    for name, a, b in rows:
        md.append(f"| {name} | {fmt(a)} | {fmt(b) if b is not None else 'not applicable'} |")
    fa_cnn = col(lambda m: m["cnn"]["crash_vs_rest_argmax"]["false_alarm_rate_loud_not_crash"])
    fa_loud = col(lambda m: m["loudness_rule"]["tuned"]["false_alarm_rate_loud_not_crash"])
    md.append("\n### Key point: false alarms on loud sounds that are not crashes\n")
    md.append(f"> On `loud_not_crash` test windows (door knocks, fireworks, glass breaking, horns, sirens, "
              f"gun shots and more) the **loudness rule raised a crash alarm on {fmt(fa_loud)}** of windows, "
              f"the **CNN on {fmt(fa_cnn)}** (fraction of windows, mean ± std over the 4 folds).\n")
    md.append("\nFalse alarm rate per loud_not_crash source (all 4 test folds pooled):\n")
    md.append("| source | windows | CNN false alarm rate | loudness rule false alarm rate |\n|---|---|---|---|")
    for s_name in sorted(set(src_all[y == names.index("loud_not_crash")])):
        sel = (src_all == s_name) & (y == names.index("loud_not_crash"))
        md.append(f"| {s_name} | {int(sel.sum())} | {cnn_crash_pred[sel].mean():.3f} | {loud_crash_pred[sel].mean():.3f} |")
    md.append("\nOther operating points, crash as positive, mean ± std over folds:\n")
    md.append("| rule | crash precision | crash recall | crash F1 | false alarm rate on loud_not_crash |\n|---|---|---|---|---|")
    for label, fn in [("CNN, crash probability > 0.7", lambda m: m["cnn"]["crash_vs_rest_p_above_0.7"]),
                      ("loudness rule, app default threshold 85", lambda m: m["loudness_rule"]["app_default_threshold_85"])]:
        md.append(f"| {label} | " + " | ".join(
            fmt(mean_std([fn(m)[key] for m in fold_metrics]))
            for key in ("crash_precision", "crash_recall", "crash_f1", "false_alarm_rate_loud_not_crash")) + " |")
    md.append("\n## CNN per class (mean ± std over folds)\n")
    md.append("| class | precision | recall | F1 |\n|---|---|---|---|")
    for n in names:
        md.append(f"| {n} | " + " | ".join(
            fmt(mean_std([m["cnn"]["per_class"][n][key] for m in fold_metrics if m["cnn"]["per_class"][n]["support"] > 0]))
            for key in ("precision", "recall", "f1")) + " |")
    md.append("\n## Per fold\n")
    md.append("| fold | test windows | CNN accuracy | CNN macro F1 | CNN crash recall | loudness threshold | loudness crash F1 | epochs run |\n|---|---|---|---|---|---|---|---|")
    for m in fold_metrics:
        md.append(f"| {m['fold']} | {m['n_test']} | {m['cnn']['accuracy']:.3f} | {m['cnn']['macro_f1']:.3f} | "
                  f"{m['cnn']['crash_recall']:.3f} | {m['loudness_rule']['tuned_threshold']:.1f} | "
                  f"{m['loudness_rule']['tuned']['crash_f1']:.3f} | {m['epochs_run']} |")
    md.append("\nFigures: `foldN_confusion.png` and `foldN_curves.png` in this folder. Raw numbers: `foldN_metrics.json`.\n")
    md_text = "\n".join(md)
    summary_name = "SMOKE_TEST_LOG.md" if synthetic else ("RESULTS.md" if args.tag == "base" else "RESULTS_variant.md")
    with open(os.path.join(out_dir, summary_name), "w", encoding="utf-8") as f:
        f.write(md_text + "\n")
    macro = mean_std([m["cnn"]["macro_f1"] for m in fold_metrics])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": args.tag, "synthetic": synthetic, "macro_f1_mean": macro[0], "macro_f1_std": macro[1],
                   "best_epochs": best_epochs, "args": vars(args), "n_params": n_params}, f, indent=1)
    print("\n" + md_text)
    print(f"\nwritten: {os.path.join(out_dir, summary_name)}")

    if args.skip_final:
        return

    # ---------------- final model on all folds
    epochs_final = max(int(round(float(np.mean(best_epochs)))), 1)
    print(f"\ntraining final model on all folds for {epochs_final} epochs (mean best epoch of the 4 folds)")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(n_classes, args.width, args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    final_weights = balanced_weights(y, n_classes)
    bg = X[y == names.index("background")]
    if len(bg) > 2000:
        bg = bg[np.random.default_rng(SEED).choice(len(bg), 2000, replace=False)]
    ds = make_train_dataset(X, y, bg, n_classes, args.batch, not args.no_specaug)
    model.fit(ds, epochs=epochs_final, verbose=2,
              class_weight=final_weights)
    os.makedirs(C.MODELS_DIR, exist_ok=True)
    model.save(os.path.join(C.MODELS_DIR, "crash_cnn.keras"))
    model.save(os.path.join(C.MODELS_DIR, "crash_cnn.h5"))
    stamp = datetime.date.today().strftime("%Y%m%d")
    version = f"crashcnn-{'SYNTHETIC-TEST' if synthetic else meta['road_dataset']}-{stamp}"
    model_meta = {
        "modelVersion": version, "synthetic": synthetic, "roadDataset": meta["road_dataset"],
        "classes": names, "parameters": int(n_params), "epochs": epochs_final,
        "trainWindows": int(len(X)), "tag": args.tag, "width": args.width, "dropout": args.dropout,
        "note": ("PLACEHOLDER trained on SYNTHETIC audio. Only for testing the code path. "
                 "Not a crash detector.") if synthetic else "Trained by the PulseTrack team.",
    }
    with open(os.path.join(C.MODELS_DIR, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=1)
    print("final model saved:", os.path.join(C.MODELS_DIR, "crash_cnn.keras"), "version", version)


if __name__ == "__main__":
    main()
