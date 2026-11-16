# SMOKE TEST ON SYNTHETIC DATA. THESE NUMBERS ARE NOT RESULTS.

The road audio here was made by `ml/make_synthetic.py` from noise and sine tones. This file only proves that the training and evaluation code runs. Do not copy these numbers into the report.

Run tag: `base`. Road data: `synthetic`. Windows: 5028. Model parameters: 79700. Max epochs: 60. Date: 2026-11-16.

All numbers below were printed by `ml/train.py`. Mean ± standard deviation over the 4 test folds.

## CNN vs loudness rule

| metric | CNN (argmax) | loudness rule (threshold tuned on train folds) |
|---|---|---|
| accuracy (4 classes) | 0.983 ± 0.014 | not applicable |
| macro F1 (4 classes) | 0.980 ± 0.011 | not applicable |
| crash vs rest accuracy | 0.995 ± 0.002 | 0.739 ± 0.035 |
| crash precision | 0.980 ± 0.017 | 0.208 ± 0.025 |
| crash recall | 0.956 ± 0.027 | 0.775 ± 0.055 |
| crash F1 | 0.967 ± 0.013 | 0.327 ± 0.035 |
| false alarm rate on loud_not_crash | 0.003 ± 0.003 | 0.458 ± 0.066 |
| false alarm rate on all non crash windows | 0.002 ± 0.001 | 0.264 ± 0.037 |

Other operating points, crash as positive, mean ± std over folds:

| rule | crash precision | crash recall | crash F1 | false alarm rate on loud_not_crash |
|---|---|---|---|---|
| CNN, crash probability > 0.7 | 0.992 ± 0.009 | 0.898 ± 0.032 | 0.942 ± 0.020 | 0.001 ± 0.001 |
| loudness rule, app default threshold 85 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.016 ± 0.013 |

## CNN per class (mean ± std over folds)

| class | precision | recall | F1 |
|---|---|---|---|
| background | 0.960 ± 0.042 | 1.000 ± 0.000 | 0.979 ± 0.022 |
| crash | 0.980 ± 0.017 | 0.956 ± 0.027 | 0.967 ± 0.013 |
| skid | 0.998 ± 0.003 | 0.982 ± 0.016 | 0.990 ± 0.007 |
| loud_not_crash | 0.994 ± 0.002 | 0.978 ± 0.029 | 0.986 ± 0.014 |

## Per fold

| fold | test windows | CNN accuracy | CNN macro F1 | CNN crash recall | loudness threshold | loudness crash F1 | epochs run |
|---|---|---|---|---|---|---|---|
| 1 | 1226 | 0.984 | 0.975 | 0.909 | 43.2 | 0.279 | 38 |
| 2 | 1230 | 0.996 | 0.993 | 0.963 | 43.2 | 0.329 | 51 |
| 3 | 1314 | 0.992 | 0.988 | 0.970 | 43.2 | 0.324 | 26 |
| 4 | 1258 | 0.960 | 0.966 | 0.980 | 43.2 | 0.377 | 23 |

Figures: `foldN_confusion.png` and `foldN_curves.png` in this folder. Raw numbers: `foldN_metrics.json`.

