# Results: crash sound CNN vs loudness rule

Run tag: `base`. Road data: `open_freesound_us8k_esc50`. Windows: 19084. Model parameters: 79700. Max epochs: 60. Date: 2026-11-18.

All numbers below were printed by `ml/train.py`. Mean ± standard deviation over the 4 test folds.

## CNN vs loudness rule

| metric | CNN (argmax) | loudness rule (threshold tuned on train folds) |
|---|---|---|
| accuracy (4 classes) | 0.673 ± 0.062 | not applicable |
| macro F1 (4 classes) | 0.503 ± 0.024 | not applicable |
| crash vs rest accuracy | 0.887 ± 0.037 | 0.755 ± 0.069 |
| crash precision | 0.309 ± 0.070 | 0.108 ± 0.016 |
| crash recall | 0.475 ± 0.034 | 0.387 ± 0.203 |
| crash F1 | 0.370 ± 0.058 | 0.151 ± 0.025 |
| false alarm rate on loud_not_crash | 0.104 ± 0.045 | 0.220 ± 0.090 |
| false alarm rate on all non crash windows | 0.084 ± 0.036 | 0.214 ± 0.099 |

### Key point: false alarms on loud sounds that are not crashes

> On `loud_not_crash` test windows (door knocks, fireworks, glass breaking, horns, sirens, gun shots and more) the **loudness rule raised a crash alarm on 0.220 ± 0.090** of windows, the **CNN on 0.104 ± 0.045** (fraction of windows, mean ± std over the 4 folds).


False alarm rate per loud_not_crash source (all 4 test folds pooled):

| source | windows | CNN false alarm rate | loudness rule false alarm rate |
|---|---|---|---|
| esc50_can_opening | 107 | 0.308 | 0.879 |
| esc50_car_horn | 114 | 0.079 | 0.167 |
| esc50_clapping | 120 | 0.250 | 0.758 |
| esc50_dog | 112 | 0.402 | 0.277 |
| esc50_door_wood_knock | 107 | 0.252 | 0.150 |
| esc50_fireworks | 119 | 0.630 | 0.294 |
| esc50_glass_breaking | 85 | 0.400 | 0.741 |
| esc50_siren | 120 | 0.042 | 0.175 |
| esc50_thunderstorm | 120 | 0.200 | 0.092 |
| us8k_car_horn | 913 | 0.034 | 0.150 |
| us8k_gun_shot | 569 | 0.380 | 0.715 |
| us8k_siren | 2742 | 0.006 | 0.083 |

Other operating points, crash as positive, mean ± std over folds:

| rule | crash precision | crash recall | crash F1 | false alarm rate on loud_not_crash |
|---|---|---|---|---|
| CNN, crash probability > 0.7 | 0.452 ± 0.147 | 0.128 ± 0.067 | 0.174 ± 0.073 | 0.026 ± 0.019 |
| loudness rule, app default threshold 85 | 0.428 ± 0.267 | 0.039 ± 0.026 | 0.071 ± 0.046 | 0.007 ± 0.005 |

## CNN per class (mean ± std over folds)

| class | precision | recall | F1 |
|---|---|---|---|
| background | 0.884 ± 0.048 | 0.713 ± 0.076 | 0.787 ± 0.050 |
| crash | 0.309 ± 0.070 | 0.475 ± 0.034 | 0.370 ± 0.058 |
| skid | 0.095 ± 0.039 | 0.441 ± 0.150 | 0.148 ± 0.045 |
| loud_not_crash | 0.775 ± 0.041 | 0.656 ± 0.055 | 0.707 ± 0.018 |

## Per fold

| fold | test windows | CNN accuracy | CNN macro F1 | CNN crash recall | loudness threshold | loudness crash F1 | epochs run |
|---|---|---|---|---|---|---|---|
| 1 | 5247 | 0.595 | 0.506 | 0.533 | 63.1 | 0.114 | 35 |
| 2 | 4613 | 0.659 | 0.467 | 0.452 | 56.8 | 0.141 | 24 |
| 3 | 4612 | 0.666 | 0.504 | 0.464 | 46.2 | 0.170 | 13 |
| 4 | 4612 | 0.770 | 0.534 | 0.449 | 45.0 | 0.178 | 18 |

Figures: `foldN_confusion.png` and `foldN_curves.png` in this folder. Raw numbers: `foldN_metrics.json`.

