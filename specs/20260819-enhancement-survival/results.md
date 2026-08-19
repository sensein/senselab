# Results — what survives enhancement, read by HeAR and YAMNet

Job 20795138 on ORCD `pi_satra` (1x A100), 1362 rows, 29:22 elapsed. Input:
`streaming-audio-2026-07-30T04-21-56-487Z.wav`. Scores are the peak over reader windows
overlapping each verified event; the unenhanced rows are the reference.

## The `verified_empty` column is invalid as computed — do not read it

It reports the peak over windows overlapping the verified-empty stretches, intending to separate
*preserving* an element from *inventing* one. It cannot: the longest verified-empty stretch is 1.80 s
and YAMNet's window is 0.96 s (HeAR's is 2 s), so every window overlapping a gap also overlaps an
adjacent real event. The proof is in the numbers — for every row the column equals that row's maximum
event score exactly (input 1.000 = cough 1.000; `MossFormer2_SE_48K` 0.994 = cough 0.994). It measures
window bleed, not false positives.

Measuring invented events needs a recording with gaps longer than the widest reader window. This one
does not have them, which is the same limitation that blocked the branch-1 evaluation.


### yamnet — as captured

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.001 | 0.726 | 0.893 | 0.869 | 1.000 | 0.994 | 1.000 |
| input@48k | 0.001 | 0.726 | 0.893 | 0.869 | 1.000 | 0.994 | 1.000 |
| sepformer-wham16k-enh | 0.005 | 0.000 | 0.925 | 0.599 | 0.000 | 0.998 | 0.998 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.007 | 0.318 | 1.000 | 0.000 | 0.000 | 0.904 | 1.000 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.902 | 0.967 | 0.955 | 0.996 | 0.996 |
| FRCRN_SE_16K | 0.000 | 0.640 | 0.865 | 0.998 | 1.000 | 0.998 | 1.000 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.000 | 0.016 | 0.989 | 0.989 |
| MossFormer2_SE_48K | 0.000 | 0.000 | 0.000 | 0.903 | 1.000 | 0.989 | 1.000 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.712 | 0.686 | 0.986 | 0.653 | 0.992 | 0.992 |

### yamnet — +20 dB SNR

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.001 | 0.953 | 0.975 | 0.954 | 1.000 | 0.996 | 1.000 |
| input@48k | 0.001 | 0.953 | 0.975 | 0.954 | 1.000 | 0.996 | 1.000 |
| sepformer-wham16k-enh | 0.005 | 0.020 | 0.919 | 0.253 | 0.034 | 0.991 | 0.991 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.044 | 0.478 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.129 | 0.930 | 0.051 | 0.994 | 0.986 |
| FRCRN_SE_16K | 0.000 | 0.000 | 0.022 | 0.000 | 0.000 | 0.999 | 0.999 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 | 0.996 |
| MossFormer2_SE_48K | 0.000 | 0.000 | 0.628 | 0.845 | 0.002 | 0.995 | 0.995 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.317 | 0.925 | 0.890 | 0.782 | 0.999 | 0.999 |

### yamnet — +10 dB SNR

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.001 | 0.782 | 0.545 | 0.957 | 0.997 | 0.959 | 0.997 |
| input@48k | 0.001 | 0.782 | 0.545 | 0.957 | 0.997 | 0.959 | 0.997 |
| sepformer-wham16k-enh | 0.000 | 0.014 | 0.060 | 0.000 | 0.000 | 0.972 | 0.972 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.053 | 0.174 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.004 | 0.299 | 0.016 | 0.996 | 0.991 |
| FRCRN_SE_16K | 0.000 | 0.000 | 0.714 | 0.000 | 0.001 | 0.988 | 0.977 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.997 | 0.997 |
| MossFormer2_SE_48K | 0.000 | 0.000 | 0.000 | 0.617 | 0.984 | 0.989 | 0.989 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.268 | 0.719 | 0.986 | 0.152 | 0.997 | 0.997 |

### yamnet — +5 dB SNR

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.000 | 0.828 | 0.011 | 0.730 | 0.957 | 0.882 | 0.957 |
| input@48k | 0.000 | 0.828 | 0.011 | 0.730 | 0.957 | 0.882 | 0.957 |
| sepformer-wham16k-enh | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.759 | 0.867 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.001 | 0.001 | 0.002 | 0.000 | 0.000 | 0.032 | 0.306 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.000 | 0.027 | 0.091 | 0.988 | 0.988 |
| FRCRN_SE_16K | 0.000 | 0.000 | 0.406 | 0.006 | 0.000 | 0.990 | 0.985 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.981 | 0.981 |
| MossFormer2_SE_48K | 0.000 | 0.260 | 0.000 | 0.354 | 0.997 | 0.985 | 0.997 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.003 | 0.000 | 0.972 | 0.091 | 0.992 | 0.992 |

### yamnet — +0 dB SNR

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.000 | 0.253 | 0.000 | 0.034 | 0.247 | 0.706 | 0.706 |
| input@48k | 0.000 | 0.253 | 0.000 | 0.034 | 0.247 | 0.706 | 0.706 |
| sepformer-wham16k-enh | 0.000 | 0.004 | 0.000 | 0.000 | 0.000 | 0.693 | 0.693 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.581 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.000 | 0.170 | 0.401 | 0.951 | 0.951 |
| FRCRN_SE_16K | 0.000 | 0.007 | 0.000 | 0.000 | 0.005 | 0.998 | 0.975 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.954 | 0.954 |
| MossFormer2_SE_48K | 0.000 | 0.022 | 0.000 | 0.530 | 0.991 | 0.983 | 0.991 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.000 | 0.000 | 0.987 | 0.759 | 0.996 | 0.996 |

### yamnet — -5 dB SNR

| enhancer | mouth | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|---|
| input@16k | 0.000 | 0.001 | 0.001 | 0.018 | 0.010 | 0.682 | 0.682 |
| input@48k | 0.000 | 0.001 | 0.001 | 0.018 | 0.010 | 0.682 | 0.682 |
| sepformer-wham16k-enh | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.539 | 0.539 |
| sepformer-whamr16k | - | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.027 | 0.530 |
| metricgan-plus-voicebank | 0.000 | 0.000 | 0.000 | 0.084 | 0.077 | 0.650 | 0.812 |
| FRCRN_SE_16K | 0.000 | 0.034 | 0.000 | 0.002 | 0.000 | 0.990 | 0.990 |
| MossFormerGAN_SE_16K | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.831 | 0.666 |
| MossFormer2_SE_48K | 0.000 | 0.000 | 0.000 | 0.523 | 0.828 | 0.915 | 0.915 |
| DriftSE_v1 | - | - | - | - | - | - | - |
| DriftSE_v2 | 0.000 | 0.000 | 0.000 | 0.953 | 0.934 | 0.992 | 0.992 |

### hear — as captured

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.998 | 0.997 | 1.000 | 1.000 | 0.348 | 1.000 |
| input@48k | 0.998 | 0.997 | 1.000 | 1.000 | 0.348 | 1.000 |
| sepformer-wham16k-enh | 0.592 | 0.712 | 0.946 | 0.946 | 0.524 | 0.946 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.395 | 0.498 | 0.103 | 0.509 | 0.179 | 0.509 |
| metricgan-plus-voicebank | 0.140 | 0.808 | 0.660 | 0.674 | 0.410 | 0.808 |
| FRCRN_SE_16K | 0.996 | 0.999 | 1.000 | 0.995 | 0.403 | 1.000 |
| MossFormerGAN_SE_16K | 0.253 | 0.508 | 0.005 | 0.022 | 0.475 | 0.508 |
| MossFormer2_SE_48K | 0.289 | 0.051 | 0.994 | 0.994 | 0.552 | 0.994 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.919 | 0.988 | 0.983 | 0.980 | 0.605 | 0.988 |

### hear — +20 dB SNR

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.992 | 0.973 | 1.000 | 1.000 | 0.315 | 1.000 |
| input@48k | 0.992 | 0.973 | 1.000 | 1.000 | 0.315 | 1.000 |
| sepformer-wham16k-enh | 0.328 | 0.382 | 0.697 | 0.983 | 0.507 | 0.983 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.001 | 0.002 | 0.067 | 0.067 | 0.006 | 0.067 |
| metricgan-plus-voicebank | 0.086 | 0.248 | 0.632 | 0.632 | 0.104 | 0.632 |
| FRCRN_SE_16K | 0.280 | 0.894 | 0.993 | 0.814 | 0.348 | 0.993 |
| MossFormerGAN_SE_16K | 0.026 | 0.037 | 0.001 | 0.026 | 0.301 | 0.301 |
| MossFormer2_SE_48K | 0.780 | 0.889 | 0.953 | 0.896 | 0.414 | 0.953 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.498 | 0.977 | 0.971 | 0.971 | 0.411 | 0.977 |

### hear — +10 dB SNR

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.834 | 0.476 | 1.000 | 0.999 | 0.363 | 1.000 |
| input@48k | 0.834 | 0.476 | 1.000 | 0.999 | 0.363 | 1.000 |
| sepformer-wham16k-enh | 0.255 | 0.708 | 0.992 | 0.934 | 0.255 | 0.992 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.002 | 0.001 | 0.497 | 0.497 | 0.010 | 0.497 |
| metricgan-plus-voicebank | 0.018 | 0.086 | 0.098 | 0.098 | 0.101 | 0.101 |
| FRCRN_SE_16K | 0.574 | 0.979 | 0.988 | 0.306 | 0.379 | 0.988 |
| MossFormerGAN_SE_16K | 0.028 | 0.090 | 0.003 | 0.016 | 0.438 | 0.438 |
| MossFormer2_SE_48K | 0.940 | 0.552 | 0.995 | 0.983 | 0.437 | 0.995 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.249 | 0.866 | 0.965 | 0.965 | 0.586 | 0.965 |

### hear — +5 dB SNR

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.534 | 0.225 | 1.000 | 0.999 | 0.167 | 1.000 |
| input@48k | 0.534 | 0.225 | 1.000 | 0.999 | 0.167 | 1.000 |
| sepformer-wham16k-enh | 0.078 | 0.107 | 0.661 | 0.661 | 0.123 | 0.661 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.003 | 0.004 | 0.004 | 0.005 | 0.010 | 0.028 |
| metricgan-plus-voicebank | 0.013 | 0.014 | 0.040 | 0.032 | 0.046 | 0.046 |
| FRCRN_SE_16K | 0.818 | 0.935 | 0.968 | 0.727 | 0.504 | 0.970 |
| MossFormerGAN_SE_16K | 0.037 | 0.067 | 0.002 | 0.011 | 0.448 | 0.448 |
| MossFormer2_SE_48K | 0.985 | 0.740 | 0.987 | 0.987 | 0.564 | 0.989 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.097 | 0.370 | 0.963 | 0.963 | 0.724 | 0.963 |

### hear — +0 dB SNR

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.206 | 0.018 | 0.999 | 1.000 | 0.083 | 1.000 |
| input@48k | 0.206 | 0.018 | 0.999 | 1.000 | 0.083 | 1.000 |
| sepformer-wham16k-enh | 0.026 | 0.035 | 0.131 | 0.879 | 0.175 | 0.879 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.001 | 0.003 | 0.001 | 0.018 | 0.009 | 0.040 |
| metricgan-plus-voicebank | 0.009 | 0.018 | 0.262 | 0.025 | 0.008 | 0.262 |
| FRCRN_SE_16K | 0.727 | 0.360 | 0.938 | 0.739 | 0.683 | 0.938 |
| MossFormerGAN_SE_16K | 0.063 | 0.044 | 0.010 | 0.011 | 0.250 | 0.250 |
| MossFormer2_SE_48K | 0.981 | 0.881 | 0.987 | 0.987 | 0.646 | 0.999 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.059 | 0.023 | 0.968 | 0.968 | 0.709 | 0.968 |

### hear — -5 dB SNR

| enhancer | breath_1 | breath_2 | cough_1 | cough_2 | speech | verified_empty |
|---|---|---|---|---|---|---|
| input@16k | 0.074 | 0.016 | 1.000 | 1.000 | 0.038 | 1.000 |
| input@48k | 0.074 | 0.016 | 1.000 | 1.000 | 0.038 | 1.000 |
| sepformer-wham16k-enh | 0.008 | 0.055 | 0.176 | 0.863 | 0.087 | 0.955 |
| sepformer-whamr16k | - | - | - | - | - | - |
| sepformer-dns4-16k-enh | 0.001 | 0.002 | 0.013 | 0.025 | 0.006 | 0.032 |
| metricgan-plus-voicebank | 0.003 | 0.020 | 0.837 | 0.056 | 0.011 | 0.837 |
| FRCRN_SE_16K | 0.569 | 0.362 | 0.719 | 0.779 | 0.889 | 0.889 |
| MossFormerGAN_SE_16K | 0.114 | 0.053 | 0.017 | 0.018 | 0.021 | 0.114 |
| MossFormer2_SE_48K | 0.787 | 0.152 | 0.983 | 0.983 | 0.566 | 0.983 |
| DriftSE_v1 | - | - | - | - | - | - |
| DriftSE_v2 | 0.026 | 0.038 | 0.978 | 0.965 | 0.244 | 0.978 |
