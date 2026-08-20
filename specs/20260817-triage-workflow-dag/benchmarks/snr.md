# Noise robustness

Pink noise, one seed (17), SNR defined over the speech span, mixture rescaled to avoid clipping — which
is itself a change to the signal.

## Speech span extraction, local floor at `K` = 12 dB

| SNR over the speech | span proposed? | IoU with the label | YAMNet coverage | verdict |
| --- | --- | --- | --- | --- |
| as captured | yes | 0.89 | 80% | speech |
| +20 dB | yes | 0.17 | 100% | speech |
| +10 dB | yes | 0.17 | 100% | speech |
| +5 dB | merged with a cough | 0.10 | 67% | flag |
| 0 dB | merged with a cough | 0.10 | 50% | flag |
| −5 dB | no contrast — speech is 3.9 dB above the floor | — | — | fail |

**The classifier is never the limit.** YAMNet `Speech` holds 0.987–0.998 with full coverage across every
SNR where the envelope can no longer find the span. Effort on noisy-speech extraction belongs in
proposal, not classification.

**Extent dies before detection.** IoU 0.89 → 0.17 by +10 dB while the verdict is still confidently
speech. A span locates speech; it does not bound it.

**Below +5 dB the failure changes character** from missing to merging: the surviving span runs
9.61–12.00 s, joining cough 2 to the speech, and its STOI collapses to 0.254. It flags because the two
instruments disagree exactly as they should — YAMNet still seeing speech in the window, SQUIM seeing a
mixture.

## Per-event contrast above the floor, as noise rises

| SNR | floor | mouth | exhal 1 | exhal 2 | cough 1 | cough 2 | speech |
| --- | --- | --- | --- | --- | --- | --- | --- |
| orig | −53.5 | 16.7 | 35.3 | 28.4 | 53.5 | 52.4 | 31.4 |
| +20 | −46.5 | 10.5 | 28.3 | 21.4 | 46.5 | 45.4 | 24.5 |
| +10 | −37.8 | 5.0 | 19.6 | 12.9 | 37.8 | 36.6 | 15.9 |
| +5 | −32.9 | 3.2 | 14.9 | 8.5 | 32.9 | 31.7 | 11.3 |
| 0 | −28.0 | 2.0 | 10.3 | 4.9 | 28.0 | 26.8 | 7.1 |
| −5 | −23.1 | 1.8 | 6.2 | 2.5 | 23.1 | 21.9 | 3.9 |

Speech is the low-contrast event of interest, at roughly 22 dB below the coughs throughout. The mouth
sound is lower still and is unrecoverable beyond +20 dB.

## Enhancement survival — speech event only

Job 20795138, 1× A100, 1362 rows. `*` = ClearVoice (`alibabasglab/*`).

| model | orig | +20 | +10 | +5 | 0 | −5 | drop |
| --- | --- | --- | --- | --- | --- | --- | --- |
| input | 0.994 | 0.996 | 0.959 | 0.882 | 0.706 | 0.682 | −0.312 |
| FRCRN_SE_16K * | 0.998 | 0.999 | 0.988 | 0.990 | 0.998 | 0.990 | **−0.008** |
| MossFormerGAN_SE_16K * | 0.989 | 0.996 | 0.997 | 0.981 | 0.954 | 0.831 | −0.158 |
| MossFormer2_SE_48K * | 0.989 | 0.995 | 0.989 | 0.985 | 0.983 | 0.915 | −0.074 |
| DriftSE_v2 | 0.992 | 0.999 | 0.997 | 0.992 | 0.996 | 0.992 | **+0.000** |
| sepformer-wham16k-enh | 0.998 | 0.991 | 0.972 | 0.759 | 0.693 | 0.539 | −0.459 |
| sepformer-dns4-16k-enh | 0.904 | 0.044 | 0.053 | 0.032 | 0.012 | 0.027 | −0.877 |
| metricgan-plus-voicebank | 0.996 | 0.994 | 0.996 | 0.988 | 0.951 | 0.650 | −0.346 |

## Enhancement survival — airway events (mean of both breaths, both coughs)

| model | orig | +20 | +10 | +5 | 0 | −5 |
| --- | --- | --- | --- | --- | --- | --- |
| input | 0.872 | 0.970 | 0.820 | 0.631 | 0.134 | 0.007 |
| FRCRN_SE_16K * | 0.876 | 0.005 | 0.179 | 0.103 | 0.003 | 0.009 |
| MossFormerGAN_SE_16K * | **0.004** | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| MossFormer2_SE_48K * | 0.476 | 0.369 | 0.400 | 0.403 | 0.386 | **0.338** |
| DriftSE_v2 | 0.759 | 0.729 | 0.531 | 0.267 | 0.436 | 0.472 |
| sepformer-wham16k-enh | 0.381 | 0.306 | 0.018 | 0.000 | 0.001 | 0.000 |
| metricgan-plus-voicebank | 0.706 | 0.278 | 0.080 | 0.029 | 0.143 | 0.040 |

Speech survives every enhancer at every SNR (0.83–0.999), so these differences are entirely about
airway content. `MossFormerGAN_SE_16K` annihilates airway events on the **clean** input — per event:
both breaths 0.000, cough 1 0.000, cough 2 0.016, speech 0.989. It is a speech isolator.
`MossFormer2_SE_48K` is the only enhancer beating the raw input at −5 dB, and it keeps coughs while
destroying breaths. FRCRN's non-monotonicity (0.876 clean, 0.005 at +20 dB) is unexplained and should be
treated as suspect.

**The `verified_empty` column in that job's output is invalid** and must not be read: the longest
verified-empty stretch is 1.80 s against YAMNet's 0.96 s and HeAR's 2 s window, so every window
overlapping a gap also overlaps a real event. It measures window bleed, not invented events. Nothing in
this file speaks to false positives.

**`MossFormer2_SS_16K` was never measured across SNR.** It is declared in `bench.py:89` under
`SEPARATORS`, which `bench.py:296` runs only when `--arm != enhance`. The single completed job wrote
1362 rows accounting exactly for the enhancers (9 × 150 + 2 × 6), with no separation rows and no
separation result file. Given how far the two enhancement variants diverge, its behaviour should not be
assumed.
