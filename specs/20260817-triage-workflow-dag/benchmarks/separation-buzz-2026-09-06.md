# Separation against a background buzz — seven models, one recording

Recording: `sub-17578482-…_task-Story-recall.wav`, 25.542 s, 16 kHz, from `b2ai_v31_bids_07_01_v3`.
Not the standing reference file. It carries a continuous electrical buzz; its first 2.29 s are that
buzz alone, with no speech over it.

Audio for every row is kept at `~/Downloads/buzz_separation_20260906/` (raw, `rms_matched/`,
`head_2s/`), with the ORCD run's manifest under `provenance/unasdiff/`.

## What each model did to the buzz

YAMNet peaks over the whole stream. Original: `Buzz` 0.808, `Insect` 0.904, `Mains hum` 0.639,
`Electric shaver, electric razor` 0.431, `Speech` 0.994.

| model | task | buzz outcome |
| --- | --- | --- |
| `FRCRN_SE_16K` | enhancement | removed — every buzz label to 0.000 |
| `MossFormerGAN_SE_16K` | enhancement | removed |
| `MossFormer2_SE_48K` | enhancement | removed |
| `DriftSE` | enhancement | removed |
| `MossFormer2_SS_16K` s0/s1 | separation | removed; background in neither stream |
| `sepformer-wham16k` | enhancement | **leaked and amplified** — `Buzz` 0.682 → 0.934, `Insect` 0.351 → 0.891 |
| `unasdiff` speech_speech s0/s1 | separation | split and lost it — r 0.402/0.679, all buzz labels ≈ 0 |
| `unasdiff` speech_sound s0/s1 | separation | partly isolated into s1 — `Mains hum` 0.989 |

Both unasdiff decompositions are real: `s0+s1` reconstructs the input at −51/−53 dBFS residual,
r = 0.994/0.996. The failures below are not reconstruction failures.

## unasdiff is not clean enough for downstream use — 2026-09-06

`speech_sound` is the only model here that *isolated* the buzz rather than removing it, which is the
behaviour a conditioned separator is wanted for. It is still not usable downstream, on four grounds:

1. **Speech leaks into both slots**, `Speech` 1.000 in each. A "noise" stream carrying full-scale
   speech cannot be subtracted from, or measured as, background.
2. **It keeps only half the buzz.** s1's 4–8 kHz fraction is 0.000 against the original's 0.020, so
   the 60 Hz stack survives (head-block r = 0.889) while the raspy upper band is destroyed. That is
   why `Mains hum` rises to 0.989 while `Buzz` and `Insect` collapse — the label movement is an
   artefact of losing a band, not of isolating a source.
3. **The class list cannot name the target.** The 41-class FSD prior has no `Buzz`, `Hum` or `Noise`
   member. This run was steered with `["Microwave_oven"]` as the nearest sustained electrical drone.
   Conditioning on a proxy class is not a contract any downstream consumer can rely on.
4. **`speech_speech` has no noise slot at all** and loses the buzz outright, as upstream's README
   warns. Of the two modes, only one is even a candidate.

**Cost, separately disqualifying for triage.** On an A100 80 GB (ORCD job 22150074, node3805, CUDA):
`speech_speech` 793.5 s and `speech_sound` 1097.7 s for 25.542 s of audio — **43× slower than real
time on the largest GPU available**, at a locked 200 diffusion steps over 12 windows. Node was
shared, so these are upper bounds; the order of magnitude is not in doubt. On CPU it is >14 min per
4 s window, and the MPS path fails on a float64 schedule tensor.

**Ruling.** unasdiff is not a separation backend for the triage path. Where the buzz needs to go,
any of the four enhancement models removes it at a fraction of the cost. Its remaining interest is
the `span_reconfirm` class-filter idea in [`../model-to-branch.md`](../model-to-branch.md), which
this measurement does not support and does not refute — that candidacy rests on conditioning on
`Cough`, a class the prior does have, and was never about background.

**`sepformer-wham16k` is the other finding worth carrying.** It makes the buzz *worse*: anything
measured downstream of it carries a stronger buzz than the recording did. It should not be a default
anywhere.

## What this does not license

One recording, one buzz, one condition class. It says nothing about unasdiff on the classes its prior
does carry, nothing about overlapping speech, and nothing about any of these models on a recording
where the interferer is not stationary.
