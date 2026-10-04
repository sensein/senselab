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
| `unasdiff` speech_sound s0/s1 | separation | separated nothing; s1 is the input, lowpassed |

Both unasdiff decompositions are real: `s0+s1` reconstructs the input at −51/−53 dBFS residual,
r = 0.994/0.996. The failures below are not reconstruction failures.

## unasdiff separated nothing — 2026-09-06

An earlier version of this file said `speech_sound` "isolated the mains-hum half" of the buzz. That
was wrong, and the error is worth recording because it is easy to repeat.

The claim rested on the 2.29 s head block, measured against the original:

| stream | r vs original | level | 0–200 Hz | 4–8 kHz |
| --- | --- | --- | --- | --- |
| `00_original` | 1.000 | −49.2 dBFS | 0.407 | 0.0199 |
| `speech_sound` s1 | **0.889** | **−50.3 dBFS** | 0.535 | 0.0005 |
| `speech_sound` s0 | 0.142 | −70.1 dBFS | 0.523 | 0.2062 |

s1 is the input, within 1.1 dB and at r = 0.889. s0 is 20 dB down. **The head block contains one
source and no speech**, so routing all of it to one stream is what any separator would do and
demonstrates no separation at all. Reading "s1 ≈ input, s0 ≈ silent" as isolation is reading the
absence of a second source as evidence of splitting one.

The `Mains hum` 0.989 that looked like a result is the same artefact from the other side: s1's
4–8 kHz share is 0.0005 against the original's 0.0199, so s1 is lowpassed. Strip the upper band from
a buzz and YAMNet's `Buzz` and `Insect` labels fall while `Mains hum` rises. That is degradation
being read as selectivity.

Nothing here supports a buzz being separated from speech, which is the only thing that would have
been useful. **unasdiff is not a separation backend for the triage path.**

**Cost, independently disqualifying.** On an A100 80 GB (ORCD job 22150074, node3805, CUDA):
`speech_speech` 793.5 s and `speech_sound` 1097.7 s for 25.542 s of audio — **43× slower than real
time on the largest GPU available**, at a locked 200 diffusion steps over 12 windows. Node was
shared, so these are upper bounds. On CPU it is >14 min per 4 s window; the MPS path fails on a
float64 schedule tensor.

**The class list also cannot name the target.** The 41-class FSD prior has no `Buzz`, `Hum` or
`Noise` member; this run was steered with `["Microwave_oven"]` as the nearest sustained electrical
drone. `speech_speech` takes no class list at all.

This ruling is about background separation. It leaves the `span_reconfirm` class-filter candidacy in
[`../model-to-branch.md`](../model-to-branch.md) untouched — that one conditions on `Cough`, a class
the prior does carry, and was never about background.

**`sepformer-wham16k` is the finding to carry forward.** It makes the buzz worse: `Buzz` 0.682 →
0.934, `Insect` 0.351 → 0.891. Anything measured downstream of it carries a stronger buzz than the
recording did. It must not be a default anywhere.

## What this does not license

One recording, one buzz, one condition class. It says nothing about unasdiff on the classes its prior
does carry, nothing about overlapping speech, and nothing about any of these models on a recording
where the interferer is not stationary.
