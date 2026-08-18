# What a real recording resolves — measured, 2026-08-17

One 14.03 s file, 48 kHz mono, close-miked (Brouhaha C50 median 28.5 dB, so essentially dry), holding
two coughs, two breaths, one short utterance and a handling click. Asserted by the file-level taxonomy
to contain breathing, coughing and speech. Detectors run locally; outputs under the job scratch dir.

## The events

| event | onset (s) | 10-90% rise | level step | detected as |
| --- | --- | --- | --- | --- |
| handling click | 0.893 | — | ~13 dB | transient |
| breath (exhale) | 2.275 | 60 ms | 28.6 dB | breath |
| breath (exhale) | 5.308 | 127 ms | 20.0 dB | breath |
| cough 1 | 7.924 | 17 ms | 44.9 dB | cough |
| cough 2 | 9.609 | **9 ms** | 48.5 dB | cough |
| speech | ~11.62 | — | — | "There's something going on." |

## Finding 1 — only DSP resolves an onset; no classifier comes close

Cough 2's rise is bounded to about ±5 ms on a 1 ms envelope. Independent flux detectors at a 5.33 ms
hop land within ~20 ms. Against that:

| detector | cough response width at 10% | speech leading edge error |
| --- | --- | --- |
| YAMNet 0.96 s / 0.48 s | 0.48 s | 1.06 s early |
| AST 0.96 s / 0.48 s | 0.96 s | 1.06 s early |
| AST 0.96 s / 0.10 s | 0.90 s | 1.34 s early |
| AST 0.48 s / 0.05 s | 0.65 s | 1.58 s early |

Shrinking the hop tenfold made the leading edge *worse*, not tighter: response width is set by the
window and the model's context, not by the hop. So a classifier cannot localise, at any hop, and
sliding it faster only buys sample density while looking like precision.

## Finding 2 — rise time separates cough from breath with no model at all

9-17 ms and 45-49 dB for the coughs; 60-127 ms and 20-29 dB for the breaths. The separation is
physiological — a cough is an explosive release against a closed glottis, a breath is turbulent flow —
and it is available from the envelope alone.

## Finding 3 — breath duration is not measurable, and neither is any offset here

Moving the offset threshold from floor+12 dB to floor+3 dB moves the breath offset by **2.03 s** and
**1.76 s**. The coughs carry 1.04-1.10 s of offset ambiguity. Any breath duration reported from this
file describes the threshold, not the breath. This is the same shape as the phonation-offset problem
in D12: for turbulent and aspirate events the offset is definitional, and a single-threshold rule
reports a choice as a measurement.

## Finding 4 — `pyannote/segmentation-3.0` calls the coughs speech

P(speech) saturates at 1.0 across [7.898, 10.226], covering both coughs, while Brouhaha's VAD stays
near 0.01 there and fires only on the real utterance. Brouhaha is right: SQUIM STOI is 0.18-0.44
across that region and two independent Whisper models transcribe nothing there. A cough's second
phase is voiced human sound carrying speaker identity, which is exactly what a speaker-segmentation
model is built to fire on. Used as a VAD it produces a 2.3 s false speech span, on a file whose real
speech is 1.5 s.

## Finding 5 — AST and YAMNet disagree sharply on the same event

YAMNet: `Cough` 1.000. AST: `Throat clearing` 0.93-0.96, `Cough` 0.11. Given a 9 ms rise and a 48 dB
step, YAMNet is right. Two consequences: the correlation risk accepted in D6 does not show up here —
these two failed differently, which is what makes them two families — and the taxonomy cannot assume
its confusable classes are separable by classifier vote, because on this file they are not.

## Finding 6 — periodicity measures are unavailable outside speech

Praat HNR returns nan nearly everywhere, with valid values only at the two cough onsets. pyin rails at
its 60 Hz floor through the quiet stretches, locking onto low-frequency rumble. Any design leaning on
HNR or F0 as a general vocal-evidence family must account for their being undefined wherever there is
no periodic content — which is most of an airway-branch recording.

## Also measured

No background talkers on this file: pyin voicing probability never exceeds 0.31 outside the utterance
and segmentation-3.0 shows no second speaker. 81.7% of energy sits below 1 kHz, consistent with
proximity effect. Stationary tones at 85.0, 108.4, 164.1, 1564.5 and 1757.8 Hz. Clipped fraction 0.000.

`pyannote/voice-activity-detection` is **gated (403)** for this account, so the dedicated VAD pipeline
could not run; raw `segmentation-3.0` frame posteriors were substituted, which is what surfaced
Finding 4.

---

# Extraction and HeAR on the same recording — measured, 2026-08-18

Nine separation/enhancement checkpoints and Google's HeAR, all pinned to commit SHAs. Intended for
Engaging; a `monthly_maint` reservation covered 1423 nodes with no GPU node outside it, so it ran
locally instead — the file is 14 s and the largest model took 13 s.

## Finding 7 — every SepFormer checkpoint fails on this recording

Five checkpoints (`sepformer-whamr16k`, `-wsj02mix`, `-libri2mix`, `-dns4-16k-enhancement`,
`-wham16k-enhancement`). Their streams explain only 8-50% of input energy while emitting **10-27 dB
more energy than the input**, at zero lag, with Whisper reading the same sentence off both streams.
That is duplication, not separation, and the residual after least-squares fitting is indistinguishable
from the original. Peak-normalising the input first changed nothing to within 0.04 dB.

This matters beyond the probe: `speech_enhancement/` currently wires
`speechbrain/sepformer-wham16k-enhancement`, which is in the failing set, and that model is what the
existing `speech_enhancement` perturbation applies.

## Finding 8 — enhancement sorts cough and breath differently, and the split is usable

Energy retained per event, relative to the input, streams least-squares gain-fitted:

| model | breaths | coughs | speech |
| --- | --- | --- | --- |
| `MossFormer2_SE_48K` | **−39 to −45 dB** | **−0 to −1 dB** | −0 dB |
| `MossFormerGAN_SE_16K` | −51 dB | −42 to −53 dB (2 of 4) | ~0 dB |
| `FRCRN_SE_16K` | −0 to −13 dB | −0 to −5 dB | 0 dB |
| `MossFormer2_SS_16K` | split stream 1 / residual | split alternately between streams | stream 1 |

`MossFormer2_SE_48K` destroys breaths and keeps coughs. Every breath lands whole in the residual,
where the detector scores Breathe = 1.00. So breath **is** recoverable from what enhancement leaves
behind — but cough is not, because cough survives into the speech stream. D8 assumed a single
residual would carry all non-speech vocal material; it does not, and the two elements need different
routes.

`MossFormer2_SS_16K` assigns each cough burst to whichever stream is free rather than isolating cough
as a class, so a 2-source separator is not a class decomposer.

## Finding 9 — HeAR needs 2 s of real context, and padding destroys it

Declared input is 2 s mono 16 kHz to a 512-d embedding. It **silently accepts shorter input** — no
error, no padding, no NaN, at every length from 0.01 s to 4 s — so the static shape is not enforced
and a caller can feed it a 0.3 s cough and get a plausible-looking vector back.

Length and framing then dominate content:

- **Padding versus real context**: centred cosine between the same event under different framings runs
  0.0-0.5, and `native|real_context` ranges −0.21 to +0.26, against a class margin of ~0.9. Padding a
  0.3 s cough out to 2 s moves its embedding about as far as substituting unrelated audio.
- **Window shift is benign**: ±50-200 ms gives 0.93-0.98. So a boundary error of 100 ms costs almost
  nothing, while the padding decision costs an order of magnitude more.
- **Amplitude invariant**: gains from ×0.1 to ×10 give cosine 1.0000.

Minimum usable length, from fixed-length crops of real audio centred on 13 events, as centred
within-minus-between class margin and leave-one-out nearest-neighbour accuracy over 4 classes
(chance 0.19):

| duration | 0.10 s | 0.15 s | 0.30 s | 0.50 s | 1.0 s | 1.5 s | 2.0 s | 3.0 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| margin | +0.12 | +0.32 | +0.29 | +0.28 | +0.46 | +0.81 | **+0.91** | +0.67 |
| LOO-NN | 0.46 | 0.77 | 0.62 | 0.62 | 0.77 | **0.85** | **0.85** | 0.77 |

A 0.3 s cough retains about a third of the separation available at 2 s; 3 s is worse than 2 s. The
elements do separate cleanly at 2 s — within-class +0.653, between-class −0.256, LOO-NN 0.846 — but
**only after mean-centring**: raw cosines are 0.977 within and 0.918 between, which would report
everything as similar to everything.

HeAR's own detector found a quiet breath at 6.60-7.10 s that had been hand-labelled silence.

---

# The configured enhancement model, tested across 17 recordings — 2026-08-18

## Two corrections to the earlier finding first

**The "10-27 dB more energy than the input" figure was inflated by the harness.** Both the earlier
scripts and the first pass of this test wrote streams with soundfile's default WAV subtype, PCM_16,
while SepFormer output routinely peaks at 2-30 — so up to 26% of samples were clipped at write time.
Re-run with `subtype="FLOAT"`. On the original quiet recording the distortion was small (residual
−0.83 → −1.22 dB) so that observation survives, but the numbers on louder recordings were wrong.

**The energy inflation is real, universal and harmless in the shipped pipeline.** SepFormer is exactly
scale-equivariant — a 50 dB input gain sweep reproduces the output waveform to corr 1.0000 and the
energy ratio to 0.05 dB — and SpeechBrain's own `separate_file()` peak-normalises. `speechbrain.py`'s
attenuate-only normalisation brings the repo's output to a median −1.7 dB against the input. Running
the shipped `enhance_audios()` reproduces a direct call exactly. **The workflow does not propagate a
blow-up**, and the earlier suggestion that it might was wrong.

## The real defect: an output-fidelity ceiling that makes the default net-harmful

Against 13 synthetic mixtures with kept clean references, output SI-SDR in dB:

| input | input SI-SDR | wham16k-enh (repo default) | whamr16k | dns4-16k-enh | MossFormer2 | FRCRN |
| --- | --- | --- | --- | --- | --- | --- |
| clean speech | ∞ | **4.79** | 3.03 | 13.34 | 11.39 | 11.83 |
| +20 dB SNR | 18.4 | **4.69** | 3.27 | 13.12 | 11.29 | 11.54 |
| +10 dB SNR | 8.5 | **4.17** | 2.43 | 10.40 | 9.86 | 9.85 |
| +5 dB SNR | 3.5 | **3.50** | 1.94 | 8.27 | 8.01 | 7.92 |
| 0 dB SNR | −1.5 | **2.43** | 0.44 | 5.75 | 5.88 | 5.87 |
| −5 dB SNR | −6.5 | **−1.00** | −0.65 | 3.36 | 3.48 | 3.71 |

`sepformer-wham16k-enhancement`'s output is pinned near 4.8 dB SI-SDR **however clean the input is**.
As improvement, it is **net-harmful at every input SNR above ≈5 dB**: −13.8 dB at 20 dB SNR, −4.3 dB
at 10 dB, break-even at 5 dB, +3.9 dB at 0 dB. The model is not broken — it denoises correctly inside
its WHAM training distribution — it simply cannot pass clean speech through.

**Stated testably:** the configured default degrades any recording whose speech is already cleaner
than roughly 5 dB SNR. On the assembled corpus SepFormer explained ≥3 dB less of the input than the
best control on **13 of 17 recordings**; the four exceptions are the ones with the most background
energy, including SepFormer's own WHAM demo file, where it matches the controls to 0.2 dB. **All four
quiet `streaming-audio-*` captures — the workflow's actual input class — meet the harmful condition.**

Audible consequence, Whisper on level-normalised output: the repo default turns "There's something
going on." into "Something going on", and substitutes "Ranger" for "And Josh" in the tutorial clip.
`whamr16k` duplicates the same sentence into both streams on non-overlapping input. FRCRN reproduces
both verbatim.

**Not the cause, all tested:** input level (0.000 dB effect over 50 dB of gain), sample rate and
resampling path (within 0.5 dB), duration (flat 1-20 s; only <0.5 s collapses), speech-to-silence
ratio. Digital silence yields exact zeros, but a −70 dBFS dither floor is amplified by +26.8 dB.

**The ceiling is checkpoint-specific, not architectural**: `sepformer-dns4-16k-enhancement` reaches
13.3 dB with the same architecture and harness — though it carries the largest scale inflation of all,
+48 dB, which the pipeline's normalisation would absorb.

Controls `MossFormer2_SE_48K` and `FRCRN_SE_16K` conserve energy on every recording, stay in phase at
zero lag, and leave clean speech essentially untouched (−46 and −62 dB residual on a clean two-talker
conversation).

**Untested and would change the conclusion:** a normalisation the SpeechBrain recipe applies at
inference that neither `separate_batch` nor `separate_file` applies. Also untested: reverberant
mixtures (no RIR applied), the 8 kHz variants, MossFormerGAN, and DriftSE.
