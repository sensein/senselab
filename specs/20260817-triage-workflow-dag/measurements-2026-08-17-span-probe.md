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

---

# unasdiff's conditioning vocabulary, and its licence — 2026-08-18

Answered without inference, from `source_separation/data/fsd41_classes.json` and upstream.

## 41 classes, and only four are vocal

`Cough`, `Laughter`, `Burping_or_eructation`, `Fart`. Nothing else human.

**Absent:** `Breathing`, `Sneeze`, `Sigh`, `Gasp`, `Whispering`, `Crying`, `Screaming`,
`Throat clearing`, `Singing`. Speech is the separate prior, so it is not in this list by design.

**Nineteen of the 41 are musical instruments** — acoustic guitar, bass drum, cello, chime, clarinet,
cowbell, double bass, electric piano, flute, glockenspiel, gong, harmonica, hi-hat, oboe, saxophone,
snare drum, tambourine, trumpet, violin. The remainder are domestic and mechanical events (applause,
fireworks, gunshot, knock, shatter, squeak, tearing, telephone, writing, scissors, keys jangling,
computer keyboard, drawer, microwave, bus) plus two animals (bark, meow).

This is the DCASE / FSD Kaggle 2018 41-class set. `unasdiff.py`'s description of the prior as
"FSD50K-conditioned" is loose: FSD50K carries ~200 classes including most of the vocal ones missing
here.

## What that settles

**Cough is conditionable, breath is not.** So `speech_sound` conditioned on `Cough` can plausibly pull
a cough out **as a named source**, which no supervised separator tested here can do — SepFormer
duplicated one voice across two streams, and MossFormer2's two-source mode assigned cough bursts to
whichever slot was free. Naming is the capability that matters, because a named channel is
simultaneously a detector and a span.

**Breath has no conditioning slot**, so it stays on the DSP-envelope and HeAR route. That is now forced
by three independent findings: neither VAD responds to it, enhancement puts it in a residual shared with
music, and the only conditioned separator in the repo cannot name it.

**Laughter is conditionable**, which covers one of D11's always-measured confounds.

**The music has named channels.** With 19 instrument classes and music confirmed in the probe recording,
the tonal content is separable by name rather than being an unlabelled residual — which is what made
the D8 residual route ambiguous. Whether the specific background music here matches any of the 19 is
unmeasured.

**As a general taxonomy instrument it is narrow.** Four vocal classes out of 41, with the vocabulary
weighted toward instruments, means this cannot carry the taxonomy — it is a targeted tool for the
classes it happens to hold.

## Licence: our note is accurate, unlike DriftSE's

Upstream `RunwuShi/unasdiff` reports `license: null`, has **no LICENSE file**, and its **only issue is
open** — the same "Request: an explicit license (and optionally a HuggingFace weights mirror)" that was
filed for DriftSE, here unanswered. Repository last updated 2026-06-03. So `unasdiff.py`'s and
`doc.md`'s "mirror is public, with the licence still unknown" is correct and stays; the DriftSE
parallel does not hold, and the guess that it might be stale was wrong.

---

# DriftSE returns unnormalised output — verified defect, 2026-08-18

Measured peaks of each model's returned waveform, and the fraction of samples a PCM_16 write would clip:

| model | output peak | samples clipped by PCM_16 | clipping error energy re signal |
| --- | --- | --- | --- |
| **DriftSE v1** (`distillhubert_three_layers_with_z`, senselab's default) | **51 741.7** | **98.47%** | −0.0 dB |
| DriftSE v2 (`..._pesq_sisdr_ccmse_with_z`) | 0.996 | 0.0% | −65.8 dB |
| `sepformer-dns4-16k-enhancement` | 14.01 | 8.71% | −3.18 dB |
| `sepformer-libri2mix` src0 | 34.44 | 2.16% | −1.47 dB |
| `sepformer-wsj02mix` src1 | 21.16 | 1.06% | −2.11 dB |
| `sepformer-wham16k-enhancement` (repo default) | 5.43 | 0.08% | −9.32 dB |
| all ClearerVoice models | ≤ 0.95 | 0.0% | ≤ −53.7 dB |

## The defect

`speech_enhancement/speechbrain.py:165-177` peak-normalises the enhanced waveform back to the input's
peak, and its comment states why: SpeechBrain enhancement models produce arbitrarily-scaled output,
and downstream consumers that assume [−1, +1] — PPG and wav2vec2 CTC alignment, mel-spectrogram
extractors without internal normalisation, openSMILE LLDs — degrade on out-of-distribution amplitudes.

`speech_enhancement/driftse.py` does **not** do this. Its only normalisation, at `:193`, divides the
*input* by its own peak before the STFT and never restores the scale, so the returned waveform is in
the model's internal scale. DriftSE was added on 2026-08-13, after the SpeechBrain backend had already
recorded the lesson.

Consequence, from the preserve/destroy matrix: DriftSE v1 through a PCM_16 write shows cough 1 and
cough 2 **destroyed at −21.8 and −25.2 dB**, where the same output written as FLOAT keeps them at +0.3
and −0.4 dB. The clipping is silent — nothing raises, and the waveform looks bounded afterwards.

For v2 the peak is 0.996, so the defect is **latent rather than absent**: it survives by luck of scale,
not by design.

## Two things this also settles

**The earlier PCM_16 harness bug was material, not theoretical.** `dns4-16k-enhancement` lost 8.71% of
its samples with clipping error only 3.18 dB below the signal; `libri2mix` src0 2.16% at −1.47 dB;
`wsj02mix` src1 1.06% at −2.11 dB. Any earlier figure for those three is wrong, not merely imprecise.

**The fix is precedented and local**: apply `speechbrain.py`'s approach — peak-normalise back to the
input's peak, above a silence threshold, recording the applied gain so the operation is reversible and
auditable. It belongs in `driftse.py` before the branch is merged.

---

# unasdiff, run — the label is inert where it looked decisive, 2026-08-18

Four runs through `separate_audios`, `n_sources=2`, seed 17, CPU, **60 diffusion steps**, on one 4 s
window 9.5-13.5 s holding cough 2 and the whole speech span. Upstream's 200 steps was attempted twice
and lost both times. Timings deliberately not reported — the machine was under heavy contention.

## The control kills the naming claim

| run | mode | cough 2 share into the sound slot |
| --- | --- | --- |
| C1 | `speech_sound`, conditioned `Cough` | **0.971** |
| C2 | `speech_sound`, conditioned **`Computer_keyboard`** | **0.967** |

A class with no relation to a cough performs the same. In `speech_sound` the separation is produced by
the **speech-prior versus sound-prior asymmetry**, not by the conditioning label — which is inert. The
wrong label was in one respect cleaner: 0.01% speech leakage into the sound slot against `Cough`'s 6.8%.

**This refutes the earlier reading** that unasdiff could "name a cough". It routes non-speech into the
sound slot. It does not identify what it routed.

## Where the label does act — `sound_sound`, unconfirmed

Both slots use the same sound prior, so only the conditioning differs. Cough 2: `Cough` slot −0.11 dB,
`Electric_piano` slot −53.39 dB, share **1.0000 / 0.0000**. Speech window: 0.0001 / 0.9999 into the
instrument slot. The music followed the instrument slot on four of five partials (0.95-0.9998), the
exception being 1757.8 Hz at 0.78 into the `Cough` slot.

A 53 dB separation from nothing but an integer index. **Caveat, from the agent and not closed:** the
slot-swapped pair (`[Electric_piano, Cough]`) was not run, so label-driven and slot-order-driven remain
formally unseparated. That single control decides whether this capability is real.

`speech_speech` is not a decomposition: slot 0 takes 98.4% of the cough, and the speech span splits
37/63 with no interpretable structure — as upstream's README warns.

## No configuration preserves intelligible speech

The unseparated 4 s window transcribes the sentence. **None of the eight separated streams does.**
C1 speech slot → `Oh.` + `[laughter]`; C2 speech slot → `Yeah.` + `Sentence of the light.` Meanwhile
every configuration preserved the cough well enough for CrisperWhisper to name it — including B1's
instrument slot, which holds 99.99% of the speech-window *energy* and transcribes as `[breath]`: the
energy is there and the words are not.

Confounded with the 60-step reduction; upstream's 200 was not reached.

## Two defects in shipped senselab code

**`unasdiff.py:726` discards an entire run on timeout.** `subprocess.run(..., timeout=3600)` is
hard-coded, and `TimeoutExpired` loses everything — no partial output survives. This killed a 200-step
attempt outright.

**The worker writes intermediate per-window files at soundfile's default PCM_16 subtype**, so any sample
beyond ±1 is clipped before the host reads it back. Peaks were 0.053-0.958 here so nothing clipped, but
this is the same exposure that invalidated an earlier measurement harness, now in library code rather
than in a probe.

Also undocumented: `diffusion_steps` has a usable range of roughly **52-200**. Below 52 the sampler
fails inside `q_sample` (it calls `t=t_last-50`); `1` fails earlier in `GaussianDiffusion.__init__`.

## A metric being asked the wrong question

SQUIM objective (STOI, PESQ, SI-SDR) is reference-free; subjective MOS needs a non-matching reference.
Applied to these streams the numbers are not interpretable: **MOS 4.259 on a stream containing one
isolated cough**, against 3.058 for the input. They are speech-quality estimators, and most of these
streams are not speech.

Pins: upstream `RunwuShi/unasdiff` @ `5a5d70cd…`; weights `sensein/unasdiff-diffusion-priors` @
`8d7c3220…`; torch 2.6.0 in the isolated venv.

---

# Retraction: DriftSE's plain checkpoint is not a hallucinating model — 2026-08-18

Established from the paper (arXiv 2604.24199), upstream's code at the pinned commit, and both issues.

**Our worker omits upstream's output rescale.** `driftse.py:208` returns `x * norm`. Upstream's
`enhancement.py:224-229` computes `x_hat / x_hat.abs().max() * norm_factor` — rescale by the output's
own peak, then back to the input's. Those lines have been present since upstream's first commit; our
worker never had them, so this is not drift we failed to track.

**The plain checkpoint's gain is undefined by its training, and that is expected.** Its config sets
`pesq_weight`, `sisdr_weight` and `ccmse_weight` to 0 with `latent_drift_weight: 1.0`, and
`train.py:297-304` standardises the generated waveform to zero mean and unit variance before the only
active loss term. The objective is exactly scale-invariant. Measured raw ISTFT peaks: **4 400 and
4 896** on two clips — content-dependent, so an earlier observation of 51 741 is the same phenomenon,
not a different one.

**The † checkpoint masks the bug.** Its CCMSE and PESQ terms are computed on absolute amplitudes
(`train.py:292-294`), pinning its raw peak at 0.997. Hence:

| checkpoint | raw ISTFT peak | upstream rescale | our `x * norm` | ours after the PCM_16 write |
| --- | --- | --- | --- | --- |
| plain `..._with_z` | **4 399.99** | peak 0.572, corr **0.9950** | peak 2 517.8, corr 0.9950 | peak 1.0, corr **0.6284**, 91.8% of samples clipped |
| † `..._pesq_sisdr_ccmse_with_z` | 0.9968 | peak 0.572, corr 0.9997 | peak 0.570, corr 0.9997 | corr 0.9997, 0% clipped |

The missing rescale is a scalar gain and cannot by itself change correlation. What destroys the signal
is the scalar gain **followed by a PCM_16 write**, which hard-clips a ~2 500× overshoot into a square
wave. That is what an ASR read as fabricated text.

**Corroboration from upstream's own release:** the published output WAVs for *both* variants have peak
**0.513519** — bit-identical peaks from models whose internal scales differ by ~4 000×. That is the
signature of the rescale we drop.

**So "the hallucinating variant" is withdrawn.** With upstream's line restored the plain checkpoint
gives correlation 0.9950, and at 1 NFE it scores PESQ 3.00 / SI-SDR 15.8 — beating a 30-step SGMSE+
baseline on PESQ. It is a working enhancer that our integration was breaking.

## Two further corrections

**`driftse.py:276-277` calls the default variant "the paper's headline model". It is not.** The
headline PESQ 3.15 / SI-SDR 16.1 figure is the σ=0 `no_z` checkpoint, **which was never released**.
Relatedly, `with_z`/`no_z` is not the paper's conditional-generator versus direct-mapping split — it is
only `train_add_gaussian` true/false. The conditional generator (`backbones/ncsnpp_v2.py`) is unreleased
and nothing shipped uses it.

**Per-chunk normalisation is a second defect behind the first.** Our 20 s overlap-add chunking
normalises each chunk by its own peak (`driftse.py:216-233`), so even once the rescale is restored,
chunk boundaries will track the local peak envelope rather than upstream's per-file peak matching. And
`if seg.shape[-1] < n_fft: break` (`:222`) silently drops a trailing remainder shorter than 510 samples
(~32 ms), leaving that tail at zero.

Everything else in the worker is faithful line for line: input normalisation, STFT geometry,
`spec_fwd`/`spec_bwd`, `pad_spec`, `t = ones(B)`, σ=0.01, the ema→model→raw priority, ISTFT
`length=T_orig`, and the 16 kHz target.

---

# unasdiff at full steps on Engaging — the naming capability is closed, 2026-08-18

Exclusive A100 node, 200 diffusion steps, full 14.027 s file, all modes.

## The slot-swap control refutes the label effect

`sound_sound [Electric_piano, Cough]` against `sound_sound [Cough, Electric_piano]`, identical
otherwise. Measured on **raw per-window sampler outputs** rather than stitched streams, because
`align_permutations` margins were 0.03-0.25 — inside the band `data/permutation_alignment.json` itself
calls ambiguous, so a stitched answer would have been confounded by the chunker.

| event | `[E.piano, Cough]` slot0 / slot1 | `[Cough, E.piano]` slot0 / slot1 |
| --- | --- | --- |
| cough 1 | **−0.31** / −36.68 | **−0.32** / −37.20 |
| cough 2 | **−0.07** / −52.23 | **−0.06** / −33.37 |

Both coughs land in slot 0 whatever slot 0 is labelled. Across 53 comparable cells: 24 label-consistent,
29 slot-consistent, organised by **window** rather than by content — w5 is 7/0 label-consistent, w2 7/1,
while w3, w4 and w6 are entirely slot-consistent. The fitting picture is two near-identical
decompositions with a per-window binary indeterminacy about which slot a component emerges in.

**So unasdiff cannot name a source.** `speech_sound` was already shown inert by a matched control
(`Computer_keyboard` 96.7% vs `Cough` 97.1%); `sound_sound` is ordering. The capability that made this
backend interesting for the taxonomy does not exist.

## Retraction: the 60-step "no configuration preserves speech" finding

At 200 steps CrisperWhisper recovers "There's something going on." with correct word timings from five
streams. The earlier result was a step-count artefact, not a property of the method. Speech-window
energy sits at −0.36 dB and −0.01 dB in the speech-prior slot.

Label behaviour at 200 steps is not inert but does not act in its own direction either: `Cough`
conditioning put cough 1 at −16.99 dB in the **speech** slot against −40.14 in the Cough slot, while
`Computer_keyboard` put it at −1.34 dB in the sound slot. Caveat carried from the report: in two runs
both slots sit 17-18 dB below the input for cough 1, so a high share there is a share of very little.

Instrument conditioning does **not** extract the music: the 164.1, 1564.5 and 1757.8 Hz partials sit at
0.990, 0.967 and 0.991 share in the **speech-prior** slot.

## A third defect, upstream and verified

`models/atten_unet.py:6` and `diffusion/gaussian_diffusion.py:34` both execute
`os.environ["CUDA_VISIBLE_DEVICES"] = "0"` **at module import**. Four workers launched with distinct
device assignments — verified in `/proc/<pid>/environ` — all ran on physical GPU 0. **unasdiff cannot be
fanned out across GPUs on a node.** senselab's worker already avoids upstream's benchmark scripts
because of `torch.cuda.set_device(0)`; the library modules it does import carry the same pin.

## GPU reproduces CPU, and PCM_16 bit a third time

91 paired window cells, median |Δ| **0.09 dB**, max 2.21 dB on a −37 dB cell; no preserve/destroy
verdict changes. Outputs are **byte-identical** across A100 80GB PCIe and A100-SXM4-80GB.

The first GPU pass reused a script writing at soundfile's default PCM_16, clipped up to 8.9% of samples
on three SepFormer streams, and disagreed with CPU by as much as 9.5 dB. Third occurrence of that
default silently corrupting a measurement in this session.

In unasdiff's own worker, all 84 per-window files are PCM_16 (the defect stands) but the largest sample
across 126 files is 0.9949 with zero at full scale — the worker's per-window peak normalisation keeps it
in range for this recording, so the exposure did not fire.

## The first valid timing of the session

Exclusive node, 128/128 CPUs, 4/4 A100-SXM4-80GB, `OverSubscribe=NO`, no other job present, runs serial:
G1 560.71 s, G2 559.38 s, G3 576.67 s — RTF ≈ **40×** on 14.027 s, ≈0.40 s per window-step. The job was
preempted during G4, so G4-G6 carry no valid timing and none is claimed.

## My CrisperWhisper reference is the outlier

Three independent input routes on the cluster — scipy, senselab decode, senselab from samples — produce
**byte-identical** token sequences: `[cough] 7.92-8.08` plus the four speech tokens with timings matching
mine exactly. **No `[breath]` tokens, no `[UH]`, and the cough ends at 8.08 rather than 8.48.**

So the discrepancy is not a resampling artefact and not a harness difference between two machines: the
five-token sequence recorded earlier in this file is the one that does not reproduce, on either the CPU
harness or the cluster. Two consequences. The earlier scoring of CrisperWhisper's breath coverage
(26.2% and 10.2%) rests on tokens that other runs do not produce. And `span_refine` in
`branch-1-airway.md`, which consumes CrisperWhisper token edges as span candidates, has a much weaker
input than assumed — reliably `[cough]` and little else.

### Correction: the CUDA pin is one site, not two, and the fix is ours

Read directly rather than taken from the report above.

**`models/atten_unet.py:6` is the defect** — unconditional, module level, and placed **before that
file's own `import torch` on line 7**, so it executes at import and overwrites whatever the launcher
set.

**`diffusion/gaussian_diffusion.py:34` is not at module import.** It sits inside
`load_spk_model(config_path, model_filename, device=None)` guarded by `if device is None:` — a
default-gathering branch that an explicit device bypasses. Our worker never calls `load_spk_model`, so
it does not apply to us at all. The earlier claim that both fire at import was wrong.

**Why it bites us specifically**, from `source_separation/unasdiff.py:320-328`:

```python
import torch                          # :320  — imported first, but CUDA is not initialised yet
import models                         # :325  — triggers atten_unet.py:6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # :328 — first CUDA call
```

CUDA initialises lazily at line 328, by which time the variable has been overwritten, so it enumerates
physical GPU 0 only — and line 328 requests `"cuda"` with no index, taking visible device 0 regardless.

**The fix is two lines on our side**: capture `CUDA_VISIBLE_DEVICES` before line 325 and restore it
after line 326. No CUDA API has been touched at that point, so restoring before line 328 defeats the
pin entirely. Selecting an explicit `cuda:N` would additionally make the intent visible.

Worth recording for its own sake: the comment at `:323-324` states that this worker avoids upstream's
three `test_*.py` scripts because they call `torch.cuda.set_device(0)` at import, and a module-level
test checks this file for those substrings. The hazard class was understood and guarded against by one
mechanism, while a library module the worker does import carried the same pin by another.

**And senselab does not select a device for unasdiff either.** `separate_with_unasdiff` accepts
`device`, passes it to `_select_device_and_dtype` for validation, and **discards the return value** — it
never reaches the worker payload. The docstring states this ("accepted for signature parity ... the
worker selects CUDA when available and CPU otherwise"), so it is a documented limitation, not a silent
bug. The worker then requests `torch.device("cuda")` with no index.

So two independent things block GPU selection, and both must be fixed for multi-GPU fan-out: our API
drops the caller's device, and upstream's import-time pin overwrites whatever the launcher set. Fixing
either alone changes nothing. The cluster run is the case that wants it — four workers on a four-GPU
node all ran on GPU 0.
