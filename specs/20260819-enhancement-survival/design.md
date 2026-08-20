# What survives enhancement, read by HeAR and YAMNet

## Why this exists

Two comparisons already existed and neither answers the question the triage work needs.

The **SI-SDR sweep** measures speech fidelity against a clean reference: `sepformer-wham16k-enhancement`
sits near 4.8 dB however clean the input, making it net-harmful above roughly 5 dB input SNR. That is a
statement about speech, and the airway elements are exactly what SI-SDR is indifferent to.

The **preserve/destroy matrix** measures per-element energy and legibility, but at **one condition**, and
its legibility reader was CrisperWhisper. Energy and legibility already disagree there — `FRCRN_SE_16K`
keeps breath energy at −2.0 dB while CrisperWhisper stops annotating breaths at all — so neither is a
proxy for the other, and neither is a proxy for what HeAR or YAMNet will report.

This measures the missing cell: **per-element survival across the SNR sweep, read by the two models the
triage design would actually use.**

## Design

One input: `streaming-audio-2026-07-30T04-21-56-487Z.wav`, the only human-verified recording, with six
labelled events and seven verified-empty stretches.

Six SNR conditions (as captured, +20, +10, +5, 0, −5 dB white noise, seeded). Both readers score every
condition on each verified event **and** on the verified-empty stretches, so *preserving* an element is
separated from *inventing* one — an enhancer that raises `Cough` inside silence is not preserving a
cough.

**Separation is included, scored per output channel.** A separator's channel is a candidate like any
other output, and the old matrix's most interesting finding was a separator behaving as an element
filter (`MossFormer2_SS_16K` src1 keeping cough at −1.2 dB while everything else fell 31–50 dB).
Averaging channels would hide exactly that, so each is its own row.

unasdiff is run in `speech_sound` mode twice: conditioned on `Cough`, and on `Computer_keyboard` as a
control. The control is the measurement — a prior run found the conditioning label **inert**, with the
unrelated class matching `Cough` to 0.4%, so the separation came from the speech-prior/sound-prior
asymmetry rather than the label. That run predates PR #564, which fixed a PCM_16 worker hand-off and a
dropped device, so it is repeated here rather than cited.

## Three silent-zero traps, found before running

Each of these produces a confident wrong table rather than an error, which is why they are recorded:

1. **HeAR emits `label_scores`, not `scores`.** The first draft read the wrong key. Every score would
   have been 0.0 — indistinguishable from "every enhancer destroyed every element".
2. **HeAR refuses audio shorter than 2 s**, and every verified event is shorter. Clipping to the event
   window would have failed all six. Both readers therefore run over the whole recording and take the
   peak over the windows overlapping each event, which is also how the models are meant to be used.
3. **YAMNet's `top_k` defaults to 5.** `Breathing` falling outside the top five reads as 0.0. The run
   requests all 521 labels, verified as 521 per window.

## Defects this surfaced

**DriftSE's default variant cannot load.** `_DRIFTSE_DEFAULT_VARIANT` is
`distillhubert_three_layers_with_z`, whose checkpoint is **nf = 64**, while both configs in the pinned
code clone at `0a489dadfa27` are **nf = 128**. Every layer mismatches by exactly 2×
(`[64, 2, 3, 3]` against `[128, 2, 3, 3]`), and no nf=64 config exists at that commit. So
`enhance_audios_with_driftse()` with no variant raises a state_dict size mismatch. `DriftSE_v2`
(`pesq_sisdr_ccmse_with_z`) loads and runs.

This also unsettles a prior result: the model-to-branch matrix reports v1 measurements ("hallucinates
content; output peaks at 51 741×"), which cannot have come from this checkpoint/config pair. Either the
weights mirror moved upstream or what was labelled v1 was something else. Not resolvable from the run
logs, and not guessed at here.

**`sepformer-whamr16k` is a separator, and the guard catches it.** It returns 2 sources and is refused
by `_single_source` (PR #569). That retroactively voids its column in the SI-SDR sweep, which was
measured when the backend flattened `(batch, samples, sources)` with `reshape(1, -1)` — so those figures
were computed on two sources interleaved sample-by-sample, not on audio. It read as "consistently the
worst enhancer".

**There is no SpeechBrain separation backend.** `separate_audios` dispatches to ClearVoice and unasdiff
only. PR #569's error message tells the caller to use `senselab.audio.tasks.source_separation` for a
separation checkpoint, and that module cannot load one. The advice points nowhere, which is why
`sepformer-whamr16k` and `sepformer-wsj02mix` are absent from the separation arm rather than chosen
against.

## Cluster notes

Run on ORCD (`pi_satra`, 1× A100). Three environmental lessons, each of which cost a job:

- **`$SCRATCH` is not set in a Slurm batch shell.** The shared-locations recipe assumes it is exported;
  it must be resolved with `readlink -f ~/orcd/scratch`, whose target is sharded per user.
- **Venvs and the uv cache must not live on SCRATCH.** Unpacking one 782 MB torch wheel exhausted the
  **1 M inode cap** — 984.2K files used, 98.4%, with half the space free. It surfaces as
  `Disk quota exceeded (os error 122)` mid-extract, and `df` reports hundreds of free terabytes. They
  live on POOL, which was empty.
- Every enhancer refuses audio at another rate, so the input is resampled per row, and a
  **rate-matched unenhanced baseline** is emitted for each rate present rather than comparing a 16 kHz
  output against a 48 kHz reference.

No timing or memory figures are reported: other work shared the host, and a contended resource number
is worse than none.

## Correction: HeAR needs ~160 ms of a cough, not 40 ms

The branch-1 draft chose HeAR as *confirmer* on the premise that "40 ms suffices to fire it, so very
few false negatives". Measured directly, that is wrong by about 4x.

A 40 ms rectangular window swept across the recording at a 20 ms hop, each window embedded alone at
the centre of an otherwise-silent 2 s buffer so the response is attributable to those 40 ms:
**not one of the 700 positions exceeds 0.5.** Breathe peaks at 0.307 at the 2.275 s breath onset,
Cough at 0.077, Speech at 0.023. See `hear_sweep_40ms.png`.

The zero-padding is not what suppresses it. Holding the same construction and growing only the
excerpt:

| event | 40 ms | 80 ms | 160 ms | 320 ms | 640 ms | 1280 ms | 2000 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| breath @2.275 | 0.026 | 0.138 | 0.462 | 0.859 | 0.922 | 0.957 | 0.997 |
| breath @5.308 | 0.026 | 0.469 | 0.725 | 0.832 | 0.926 | 0.980 | 0.980 |
| cough @7.926 | 0.063 | 0.220 | **0.999** | 0.995 | 0.994 | 0.994 | 0.992 |
| cough @9.610 | 0.049 | 0.049 | **0.996** | 0.992 | 0.989 | 0.991 | 0.652 |
| speech @11.62 | 0.009 | 0.004 | 0.002 | 0.009 | 0.234 | 0.433 | 0.200 |

160 ms of cough in a silent buffer scores 0.999, so a burst surrounded by digital silence is well
within what the model responds to. The threshold is a **cliff between 80 and 160 ms** for cough, and
between 160 and 320 ms for breath. Below it the model is not merely uncertain, it is flat.

Two consequences.

**The original claim was measured on a different construction.** "40 ms anywhere inside the window"
must have held the remaining ~1.96 s as real recording, in which case what fired the detector cannot
be attributed to the 40 ms. Either reading damages the confirmer argument: if 40 ms of event plus
1.96 s of context fires it, the response is a property of the context.

**Speech never crosses 0.5 at any duration**, peaking at 0.433. That is consistent with the
enhancement sweep, where HeAR scored the verified speech event 0.348 unenhanced while YAMNet gave
0.994, and with enhancement *raising* HeAR's speech score above the unenhanced value.

`cough @9.610` falling from 0.991 at 1280 ms to 0.652 at 2000 ms is the merging effect from the other
direction: the last 720 ms admits the following quiet stretch and dilutes the response.

## At 160 ms, HeAR localises and stops merging — the draft's claim is wrong

Repeating the sweep with a 160 ms window (the duration at which cough crosses to 0.999) at the same
20 ms hop, same silent-buffer construction. See `hear_sweep_160ms.png`.

`Cough` fires in two sharp plateaus, ~7.98-8.28 s and ~9.66-10.02 s, peaking at **0.998**, and stays
below 0.09 everywhere else — 24 windows above 0.5, all inside the two verified coughs.

**The two coughs are 1.1 s apart.** At the 2 s input they merge into one plateau, which is what the
branch-1 draft concluded from:

> `group_events` cannot be solved by better thresholding of HeAR's posterior, because events closer
> than 2 s are merged before any threshold sees them. The information is not in the track.

That is refuted. The information is in the model; it was absent from the track because of the input
length. Merging is a property of feeding HeAR 2 s of continuous recording, not of HeAR. A short
excerpt in a silent buffer both localises the event and separates neighbours the 2 s window cannot.

Secondary observations from the same sweep:

- **Breathe is weaker and raggeder than cough** — 0.845 at 2.36 s, 0.80 at 5.38 s, 15 windows above
  0.5 — consistent with breath needing 160-320 ms where cough needs 160.
- **Three consecutive windows above 0.5 at 3.94-4.02 s, peaking 0.63, inside verified-empty audio.**
  Either a false positive or an unlabelled inhalation. The same ambiguity was resolved by ear at
  6.60-7.10 s, where HeAR's Breathe 0.49 was wrong; this one is unresolved and wants a listen.
- **Speech is flat at <=0.009 across all 694 windows.** With the 0.348 on the full verified event and
  the duration table never crossing 0.5, that is three independent measurements agreeing that HeAR
  does not report this speech.

## All eight classes at 500 ms: Snore is the second-most-active class, and every hit is false

A 500 ms window at 100 ms hop, 136 positions, rastered over all eight detector labels rather than the
four plotted before. See `hear_raster_500ms.png`.

| class | peak | windows >0.5 | where |
| --- | --- | --- | --- |
| Cough | 1.000 | 12 | all 12 inside the two verified coughs |
| Breathe | 0.928 | 13 | breath_1, breath_2, and 4.05-4.25 |
| **Snore** | **0.864** | **16** | **all 16 false** |
| Throat Clear | 0.722 | 1 | 12.45 s, inside speech |
| Baby Cough | 0.637 | 1 | 8.45 s, inside the adult cough |
| Laugh | 0.102 | 0 | — |
| Sneeze | 0.092 | 0 | — |
| Speech | 0.080 | 0 | — |

**`Snore` produces more detections than `Cough` and there is no snoring in the recording.** It fires
at 1.15, 3.05, 3.45, 3.85-4.15, 7.45-7.55, 8.75 and 12.55-12.75 s — in verified-empty audio, inside
breath_1, and inside the speech span. Restricting earlier plots to four labels hid this entirely, and
it is the strongest evidence against reading these posteriors as corroboration: the model's most
active class after cough is one with no instances present.

Two smaller confusions, both timed correctly: `Baby Cough` 0.637 on an adult cough, and
`Throat Clear` 0.722 inside speech.

**A candidate unlabelled event at 3.9-4.25 s.** `Breathe` exceeds 0.5 there at 160 ms (3 windows),
320 ms and 500 ms, and `Snore` peaks at the same place. Four window lengths and two classes agree on a
stretch the branch-1 document lists as verified-empty. Either the list is wrong at that point or both
classes share a false positive; it wants a human listen, and until then the empty-stretch list should
not be treated as verified there.
