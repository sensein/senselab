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
