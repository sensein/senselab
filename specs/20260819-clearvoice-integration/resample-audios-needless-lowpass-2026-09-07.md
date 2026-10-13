# `resample_audios` low-passes same-rate audio, unconditionally, on every ClearerVoice call

**Confirmed.** `resample_audios` has no same-rate short-circuit: every call designs a Butterworth
low-pass at `resample_rate/2 - 100` Hz and runs `sosfiltfilt` over every channel, whether or not
`torchaudio.transforms.Resample` will do anything. `prepare_audios_for_clearvoice` calls it
unconditionally, so a 16 kHz input handed to `FRCRN_SE_16K`, `MossFormerGAN_SE_16K`,
`MossFormer2_SS_16K` or `AV_MossFormer2_TSE_16K` — four of the six ClearerVoice checkpoints
(`design.md` §1) — is low-passed at 7900 Hz before the model ever sees it, for no reason connected
to any actual rate change. b2ai recordings are commonly already 16 kHz, so this is the common case,
not an edge case.

The waveform-level perturbation this filter causes is minuscule on the four files measured here
(correlation ≥0.999, most ≥0.999999). But on the one file whose content sits at FRCRN's
already-documented null/pass-through decision boundary (`residual-without-speech-2026-09-08.md`), a
perturbation that small is enough to move the model's output by double-digit percentage points of
energy fraction — and it exactly reproduces a 40%-relative divergence between "senselab" and
"upstream IO" numbers that a prior investigation in this tree (`frcrn-torch-vs-arch-2026-09-08.md`)
measured, flagged as undocumented, and could not explain. This spec explains it: it is this filter,
not the `decode()`-vs-batch bypass that report suspected.

## 1. The claim, checked against the code

`prepare_audios_for_clearvoice` calls `resample_audios` with no rate check:

```
src/senselab/audio/tasks/clearvoice.py:38    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
src/senselab/audio/tasks/clearvoice.py:40    return downmix_audios_to_mono(resample_audios(audios, resample_rate=spec.sampling_rate))
```

`resample_audios` itself has no same-rate branch — the low-pass design and the `sosfiltfilt` call
run for every audio, every call, regardless of `audio.sampling_rate`:

```
src/senselab/audio/tasks/preprocessing/preprocessing.py:73    for audio in audios:
src/senselab/audio/tasks/preprocessing/preprocessing.py:78        _lowcut = lowcut if lowcut is not None else (resample_rate / 2 - 100.0)
src/senselab/audio/tasks/preprocessing/preprocessing.py:79        sos = signal.butter(order, _lowcut, btype="low", output="sos", fs=resample_rate)
src/senselab/audio/tasks/preprocessing/preprocessing.py:82        resampler = Resample(orig_freq=audio.sampling_rate, new_freq=resample_rate)
src/senselab/audio/tasks/preprocessing/preprocessing.py:86            filtered = signal.sosfiltfilt(sos, ch.detach().cpu().numpy()).copy()
```

`torchaudio.transforms.Resample(orig_freq=X, new_freq=X)` at line 82 is an identity in the
`orig_freq == new_freq` case; the `sosfiltfilt` at line 86 is not. There is no branch anywhere in the
function that skips the filter design or the filter application when `audio.sampling_rate ==
resample_rate`.

**No test asserts same-rate is a no-op.** `test_resample_audios`
(`src/tests/audio/tasks/preprocessing_test.py:21-36`) only checks the output sample *count*; its
fixtures resample a 48 kHz file to 16 kHz
(`src/tests/audio/conftest.py:23,41-43`, confirmed via `soundfile.info` — native rate 48000 Hz), so
the same-rate path is never exercised by any test in `src/tests/`.

## 2. Signal effect, model-free

`resample_audios([audio], resample_rate=16000)` against the unmodified input, for four files
verified natively 16 kHz by `soundfile.info` (not assumed):

| file | native rate | samples in → out | correlation | total energy ratio | E>7900 Hz, in | E>7900 Hz, out |
|---|---|---|---|---|---|---|
| `s3_breath__original.wav` | 16000 Hz | 317649 → 317649 | 0.99999986 | 0.99999962 | 0.00000030 | 0.00000000 |
| `s3_hardcough__original.wav` | 16000 Hz | 39382 → 39382 | 0.99998559 | 0.99996432 | 0.00003300 | 0.00000080 |
| `s1_fivebreaths1__original.wav` | 16000 Hz | 329167 → 329167 | 0.99999110 | 0.99997867 | 0.00002130 | 0.00000030 |
| `00_original.wav` (buzz separation) | 16000 Hz | 408672 → 408672 | 0.99911519 | 0.99770828 | 0.00198770 | 0.00004420 |

Sample count is unchanged in every case (an identity resample, as expected). The filter does what
its design says: whatever energy sat above 7900 Hz is removed almost completely (a 46-99% reduction
of an already-small band, down to a further 4-97% of what was there). But on three of these four
recordings that band held under 0.003% of the file's total energy to start with, so the *waveform*
effect is close to imperceptible — correlation ≥0.99999 on three of four files. `00_original.wav`
carries more genuine high-frequency content (a buzz/interference tone) and shows the largest
waveform-level change of the four: correlation 0.9991, 0.23% of total energy removed.

**On these four files, the filter's direct waveform perturbation is real but small.** The next
section shows that "small" does not mean "harmless" once FRCRN is in the loop.

## 3. Downstream effect: FRCRN_SE_16K, with and without the needless filter

Ran `alibabasglab/FRCRN_SE_16K` via `senselab.utils.clearvoice.run_clearvoice_audio` twice per file —
`raw` bypasses `prepare_audios_for_clearvoice` entirely (the input is saved and handed to the worker
as-is, since it is already at the checkpoint's 16 kHz); `processed` is the actual production path,
i.e. `raw` fed through one `resample_audios(..., resample_rate=16000)` pass first, matching exactly
what `enhance_audios(...)` does today for every already-16kHz input. Both outputs are related back to
the same unfiltered reference with `senselab.audio.tasks.speech_enhancement.residual.compute_residual`
(`max_lag_ms=20`), the module this repository already uses for this exact measurement in
`workflows/triage/nodes/preprocess.py`.

| file | condition | enhanced-energy fraction | residual-energy fraction | corr(input, enhanced) | gain (dB) |
|---|---|---|---|---|---|
| `s3_breath` | raw (no filter) | 0.1214 | 0.4104 | 0.7680 | 6.86 |
| `s3_breath` | processed (production) | 0.0671 | 0.5724 | 0.6543 | 8.05 |
| `s3_hardcough` | raw (no filter) | 0.9815 | 0.00077 | 0.9996 | 0.08 |
| `s3_hardcough` | processed (production) | 0.9654 | 0.00154 | 0.9992 | 0.15 |
| `s1_fivebreaths1` | raw (no filter) | 0.00142 | 0.9383 | 0.2485 | 16.38 |
| `s1_fivebreaths1` | processed (production) | 0.00181 | 0.9364 | 0.2522 | 15.45 |
| `00_original` | raw (no filter) | 0.9802 | 0.01424 | 0.9929 | 0.02 |
| `00_original` | processed (production) | 0.9780 | 0.01574 | 0.9921 | 0.03 |

**`s3_breath` moves by 16.2 percentage points of residual-energy fraction (0.4104 → 0.5724, +39%
relative) and the enhanced/input correlation drops from 0.768 to 0.654 — the largest change of the
four, and it lands on the one file whose content (a 20 s breath recording, duration-filling rather
than a brief burst) `residual-without-speech-2026-09-08.md` already documented as sitting at FRCRN's
null/pass-through decision boundary** ("no stable middle ground" for this recording shape). The other
three files, none of which sit at that boundary, move by ≤0.2 percentage points.

**This resolves a previously open, previously undocumented divergence in this repository's own
records.** `frcrn-torch-vs-arch-2026-09-08.md` measured, for the same recording, upstream's own
supported IO path giving a residual-energy fraction of **0.4105** against senselab's
`net.decode()`-bypass giving **0.5724** — flagged there as "differs by 40% relative... undocumented.
It should be an explicit decision rather than an accident," with the `decode()`-vs-batch-path
difference as the suspected (untested) cause. This spec's `raw` condition, measured independently
here and by a different method (bypassing `prepare_audios_for_clearvoice` rather than bypassing
`decode()`), gives **0.4104** — matching upstream's IO path to four significant figures. Upstream's
reader never calls `resample_audios`, so it never sees the low-pass; senselab's production path
always does. The gap is this filter, not the decode() bypass.

**This also refines, rather than contradicts, this repository's earlier verdict that the filter "is
not the mechanism."** `residual-without-speech-2026-09-08.md`'s 2026-09-07 update compared `raw`
(one filter pass, via `enhance_audios`'s own internal `prepare_audios_for_clearvoice`) against
`plain` (two filter passes: `workflows/triage/nodes/preprocess.py:419` applies one pass building the
`plain` stream, then `enhance_audios` applies a second) and found them to "agree to within noise"
(`s3_breath`: 0.5724 one pass vs 0.5928 two passes, +0.0204). That comparison is correct as far as it
goes, but it never tested the true zero-filter baseline — both of its conditions already carry at
least one pass. The 16.2-point jump measured here, from zero passes to one, is eight times larger
than the 2.0-point jump from one pass to two. The physical picture is consistent: a fourth-order
Butterworth low-pass removes nearly all removable energy above its cutoff on its first application,
so a second, equally needless pass has almost nothing left to remove — which is exactly why the
prior comparison (1 pass vs 2) read as noise, while the comparison that matters (0 pass vs 1) is not.

**Null/pass-through classification does not flip on these four files.**
`residual-without-speech-2026-09-08.md` documented a clean gap for FRCRN's non-speech behaviour:
pass-through recordings sit at 0.15-1.6% residual-energy fraction, null recordings at 57-94%, "nothing
falls in between." Both conditions measured here stay on their documented side for every file:
`s3_hardcough` (0.08% / 0.15%) and `00_original` (1.42% / 1.57%) stay pass-through;
`s1_fivebreaths1` (93.8% / 93.6%) and `s3_breath` (41.0% / 57.2%) stay null. `s3_breath`'s `raw`
value (41.0%) sits below every previously-measured null case (57-94%) and below `s3_breath`'s own
previously-reported number — it is the most pass-through-like null recording measured across both
this session and the prior ones, which matters if a threshold is ever reintroduced near that gap.
`00_original`'s `processed` value (1.574%) sits closer to the documented pass-through ceiling
(≤1.6%) than its `raw` value (1.424%) — for a file nearer that boundary than this one, the filter's
effect could plausibly cross it. No gate is currently configured to test this against (removed per
`config-derivations.md`'s `residual` section), so "does not flip" is reported for these four files as
measured, not as a guarantee for every recording.

## 4. Scope: every `resample_audios` call site

| call site | guarded? | same-rate plausible in normal use |
|---|---|---|
| `src/senselab/audio/tasks/clearvoice.py:40` (`prepare_audios_for_clearvoice`, every ClearerVoice capability) | No | Yes — 4 of 6 checkpoints are 16 kHz; b2ai audio is commonly already 16 kHz |
| `src/senselab/audio/tasks/classification/huggingface.py:210-211` | Yes (`if audio.sampling_rate != expected_sampling_rate:`) | N/A — guarded |
| `src/senselab/audio/tasks/classification/yamnet.py:175` | No | Yes — YAMNet is 16 kHz native |
| `src/senselab/audio/tasks/classification/speech_emotion_recognition/api.py:801-802` | Yes (`if audio.sampling_rate != expected_sr:`) | N/A — guarded |
| `src/senselab/audio/tasks/speech_to_text/granite.py:106-109` | Yes (`if audio.sampling_rate != 16000:`) | N/A — guarded |
| `src/senselab/audio/tasks/features_extraction/sparc.py:273-277` | Partial — only calls when `any(a.sampling_rate != expected_sr for a in audios)`, but then resamples **every** audio in that batch, including already-correct ones | Yes, for the already-correct members of a mixed-rate batch |
| `src/senselab/audio/tasks/health_acoustics/hear.py:482-486` (`prepare_audio_for_hear`) | Yes (`if audio.sampling_rate == HEAR_SAMPLING_RATE: return audio`) | N/A — guarded |
| `src/senselab/audio/tasks/source_separation/unasdiff.py:885` | No | Yes — target is 16 kHz |
| `src/senselab/audio/tasks/speech_enhancement/driftse.py:431` | No | Yes — target is 16 kHz |
| `src/senselab/audio/tasks/audio_understanding/audio_flamingo.py:240-243` | Yes (`if audio.sampling_rate != TARGET_SAMPLING_RATE:`) | N/A — guarded |
| `src/senselab/audio/workflows/explore_conversation.py:80` | No | Yes — hardcoded `target_sr = 16000` |
| `src/senselab/audio/workflows/audio_analysis/adaptive/plot.py:631` | No | Yes, but output feeds a plot only, not a model |
| `src/senselab/audio/workflows/audio_analysis/adaptive/audio_io.py:73` | Yes (`if audio.sampling_rate != TARGET_SR:`) | N/A — guarded |
| `src/senselab/audio/workflows/triage/nodes/preprocess.py:419` (`[plain] = resample_audios([mono], target_hz)`) | No | Yes — `target_hz: 16000` in `data/config/default.yaml:19`, whose own comment says "YAMNet, HeAR, AST and the recognizers are all native here" |
| `src/senselab/audio/workflows/triage/nodes/preprocess.py:1674` | Yes (`if int(enhanced.sampling_rate) != target_hz:`) | N/A — guarded |

Seven of fifteen call sites are unguarded, and every one of the seven runs on a hardcoded or commonly-
already-matching target rate: 16000 for six of them. The most consequential is
`workflows/triage/nodes/preprocess.py:419`: for a 16 kHz b2ai recording, the `plain` reference stream
that every triage measurement is computed against is already filtered once before FRCRN sees it via
`enhance_audios`'s own second, independent pass through the same needless filter (§3's "refines"
paragraph) — the double-application this repository already measured and correctly judged
negligible, sitting on top of the single, larger, unmeasured-until-now application both passes share.

`design.md`'s own checkpoint table (§1) lists four 16 kHz checkpoints of six
(`FRCRN_SE_16K`, `MossFormerGAN_SE_16K`, `MossFormer2_SS_16K`, `AV_MossFormer2_TSE_16K`); the other
two (`MossFormer2_SE_48K`, `MossFormer2_SR_48K`) run at 48 kHz, where a 16 kHz b2ai recording would
need a genuine resample and the filter is doing legitimate anti-aliasing work.

## 5. What the fix would be (not implemented here)

A same-rate short-circuit in `resample_audios`
(`src/senselab/audio/tasks/preprocessing/preprocessing.py:73-92`): when `audio.sampling_rate ==
resample_rate`, skip both the filter design (line 78-79) and the `sosfiltfilt` call (line 86), not
just the resampler (which is already an identity in that case via `torchaudio`). This is a change to
`preprocessing.py`, which this investigation was scoped not to make. It would also let
`test_resample_audios` (`src/tests/audio/tasks/preprocessing_test.py:21-36`) gain a same-rate case
asserting bit-for-bit (or near-bit-for-bit) identity, which no test currently exercises.

## Reproduction

- Signal-effect table (§2): `resample_audios` called directly against `Audio(filepath=...)` for each
  file, compared with `numpy`/`scipy.fft` — no model, no subprocess venv.
- Downstream table (§3): `senselab.utils.clearvoice.run_clearvoice_audio` called directly with
  `alibabasglab/FRCRN_SE_16K`, `device=DeviceType.CPU`, bypassing `run_clearvoice_over_audios`'s call
  to `prepare_audios_for_clearvoice` for the `raw` condition, and with one `resample_audios` pass
  first for `processed`. Both conditions related to the same unfiltered reference via
  `senselab.audio.tasks.speech_enhancement.residual.compute_residual(..., max_lag_ms=20)`. The
  pipeline's own config (`workflows/triage/data/config/default.yaml:210`) uses `max_lag_ms: 200.0`;
  every row measured here found `lag_samples == 0`, so the narrower window changes nothing reported.
  Checkpoint commit `3766e6a64b0d8cb58f08d913d617bf129f11ed53`, the same one every other measurement
  in this tree resolves to.
- All four files verified 16 kHz mono PCM_16 via `soundfile.info` before use, not assumed.
