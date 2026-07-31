# L1 post-processing register: every interpretation to be moved to L2

Companion to `l1-signal-contract.md` (the principle) and `layered-architecture.md` (the full
five-layer design and the decisions D-1 – D-15 governing it). This one is the complete list of
places the current code violates the principle, so each can be reviewed and moved deliberately
rather than in bulk.

Decisions from `layered-architecture.md` that dispose of entries below: **D-3** drops items 8–9's
signals outright in favour of absolute replacements; **D-5** requires item 16's channels intact;
**D-9** determines what items 1–2 become once softened by measured reliability rather than voting
`(1, 0)`.

**Why one-by-one rather than a sweep.** Every entry below is a decision someone made for a reason,
and the reason is usually still valid — it is the *layer* that is wrong, not the arithmetic. Moving
a threshold to L2 unexamined would carry its unstated assumptions along with it. Several entries
turn out not to survive review at all (see `quality_uncertainty`, item 21), and a bulk move would
have preserved them.

## What counts as post-processing

A step is post-processing, and therefore L2's, if it does any of:

- **Thresholds** a continuous measurement into a decision (`>= 0.5`, `speaks: bool`).
- **Rescales** to `[0, 1]` against an anchor — fixed (`(25 - snr)/20`) or data-derived (percentile
  rank). The anchor is a claim about what counts as normal, which is calibration.
- **Reduces** across a dimension the tool reported separately (max/noisy-or over speaker channels,
  mean over frames, argmax over 527 labels).
- **Selects** among estimators (`brouhaha else mean of DSP`), which is fusion.
- **Inverts** a measurement into a badness score (`1 - rolloff/nyquist`, `1 - no_speech_prob`).
- **Resamples** onto a reporting grid that is not the tool's native resolution.

A step is *not* post-processing if it recovers the measurement the model actually produced:
applying the model's own trained output activation, converting sample indices to seconds,
stitching sliding-window chunks back into one timeline, or naming units.

## Evidence that the layering is the defect

Brouhaha's three heads, measured directly on synthetic inputs whose correct answers differ
maximally (`/tmp/bprobe`, CPU, `BrouhahaInference`):

| input | VAD median | SNR median | C50 median |
|---|---|---|---|
| digital silence, 8 s | 0.0001 | 43.5 dB | 31.1 dB |
| white noise, −40 dBFS, 8 s | 0.0028 | −5.0 dB | 22.9 dB |
| clean TTS speech, 21.5 s | 0.9980 | 70.1 dB | 59.8 dB |
| 4 s speech + 4 s silence | speech half **0.9981** / silence half **0.0167** | — | — |

Within-file speech-vs-silence discrimination is **+0.9815**. The heads are sound and have full
dynamic range — 75 dB of SNR span across inputs. What reached the parquet was `quality_snr = 0.0`
in every bucket, because `clip((25 − 70.1)/20, 0, 1)` is zero. **The measurement was never the
problem; the clamp on top of it destroyed a working signal.**

A corrected earlier finding, recorded because it shaped several decisions: an intermediate
analysis reported brouhaha's VAD as having −0.003 discrimination and being "unusable". That test
compared loud against quiet *frames within continuous speech* and used dBFS as ground truth for
speech presence. Short inter-word gaps are not non-speech, and a frame VAD with 6 s of LSTM
context is correct to read through them. The signal was fine; the test was wrong.

## The register

Status: `open` = still in L1. Each entry names the L1 replacement — what the tool should emit
instead — and the L2 question the moved step becomes.

### Presence voters (`presence.py`) — the whole module is post-processing

`harvest_presence_votes` returns `{"speaks": bool, "native_confidence": float}` per model. Both
fields are conclusions. Per the governing instruction *"there is no presence at L1 just good
signals"*, this module has no L1 role: it should become an L2 stage reading L1 measurements.

| # | site | current post-processing | L1 should emit | L2 question | status |
|---|---|---|---|---|---|
| 1 | diarization voters | `diar_speaks_in_window` → `speaks` bool | segments `(start_s, end_s, speaker_label)` per model | does any segment cover this bucket, and whose? | open |
| 2 | ASR token overlap | `token_overlaps_window` → `speaks` bool | words `(start_s, end_s, text)` | does a word land here? | open |
| 3 | ASR hallucination gate | `speaks and not (nsp >= 0.5)` — threshold + override | `no_speech_prob` per segment, raw | is a transcript over probable silence trustworthy? | open |
| 4 | Whisper confidence | `avg_logprob` → bucket `native_confidence` | `avg_logprob` per segment, units log-probability | how does token logprob map to belief? | open |
| 5 | `::no_speech_prob` voter | `speaks = nsp < 0.5`, `nc = 1 − nsp` — threshold + inversion | same raw field as #3 (one measurement, not two voters) | same as #3 | open |
| 6 | AST | top-1 argmax over 527 labels, then `label in speech_labels` | full label→score map per native window | which categories are present, and is any of them speech? | open |
| 7 | YAMNet | top-1 argmax over 521 labels, then `label in speech_labels` | full label→score map per 0.48 s hop | same as #6 | open |
| 8 | `acoustic_loudness` | per-pass **percentile band** p10→p75 → `[0,1]` → direction flip | `Loudness_sma3` per 10 ms frame + absolute `LUFS` | what loudness counts as audible here? | open |
| 9 | `acoustic_spectral_activity` | per-pass percentile band on `spectralFlux_sma3` | raw flux per 10 ms frame | what flux counts as non-stationary, relative to the measured noise floor? | open |
| 10 | `acoustic_hnr` | fixed 2→10 dB ramp; low maps to `p = 0.5` (abstain) | `HNRdBACF_sma3nz` in dB per 10 ms frame | what HNR indicates voicing, and when is it uninformative? | open |
| 11 | `ppg_voice_fraction` | per-frame argmax, count `!= "<silent>"`, ÷ n, then `>= 0.5` | PPG posterior frames (or argmax label + its probability) | what fraction of non-silent frames means speech? | open |
| 12 | `embedding_silhouette` | cluster all windows, silhouette coefficient, `>= 0.5` | embedding vectors per window | does clustering support a coherent speaker here? | open |
| 13 | frame posteriors | bucket-mean over frames, then `>= 0.5` | per-frame posterior at native 17 ms | how do frames aggregate to a bucket? | open |
| 14 | `frame_instability` | `clip(2 × mean(per-bucket std), 0, 1)` — ×2 rescale + clip | per-frame values | how is temporal dispersion measured? | open |
| 15 | coarse-voter demotion | `weight = 0.25` when grid < 0.5 s | native window/hop in provenance | how should resolution mismatch be weighted? | open |
| 16 | segmentation-3.0 reduction | noisy-or over per-speaker channels (was `max`) | per-speaker activation matrix, channels intact | how do per-speaker activations combine? | open |

Item 16 is the sharpest case, and its diagnosis changed once measured. The model reports one
activation per speaker, and the collapse returned **exactly 1.0000 in all 1070 buckets** on the
Higgs conversation — but the cause was the output format being *misidentified*, not the choice of
reduction. `segmentation-3.0` declares `powerset=True` while pyannote 4.x returns per-speaker
columns; a single active speaker makes those rows sum to 1.0, so the row-sum heuristic took the
powerset branch and computed `1 − data[:, 0]`, treating speaker#1 as the no-speaker class. Reading
the declaration against the output width instead took discrimination from 0.0000 to **+0.9364** on
a speech-then-silence probe. Closed; see D-5 in `layered-architecture.md` for the measurements.

Items 1 and 12 together are why diarization rows render one colour: L1 never emits the labels, so
nothing downstream can distinguish speakers.

### Scene quality (`quality.py`)

| # | site | current post-processing | L1 should emit | L2 question | status |
|---|---|---|---|---|---|
| 17 | `quality_snr` | `clip((25 − snr_db)/20, 0, 1)` → **0.0 in every bucket** | `brouhaha_snr_db`, units dB | what SNR counts as clean for this task? | open |
| 18 | `quality_reverb` | `clip((30 − c50_db)/35, 0, 1)` → **0.0 in every bucket** | `brouhaha_c50_db`, units dB | what C50 counts as dry? | open |
| 19 | `quality_bandwidth` | `clip(1 − rolloff/nyquist, 0, 1)` — inversion | `rolloff_95_hz`, units hertz | is this band-limited for the sample rate? | open |
| 20 | `quality_clip` | `clip(proportion_clipped, 0, 1)`, renamed as degradation | `proportion_clipped`, units proportion | how much clipping matters? | open |
| 21 | `quality_uncertainty` | `clip(std(snr_estimates)/15, 0, 1)` | each estimator separately, own name and units | **does not survive review — see below** | open |
| 22 | `primary_snr_db` | brouhaha, else mean of DSP estimators | all estimators, unreduced | which estimator to trust where? | open |
| 23 | silence gate | `rms < 1e-4` → nulls all quality columns | `rms`, units dBFS or linear | where is quality undefined? | open |
| 24 | grid broadcast | nearest-analysis-window value copied to each reporting bucket | values at native 0.5 s / 0.25 s analysis resolution | how to resample to the reporting grid? | open |

**Item 21 fails on its own terms, not just on layering.** It takes the standard deviation of
brouhaha's SNR head, `spectral_gating_snr_metric`, and `peak_snr_from_spectral_metric` — three
quantities that are not the same measurement. Brouhaha reads ~70 dB on clean TTS where the DSP
metrics use different noise-floor definitions entirely, so their spread reflects *definitional
disagreement*, not measurement uncertainty. Divided by a 15 dB reference it pins at 1.0
structurally, and it would do so on perfect audio. Per `statistics.py`, `variability` is the
dispersion of *repeated measurements of one quantity*; these are three different quantities. This
is not a variability estimate and should not be re-derived at L2 under a new name. What is worth
keeping is the underlying observation — that the estimators disagree — reported as the estimators
themselves (item 22), letting L2 decide whether disagreement is informative.

Item 24 interacts with the resolution work: `resolution.py` exists and is not yet wired in, so
quality is currently broadcast rather than resampled.

## Cross-cutting

- **Nothing emits through `signal.measurement(...)` yet.** The provenance envelope exists
  (`signal.py`: units vocabulary, model, revision, resolution, window, reduction, backend, status)
  and is unused. Until L1 emits through it, "what units is this?" has no answer in the data, only
  in the code that produced it.
- **`resolution.py` is not wired in.** Declared native resolutions (17 ms frame signals, 10 ms
  acoustic, 480 ms YAMNet, 10.24 s AST) and `resample_series` are implemented but unused; L1 still
  reports on the reporting grid.
- **`acoustic.py` LUFS is not wired into `presence.py`**, and its `loudness_confidence` dB ramp is
  itself an L1 interpretation (item 8's replacement) that belongs at L2.
- **`scene_quality_coupling` is null in all 1070 rows** — separate defect, not a layering issue;
  tracked in `l1-signal-contract.md` open items.
- **Fixtures.** Clean TTS cannot validate SNR/C50 because it genuinely sits at 70 dB / 59.8 dB.
  Degraded fixtures exist at `/tmp/bprobe/` (`noisy_{0,10,20,40}.wav`, `reverb_{0.3,0.8}.wav`) and
  should become checked-in test fixtures, since the useful range of items 17–18 is only observable
  on them.

## Review order

Grouped so each group can be validated by one measurement rather than a full e2e run.

1. **Items 17–24, scene quality.** Self-contained, and the degraded fixtures make the fix
   verifiable immediately. Item 21 is a deletion, not a move.
2. **Item 16, then 13–14, frame signals.** Unblocks per-speaker presence and identity; the
   saturation is measurable on the existing run.
3. **Items 6–7, scene classifiers.** Keeping full posteriors also serves background
   characterization, which currently re-derives what argmax threw away.
4. **Items 8–10, acoustic.** Requires deciding the absolute anchor: LUFS for loudness, measured
   noise floor from `noise_floor.py` for flux. HNR already has a defensible absolute anchor.
5. **Items 1–5, 11–12, 15.** Largest surface: dissolves `presence.py` into an L2 stage. Do last,
   once the signals it consumes are emitting raw.
