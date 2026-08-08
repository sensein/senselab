# L1 signal contract: report measurements, not interpretations

**Status:** observations from end-to-end runs on `english_conversation_higgs_audio_v2.wav`
(21.5 s, 2 speakers) and `audio_48khz_mono_16bits.wav` (4.9 s, 5 named speakers), 2026-07-31.

## The single root cause

Every L1 defect found in this sequence is the same architectural error: **L1 applied an
interpretation instead of reporting what the tool measured.** Each interpretation happened to
be a reduction that *saturates independently of the input*, so each one destroyed the signal
before anything downstream could weight it.

| signal | what L1 emitted | what the tool produced | measured symptom |
|---|---|---|---|
| `frame_segmentation` | `max` over speaker channels | per-speaker activation probabilities | posterior **exactly 1.0000** in all 1070 buckets |
| `frame_brouhaha_vad` | same | same | min 0.7383, median 1.0 |
| `acoustic_loudness` | percentile rank within the file | loudness in dB | ~10% of frames pinned at 0, ~25% at 1.0 *by construction*; quiet frames read as loud |
| `acoustic_spectral_activity` | percentile rank | spectral flux | same |
| every presence voter | `speaks: bool` at a 0.5 threshold | a continuous score | information discarded before L2 sees it |
| support evidence pooling | `max` over signals | per-signal values | support inert at 0.967–1.000 for every claimant |

These are not six bugs. They are one mistake surfacing six times, and it will keep surfacing
until L1 stops interpreting.

**Corollary that cost real time:** these defects were only findable by *rendering the figure and
looking at it*. A signal declaring `units: "LUFS"` arriving at −70 is obviously quiet; the same
signal arriving as `0.97` after a percentile transform is not obviously anything. Making units
explicit turns a class of silent bug into an inspectable one.

## The contract

L1 emits, per signal, per its own native resolution:

- the **value in its native units** — no thresholding, no normalisation, no rescaling to `[0,1]`
- a **provenance JSON** travelling with the signal

```json
{
  "signal": "frame_segmentation",
  "model": "pyannote/segmentation-3.0",
  "revision": "<hf revision or null>",
  "units": "probability",
  "value_range": [0.0, 1.0],
  "resolution_s": 0.017,
  "window_s": 0.017,
  "reduction": "noisy_or_over_speaker_channels",
  "backend": "pyannote-audio",
  "status": "ok"
}
```

`units` is load-bearing. Without it L2 cannot know whether `0.7` is a probability, a dB value or
a rank — which is precisely how a rank came to be aggregated as though it were a probability.

`reduction` is equally load-bearing: it records what L1 *did* do, so a saturating reduction is
visible in the output rather than only in a plot.

Per-signal native resolutions established by measurement:

| signal | resolution | window | note |
|---|---|---|---|
| `frame_segmentation`, `frame_brouhaha_vad` | 17 ms | 17 ms | pyannote receptive-field step |
| `Loudness_sma3`, `spectralFlux_sma3` | 10 ms | 20 ms (50% overlap) | openSMILE eGeMAPSv02 LLD |
| `HNRdBACF_sma3nz` | 10 ms | 60 ms (83% overlap) | same |
| `yamnet` | 480 ms | 960 ms | |
| `ast` | 10.24 s | 10.24 s | finer when run windowed; hop wins over the table |
| diarization | segment boundaries | — | not a fixed grid |

## What moves where

**Out of L1:** all thresholding (`speaks`), all normalisation (percentile bands), all
`[0,1]` rescaling, all fixed dB→confidence ramps. The `loudness_confidence` ramp added on
2026-07-31 is itself an instance of the error being fixed and moves out with the rest.

**Into L2:** calibration against a versioned profile, thresholding, and resolution conversion —
coarser→finer holds (a 10.24 s decision applies across its window; interpolating invents detail
the model never produced), finer→coarser integrates (point-sampling a 17 ms posterior at 250 ms
discards fourteen of every fifteen measurements and which survives is arbitrary).

**Unchanged:** HNR's fixed 2–10 dB anchors are already absolute, so HNR was never affected.

## Signals as pipelines

A signal must return the best available form of itself, not the first form it has.

Established by measurement: CrisperWhisper and Qwen3-ASR **do** carry native word timings
(`chunks[].start`, 62 words each) and were correctly skipped by the aligner. The defect was a
consumer reading only `alignment.by_model`, so it showed words for the one text-only model and
none for the two that had them. `resolve_asr_result` — which returns whichever result carries
usable timings — already existed and was not being called.

Generalised: an ASR backend's postcondition is *word timings present*, satisfied natively or by
alignment. The consumer should never have to know which path produced them.

## Whole-signal vs chunked inference

`Inference(model, window="whole")` was set for Brouhaha, but the parent split the audio into
overlapping 10 s chunks first — so it was whole-*chunk*. Brouhaha estimates SNR and C50, and a
chunked estimate is normalised inside its own chunk: a stretch that is quiet relative to the
recording reads as ordinary when it is the loudest thing in its own window. Chunking is now a
memory fallback (`_MAX_WHOLE_SIGNAL_S = 600`), not the default path.

## Cache durability

The YAMNet TF-Hub cache defaulted to `$TMPDIR`. A partial download left a directory without
`saved_model.pb`, and TF-Hub reuses it forever because the directory exists — this removed
YAMNet from a run entirely with no indication the cause was a bad download rather than the
audio. Cache moved to `~/.cache/senselab/tfhub`; a corrupt entry is treated as a cache miss.

**Generalisation:** a model load failure must be distinguishable in the output from a model that
was never configured. A signal that ran and failed keeps its row, marked failed.

## Open items

1. `acoustic_spectral_activity` still percentile-normalised. Flux has no absolute reference the
   way loudness does; the anchor should be flux relative to the measured noise floor, which
   `noise_floor.py` already computes.
2. Diarization rows render one colour. **Not yet diagnosed** — unknown whether each model
   genuinely labels a single cluster or the lookup fails. Asserting "mechanism in place" without
   checking was an error.
3. Resolution table not yet wired into the harvest; signals are still pre-averaged onto the
   250 ms grid before L2 sees them.
4. Brouhaha produces no output on the current build — under investigation.
5. Every remaining L1 reduction should be audited for input-independent saturation rather than
   waiting for each to surface in a figure.
