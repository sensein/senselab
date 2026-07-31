# Speech-presence certainty for the `audio_analysis` uncertainty workflow

**Scope.** How to quantify *certainty of speech* in short temporal segments for
`src/senselab/audio/workflows/audio_analysis`, in the context of the Bridge2AI-Voice
recordings (speech-focused, plus cough/breathing tasks) processed via b2aiprep/senselab.
This is a handoff note; work continues in Claude Code against this same clone.

Context anchors already read:
- `src/senselab/audio/workflows/audio_analysis/presence.py` (the presence-axis harvester)
- `src/senselab/audio/workflows/audio_analysis/grid.py` (`BucketGrid`, default 0.5 s / 0.5 s)
- `specs/20260508-173136-compare-uncertainty/spec.md` (3-axis design: presence / identity / utterance)
- `src/senselab/audio/tasks/voice_activity_detection/` (pyannote-only VAD today)
- `src/senselab/audio/tasks/ssl_embeddings/` (wav2vec2 / HuBERT / WavLM, frame-native)

---

## 1. Reframing: "best model" → "best temporal resolution + calibration"

The user reported that off-the-shelf sound-event/tagging models "don't work well enough
for brief events." At short segment lengths the bottleneck is **not tagger accuracy** — it
is that most voters currently feeding the presence axis cannot resolve a brief event at
all, because their native time resolution is coarser than the bucket.

Audit of the current `harvest_presence_votes` voter set against a short (~50 ms) bucket:

| Voter (current) | Native resolution | Usable at ~50 ms? | Failure mode |
|---|---|---|---|
| openSMILE `Loudness/spectralFlux/HNR` | 10 ms hop | **Yes** | none — already continuous |
| PPG voice-fraction | ~10–20 ms frames | **Yes** | none |
| pyannote diarization (`diar_speaks_in_window`) | 16.7 ms posterior, **thresholded to segments upstream** | only if posterior used | hangover / `min_duration_on` smoothing erases sub-100 ms onsets |
| Sortformer | frame-level, post-processed to segments | posterior not exposed | segmentization loss; no native confidence |
| ASR token-overlap (Whisper/Granite/Canary/Qwen) | word→sentence (100s ms) | **No** | one token spans ~10 buckets |
| Whisper `no_speech_prob` | per ~30 s segment | **No** | constant across ~600 buckets |
| AST Speech-allowlist | 10.24 s window | **No** | one vote spans ~200 buckets |
| YAMNet Speech-allowlist | ~0.96 s window | **No** | one vote spans ~20 buckets |
| embedding silhouette | speaker-embedding window (~1 s+) | **No** | coarse |

Two consequences at fine grids, both explaining the "doesn't work for brief events" report:

1. **Coarse voters poison the entropy collapse.** AST, YAMNet, `no_speech_prob`, and
   sentence-level ASR cast an *identical* vote across every bucket they cover. That
   inflates apparent agreement (deflates Shannon entropy) around a brief event and smears
   the boundary across the whole window. They must not be equal voters in a fine-grid
   entropy sum — at best a slowly-varying context prior.
2. **Segmentized VAD discards the event before it votes.** The frame posterior that would
   catch a 50 ms cough onset or inter-word breath is thresholded away inside the pyannote
   VAD/diarization wrapper (onset/offset hysteresis, `min_duration_on`). Silero VAD has the
   same problem — several-hundred-ms lag on speech→non-speech transitions.

---

## 2. Models that actually resolve brief events (≤~20 ms frames)

The target is a **continuous per-frame speech probability**, aggregated *within* the
reporting bucket — not a segment boolean. Ranked for this use case:

1. **pyannote `segmentation-3.0` raw frame scores — highest leverage, zero new deps.**
   Already a dependency. Native output is a `(num_frames, num_classes)` posterior, ~16.98
   ms/frame. Take `max` over the speaker axis → continuous P(speech). Overlap-aware
   (powerset: up to 2 speakers/frame, 3/chunk) — matters when a clinician/family member
   talks over the participant. Use `Inference(model)` to get raw scores; **do NOT** route
   through the `VoiceActivityDetection` pipeline — that re-segments and smooths away the
   very thing we need.
2. **TEN VAD — current frame-level VAD SOTA for the sharp-transition regime.** 16 kHz,
   configurable hop 160/256 samples (10/16 ms). Rapidly detects speech↔non-speech
   transitions where Silero lags several hundred ms; reports superior precision/recall vs
   WebRTC and Silero on annotated sets. Tiny/cheap. Return the raw per-frame probability,
   not the thresholded decision (default 0.5 threshold needs domain tuning, but we want the
   probability anyway). **This is the new backend to add to `voice_activity_detection`.**
3. **openSMILE LLDs + PPG voice-fraction — keep as-is.** Already 10 ms. These are the
   physical-evidence voters (audible energy, spectral flux, phoneme-vs-silence) that catch
   whispered/distorted speech a learned VAD may miss.
4. **SSL frame posteriors via a light probe (WavLM ≻ wav2vec2/HuBERT).** `ssl_embeddings`
   already returns `[layers, frames, dim]` at ~20 ms. A small trained head on WavLM-large
   frames gives the one thing generic VAD cannot: **speech vs cough vs breath vs other** at
   frame resolution — which is exactly what the respiratory tasks need and where generic
   VADs *fire on* coughs/laughs instead of distinguishing them.

Silero v5/v6: fine as a robustness cross-check (de-facto OSS standard, >95% acc, <2 MB,
real-time CPU) but its hangover makes it the **wrong primary voter** for brief events —
include, don't lead with it.

Health-acoustics note (for the cough/breath sound-source axis, separate from presence):
Google **HeAR** (300M+ 2 s clips; cough/breath/throat-clear/laugh/speech event detector +
512-d embeddings) is the strongest purpose-built option; **OPERA** / Coswara / COUGHVID are
open-license fallbacks. HeAR is access-gated and 2 s-clip-oriented, so windowing must match.

---

## 3. Recommended reporting resolution — **~100 ms window / 20 ms hop** (not 50 ms)

**Decouple two decisions that 50 ms was conflating:**
- *Analysis* resolution = the model's native frame rate (~10–20 ms), kept un-quantized.
- *Reporting/aggregation* grid = the bucket on which a certainty scalar is emitted.

Onset precision comes from the frame posterior + model transition latency, **not** from the
bucket. So you can report at 100 ms and still localize a cough onset to ~16 ms via the
threshold crossing inside the bucket. Given that decoupling, choose the reporting grid for
statistical stability and for the consumer — and 50 ms is too fine on both counts:

1. **Phonetic timescale.** Mean speech sound ≈ 80–100 ms. ~100 ms ≈ "one phone" — the
   shortest span where *"is this speech?"* is well-posed. At 50 ms you're routinely
   sub-phonetic (stop closure, burst, voicing transition), where ground truth is genuinely
   undefined. This sub-phonetic ambiguity is the irreducible-uncertainty floor; 100 ms
   roughly halves how often you sit on it.
2. **Statistical stability.** At ~16.7 ms frames, a 50 ms bucket holds ~3 frames; 100 ms
   holds ~6. You estimate both a mean and a dispersion (the confidence/uncertainty split)
   per bucket — doubling frame count roughly halves the variance of both. 3 frames can't
   separate "genuinely ambiguous" from "just noisy."
3. **Consumer can't use 50 ms.** Human reviewers in Label Studio can't reliably scrub or
   annotate below ~100 ms; a 50 ms track is mostly false precision.
4. **Brief events don't need 50 ms reporting.** Whole cough ≈ 300–500 ms; inhalation ≈
   0.5–2 s; the explosive cough burst ≈ 30–50 ms is caught as a *transition in the
   posterior*, not as a bucket.

The 20 ms hop (overlapping windows) yields a smooth certainty curve, keeps onsets
localizable to the hop, and is nearly free since the posterior is already computed.

### Better still: per-axis grids (each axis has its own natural timescale)

| Axis / target | Window | Hop | Rationale |
|---|---|---|---|
| **Speech presence** | **100 ms** | 20 ms | phone-scale; "is this speech" well-posed |
| **Cough** | ~250 ms | 50 ms | matches single-cough duration; burst onset from posterior |
| **Breathing** | ~500 ms | 100 ms | inhale/exhale are 0.5–2 s |
| **Speaker identity** | ~1 s+ | 250 ms | embeddings need ~1 s to be meaningful |
| **Utterance / transcription** | word-scale ~200–500 ms | word-aligned | ASR confidence lives at the token |

**Single smallest change if only one is made today:** set the presence grid to
`win_length=0.1, hop_length=0.02`; leave other axes at 0.5 s.

---

## 4. Concrete senselab changes

1. **Expose frame posteriors (the unlock).** Add a function alongside
   `detect_human_voice_activity_in_audios` in
   `src/senselab/audio/tasks/voice_activity_detection/` that returns the **per-frame
   probability array + frame hop**, not `ScriptLine` segments, for:
   - pyannote `segmentation-3.0` via `Inference(model)` (raw scores, `max` over speaker axis);
   - a **new TEN VAD backend** (per-frame probability).
   Everything else depends on having a continuous ≤~17 ms signal.
2. **Make the presence harvester resolution-aware.** Gate voters by native resolution vs
   `grid.win_length`. At ≤100 ms: frame-VAD posteriors + openSMILE + PPG + optional
   SSL-probe vote. Demote AST / YAMNet / `no_speech_prob` / sentence-level ASR to a
   separate slowly-varying prior (or drop from the fine-grid entropy) — do not sum them as
   equal voters.
3. **Make `BucketGrid` per-axis.** Replace the single `--cross-stream-win-length 0.5` /
   `--cross-stream-hop-length 0.5` with per-axis defaults (presence 0.1/0.02; see table).
   Record grid params in each parquet's provenance (already required by FR-010/FR-014).
4. **Change the collapse for fine grids.** Binary-vote Shannon entropy discards the
   continuous confidence now available. Aggregate frame probabilities *within* the bucket
   and report two separable quantities:
   - **confidence** = calibrated mean P(speech) across voters/frames;
   - **uncertainty** = (a) cross-voter disagreement + (b) within-bucket temporal instability
     (a bucket straddling an onset has high frame-variance).
   Temperature-scale the mean on a small labeled set for calibration.
5. **Sound-source axis (cough/breath/other).** Train a small WavLM-frame head on
   domain-labeled speech/cough/breath/other. This is what the respiratory tasks need and
   what no off-the-shelf VAD gives at frame resolution.

### Intrinsic caveat to preserve in the output
At sub-phonetic spans, P(speech | window) has a genuine floor of irreducible uncertainty
that is **phonetic-class-conditioned**, not model error (a stop closure inside a word is
"silence within speech"; 50 ms of unvoiced fricative looks like noise). Keep PPG phoneme
posteriors + SSL frames in the mix so the output can distinguish "stop closure inside
speech" from "true silence," and let the uncertainty signal reflect that part of the
ambiguity lives in the ground truth, not the model.

---

## 5. Suggested first implementation step (for Claude Code)

Prototype the highest-leverage piece against one clip in `tutorial_audio_files/`:
a `segmentation-3.0` (+ optional TEN VAD) frame-posterior extractor that emits calibrated
per-100 ms (20 ms hop) P(speech) with the confidence/uncertainty split, plotted alongside
the current 0.5 s entropy output on the same clip — to see the stability difference
directly. Start with pyannote raw scores only (zero new deps); add TEN VAD as a real
backend as a second step.

Open questions to confirm with the user:
- Add TEN VAD as a real dependency/backend now, or pyannote-raw-scores-only first?
- Is there a labeled clip with brief cough/breath events to validate against, or demo on
  tutorial audio first?
