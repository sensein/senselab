# SPEECH branch

## Signature

```
speech(store, hint?) -> fail(reason) | flag(reason, partial) | pass(product)
```

Reads and writes the [element store](store.md). **Speech spans come from detected speech**: ASR word
timings propose them, and PREPROCESS's envelope spans are `refine`d where they overlap rather than being
the source.

What it reads from the store:

| element | author | used for |
| --- | --- | --- |
| `asr_crisperwhisper`, `asr_qwen` words | PREPROCESS | the transcript, its edges, and the agreement between them |
| `alignment` | PREPROCESS | published word and phone edges |
| `energy_envelope`, `silence` | PREPROCESS | the fabrication test in step 1 |
| `span` elements | PREPROCESS | `refine`d against word timings; a span with no words is left alone |
| airway-labelled spans | AIRWAY, if present | withdrawing a diarizer segment inside the speech interval |
| `hint` | caller, optional | task context; a target embedding **with the model and revision that produced it** |

This branch runs pyannote, an optional second diarizer, and an optional speaker-embedding comparison.
**It runs no ASR.**

## Flow

```
  1. TRANSCRIPT   asr_crisperwhisper ⋈ asr_qwen  ──►  per-word confidence
                  fabrication test vs envelope + local floor
        │  no words → fail
        ▼
  2. SPEECH SPANS from word timings
        │
        ▼
  3. CORROBORATE  YAMNet Speech coverage + SQUIM per span
        │  disagree → flag
        ▼
  4. DIARIZE      pyannote over [first word start, last word end] only
        │  count ≠ 1 → second diarizer, report disagreement
        ▼
  5. SEPARATE     only when count ≠ 1 → MossFormer2_SS_16K, one stream per speaker
        │
        ▼
  6. IDENTIFY     words → speakers; target match only if the hint carries one
        │
        ▼            ┌──────────────────────────────────────┐
     pass(product) ◄─┤ 7. QUALITY — parallel, reported only │
                     └──────────────────────────────────────┘
```

## 1. Transcript

- **Word agreement** between the two recognizers gives per-word confidence. **This is the clean
  transcript** — there is no separate cleaning step.
- **Edges** come from `alignment`; CrisperWhisper alone supplies them where alignment is absent.
- **A word over no energy and no periodicity is a fabrication candidate**, tested against
  `energy_envelope` and its local floor.

Agreement bounds confidence from above and is reported as agreement, never as correctness.

`fail` when neither recognizer returns a word: no speech was detected, so this branch has no subject.

## 2. Speech spans

Words are grouped into spans by their timings. A span is the extent of a run of words.

## 3. Corroboration

Two instruments per span, each over the whole span:

| instrument | measure |
| --- | --- |
| YAMNet `Speech` | coverage — fraction of overlapping 0.96 s windows ≥ 0.5 |
| SQUIM over the span | STOI, SI-SDR, as a **test of whether the span is speech** |

Both agreeing confirms the span. Disagreement is a **`flag`**, and the measure that made it ambiguous
travels with the flag.

## 4. Diarization

`pyannote/speaker-diarization-community-1`, applied **only to `[first word start, last word end]`**.
Codomain **{1, 2, ≥3}**.

- Restricting the interval is what keeps non-speech events out of the speaker count.
- A segment inside the interval that overlaps an `airway_spans` entry is **withdrawn**, not relabelled.
- Count ≠ 1 consults a second diarizer and **reports disagreement**; it does not replace pyannote. The
  product still carries per-speaker spans.

## 5. Separation — to pull a speaker out

**Runs when the speaker count is not 1.** `MossFormer2_SS_16K`, two streams, over the speech interval.
With one speaker there is nothing to separate and it does not run.

Its purpose is to isolate a speaker so that steps 6 and 7 measure one voice rather than a mixture. Each
stream is written to the store as its own element, and every measurement taken on a stream records which
stream it came from — a transcript or a quality reading from a separated stream is not the same claim as
one from the recording.

`n_sources` is fixed at 2 by the checkpoint. A count of ≥3 therefore cannot be served by this model, and
the branch reports that rather than separating into the wrong number.

Whether separation improves anything is **unmeasured on overlapping speech** — see
[`benchmarks/separation.md`](benchmarks/separation.md). It is specified here because the flow needs a
way to isolate a speaker, not because its benefit is established.

## 6. Speaker identification

Words are attributed to speakers by their timings against the diarizer's segments. A word straddling a
boundary, or falling inside a withdrawn segment, is marked rather than assigned.

**Target comparison happens only when the hint supplies a target embedding**, and the hint must carry
**the model and revision that produced it**. Embeddings from different models are not comparable, so a
target without provenance is refused rather than compared. Absent a target, speakers are `SPEAKER_*` and
no identity is claimed.

## 7. Quality — parallel, reported

SQUIM objective head over the **target speaker's** speech spans — on that speaker's separated stream
when separation ran, on the recording when it did not. Over every speech span when no target was given. The subjective head is not used: it needs a non-matching reference, which is a config
artifact nobody has declared.

**Reported, never gated.** No threshold has been derived, so no recording is dismissed on quality. This
step is a parallel branch of the graph and blocks nothing. It becomes a gate when thresholds exist.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | no words from either recognizer |
| `flag` | step 3's instruments disagreed; the count is ≥3 so separation cannot isolate a speaker; speaker count ≠ 1; the recognizers disagree beyond threshold; fabrication candidates survive; a target was given without model provenance, or with provenance and no speaker matches; a hint asserts speech not found |
| `pass` | a transcript with per-word confidence, spans attributed to speakers, quality reported |

## Product

```
transcript:    [ { word, start, end, confidence, speaker } ]
speech_spans:  [ { start, end, corroborated, yamnet_coverage, refines? } ]
streams:       [ { id, speaker, source: "recording" | "separated" } ]
speaker_spans: [ { start, end, speaker, withdrawn, withdrawn_because } ]
speaker_count: 1 | 2 | ">=3"
quality:       { per_span: [ {start, end, stoi, pesq, si_sdr, stream} ], scope: "target" | "all" }
target_match:  { speaker, similarity, model, revision } | absent
figure:        one aligned figure per recording
```

## Out of scope

ASR (PREPROCESS runs it), airway detection (reads `airway_spans`), speaker identity without a target,
emotion, language identification, diarizer ranking, quality gating.

Every element and assertion above goes to the [element store](store.md) with its provenance.
Derivations live in [`benchmarks/`](benchmarks/).
