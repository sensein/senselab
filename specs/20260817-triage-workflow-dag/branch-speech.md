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
        ▼
  7. PII SCAN     scan the transcript; scope the decision by speaker
        │  PII in target spans, or target unknown, or a detector failed → flag
        ▼            ┌──────────────────────────────────────┐
     pass(product) ◄─┤ 8. QUALITY — parallel, reported only │
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

## 7. PII

`senselab.text.tasks.pii_detection.scan_for_pii` over the transcript, then **this branch's own decision
rule** rather than the module's default — the module splits scanning from deciding precisely so a caller
can impose its own.

**Both recognizers' transcripts are scanned.** Each detection carries the `asr_model` that produced it,
so a finding present in both is corroborated and one present in a single hypothesis is not. That is
evidence to record, not a threshold to apply here.

### The decision is scoped by speaker

| finding | outcome |
| --- | --- |
| PII overlapping a **target speaker** span | **`flag`** |
| PII overlapping only a **non-target** speaker's spans | no flag |
| PII when **no target is known** | **`flag`** — there is no speaker to exempt |
| a detector **failed to run** | **`flag`** — "could not check" is not "clean" |

`PiiScan.failures` exists for that last row and must be honoured: an empty `spans` with a populated
`failures` means the scan did not happen, and reading it as a clean result is the one outcome worse than
not scanning.

### Three limits on what a clean scan means

**Speaker scope catches who *spoke* it, not who it is *about*.** A clinician saying the participant's
name is the participant's PII spoken by a non-target speaker, and the rule above does not flag it. This
is a known gap in the rule as specified, not an oversight in the implementation.

**The scan reads a transcript, so it is a lower bound.** A mis-transcribed name is missed while the
**audio still contains it**. A clean scan is a statement about the text, never about the recording, and
nothing downstream may treat it as clearance to release audio.

**The store now holds PII.** Once a transcript is written, the store is sensitive, and being append-only
it stays that way. So a PII finding `label`s the offending `word` elements and **every artifact must
respect that marking** — in particular the figure, which renders words and would otherwise leak what the
scan just found. Producing a releasable derivative is [`REDACT`](redact.md)'s job, and it cannot make the
store itself releasable.

### What the product may carry

`verdict` carries **category and extent, never the matched text**. A verdict that quotes the PII it found
has published it into whatever reads the verdict, which is the opposite of the point.

## 8. Quality — parallel, reported

Two independent readings per relevant span — the **target speaker's** speech spans, on that speaker's
separated stream when separation ran and on the recording when it did not; every speech span when no
target was given.

| reading | what it answers |
| --- | --- |
| `squim` objective head — STOI, PESQ, SI-SDR | how intelligible the speech is |
| `disruptions` — clipping, dropouts, discontinuities, DC offset | whether the recording is intact |

**These are not redundant, and the second is not derivable from the first.** SQUIM is a *speech*-quality
estimator trained on particular degradations, so hard clipping can read as acceptable or as generic
noise rather than as the specific defect it is. A clipped span and a reverberant span can score alike
while needing opposite responses — one is a capture fault to fix at source, the other is a property of
the room.

Per span, `disruptions` reports **counts and extents, not a score**: clipped sample runs and their total
duration, zero-run dropouts, sample-to-sample discontinuities, and DC offset. A span with none reports
zero, which is a different statement from a span nobody measured.

The subjective SQUIM head is not used: it needs a non-matching reference, which is a config artifact
nobody has declared.

**Reported, never gated.** No threshold has been derived for either reading, so no recording is dismissed
on quality. This step is a parallel branch of the graph and blocks nothing. It becomes a gate when
thresholds exist.

Disruption *counts* are exact and need no threshold — a clipped run either happened or it did not. What
has no derived value is **how much is too much**, and that is the gate, not the measurement.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | no words from either recognizer |
| `flag` | PII in a target speaker's spans, or PII with no known target, or a PII detector failed to run; step 3's instruments disagreed; the count is ≥3 so separation cannot isolate a speaker; speaker count ≠ 1; the recognizers disagree beyond threshold; fabrication candidates survive; a target was given without model provenance, or with provenance and no speaker matches; a hint asserts speech not found |
| `pass` | words, spans, speakers and quality are in the store, and the verdict below says what the branch concluded |

## Product

**The store holds the content; the product is the verdict and a named view over it.** Everything below
is authored into the store as it is produced, so returning a copy would create a second version of the
same facts that can drift from the first.

```
outcome:  fail(reason) | flag(reason, partial) | pass
verdict:  { speaker_count, target_speaker?, words_n, speech_s, pii{categories[], n, scanned_by[], failed[]}, flags[] }
view:     the element ids this branch authored or asserted over
figure:   one aligned figure per recording          # an artifact, not in the store
```

`verdict` is the only new information: it is this branch's summary judgement, which is not derivable from
the elements without knowing which fold this branch intends. Everything else is a pointer.

What a consumer reads through the view, by element kind:

| kind | what it carries | authored in |
| --- | --- | --- |
| `word` | text, extent, confidence from agreement, speaker, stream, `pii` marking if any | 1, 6, 7 |
| `span` (speech) | extent, corroboration, YAMNet coverage, `refines` a PREPROCESS span where one overlapped | 2, 3 |
| `interval` | the diarizer's window, `[first word start, last word end]` | 4 |
| `speaker` | diarizer segments, `withdraw`n ones retained with their reason | 4, 6 |
| `stream` | one per separated source, or the recording itself | 5 |
| `pii` | category and extent per finding, the detectors that ran, the detectors that failed, and which recognizer's hypothesis carried it. **Never the matched text** | 7 |
| `measurement` | SQUIM per span, tagged with the stream it was taken on | 8 |
| `measurement` | disruption counts and extents per span — clipped runs and duration, dropouts, discontinuities, DC offset | 8 |
| `target_match` | speaker, similarity, and the model + revision of both embeddings | 6 |

**`partial` on a `flag` is a view, not a payload** — the same element ids, with the contested assertions
included so a reader sees both sides rather than the branch's preferred side.

**The figure is the one thing that is not an element.** It is a rendering, so it is an artifact beside
the store rather than in it, and it carries the run's element ids so a reader can trace any mark on it
back to the assertion that produced it.
## Out of scope

ASR (PREPROCESS runs it), airway detection (reads `airway_spans`), speaker identity without a target,
emotion, language identification, diarizer ranking, quality gating, and redaction — this branch
*detects* PII and marks it; [`REDACT`](redact.md) acts on the marking.

Every element and assertion above goes to the [element store](store.md) with its provenance.
Derivations live in [`benchmarks/`](benchmarks/).
