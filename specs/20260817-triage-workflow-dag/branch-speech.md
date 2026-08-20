# SPEECH branch

## Signature

```
speech(spans, squim, asr_crisperwhisper, asr_qwen, alignment, silence,
       airway_spans?, target_embedding?, hint?)
    -> fail(reason) | flag(reason, partial) | pass(product)
```

| input | from | used for |
| --- | --- | --- |
| `spans` | PREPROCESS, `K` = 12 dB | the candidate spans this branch interprets |
| `silence` | PREPROCESS | the floor the spans were derived against |
| `asr_crisperwhisper`, `asr_qwen`, `alignment` | PREPROCESS | transcript, edges, agreement |
| `squim` | computed **per span**, not per file | the speech test in step 1, the quality gate in step 3 |
| `airway_spans` | AIRWAY | withdrawing pyannote segments that are airway events |
| `target_embedding` | caller, optional | target attribution |
| `hint` | caller | conditions the outcome only |

This branch runs pyannote, an optional second diarizer, an optional speaker-embedding comparison, and
optional separation. **It runs no ASR.**

## 1. Extract speech — interpret the spans

Two instruments vote on each span, both over the whole span:

| instrument | measure |
| --- | --- |
| YAMNet `Speech` | coverage — fraction of overlapping 0.96 s windows ≥ 0.5 |
| SQUIM over the span | STOI, SI-SDR. Used here as a **test of whether the span is speech**, not as quality |

| both vote | outcome |
| --- | --- |
| speech | the span is a speech span, to step 2 |
| not speech | the span is not this branch's subject |
| **disagree** | **`flag`** — the measure that made it ambiguous travels with the flag |

`fail` when no span carries a speech vote, or PREPROCESS reported `no_contrast`. The two causes are
distinguished in the reason.

## 2. Speaker count — pyannote

`pyannote/speaker-diarization-community-1`. Codomain **{1, 2, ≥3}**.

- A pyannote segment overlapping an `airway_spans` entry is **withdrawn**, not relabelled.
- Count ≠ 1 consults a second diarizer and **reports disagreement**; it does not replace pyannote.

## 3. Quality — over speech spans

SQUIM objective head, per speech span. The subjective head is not used: it needs a non-matching
reference, which is a config artifact nobody has declared.

Reported, not gated: no threshold on STOI or PESQ has been derived, so `fail` is unreachable by this
route until one exists.

## 4. Transcript — two recognizers, compared

- **Word agreement** between `asr_crisperwhisper` and `asr_qwen` gives per-word confidence.
- **Edges** come from CrisperWhisper alone, not an average.
- **A word slot over no energy and no periodicity** is a fabrication candidate, tested against
  `energy_envelope` and its local floor.

Agreement bounds confidence from above and is reported as agreement, never as correctness.

## 5. Separation — optional, off by default

`MossFormer2_SS_16K`, two streams. Available for recordings where speech and airway events overlap.
Off by default.

## 6. Target comparison — optional

Only when `target_embedding` is supplied. Absent one, speakers are `SPEAKER_*` and no identity is
claimed.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | no speech span in step 1, or `no_contrast` |
| `flag` | step 1's instruments disagreed; speaker count ≠ 1; the recognizers disagree beyond threshold; fabrication candidates survive; a target was given and no speaker matches; a hint asserts speech not found |
| `pass` | a transcript with per-word confidence, spans attributed to speakers, quality measured over the speech spans |

## Product

```
transcript:    [ { word, start, end, confidence, speaker } ]
speaker_spans: [ { start, end, speaker, withdrawn, withdrawn_because } ]
speaker_count: 1 | 2 | ">=3"
quality:       { per_span: [ {start, end, stoi, pesq, si_sdr} ], spans_measured_s }
target_match:  { speaker, similarity } | absent
figure:        one aligned figure per recording
```

Span extents are **locators, not edges**. Published word edges come from `alignment`.

## Out of scope

ASR (PREPROCESS runs it), airway detection (reads `airway_spans`), speaker identity without a target,
emotion, language identification, diarizer ranking.

Derivations live in [`benchmarks/`](benchmarks/).
