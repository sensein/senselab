# SPEECH branch

Runs when [`routing.md`](routing.md) says the speech kind is present or uncertain, or when a hint
forces it. [`REDACT`](redact.md) is a step of this branch, not a node beside it.

## Signature

```
speech(store, enrollment?, hint?) -> fail(reason) | flag(reason, partial) | pass(product)
```

Reads and writes the [element store](store.md). **Speech spans come from detected speech**: consensus
word timings propose them, and PREPROCESS's envelope spans are `refine`d where they overlap rather
than being the source.

What it reads from the store:

| element | author | used for |
| --- | --- | --- |
| `consensus_transcript` and its `word` elements | PREPROCESS | the transcript, its edges, the agreement behind each word, and **the only text the PII scan reads** |
| `asr_crisperwhisper`, `asr_qwen` words | PREPROCESS | the per-recognizer evidence the consensus was fused from |
| `alignment` | PREPROCESS | published word and phone edges |
| `yamnet_windows` | PREPROCESS | `Speech` corroboration per span |
| `squim` | PREPROCESS | per-span quality and the speech test |
| `energy_envelope`, `silence` | PREPROCESS | the local floor |
| `disruptions_file` | PREPROCESS | clipping and zero-crossing rate, measured on the original stream |
| `span` elements | PREPROCESS | `refine`d against word timings; a span with no words is left alone |
| `enrollment` | caller | the target speaker, with the model and revision behind it |
| `hint` | caller, optional | task context |

This branch runs pyannote, a second diarizer under one condition, separation under another, and a
speaker-embedding comparison. **It runs no ASR, and it never re-transcribes.**

**It does not read AIRWAY.** Diarization is a speech-only instrument and nothing in this branch is
conditioned on what AIRWAY found.

## Flow

```
  1. TRANSCRIPT   consensus words  ──►  per-word agreement
        │  no words → fail
        ▼
  2. SPEECH SPANS from consensus word timings
        │
        ▼
  3. CORROBORATE  YAMNet Speech coverage + SQUIM per span
        │  disagree → flag
        ▼
  4. DIARIZE      pyannote over [first word start, last word end] only
        │  count ≠ 1 → second diarizer, report disagreement
        ▼
  5. SEPARATE     only when the foreground must be extracted
        │
        ▼
  6. IDENTIFY     words → speakers; target by enrollment
        │
        ▼
  7. PII SCAN     the consensus transcript; scope the decision by speaker
        │  PII found → REDACT
        ▼            ┌──────────────────────────────────────────────┐
     pass(product) ◄─┤ 8. QUALITY  9. NON-TARGET — parallel, reported│
                     └──────────────────────────────────────────────┘
```

## 1. Transcript

- **The consensus transcript is the transcript.** PREPROCESS produced it with
  `fuse_consensus_words`; this branch reads it and does not re-fuse, re-clean or re-decode.
- **Per-word agreement** between the recognizers is the word's confidence. It bounds confidence from
  above and is reported as agreement, never as correctness.
- **Edges** come from `alignment`; the consensus word's own timings supply them where alignment is
  absent.
- **Bracketed and onomatopoeic events are not words.** PREPROCESS wrote them as `event` elements, so
  nothing here counts them toward word totals, span extents, or the PII scan's subject.

**A word carried by one recognizer alone is not a consensus word**, and a single-recognizer word is
the fabrication evidence this branch records — a shared hallucination across two independent
recognizers is a different and much rarer event. Each word carries which recognizers produced it.

A recording whose non-lexical content is a sustained production reaches [`VOICE`](branch-voice.md)
through [`routing.md`](routing.md); this branch does not defend against it with an energy test.

`fail` when the consensus carries no word: no speech was detected, so this branch has no subject. No
PII scan is written on that path, no REDACT step runs, and the file's release axis reads
`not_assessed`.

## 2. Speech spans

Consensus words are grouped into spans by their timings. A span is the extent of a run of words.

## 3. Corroboration

Two instruments per span, each over the whole span:

| instrument | measure |
| --- | --- |
| YAMNet `Speech` | the fraction of the span's `yamnet_windows` whose label set contains a speech-family member |
| SQUIM over the span | STOI, SI-SDR, as a **test of whether the span is speech** |

Both agreeing confirms the span. Disagreement is a **`flag`**, and the measure that made it ambiguous
travels with the flag.

The SQUIM floors are estimated **across speech-containing spans** and are config keys
(`speech.speech_test_stoi_floor`, `speech.speech_test_si_sdr_floor`), null until that estimation
exists.

## 4. Diarization

**The default assumption is one speaker at a time.** Overlapping voices occur and are not the case
this branch is built for; what it must get right is the **speaker count**.

`pyannote/speaker-diarization-community-1` runs first, applied **only to `[first word start, last
word end]`**.

| pyannote's count | what happens |
| --- | --- |
| 1 | that is the count. **No second diarizer runs** |
| ≠ 1 | a second diarizer is consulted and the disagreement is reported; it does not replace pyannote |

- Restricting the interval is what keeps non-speech events out of the speaker count.
- **No segment is withdrawn for overlapping an airway span.** Diarization answers a question about
  speech, and an airway event inside a speaker turn does not remove the turn. This supersedes the
  withdrawal rule N10; the codomain is the counts pyannote can return, and 0 is one of them.
- Overlap is not a product of this step: pyannote's exclusive view caps per-instant speaker count at
  1 by construction, and the branch reports the count rather than an overlap track.
- **The count is not compared against a declared count.** `hint.targeted_speaker_count` is the
  acquisition protocol's intent, and no corpus this graph runs on establishes where that number came
  from. It is not read here, and a measured count that differs from a declared one is not evidence of
  anything until the declaration's provenance is known.

## 5. Separation — to extract the foreground

Runs when the foreground must be extracted from a background: a speaker count above 1, or a
non-target source the proximity leg (step 9) places behind the target.

| backend | how it is invoked |
| --- | --- |
| `unasdiff` in **`speech_sound` mode** | slot 0 is the speech prior; the sound slot stands for **any background**, so the mode is used without conditioning the background on a class |
| `MossFormer2_SS_16K` | two speech streams, as the alternative |

Both are **measurement-gated**: neither is selected by default, the choice is the config key
`speech.separation_backend`, and it ships null until a measurement over this corpus ranks them.
Whether separation improves anything on overlapping speech is unmeasured —
[`benchmarks/separation.md`](benchmarks/separation.md).

Each stream is written to the store as its own element, and every measurement taken on a stream
records which stream it came from: a quality reading from a separated stream is not the same claim as
one from the recording. `MossFormer2_SS_16K` fixes `n_sources` at 2, so a count of ≥3 is reported
rather than separated into the wrong number.

## 6. Speaker identification — the target is enrolled, not hinted

Words are attributed to speakers by their timings against the diarizer's segments. A word straddling
a boundary is marked rather than assigned.

**The target speaker is identified by an embedding enrolled across all of the subject's provided
recordings**, not by a per-file target hint. Enrollment is a caller-supplied input and a store
element:

```
enrollment: {
  subject_id:  str,
  vector:      [float],            # unit-norm, estimated across the subject's recordings
  provenance:  { model_id, revision, task },     # REQUIRED
  sources:     [ { recording, extent? }, ... ],  # every recording that contributed
  distribution: { ... }?           # spread over the contributing windows, when available
}
```

- **Provenance is required and is model + revision.** Embeddings from different models, or from two
  commits of one model, are not comparable, so an enrollment without both is **refused rather than
  compared**, and the branch flags.
- `sources` names every recording behind the vector, so an enrollment is reproducible and a file's
  own contribution to its target is visible.
- The embedding model is the config key `speech.enrollment_model`, with its revision; it ships null.
- Similarity to a diarized speaker is compared against `speech.target_match_cosine`, null until
  derived.
- Absent an enrollment, speakers are `SPEAKER_*`, no identity is claimed, and the branch flags if PII
  was found.

**A span attributed to a non-target speaker is flagged and removable.** Once attribution exists, each
speech span carries `attributed_to`, and a span whose speaker is not the target carries a
`nontarget` marking that a consumer may act on — excluding it from a measurement, or removing it from
a derivative. This branch marks; it removes nothing.

## 7. PII

`senselab.text.tasks.pii_detection.scan_for_pii` **over the consensus transcript**, then this
branch's own decision rule rather than the module's default.

**One scan, one text.** The consensus transcript is the only text scanned, and it is the same text
[`REDACT`](redact.md) plans and verifies against. Each finding carries which recognizers' hypotheses
carried the word, from the consensus, so a finding resting on one recognizer alone is legible as
such.

### The decision is scoped by speaker

| finding | outcome |
| --- | --- |
| PII overlapping a **target speaker** span | **`flag`** |
| PII overlapping only a **non-target** speaker's spans | no flag |
| PII when **no target is known** | **`flag`** — there is no speaker to exempt |
| a detector **failed to run** | **`flag`** — "could not check" is not "clean" |
| a **required** detector was **never attempted** | **`flag`** — same reason, and it is the silent one |

Completeness is `required ⊆ scanned_by` **and** `failed` empty, where `required` is the config key
`pii.required_detectors`. A detector in `required` but neither scanned nor failed is recorded in the
measurement's `missing` and flags.

**Any finding at all sends the branch to [`REDACT`](redact.md)**, whatever the speaker scope: flagging
asks whether a human is needed, redaction asks whether an artifact is releasable, and a non-target
speaker naming the participant is exactly as unsafe.

### Three limits on what a clean scan means

**Speaker scope catches who *spoke* it, not who it is *about*.** A clinician saying the participant's
name is the participant's PII spoken by a non-target speaker.

**The scan reads a transcript, so it is a lower bound** — a mis-transcribed name is missed while the
audio still contains it — **and an upper bound**: a hallucinated identifier is a finding about text
that was never uttered. A clean scan is a statement about the text, never about the recording.

**The store now holds PII.** A PII finding `label`s the offending `word` elements and every artifact
must respect that marking — in particular the [report](report.md), which renders words.

### What the product may carry

`verdict` carries **category and extent, never the matched text**.

## 8. Quality — parallel, reported

Two readings per relevant span — the target speaker's speech spans, on that speaker's separated
stream when separation ran and on the recording when it did not; every speech span when no
enrollment was given.

| reading | stream | why that stream |
| --- | --- | --- |
| `squim` — STOI, PESQ, SI-SDR | **plain** | SQUIM is trained on conditioned 16 kHz speech |
| clipping, zero-crossing rate | **recording** | peak normalisation and resampling destroy the flat plateaus and the crossing rate the instruments read |
| dropouts, discontinuities, DC offset | **recording** | same |

**Every span reading names its stream.** A reading taken on the wrong stream is not a weaker
measurement of the same quantity; it is a measurement of something else.

Per span, the disruption reading reports **counts and extents, not a score**. A span with none
reports zero, which is a different statement from a span nobody measured. The subjective SQUIM head
is not used.

**Reported, never gated.** Disruption counts are exact and need no threshold; how much is too much is
the gate, and no such value is derived.

## 9. The non-target axis

A presence-level product, independent of transcription and of speaker embeddings:

| leg | measure, per span |
| --- | --- |
| level | span RMS and peak against the file's own reference level |
| spectral tilt | the long-term spectral slope over the span |
| direct-to-reverberant | the span's direct-to-reverberant energy ratio |

Together these are the **proximity leg**: the participant is close-miked and an examiner or bystander
is not. It is speaker-independent and element-independent, so it applies to a span the embedder
cannot characterise.

The product is `nontarget_speech_s` — the total duration of speech spans the proximity leg places
away from the target — reported in the verdict beside the count.

**Measurement-gated.** Every threshold on every leg is a config key under `speech.nontarget` and
ships **null**; until they are derived the legs are measured and reported per span, `nontarget_speech_s`
is written as null rather than zero, and no span is excluded on this evidence. A close examiner may
be indistinguishable from the target on all three legs, and the product says so rather than claiming
a separation it does not have.

## 10. REDACT

When step 7 found PII, [`REDACT`](redact.md) runs as this branch's last step. When it found none, or
when the branch failed for want of words, REDACT does not run and the file's release axis reads
`not_assessed`.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | no consensus word |
| `flag` | PII in a target speaker's spans, or PII with no known target, or a PII detector failed to run; step 3's instruments disagreed; the speaker count is not 1; the two diarizers disagree; the count is ≥3 so separation cannot isolate a speaker; single-recognizer words survive as fabrication candidates; an enrollment was given without model and revision, or with them and no speaker matches |
| `pass` | words, spans, speakers and quality are in the store, and the verdict below says what the branch concluded |

## Product

**The store holds the content; the product is the verdict and a named view over it.**

```
outcome:  fail(reason) | flag(reason, partial) | pass
verdict:  { speaker_count, target_speaker?, enrollment_id?, words_n, speech_s,
            nontarget_speech_s?, pii{categories[], n, scanned_by[], failed[], missing[]}, flags[] }
view:     the element ids this branch authored or asserted over
```

What a consumer reads through the view, by element kind:

| kind | what it carries | authored in |
| --- | --- | --- |
| `word` | text, extent, confidence from consensus agreement, the recognizers behind it, speaker, stream, `pii` marking if any | 1, 6, 7 |
| `span` (speech) | extent, corroboration, YAMNet coverage, `attributed_to`, `nontarget` marking if any, `refines` a PREPROCESS span where one overlapped | 2, 3, 6, 9 |
| `interval` | the diarizer's window, `[first word start, last word end]` | 4 |
| `speaker` | diarizer segments, per diarizer, with the disagreement where two ran | 4, 6 |
| `stream` | one per separated source, or the recording itself | 5 |
| `enrollment` | the target vector, its model and revision, and every recording behind it | 6 |
| `pii` | category and extent per finding, the detectors that ran, the detectors that failed, and which recognizers' hypotheses carried it. **Never the matched text** | 7 |
| `measurement` | SQUIM per span, tagged with the stream it was taken on | 8 |
| `measurement` | clipping and zero-crossing rate per span, on the original stream | 8 |
| `measurement` | level, spectral tilt, direct-to-reverberant ratio per span | 9 |
| `target_match` | speaker, similarity, and the model + revision of both embeddings | 6 |

**`partial` on a `flag` is a view, not a payload** — the same element ids, with the contested
assertions included so a reader sees both sides.

## Out of scope

ASR and re-transcription (PREPROCESS runs the recognizers and fuses the consensus), airway detection,
speaker identity without an enrollment, emotion, language identification, diarizer ranking, quality
gating, and removing anything — this branch *marks*.

Every element and assertion above goes to the [element store](store.md) with its provenance.
Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `speech.enrollment_model` | which speaker-embedding model and revision enrollment is estimated with; **null** until chosen against a measurement |
| `speech.target_match_cosine` | the similarity at which a diarized speaker is the enrolled target; **null** |
| `speech.separation_backend` | `unasdiff` in `speech_sound` mode or `MossFormer2_SS_16K`; **null** until the two are ranked on this corpus |
| `speech.separation_sound_class` | **not owed a measurement — owed a capability.** `separate_audios` refuses `speech_sound` without a conditioning class for its sound slot ("index 0 is 'Hi-hat'"), so the unconditioned background this section describes is not expressible today. Settled by an unconditioned sound slot upstream, or by naming a defensible FSD class and saying why; **null** meanwhile, and the `unasdiff` option cannot run |
| `speech.speech_test_stoi_floor`, `speech.speech_test_si_sdr_floor` | SQUIM floors estimated across speech-containing spans; **null** |
| `speech.nontarget.level_db`, `.tilt_db_per_octave`, `.d_to_r_db` | the proximity leg's thresholds; **null** each, and `nontarget_speech_s` is null until all three exist |
| `speech.word_gap_ms` | the gap that ends a speech span; **null** |
