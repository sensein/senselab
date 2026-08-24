# VOICE branch

Vocalic activity that is not speech: sustained vowels, humming, imitation, glides, phonation.

Runs when [`routing.md`](routing.md) says the voice kind is present or uncertain — that is, when
PREPROCESS found a sustained-phonation or glide span of sufficient duration — or when a hint forces
it.

## Signature

```
voice(store, hint?) -> fail(reason) | flag(reason, partial) | pass
```

Reads and writes the [element store](store.md). **It measures; it does not classify.** No classifier
in the graph can name a member of this kind, so this branch owns no label space.

| element read | author | used for |
| --- | --- | --- |
| `phonation_spans` | PREPROCESS | **the subject of this branch.** Sustained-phonation and glide spans with their `duration_s` and production mode |
| `formant_tracks` | PREPROCESS | the formant evidence over those spans |
| `energy_envelope` | PREPROCESS | level and modulation rate |
| `silence` | PREPROCESS | the floor |
| `word` and `event` elements | PREPROCESS | which spans carry a transcript |
| `hint` | caller, optional | task identity; never creates a span |

**There is no residual.** VOICE measures the phonation spans PREPROCESS detected; it does not subtract
AIRWAY's labels or SPEECH's spans from an energy track and analyse what is left. Nothing another
branch claimed is removed from this branch's subject, and nothing this branch measures is conditioned
on what another branch concluded.

## Flow

```
  1. SUBJECT      PREPROCESS's phonation and glide spans
        │  none → fail
        ▼
  2. TRACKS       computed once on the stream, then sliced to each span
        │
        ▼
  3. PERIOD MARKS per voiced or mixed span: a point process, not a contour
        │
        ▼
  4. AMPLITUDES   one per period
        │
        ▼
  5. TASK         duration against the declared task, when one is declared
        │
        ▼
     pass
```

## 1. The subject

The spans are PREPROCESS's, and their production may be **voiced, unvoiced or mixed**: a disordered
voice sustains with little or no periodicity, and a branch that admitted only periodic production
would measure exactly the voices least in need of measurement. Each span carries which of the three
its tracks support, and no span is refused for being unvoiced.

`fail` when the store carries no phonation span. That is the same condition
[`routing.md`](routing.md) gates on, so this path is reached only when a hint forced the branch.

## 2. Tracks — computed on the stream, sliced to the span

**Every track is computed once over the whole stream and then sliced to each span's extent.** A
criterion evaluated on a fragment renormalises to that fragment's own maximum, which makes the same
signal read differently depending on how it was cut. The tracks are the stream's; the spans index
them.

| product | grid | defined where |
| --- | --- | --- |
| `energy_track`, `periodicity_track`, `hnr_track`, `f0_candidates` | the analysis hop | everywhere on the stream; F0 travels with its periodicity so a reader cannot separate them |
| `period_marks` | **none** | inside voiced and mixed spans only — **absent** elsewhere, not zero and not interpolated |

**F0 is measured on the cleanest available stream.** Where a separated or enhanced stream exists for
the extent, F0 and its derived statistics are recomputed on it, and every F0 measurement records the
stream it was taken on. Two F0 values from two streams are two measurements, never one.

## 3. Period marks — a point process, not an F0 contour

Per voiced or mixed span, an ordered sequence of glottal period boundaries, each carrying its
duration, its amplitude, and the autocorrelation peak that placed it. At 87.4 Hz one period is
11.44 ms, so any fixed-hop contour is already coarser than the quantity it samples, and jitter and
shimmer are defined between consecutive periods.

An unvoiced span carries no period marks and is not thereby a failure: its duration, formants and
level are its measurement.

## 4. Spans — the onset is a period, the offset is a criterion

A span's onset is its first period mark where one exists: an observed event. Its offset is wherever
the continuity criterion stops holding, which is a **criterion**, so the two edges are not the same
kind of quantity and the product does not present them as one.

## 5. Duration against the task

**Duration in seconds is this branch's primary quantity**, and what it means depends on the task.

| declared task | what duration is read as |
| --- | --- |
| maximum phonation time | the span's duration under a named offset criterion, reported as `longest_span_s` with the criterion beside it |
| a glide task | the trajectory's extent and its duration |
| none declared | the span's duration, unqualified |

The task comes from the hint, and it conditions **how a duration is reported, never whether a span
exists**. Where the task declares an expected duration range, a span outside it flags with the
declared range named.

`longest_span_s` is a first-class product: a task measurement that a reader has to reassemble from
fragments is not recoverable, so the branch reports the longest span and the criterion that closed
it, not only the set.

## The F0 range serves a population

The F0 search range is a **property of the declared population**, not a constant: age and sex move it,
and a range spanning too wide an interval makes any period-doubling test on it vacuous.

- The range is the config key `voice.f0_range_hz`, overridable per task and per declared demographic
  through `voice.f0_range_by_population`.
- The period-doubling check is only informative when `f0_max / f0_min` is below
  `voice.f0_range_ratio_max`; a configuration exceeding it is **refused at load**, not run and
  flagged, because a check that flags everything reports nothing.

## Two members that are not acoustic classes

| member | what it actually is |
| --- | --- |
| maximum phonation time | the **duration** of a sustained span under a named offset criterion. A task and its measurement, not a class |
| loud phonation | a **contrast between two spans** in one recording. Not a property of any single span |

Neither is a label this branch can attach, and it does not try.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | the store carries no phonation span |
| `flag` | a span's F0 sits where the declared population's range is ambiguous; a declared task's expected duration range is not met; the gate's parameters are still un-derived and a span sits near an interval's edge; a hint asserts phonation the branch did not find |
| `pass` | spans, tracks, period marks and amplitudes are in the store |

**VOICE concludes about the voice kind and no other** — [`verdict.md`](verdict.md).

## Product

```
outcome:  fail(reason) | flag(reason, partial) | pass
verdict:  { spans_n, phonation_s, longest_span_s, production{voiced,unvoiced,mixed},
            f0_median_hz?, f0_stream?, ambiguous_spans_n, flags[] }
view:     the element ids this branch authored
```

| kind | carries | step |
| --- | --- | --- |
| `span` (phonation) | extent, duration, production mode, track values at onset and offset, offset criterion named | 1, 4 |
| `measurement` (period marks) | ordered period boundaries with duration, amplitude, placing peak | 3, 4 |
| `measurement` (tracks) | energy, periodicity, HNR, F0 candidates on the analysis hop, with the stream named | 2 |
| `measurement` (formants) | F1–F4 and bandwidths over the span, glide direction and extent | 1 |

`f0_median_hz` is reported only with `f0_stream` beside it, and is absent for a span with no period
marks rather than estimated from one.

## Out of scope

Detecting the spans (PREPROCESS does), labelling members, jitter and shimmer as summary statistics
(the period marks are what a consumer computes them from), any claim that a span is a particular
vocal task beyond what the hint declared, and any conclusion about a kind that is not voice.

Derivations live in [`benchmarks/voice.md`](benchmarks/voice.md).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `voice.f0_range_hz`, `voice.f0_range_by_population` | the F0 search range per declared age and sex; **null** until fitted per population |
| `voice.f0_range_ratio_max` | the `f0_max / f0_min` above which the period-doubling check is vacuous and the config is refused; owed the derivation that fixes it |
| `voice.task_duration_ranges` | expected duration ranges per declared task, against which a span flags; **null** |
| `phonation.hnr_floor_db`, `phonation.rms_floor` | the track floors, still an interval rather than a value; **null** |
