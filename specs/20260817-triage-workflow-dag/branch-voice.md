# VOICE branch

Vocalic activity that is not speech: sustained vowels, humming, imitation, phonation.

## Signature

```
voice(store, hint?) -> fail(reason) | flag(reason, partial) | pass
```

Reads and writes the [element store](store.md). **It measures; it does not classify.** No classifier in
the screening set can name a member of this kind — `Human voice`, `Human sounds` and `Respiratory
sounds` are all absent from YAMNet's 521 labels — so this branch owns no label space.

| element read | author | used for |
| --- | --- | --- |
| `energy_envelope` | PREPROCESS | the energy half of the gate. VOICE reads the envelope directly and derives no spans, so AIRWAY's `K` does not apply to it |
| `span` elements and their labels | PREPROCESS, AIRWAY | what the residual excludes |
| speech spans | SPEECH | what the residual excludes |
| `silence` | PREPROCESS | the floor |
| `hint` | caller, optional | resolves ambiguity the acoustics cannot; never creates a span |

## Flow

```
  1. RESIDUAL     intervals with energy, minus airway-labelled spans, minus speech spans
        │  nothing left → fail
        ▼
  2. GATE         energy AND periodicity  ──►  voiced runs
        │  no run passes → fail
        ▼
  3. PERIOD MARKS per run: a point process, not a contour
        │
        ▼
  4. AMPLITUDES   one per period
        │
        ▼
     pass
```

## 1. The residual

**The residual is what the store makes computable.** It is the set of intervals carrying energy above the
floor that no other branch has claimed: not covered by an airway-labelled span, not covered by a speech
span. Earlier designs named a `residual_windows` input with no producer anywhere in the graph; with an
append-only store the residual is a fold over what the other branches asserted, so it has one.

A span that AIRWAY proposed and declined to label is **not** excluded — an unlabelled span is exactly
where unclaimed vocalic activity would sit.

`fail` when the residual is empty: every energetic interval belongs to another branch.

## 2. The gate — energy and periodicity

Normalised autocorrelation with an RMS floor, so periodic room tone cannot pass. Both conditions must
hold; either alone admits something this branch should not claim.

**The parameters are an interval, not a value.** The measurement supports a periodicity floor anywhere in
`(0.44, 0.933)` and an RMS floor anywhere in `(0.0007, 0.0161)` — a factor of 2.1 and a factor of 23 —
on one recording. There is no fitted number, and picking a midpoint would invent a decision the
measurement does not contain. The config records the interval and leaves the derivation slot **empty**
until labelled verdicts exist.

Runs are **elementary**. Nothing is merged: two voiced runs separated by an unvoiced gap are two runs,
because a merge criterion is a claim about what constitutes one vocalisation and none has been measured.

## 3. Period marks — a point process, not an F0 contour

Per voiced run, an ordered sequence of glottal period boundaries, each carrying its duration, its
amplitude, and the autocorrelation peak that placed it.

| product | grid | defined where |
| --- | --- | --- |
| `energy_track`, `periodicity_track`, `f0_candidates` | the analysis hop | everywhere; F0 is meaningless below the gate's floor and travels with its periodicity so a reader cannot separate them |
| `period_marks` | **none** | inside voiced runs only — **absent** elsewhere, not zero and not interpolated |

At 87.4 Hz one period is 11.44 ms, so any fixed-hop contour is already coarser than the quantity it
samples, and jitter and shimmer are defined *between consecutive periods* and unrecoverable from a
resampled contour.

## 4. Spans — the onset is a period, the offset is a criterion

A run's onset is the first period mark: an observed event. Its offset is wherever the gate stops holding,
which is a **criterion**, so the two edges are not the same kind of quantity and the product does not
present them as one.

## Two members that are not acoustic classes

| member | what it actually is |
| --- | --- |
| maximum phonation time | the **duration** of a sustained-vowel run under a named offset criterion. A task and its measurement, not a class |
| loud phonation | a **contrast between two runs** in one recording. Not a property of any single run |

Neither is a label this branch can attach, and it does not try.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | the residual is empty, or no run passes the gate |
| `flag` | a run's F0 sits where the search range serves two populations ambiguously; a hint asserts phonation the branch did not find; the gate's parameters are still un-derived and a run sits near the interval's edge |
| `pass` | period marks, amplitudes and runs are in the store |

## Product

```
outcome:  fail(reason) | flag(reason, partial) | pass
verdict:  { runs_n, voiced_s, f0_median_hz?, ambiguous_runs_n, flags[] }
view:     the element ids this branch authored
```

| kind | carries | step |
| --- | --- | --- |
| `span` (voiced run) | extent, gate values at onset and offset, offset criterion named | 2, 4 |
| `measurement` (period marks) | ordered period boundaries with duration, amplitude, placing peak | 3, 4 |
| `measurement` (tracks) | energy, periodicity, F0 candidates on the analysis hop | 2 |

## Out of scope

Labelling members, merging runs, jitter and shimmer as summary statistics (the period marks are what a
consumer computes them from), and any claim that a run is a particular vocal task.

Derivations live in [`benchmarks/voice.md`](benchmarks/voice.md).
