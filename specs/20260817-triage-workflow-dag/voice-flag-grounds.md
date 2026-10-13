# What VOICE had in hand when it flagged

A measurement of the two VOICE flag families that dominated the 20-recording smoke:

- `mismatch: routing routed VOICE, it found no subject` and `hint mismatch: VOICE was declared and
  did not find it` — VERDICT's agreement and hint tables, folding what VOICE *found* (the spans it
  proposed) against what routing and the declaration expected;
- `VOICE reported that what the instruction asked for did not happen on <family>` — VOICE's own
  `conformance` reaching `False`.

Every `False` and every `absent` below was traced back into the store: which amplitude spans VOICE
was handed, what it measured over each, and which gate rejected it.

This document reports and proposes. **Nothing in `src/` is changed by it.**

## What was measured

| | |
| --- | --- |
| recordings | **300** |
| strata | 10 families × 30, seed 20260919, sampled from `clipfix_20260913` / `corpus_inventory.jsonl` |
| in family for VOICE | `maximum-phonation-time`, `maximum-phonation-time-v2`, `prolonged-vowel`, `glides-low-to-high`, `glides-high-to-low`, `high-to-low` — all six |
| controls | `diadochokinesis-pa`, `loudness`, `rainbow-passage`, `respiration-and-cough-cough` |
| graph | the full triage DAG through REPORT, CPU, one recording per run directory |
| commit | **`f2729d7cc8d79bd93266560c397365f7f17985c1`**, pinned |
| config | `senselab-triage/default` v1, hash `23c7df85d999d771`, no override |
| where | ORCD `pi_satra`, 24-way array, `sbatch` |
| probe | `voice-flag-grounds-census.py` and `voice-flag-grounds-tables.py`, beside this file |

`voice.py`, `vocabulary.py`, `verdict.py`, `routing.py` and the packaged config are byte-identical
between `f2729d7c` and the branch tip this was analysed against, so the line numbers cited below
hold for both.

**Provenance note.** A first pass over the same 300 recordings ran while the shared ORCD checkout
was reset underneath it, leaving its rows commit-ambiguous. It was re-run on the pinned clone above,
with the commit asserted at slice start *and* at slice end so a mid-run move would fail the slice;
no slice reported a violation. The two passes were then compared per recording rather than assumed
equivalent:

| quantity | runs agreeing |
| --- | --- |
| `conformance[VOICE]` | 300 / 300 |
| `findings[VOICE]` | 300 / 300 |
| `routes[VOICE]` | 300 / 300 |
| `hints[VOICE]` | 300 / 300 |
| carriers that qualified | 300 / 300 |
| amplitude spans handed to VOICE | 300 / 300 |

The reset is therefore measured to be inert for everything reported here. Every number below is from
the pinned pass.

## A correction to the framing

The brief lists `loudness` among "VOICE's targeted families". Under the packaged configuration it is
not one. `taxonomy.ruleset.reference_family_set.VOICE` is the key `voice`, which resolves to
`families.VOICE_ELICITING` — the six families named above. `loudness` and `loudness-v2` sit in
`LEXICAL_SPEECH` and route to SPEECH; they appear only in
`VOICE_EXPECTATIONS_PENDING_DECLARATION`, a table `align_voice` never reads. A `loudness` recording
takes VOICE's out-of-family mode and evaluates no task. It is measured here as a control, and
`maximum-phonation-time-v2` and `high-to-low` — which the brief omitted — are covered instead, so
all six real in-family families are present.

## 1, 2, 3 — the three rates, per family

`routed` counts `routes[VOICE] == routed`; `rtd+abs` counts those that were also
`findings[VOICE] == absent`, and `rate` is that over `routed`. `claimNF` counts
`hints[VOICE] == claimed_not_found`, over all `n`. The last four columns are `conformance[VOICE]`;
`noRep` is a recording on which VOICE filed no report at all.

```
family                           n routed rtd+abs   rate claimNF   rate  True False UNDET noRep
glides-high-to-low              30     27      15   0.56      18   0.60    12    18     0     0
glides-low-to-high              30     30      19   0.63      19   0.63    11    19     0     0
high-to-low                     30     29      17   0.59      18   0.60    12    18     0     0
maximum-phonation-time          30     29      25   0.86      26   0.87     4    26     0     0
maximum-phonation-time-v2       30     27      19   0.70      22   0.73     8    22     0     0
prolonged-vowel                 30     29       6   0.21       6   0.20     4    26     0     0
diadochokinesis-pa              30     11      11   1.00       0   0.00     0     0    11    19
loudness                        30      7       7   1.00       0   0.00     0     0     7    23
rainbow-passage                 30     19      18   0.95       0   0.00     0     0    19    11
respiration-and-cough-cough     30      2       2   1.00       0   0.00     0     0     2    28
```

Three readings stand out.

**In-family conformance is `False` on 129 of 180 recordings (72%).** It is never `UNDETERMINED`
on any of the six: the tracks were present on all 180 in-family recordings, so the `UNDETERMINED`
arm the branch carries for an absent instrument never fires. (The single recording in the whole
sample with no tracks is a 0.41 s `respiration-and-cough-cough` file, on which VOICE filed no report
at all.) The branch always answered, and it mostly answered no.

**`prolonged-vowel` is the informative dissociation.** Its `absent` rate is the lowest of the six
(0.21) while its `False` rate is the joint highest (26/30). The two are not the same event there —
which is the first sign that its `False` is reached by a different route from the others. It is:
see *Defect B*.

**On the controls, when VOICE is routed it finds nothing essentially always** — 11/11, 7/7, 2/2 and
18/19. But it is routed on only 11, 7, 2 and 19 of 30. So the control result is not "VOICE flags
everything it is routed to" in the sense of a runaway detector; it is that the ruleset routes VOICE
to a minority of non-voice recordings, and on those it correctly finds no sustained phonation and
VERDICT flags the recording for it. `conformance` there is `UNDETERMINED` (the out-of-family mode
evaluates no task) and `verdict.undetermined_flags` is `false`, so the control flags come from the
agreement table alone.

## 5 — the structural trap: ruled out

Checked first, against the writers rather than from memory. VOICE reads five things. Every one has a
live writer in the current graph, and every one was present in the store:

| VOICE reads | writer | present on |
| --- | --- | --- |
| spans with `measure == "amplitude"` | `preprocess.py` `_spans`, from the pre-emphasised and normalized envelopes | 298 / 300 carried ≥ 1 |
| `phonation_tracks` (F0 + strength) | `preprocess.py:phonation_tracks`, npz sidecar + measurement | 299 / 300 |
| `continuity_trace` | `preprocess.py:_continuity_trace` | 300 / 300 |
| consensus `word` entities | `preprocess.py:_consensus` | present |
| the `recording` stream's extent | ADMIT | present |

**VOICE is not keyed to anything no writer produces.** This is the opposite shape from the SPEECH
defect: there, `_speech_free_response` read `False` because the spans it keyed on structurally could
not exist. Here the evidence is all present, of good quality, and a gate throws it away. The fix
shape is therefore different too, and the SPEECH remedy does not transfer unchanged.

## 4 — the evidence census: what VOICE actually had

This is the heart of it. For every recording whose VOICE conformance was `False` or whose finding
was `absent`, the probe re-read the store, rebuilt the exact `Evidence` record `voice.py` builds,
and replayed `qualifying_phonation` gate by gate over each amplitude span.

The decisive table. Of the recordings where **no carrier qualified**, how many had no span long
enough to be a candidate at all; how many had no candidate reaching `voiced_fraction_min`; and how
many had a candidate that *was* voiced (`voiced_fraction ≥ 0.5`) and was rejected anyway, by which
gate:

```
family                         no_carrier no_span>=.5s none_voiced  voiced_but_spread  voiced_but_cont  voiced_but_lex  glide_dom
glides-high-to-low                     18            1           7                  0                0               0         10
glides-low-to-high                     19            0           5                  0                0               0         14
high-to-low                            18            0           3                  0                0               0         15
maximum-phonation-time                 26            2           6                 18                0               0          0
maximum-phonation-time-v2              22            1           6                 15                0               0          0
prolonged-vowel                        22            0           0                 15                0               7          0
diadochokinesis-pa                     29            3          13                 13                0               0          0
loudness                               30            0          23                  7                0               0          0
rainbow-passage                        26            0           7                 19                0               0          0
respiration-and-cough-cough            30           10          20                  0                0               0          0
```

On the three sustained in-family families, **48 of the 70 no-carrier outcomes had a voiced carrier
in hand that the F0-spread gate rejected**, and a further 7 (`prolonged-vowel`) had one the lexical
separator discarded. On the three glide families, **39 of 55** had a monotone sweep located and then
discarded for covering too little of its carrier. The `continuity_min` gate rejected nothing,
anywhere — it is not load-bearing on this corpus.

The `none_voiced` column — 27 in-family recordings (7, 5, 3, 6, 6, 0 down the six in-family rows) —
is the one I cannot adjudicate from the store. Some are recordings where the participant did not
phonate; some may be tracker failures. Settling them needs listening, not another pass. The four
in-family `no_span` recordings are a separate and benign case: one is a 0.56 s
`maximum-phonation-time` file, on which "no sustained phonation" is simply true.

## Defect A — the F0-spread qualifier is an extreme-value statistic under a per-window threshold

**Site.** `src/senselab/audio/workflows/triage/nodes/voice.py:273`, inside `qualifying_phonation`:

```python
and (spread_max is None or spread_window_s is None or spread <= spread_max)
```

where `spread = max_windowed_spread(pitch, track.hop_s, spread_window_s)`
(`branches.py:1779`), `branch.f0_spread_window_s = 0.5`, `branch.f0_spread_max_semitones = 2.0`.

**Mechanism.** `max_windowed_spread` slides a 0.5 s window (50 frames at the 0.01 s hop) across the
carrier and returns the **maximum**, over every window position, of that window's 5th-to-95th
percentile F0 range. The carrier is rejected when that maximum exceeds 2.0 semitones. A 25-second
maximum-phonation-time carrier contains roughly 2,500 window positions.

The threshold's own written derivation (`config-derivations.md`, `branch.f0_spread_max_semitones`)
is: *"A whole tone: the smallest interval a listener names as a pitch change in speech rather than
as vibrato or tracker noise."* That reasons about **one** window. Applied to the maximum over
thousands of them, a bound whose stated purpose is to *tolerate* tracker noise becomes a test that
is near-certain to find some: the expected maximum grows with the number of windows, and the number
of windows grows linearly with the duration of the production.

**The production is steady; the statistic is not.** For every carrier the gate rejected, the probe
recorded the whole distribution of window spreads, not just its maximum:

```
family                           n  med_windows  med_spread_med  med_spread_max  med_frac_over  max~octave   f0_range_hz
maximum-phonation-time          18         1143            0.97           13.41          0.204           2       141-537
maximum-phonation-time-v2       25          714            1.12           13.06          0.279           5        61-246
prolonged-vowel                 17          483            0.83           10.58          0.179           2        50-600
diadochokinesis-pa              21           46            3.14            4.66          0.978           2        89-364
loudness                        12           39            6.33            8.20          1.000           4        89-600
rainbow-passage                133           71            4.69            8.37          0.971          28        60-600
```

`med_spread_med` is the median across rejected carriers of that carrier's **median** window spread;
`med_spread_max` is the median of the **maximum** the gate actually reads; `med_frac_over` is the
fraction of a carrier's windows that exceed 2.0 semitones.

On the three sustained families the typical rejected carrier is steady — **median window spread
0.83 to 1.12 semitones, comfortably inside the 2.0 bound** — while the maximum the gate reads is
**10.6 to 13.4 semitones**, and only 18 to 28% of windows exceed the threshold at all. Ten to
thirteen semitones is an octave, against F0 search ranges (read from the `phonation_tracks`
measurement) that all span more than one: that is the pitch-tracker octave-error signature, not a
participant's pitch.

**The control contrast is what rules out "participants are just unstable".** On connected speech —
`rainbow-passage`, `loudness`, `diadochokinesis-pa` — the rejected carriers have median *window*
spreads of 3.1 to 6.3 semitones and 97 to 100% of windows over threshold. There the gate is reading
real, pervasive pitch movement and rejecting correctly. The gate discriminates properly on the
material it is not for, and fails specifically on the steady productions it exists to qualify.

**The dose-response.** Rejection rate against carrier duration, over sustained in-family carriers:

```
carrier duration (s)      rejected     n    rate
0-2                             12    73    0.16
2-5                              7    18    0.39
5-10                            19    32    0.59
10-20                           17    29    0.59
20-100                           5     6    0.83
```

The gate is most likely to reject exactly when the participant held the vowel longest. On a task
whose entire measurement *is* how long the vowel was held, that is the worst possible direction for
the error to run.

**Verdict: a defect.** An instrument artefact confined to a minority of analysis windows is
converted into `conformance = False`, which `vocabulary.py:712` renders as *"VOICE reported that
what the instruction asked for did not happen on maximum-phonation-time"*. This is the contract the
owner has stated repeatedly — an extractor's imperfection must not read as a participant failure —
and it is violated here in its sharpest form, because the branch had a 25-second, 95%-voiced,
0.96-stationarity sustained phonation in hand when it said the task was not performed.

## Defect B — on `prolonged-vowel`, the conformance answers a question about the count-in

**Site.** `voice.py:452`, the last line of `_voice_sustained`:

```python
return Result(count_in_found, components, findings)
```

**Mechanism.** The task's conformance is whatever `_count_in` returned. `prolonged-vowel` is the
only `VOICE_EXPECTATIONS` row carrying `tokens` — `("one", "two", "three")`, with
`lexical_separator=True` (`branches.py:601-607`). So on a recording where a carrier *does* qualify,
the conformance of a **prolonged vowel** task is whether the ASR consensus contained an ordered run
of those three words. The held vowel — the thing the instruction asked for, and the thing VOICE has
just measured to three decimal places — contributes nothing to the answer.

Measured: of `prolonged-vowel`'s 26 `False`s, **4 were reached with a qualifying carrier in hand**.
On all four the branch proposed a `task_extent` span and wrote a measured phonation of **4.2, 7.6,
8.6 and 9.0 seconds** — and on all four the consensus transcript contained **no lexical word at
all**, so `ordered_run` matched none of `one`/`two`/`three`. The branch measured a nine-second
sustained vowel and reported that the prolonged-vowel task did not happen, because nobody counted
in.

The same row's second half compounds it: `lexical_separator=True` makes `qualifying_phonation`
(`voice.py:272`) discard any amplitude span overlapping any lexical consensus word. **7 of 30
`prolonged-vowel` recordings lost their voiced carrier that way** — an ASR word over the sustained
vowel removes the subject entirely.

This is also why `prolonged-vowel` shows the dissociation noted earlier: on those 4 recordings the
`task_extent` span makes `findings[VOICE]` read `present`, so no agreement mismatch fires, while the
conformance is `False` anyway. Its `absent` rate (0.21) and its `False` rate (0.87) come apart
because they are answering different questions — and it is the only family in which the store
carries positive evidence of the task alongside a verdict that it was not performed.

**Verdict: a defect**, and a separate one from A — a different site, a different mechanism, and it
survives any fix to the spread gate.

## Defect C — the glide reports "no sweep found" about a sweep it found

**Site.** `voice.py:527`, then `:532-533`:

```python
if dominant_min is not None and duration(sweep) / max(duration(span.extent), 1e-9) < dominant_min:
    continue
...
if best is None:
    return Result(False, [], [count("sweep_found", False, True)])
```

**Mechanism.** `_voice_glide` locates the longest monotone F0 run inside each amplitude carrier,
then discards it when it covers less than `branch.dominant_segment_min_fraction` (0.5) of that
carrier. When every candidate is discarded, the branch writes the finding `sweep_found: False` and
`conformance = False`.

That finding is contradicted by the branch's own intermediate measurement: a monotone sweep *was*
located. It was discarded for being a minority of an amplitude span whose extent VOICE does not
control — the span wraps onset, offset, and any preparatory or trailing phonation, so the
denominator is set by PREPROCESS's envelope gate, not by the glide.

Measured: **39 of 55** glide no-carrier outcomes went this way — 10/30, 14/30 and 15/30 across the
three glide families.

**Verdict: a defect in the reporting, undetermined in the threshold.** Whether 0.5 is the right
fraction needs listening, and I did not listen. But `sweep_found: False` states more than was
measured whatever the right fraction is, and `conformance = False` on that basis carries a claim
about the participant that the branch's own numbers do not support.

## The rejection leaves no trace in the store

On an in-family `False`, VOICE writes **no span and no deviation naming the gate**:

```
  glides-high-to-low:        spans_n {0: 18}; deviation sets {(): 18}
  glides-low-to-high:        spans_n {0: 19}; deviation sets {(): 19}
  high-to-low:               spans_n {0: 18}; deviation sets {(): 18}
  maximum-phonation-time:    spans_n {0: 26}; deviation sets {(): 26}
  maximum-phonation-time-v2: spans_n {0: 22}; deviation sets {(): 22}
  prolonged-vowel:           spans_n {1: 20, 0: 6}; deviation sets {(): 15, ('omission',): 9, ...}
```

`qualifying_phonation` returns a list; a rejected carrier is simply not in it. The only finding
written is `count("attempt_count", 0, ...)` with no derivation ids (`voice.py:383-386`), and
`unmeasured` is empty because every operating point *had* a value. A reader of the store cannot tell
"this recording holds no sustained phonation" from "a 25-second sustained phonation was measured and
a qualifier threw it away". This census had to recompute the gates from the raw npz derivatives to
separate them — which is the practical reason the defect survived to a 62,578-recording run.

## The two flag families are one event, counted three times

On an in-family recording, a single no-carrier outcome produces three flag records:

1. `conformance = False` → `vocabulary.py:712` → *"VOICE reported that what the instruction asked
   for did not happen on …"*
2. zero proposed spans → `findings[VOICE] = absent` (`_found`, `vocabulary.py:461`) → route `routed`
   and not found → `agreement = mismatch` → `vocabulary.py:729` → *"mismatch: routing routed VOICE,
   it found no subject"*
3. the stem's own family makes ROUTING record `declared = True` for VOICE, so
   `hints[VOICE] = claimed_not_found` → `vocabulary.py:739` → *"hint mismatch: VOICE was declared
   and did not find it"*

Measured over the 300 recordings: 139 + 129 + 109 = **377 VOICE flag records**, and the distribution
per recording is

```
0 flags: 133    1 flag: 57    2 flags: 9    3 flags: 101
```

A hundred and one recordings carry exactly three. That is why 20 smoke recordings produced 23 flag
records, and it means the flag *count* in any corpus report overstates the number of distinct
grounds by up to 3×. Whether the three should be collapsed is the owner's design call, not a defect
claim — but no corpus summary should be read as if they were independent.

## What each of the brief's two patterns is

**Pattern 2 — "every `maximum-phonation-time` flagged" — is a defect.** 26 of 30, and 18 of those
had a voiced, stationary, multi-second carrier rejected by an extreme-value statistic under a
per-window threshold. The control contrast and the duration dose-response both rule out a corpus
finding as the explanation for that subset. The remaining 6 (`none_voiced`) and 2 (`no_span`) are
**undetermined on this sample**.

**Pattern 1 — "a branch found nothing" — is two different things, and neither is a participant
finding.**

- *On VOICE's own families*, `absent` is the same no-carrier event as the `False` — Defect A, B or
  C — surfacing a second and third time through the agreement and hint tables. It adds no
  information and inherits the defect.
- *On control families*, `absent` is the detector behaving correctly: there is no sustained
  phonation in a cough or a read passage. The flag exists because ROUTING routed VOICE there. It is
  a disagreement between the ruleset and the detector, charged to the recording. Whether that should
  flag is a routing-calibration question and a design call; as evidence about the participant it is
  **uninformative**, and it is not a VOICE defect.

## How many recordings this would move

Rates from this sample, applied to the family counts in `corpus_inventory.jsonl` (62,550 rows; the
manifest's 62,578 includes 28 rows without a resolvable family). These are point estimates from
n=30 per family, not confidence-bounded.

| | in-family corpus | measured rate | recordings |
| --- | --- | --- | --- |
| `conformance[VOICE] == False`, all causes | 8,307 | 0.60–0.87 by family | **≈ 6,293** |
| …of which: spread gate rejected a voiced carrier (sustained) | 5,114 | 18/30, 15/30, 15/30 | ≈ 2,827 |
| …of which: `lexical_separator` discarded it (`prolonged-vowel`) | 1,604 | 7/30 | ≈ 374 |
| …of which: glide sweep found then discarded | 3,193 | 10/30, 14/30, 15/30 | ≈ 1,284 |
| **attributable to a named detector-side gate** | 8,307 | | **≈ 4,485** |

So roughly **4,485 of the 8,307 in-family recordings** rest on a gate rejecting evidence the branch
had in hand — and, since each such recording raises the conformance flag plus the agreement and hint
mismatches, on the order of **13,000 flag records**. The all-cause figure is higher still: ≈ 6,293
recordings and ≈ 18,000 flag records, but the difference between the two is the `none_voiced` and
`no_span` group this sample cannot adjudicate.

The control families contribute a further 38 routed-VOICE mismatch flags per 120 recordings. Their
corpus-wide total is **not estimated here**: it depends on VOICE's routing rate in the 42 families
this sample did not cover, and those rates were not measured.

## The fixes, as proposed

What follows was the proposal. The owner approved 1, 2 and the non-refit half of 3 on
2026-09-19; what was actually made, and what it moved, is in *What was fixed* below.

**A. The spread qualifier must not be an extreme-value statistic, and must not produce a `False`.**
Two separable changes, and I would want both:

1. *The statistic.* `max_windowed_spread` reads the maximum over every window. The threshold's
   derivation is a per-window bound. Replace the reduction with a robust summary — the median or a
   high percentile of the window spreads — so that a handful of octave-error windows cannot decide a
   2,500-window production. **The statistic and the threshold must be re-derived together**, against
   listened verdicts, because 2.0 was reasoned for a single window and means something different
   under any other reduction. This is a `data/`-profile fitting job, not a literal edit.
2. *The consequence.* Independently of the threshold, a carrier that fails a **qualifier** is not
   evidence that the task was not performed. A qualifier says where it is safe to measure; failing
   every one of them means the branch could not place a boundary. `voice.py:387-388` should return
   `UNDETERMINED` with an `unviable` finding naming the gate, not `False` — the same shape the
   branch already uses for absent tracks (`TRACKS_ABSENT`). This alone closes the contract violation
   even before the statistic is refitted, and it is the smaller, safer change.

**B. `prolonged-vowel`'s conformance must be about the vowel.** `voice.py:452` should answer the
`SUSTAINED` question — was a sustained production found and measured — and the count-in's absence
should stay what it already is, an `omission` deviation, which the branch is already emitting. The
`lexical_separator` exclusion should not be able to remove the only carrier: on a `SUSTAINED` row it
should trim the count-in off the measurement window rather than discard the whole span.

**C. The glide must report what it measured.** When a sweep is located and then discarded by
`dominant_segment_min_fraction`, that is a measurement with a qualification, not `sweep_found:
False`. Emit the sweep and its fraction; leave conformance `UNDETERMINED` unless no monotone run was
found at all.

**D. A rejection should be visible.** Whatever is decided about A–C, `qualifying_phonation` should
record, per rejected carrier, the gate that rejected it and the value it read. Without that, no
future reader can separate an absent production from a discarded one without re-deriving the npz
sidecars, as this document had to.

The order I would take them: D first (it is additive and makes everything after it auditable), then
A.2 (the contract fix), then B and C, and A.1 last, because it is the only one that needs listening
before it can be chosen.

## What was fixed, and what it moved

The owner approved fixes 1 and 2 outright on 2026-09-19, and the non-refit half of fix 3. All three
landed in `6a99b5f4`. The same 300 recordings were then re-run on the fixed commit, pinned the same
way, and paired per recording against the pre-fix pass — so every row below is the same recordings
on both sides.

| | before | after |
| --- | --- | --- |
| commit | `f2729d7c` | `6a99b5f4` |
| recordings | 300 | 300 (299 paired at first census; the last slice landed after) |
| pin violations | 0 | 0 |
| driver row failures | 0 | 0 |

### 1 — a discarded carrier is now reported

`qualifying_phonation` returns a `Qualification` carrying both what qualified and what it discarded,
and every discarded span becomes a `carrier_rejected` **measurement** — not a deviation, because it
is a reading of this branch's instrument rather than a departure by the speaker — naming the gate,
the value read, the bound, and the carrier's length. The branch report carries the same list under
`carriers_rejected`, so a corpus reader needs no store at all.

```
  before: 0 rejections recorded, on 0 of 300 recordings
  after: 1413 rejections recorded, on 207 of 300 recordings
```

By gate: `production_min_s` 864, `voiced_fraction_min` 386, `f0_spread_max_semitones` 78,
`dominant_segment_min_fraction` 43, `lexical_separator` 42. The question this document had to answer
by re-deriving npz sidecars — did the branch find nothing, or throw something away? — is now a field
in the report.

The glide arm no longer writes `sweep_found: False` about a sweep it located: a sweep discarded by
`dominant_segment_min_fraction` is recorded with its direction, its length and the fraction it held.

### 2 — conformance is `UNDETERMINED` where the branch could not measure, and never `False`

Taken wider than `prolonged-vowel`, as asked. `_voice_sustained` no longer returns the count-in
match as the task's conformance; the count-in's absence stays the `omission` deviation it already
was. Both arms return `UNDETERMINED` when no carrier or no sweep survived. **VOICE now writes no
`False` at all**, and a test asserts that structurally: these instruments can establish that a
sustained production happened, not that none did.

```
family                           n           True          False     UNDETERMINED
                                     before after   before after   before   after
glides-high-to-low              30       12    12       18     0        0      18
glides-low-to-high              30       11    11       19     0        0      19
high-to-low                     30       12    12       18     0        0      18
maximum-phonation-time          30        4    20       26     0        0      10
maximum-phonation-time-v2       30        8    20       22     0        0      10
prolonged-vowel                 30        4    19       26     0        0      11
diadochokinesis-pa              30        0     0        0     0       11      11
loudness                        30        0     0        0     0        7       7
rainbow-passage                 30        0     0        0     0       19      19
respiration-and-cough-cough     30        0     0        0     0        2       2
```

The three sustained families are where the substance is: `True` goes 4 → 20, 8 → 20 and 4 → 19 of
30. Those are recordings whose sustained phonation the branch was measuring and then throwing away.
The glide families are unchanged at 12, 11 and 12 — correctly, because `dominant_segment_min_fraction`
is the part that needs listening and was left alone; their `False` became `UNDETERMINED`.

### 3 — the per-window bound is read off a representative window

`max_windowed_spread` is replaced by `windowed_spreads` and `typical_windowed_spread`. The median
window's spread is compared against the same `f0_spread_max_semitones = 2.0`. A window carrying
fewer than two finite values is dropped rather than voting as a spread of zero, and an unreadable
spread no longer rejects — an unmeasured qualifier could neither admit nor reject.

**No new operating point, and none was needed.** The candidate constant was the fraction of windows
that must lie within the bound. Derived across the ten strata — positives being the best carrier on
a sustained in-family recording, negatives the same on connected speech:

```
 cut   positives kept   negatives kept   Youden J
 0.45        60/68            9/44        +0.678   <- empirical optimum
 0.50        59/68            9/44        +0.663   <- "a majority", the house convention
```

The two differ by **one recording in 112**. `median spread ≤ bound` is exactly `within-bound
fraction ≥ 0.5`, so the majority convention that `voiced_fraction_min`, `continuity_min`,
`breath_coverage_min` and `dominant_segment_min_fraction` already ship is indistinguishable from the
fitted optimum here. Adding a config key to encode 0.45 instead would have been a fitted-looking
number bought for one recording. It was not added.

**The defect's own signature is gone.** Rejection rate against carrier duration, in-family sustained:

```
carrier duration (s)    max > 2.0 st (before)    median > 2.0 st (after)
0-2                          0.40                      0.30
2-5                          0.72                      0.22
5-10                         0.75                      0.12
10-20                        0.79                      0.10
20-100                       0.83                      0.00
```

Before, the gate was most likely to reject the longest productions — on a task whose measurement is
how long the vowel was held. After, the relation inverts: a longer sustained production reads as
more clearly sustained, which is what a qualifier should do.

**The gate still rejects what it is for.** The wobble case is still discarded, pinned by a test, and
the cost on the control families is small and visible:

```
family                          routed  before  after  rate before  rate after
maximum-phonation-time              29      25      9         0.86        0.31
maximum-phonation-time-v2           27      19      7         0.70        0.26
prolonged-vowel                     29       6      1         0.21        0.03
diadochokinesis-pa                  11      11     10         1.00        0.91
loudness                             7       7      6         1.00        0.86
rainbow-passage                     19      18     15         0.95        0.79
respiration-and-cough-cough          2       2      2         1.00        1.00
```

Three control recordings out of 37 routed ones gained a phonation span they did not have before.
That is the price of the looser statistic and it is worth naming: on an out-of-family recording
`detect_voice` evaluates no task, so a spurious span there is a covariate, not an accusation.

### Tests

Nine existing tests encoded the old behaviour and were rewritten to the new contract rather than
deleted. Fifteen were added. Every fix was mutation-tested — the fix reverted one piece at a time on
top of the new code, so the suite still imports:

| mutation | undoes | caught by |
| --- | --- | --- |
| `M1` median → max | the statistic | 2 tests |
| `M2` the count-in decides again | conformance | 4 tests |
| `M3` no carrier → `False` | conformance | 7 tests |
| `M4` rejections silent | visibility | 3 tests |
| `M5` glide decides on a discarded sweep | conformance | 3 tests |

No mutation survived. `2038 passed` across `src/tests/audio/workflows/triage/`, ruff clean, mypy
clean on a purged cache.

## The triple count is VERDICT's, and this is where it arises

Not fixed, as instructed. Three flag records for one no-carrier event, and all three read **one
number**: `findings[branch]`, the count of spans the branch proposed.

| # | site | condition | text |
| --- | --- | --- | --- |
| 1 | `vocabulary.py:712` | `report.conformance is False` | `… reported that what the instruction asked for did not happen on <family>` |
| 2 | `vocabulary.py:728-731` | `agreement[branch] == MISMATCH`, from `_agreement` (`:511`) reading `findings[branch]` | `mismatch: routing routed VOICE, it found no subject` |
| 3 | `vocabulary.py:737-740` | `hints[branch] == CLAIMED_NOT_FOUND`, from `_hint_reading` (`:542`) reading the same `findings[branch]` | `hint mismatch: VOICE was declared and did not find it` |

`_found` (`:461`) computes `findings[branch]` once from the span count. `_agreement` and
`_hint_reading` then each compare it against a different expectation — the ruleset's content route,
and the stem's declared family. Those two expectations are genuinely different inputs, but on
VOICE's own families they agree almost perfectly (the ruleset routed VOICE on 29 of 30
`maximum-phonation-time` recordings and the stem declared it on 30 of 30), so in practice both fire
together, on the same span count, alongside the branch's own conformance about the same absence.

**Fix 2 already removed one of the three.** With VOICE writing no `False`, ground 1 never fires for
this branch: 378 flag records became 178, and no recording carries three.

```
  before: 378 records over 300 recordings; per recording {0: 133, 1: 57, 2: 9, 3: 101}
  after:  178 records over 300 recordings; per recording {0: 190, 1: 42, 2: 68}
```

**The smallest honest change** would take the remaining two to one. Measured on the pre-fix data,
where all three grounds were live, two candidates:

| candidate | change | records | per recording |
| --- | --- | --- | --- |
| *merge* — grounds 2 and 3 become one reason naming both disappointed expectations | one `if` in the `for branch in branches_seen` loop | 378 → 277 | no recording carries 3 |
| *report-gated* — raise 2 and 3 only where the branch left **no report** | one added condition on the same loop | 378 → 129 | at most 1 per recording |

I would take *report-gated*, and the reasoning is the report/decide split this work is built on: a
branch that filed a report has already stated its conclusion about the recording, and the span count
is the evidence for that conclusion rather than two further findings about the recording. Where a
branch is *silent*, the route and the declaration are the only signals there are, and the mismatch
is the right thing to raise. Both the `agreement` and `hints` tables stay in `FileVerdict.record()`
either way, so nothing is lost from the decision record — what changes is only what reaches
`reasons`.

One consequence to see before choosing: combined with fix 2, *report-gated* would leave VOICE
raising **no flags at all** on this sample. That is defensible today — the branch has no fitted
criterion for anything it could honestly flag on — but it means VOICE becomes purely descriptive
until one exists, and that is a decision about the graph rather than a repair, which is why it is
here and not in the commit.

## Reproducing this

```bash
# on ORCD, from a pinned clone, with PYTHONPATH pointing at its src
python voice-flag-grounds-census.py <driver-output-root> census.jsonl
python voice-flag-grounds-tables.py census.jsonl
python voice-flag-grounds-compare.py before.jsonl after.jsonl
```

The census walks `run/store.jsonl` directly rather than the driver's row files, so a run whose
post-run bookkeeping failed is still censused. It emits counts, family names, gate names and
aggregate numbers only; no transcript text and no detected string leaves the secure area.
