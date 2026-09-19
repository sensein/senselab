# DDK — which instrument's reading is the `task_extent`

What the rule is, the corpus measurement that chose it, and the one case where the candidates
differ. Landed 2026-09-19. The code is `src/senselab/audio/workflows/triage/nodes/ddk.py`
(`cv_task_extent`, `merged_task_extent`, `align_ddk`). No operating point is introduced; see
[the no-threshold ruling](#why-no-plausibility-threshold).

## The defect

The owner, reading a rendered summary of `sub-004d42e9…_task-diadochokinesis-pataka` (8.66 s):

> syllable train is incorrect. it doesn't cover the hull of all syllables.

The store held:

```
task_extent    2.635 - 2.733   (0.10 s)   <- what shipped
ppg_train      5.411 - 8.591   (3.18 s)
```

The CV instrument on the same recording reads **20 units spanning 1.530 – 8.591**, four complete
cycles, seven insertions. The shipped `task_extent` covered 1.4% of a 7.06 s performance.

### The mechanism, which is not "the carrier was short"

`ddk_carrier` found a carrier at **2.237 – 4.114** — 1.88 s, comfortably past `train_min_s`. The
extent `align_ddk` mints is not the carrier but `hull(onsets) or train.extent`, and
`events_in_span` found **exactly one** onset inside that carrier. The hull of one event is that
event: 0.0986 s.

So the envelope instrument had *a* reading and it was a bad one, and
[`ddk-cycle-counting.md`](ddk-cycle-counting.md) made presence the whole condition — `cv_task_extent`
minted the CV hull only "when the envelope instrument proposed none", and `align_ddk` dropped it
"when the envelope instrument did". The precedence was backwards and it was conditioned on
**presence** rather than **plausibility**.

It also contradicted [`ddk-instrument-over-asr.md`](ddk-instrument-over-asr.md), merged the same
day: for a declared syllable family, over the extent it covers, the CV instrument is the authority.

## How often this fires

Measured over **all 7,994 declared-DDK recordings** in the corpus, replaying both shipped
instruments against each run's own stored provenance and derivatives — `ddk_carrier` →
`events_in_span` → `hull(onsets)` on the envelope side, `ppg_reading` → `cycle_evidence` →
`cv_covered_extent` on the CV side. Slurm job 23105513 on `mit_preemptable`, 61 s, 0 failed rows.
Script and per-recording rows:
`/orcd/scratch/bcs/002/satra/checks_20260916/task_extent_sweep_20260919/`.

| | count | of what |
|---|---|---|
| declared DDK recordings | 7,994 | — |
| envelope minted a `task_extent` | 4,669 | **58.4%** of all |
| …and a readable CV hull also existed | 4,563 | **97.7%** of those |
| …and the envelope extent was **under half** the CV hull | 2,637 | **57.8%** of those, **33.0% of all** |
| CV hull but no envelope extent (the case already handled) | 3,200 | 40.0% of all |
| neither instrument read anything | 125 | 1.6% of all |

Distribution of `duration(envelope extent) / duration(CV hull)` over the 4,563 recordings where both
exist:

| p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|
| 0.033 | 0.187 | **0.408** | 0.819 | 1.011 |

| <0.1 | 0.1–0.5 | 0.5–0.9 | 0.9–1.1 | >1.1 |
|---|---|---|---|---|
| 699 | 1,938 | 1,009 | 645 | 272 |

**This is the normal path, not an edge case.** The median shipped `task_extent` covered 41% of the
performance the CV instrument read, and on a third of every declared-DDK recording in the corpus it
covered less than half. The three-syllable families are worst — `pataka` median 0.268, `buttercup`
0.225, `v2-puhtuhkuh` 0.285 — against roughly 0.5–0.6 for the single-syllable families, which is
what one expects: a three-place cycle modulates the envelope at the cycle rate, so the peak walk
resolves a third as many onsets as there are syllables.

## The rule

> **The surviving `task_extent` is the hull of both instruments' readings.**

Stated as the three cases, of which only the third changed:

1. The envelope instrument read a carrier and the CV instrument had no readable reading →
   the envelope's extent, `production` `syllable_train` or `syllable_sequence`. **Unchanged.**
2. The CV instrument read a hull and the envelope instrument found no carrier →
   the CV hull, `production` `syllable_task_from_ppg`. **Unchanged.**
3. Both read the task → one span over `min(starts)` to `max(ends)`, `production`
   `syllable_task_from_both`. **This is the change.** It was case 1's answer before.

`merged_task_extent` performs the merge and is the only place a `task_extent` for this family is
decided. The span carries the CV instrument's own measurements — `syllables_n`, `cycles`,
`consumed` — and names the envelope's contribution beside them as `envelope_syllables_n` and
`envelope_extent_s`, so a reader can see both readings and how far apart they were without going
back to the derivatives.

### Why the union, and not "the CV instrument always wins"

The authority principle is **scoped**. `ddk-instrument-over-asr.md` says the CV instrument is the
authority *over the extent it covers*. Outside that hull the CV instrument has not read low — it has
not read at all, and an instrument that makes no claim over a region cannot suppress another
instrument's claim over it. That is exactly the asymmetry the old rule got wrong in the other
direction, and adopting "CV wins outright" would have reproduced it with the instruments swapped.

The corpus says this is not hypothetical. On **1,341 of the 4,563** both-exist recordings (**29.4%**)
the envelope extent reaches past the CV hull on one side or the other, so "CV wins outright" would
have discarded produced material on nearly a third of them. The 272 recordings with a ratio above
1.1 are envelope readings *wider* than the whole CV hull.

The union also has a property worth stating plainly: **it is monotone**. The merged extent contains
the old one in every one of the three cases, so no recording loses coverage it already had. A
regression in this change can only ever be an extent that is too wide, never one that is too narrow
— and too-narrow is the defect being fixed.

### The out-of-hull case

Where "union" and "CV wins" differ is an envelope carrier lying wholly or partly outside the CV
hull, and **170 recordings (3.7% of both-exist) are fully disjoint** — no overlap at all.

Under this rule such a reading **widens the extent and bridges the gap**: the span becomes
`(min(starts), max(ends))` and covers the silence between the two readings. That is deliberate, and
it is what the role already means elsewhere. `task_extent` is a convex boundary, not a mask — the
envelope instrument's own extent is already `hull(onsets)`, which bridges every inter-syllable
silence, and `ddk-cycle-counting.md` fixed the reading of the span as *"between these times, the
subject produced consonant-vowel syllables in response to a syllable-repetition instruction"*, which
"asserts nothing about whether they were correct, regular, or continuous" and is "not claiming every
instant inside it holds speech".

The clinical case this protects is the one that motivates it: the CV walk runs off a posteriorgram
and can miss a stretch the envelope resolved — a quiet run-in, a burst after a long pause. Those are
the subject attempting the task. Excluding them is the same error as excluding the syllables no
complete cycle consumed, which that document already rejected.

What is given up is that a spurious envelope carrier far from the performance now stretches the
extent instead of replacing it. That is strictly the lesser failure: before this change such a
carrier *was* the whole extent.

### Why no plausibility threshold

The obvious alternative is to admit the envelope's contribution only when it is plausible — within
some multiple of the CV hull, or overlapping it by some fraction. Every such rule is a number, and
a number here would be a code literal nothing has fitted, which is the defect
[CLAUDE.md](../../CLAUDE.md) names directly. The sweep above measures how far apart the two readings
fall but says nothing about which of the disjoint 170 are real second performances and which are
noise — separating those needs listening verdicts that do not exist yet.

The union needs no such number: it is conditioned on nothing but whether each instrument read
anything at all. If a fitted plausibility gate is ever wanted, it belongs in
`data/config/default.yaml` with its derivation in [`config-derivations.md`](config-derivations.md),
and the verdicts have to be collected first.

## The one-span invariant, preserved

Exactly one span of role `task_extent` survives a recording. Two would make `syllable_detail`'s
`trains_n` report two trains over one performance, which is the reason the original drop existed and
is the one thing about it that was right.

`align_ddk` now partitions `cv_spans` rather than filtering it: the CV `task_extent` is pulled out
by role and handed to `merged_task_extent`, everything else is carried through untouched. Both early
returns — absent envelope derivative, no carrier past the length guard — still return `cv_spans`
directly, which holds at most one. `TRAIN_ROLES`, a one-element tuple whose docstring said "the
roles that are a train" while holding `task_extent`, is gone; `TASK_EXTENT` replaces it, matching
what `voice.py` and `airway.py` already call the same constant.

## What is not changed

- **`ppg_train`.** Its extent is the train finder's contiguous regular stretch, which is a
  segmentation by *regularity*, not a task boundary — on the owner's recording, correctly, 5.411 –
  8.591 inside a 1.530 – 8.591 hull. The merge does not reach it, and
  `test_the_ppg_train_is_unchanged_by_the_merge` pins that against a fixture whose train is a proper
  subset of its hull.
- **`cv_covered_extent`'s gate.** Still "a complete cycle or a train"; still no extent over a task
  with no evidence it was performed. Only the precedence downstream of it moved.
- **Every finding.** `PPG_RATE`, `RATE`, the dispersions, `realised_cycles`,
  `sequence_collapse_fraction`, `INSTRUMENT_READING` and the contests are all as they were, and all
  still read off their own instrument's own extent.
- **Conformance.** `_with_ppg` and `done` are untouched.

One thing did move with it: the `truncation` deviation now keys on the surviving `task_extent`
rather than on the envelope's extent alone. It is a claim about the span that ships, and leaving it
keyed on a span that no longer ships would have made a task extent reaching the recording's edge
silently stop being a truncation.
