# DDK branch

What the branch answers: **is there a repeated syllable train here, how fast is it, and how regular?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).
This document is grounded in the code and in [`dag.md`](dag.md).

## The state of this branch

**There is no node.** No `src/senselab/audio/workflows/triage/nodes/ddk.py` exists, `DDK` is absent
from `GRAPH_ORDER` (`vocabulary.py:14-25`) and absent from `run.py`'s dispatch table
(`run.py:297-301`). `DDK` *is* in `BRANCHES` (`vocabulary.py:31`), so ROUTING writes it a
`branch_decision` like any other and `_drive_branches` records it `SKIPPED` with `NO_NODE`
(`run.py:302-305`).

**And since ruleset stage 2 it is routable for the first time.** No `kind` ever mapped to DDK, so
under the old fold it could not be selected at all; the ruleset routes it on measured content. The
consequence is live: a recording whose DDK gates fire now flags the file, because VERDICT records
"was asked to run and never ran". That flag is honest — it fires only on recordings with DDK
content, not on every recording — and it is the standing argument for building the node.

This document is therefore a specification, not a description. Everything below is **not built**
unless it says otherwise.

## The tasks this branch serves

Declared families, not ground truth.

| family | n | what the task asks for |
| --- | --- | --- |
| `diadochokinesis-buttercup` | 896 | repeat a trisyllabic word |
| `diadochokinesis-ka` | 896 | repeat /ka/ |
| `diadochokinesis-pa` | 896 | repeat /pa/ |
| `diadochokinesis-pataka` | 896 | repeat /pa-ta-ka/ |
| `diadochokinesis-ta` | 896 | repeat /ta/ |
| `diadochokinesis-v2-buttercup` | 702 | as above |
| `diadochokinesis-v2-kuh` | 702 | repeat /kuh/ |
| `diadochokinesis-v2-puh` | 702 | repeat /puh/ |
| `diadochokinesis-v2-tuh` | 702 | repeat /tuh/ |
| `diadochokinesis-v2-puhtuhkuh` | 701 | repeat /puh-tuh-kuh/ |

7,989 declarations, and the protocol's own instruction says what is wanted — the
`_acoustictask-metadata.json` for a v2 DDK task reads *"imitate the speaker by repeating the
syllables 'puhtuhkuh' as quickly and consistently as possible until the timer runs out"*.

**"As quickly and consistently as possible" is the measurement.** Diadochokinetic rate is a
standard clinical measure of articulatory motor control, and it has two components a speech
scientist would expect: **rate** (syllables per second) and **regularity** (the variability of the
inter-syllable interval). The task splits into *alternating motion rate* — a single repeated
syllable, the `pa`/`ta`/`ka`/`puh`/`tuh`/`kuh` families, 4,794 declarations — and *sequential motion
rate*, a repeated sequence of different syllables, the `pataka`/`puhtuhkuh`/`buttercup` families,
3,195. The two are clinically distinct: sequential motion rate stresses the ability to switch place
of articulation and is the more sensitive of the two.

### A recording routed here whose declared task is not DDK

DDK routed 22,363 recordings against 7,989 declaring a DDK family
(`runs/ruleset-score-20260912/ruleset_score.json`), so most recordings reaching it are not DDK
tasks. Repetition occurs in ordinary speech — a stutter, a false start, a repeated word in a
sentence. The branch measures the repetition it finds and says what it is; it does not assert that
a Harvard sentence failed to be a DDK task.

**DDK's `unavailable` count is 2,345** in the same artifact — the highest of the four branches — so
a substantial population reaches the branch with its gate evidence missing rather than silent. The
design must treat an unavailable posteriorgram as an absence, not a negative.

## How DDK is routed today

Two gates, either of which routes (`default.yaml:211`, `:257-264`):

| gate | feature | threshold |
| --- | --- | --- |
| `ddk.lexical_repetition` | `transcript_repeat` — largest repeat count of any normalised transcript token | `>= 3`, and the config comment says **UNMEASURED** |
| `ddk.ppg_segment_rate_per_s` | `ppg.segment_rate_per_s` — contiguous argmax-phoneme segments per second | `>= 10` |

The second is the interesting one: it reads no transcript, so it routes a syllable train an ASR
declines to transcribe — which is most of them, since `/pa-pa-pa/` is not lexical.

**Neither threshold may be refit.** Both would be fitted against the declared DDK families, which
would encode which recordings the protocol labelled rather than which carry a syllable train. The
`>= 3` is already marked unmeasured in the config and should stay marked.

## Capabilities

### D1 — Locate the syllable train (**not built**)

**Question.** Where in the recording is the repeated production?

**Reads.** The PPG posteriorgram PREPROCESS writes, via `extract_ppg_segments`
(`tasks/features_extraction/ppg.py:349`), which returns contiguous argmax-phoneme segments each
carrying `phoneme`, `start_seconds`, `end_seconds` and `duration_seconds`.

**Computes.** The extent over which segment production is sustained. A DDK task is a burst of rapid
alternation bounded by silence at both ends; the train is the region where segments recur without a
long gap.

**Emits.** A `propose` span, `family: "DDK"`, `wasDerivedFrom` the posteriorgram measurement — the
train as one region. Where PREPROCESS already proposed a span over the same audio with a different
extent, a `refine` assertion carrying `corrected_extent` rather than a second span.

**Parameter-free?** Not entirely — "without a long gap" needs a gap criterion. But the criterion can
be stated relative to the train's own segment durations rather than as an absolute: a gap that
exceeds the train's own inter-segment interval by some factor breaks it. The factor is **owed
ground truth**; the relative formulation at least keeps it scale-free across speakers.

### D2 — Syllable rate (**not built**)

**Question.** How many syllables per second?

**Two independent sources, and they should both be reported.**

`extract_speech_rate` (`praat_parselmouth.py:91`) returns `speaking_rate` (syllables ÷ duration),
`articulation_rate` (syllables ÷ phonation time), `phonation_ratio`, `pause_rate` and
`mean_pause_dur`. For DDK, **`articulation_rate` is the clinically meaningful one** — it excludes
pause time, and DDK rate is conventionally reported over the productive portion.

The PPG segment rate is the second source: segments per second over the D1 train. It is what
`ddk.ppg_segment_rate_per_s` already routes on.

**They measure different things and will disagree.** Praat's syllable detection is intensity-peak
based; the PPG rate counts argmax-phoneme changes, so a single syllable with an onset consonant and
a vowel yields two or more segments. Reporting both, with their definitions named, is more useful
than reconciling them into one number — and reconciling them would need a calibration nobody has.

**Emits.** A per-span measurement carrying both rates with their sources.

**Serves.** All ten families.

### D3 — Regularity (**not built**)

**Question.** How consistent is the inter-syllable interval?

**Reads.** The D1 train's segment boundaries from `extract_ppg_segments`.

**Computes.** The sequence of inter-onset intervals, and its dispersion — standard deviation, and
the coefficient of variation, which is dimensionless and therefore comparable across speakers with
different rates. Irregularity of DDK is as clinically informative as slowness, and in some
presentations more so.

**Emits.** A per-span measurement. **No verdict** — mapping a coefficient of variation to normal or
disordered needs norms this project does not have.

**Parameter-free?** Yes, as a measurement. The dispersion of a sequence needs no threshold; only
interpreting it would.

**Serves.** All ten families.

### D4 — Sequence conformance (**not built**)

**Question.** Did the speaker produce the syllables the task asked for, in order?

**Reads.** The declaration — the per-task table's expected syllable sequence, derived from the
`instructions` text — and the PPG segment sequence from D1.

**Computes.** Whether the produced segment sequence is a repetition of the expected one. For
`/pa-ta-ka/` the expected cycle is three distinct places of articulation in order; a speaker who
produces `/pa-ta-pa/` has substituted, and a speaker who produces `/pa-pa-pa/` has collapsed a
sequential task into an alternating one. That collapse is a clinically meaningful finding.

**Emits.** One deviation per departure, carrying its extent.

**This is the capability that most needs the declaration**, and it is the reason DDK benefits from
the contract more than any other branch: without knowing which syllables were asked for, a segment
sequence is uninterpretable.

**Owed.** The mapping from PPG phoneme labels to the expected syllables. The posteriorgram's
inventory is its own; whether `/puh/` reliably surfaces as a particular argmax phoneme is
**unmeasured**.

### D5 — Syllable count (**not built**)

**Question.** How many syllables were produced?

The count over the D1 train, reported as a `counts` entry. Unlike the respiration families, no DDK
task declares an expected count — the instruction is "until the timer runs out" — so the `declared`
half is absent and the entry carries `found` alone. Recorded because rate × duration is not a
substitute for a count when the train is interrupted.

## Deviations

| type | evidence |
| --- | --- |
| `stimulus_mismatch` | a produced syllable that is not the one the sequence expected (D4) |
| `off_task_extent` | lexical speech, or silence, where the task expected sustained repetition |

`filler` does not apply — DDK expects no lexical content, so a disfluency has nothing to be a
disfluency against.

Syllable count is a `counts` entry (D5), not a deviation.

## What exists today

| capability | status |
| --- | --- |
| D1 locate the train | **not built** — `extract_ppg_segments` exists and nothing in a branch calls it |
| D2 syllable rate | **not built** — `extract_speech_rate` exists; no branch consumes it |
| D3 regularity | **not built** |
| D4 sequence conformance | **not built**; needs the declaration |
| D5 syllable count | **not built** |

**The routing side is built and working**: both gates are defined and the branch routes. What is
missing is everything downstream of the decision.

**What senselab already provides**: `extract_ppg_segments` (`ppg.py:349`),
`extract_mean_phoneme_durations` (`ppg.py:429`), `to_frame_major_posteriorgram` (`ppg.py:287`),
`load_ppg_posteriorgram` (`ppg.py:327`), and `extract_speech_rate` (`praat_parselmouth.py:91`).

**What is missing**: no module computes an inter-onset interval sequence or its dispersion from
segments, and no mapping exists from PPG phoneme labels to a task's expected syllables.

## What the branch would emit

```
spans        family: "DDK", one per syllable train
assertions   refine (corrected_extent) where a PREPROCESS span's extent is wrong;
             contest where a proposed span carries no repeated production
measurements per-span rate (D2, both sources), regularity (D3)
counts       syllable count {found}, declared absent
deviations   stimulus_mismatch, off_task_extent
verdict      { trains_n, longest_train_s, articulation_rate, ppg_segment_rate_per_s,
               interval_cv, flags }
```

**The verdict's basis, proposed**: `FAIL` when no syllable train was found; `FLAG` when a
measurement contradicts another — for instance a PPG segment rate that routed the branch over a
region where Praat finds no syllables at all; `PASS` otherwise. The rates and the regularity are
carried as measurements, and **no normative judgement enters the verdict**.

## Out of scope

Any normative interpretation of rate or regularity. Any refit of `ddk.lexical_repetition` or
`ddk.ppg_segment_rate_per_s` against declared families. Reconciling the Praat and PPG rates into one
number.
