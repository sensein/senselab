# The family x ruleset-gate matrix

What routes a recording today is the gates under `taxonomy.ruleset.gates` in the triage config.
This measures those, one cell per (task family, gate), and qualifies every branch that routing and
the declared task family disagree about.

## What this is not

`scripts/analyze_routing_evidence.py` and `routing_analysis/report.py` score **candidate**
detectors (`DETECTORS`, `BRANCH_DETECTORS`) against reference standards over a threshold sweep.
They predate the ruleset merge and evaluate no shipped gate: `grep gate report.py` returns nothing
from the sweep path. Their `sweeps_by_family.json` is about candidates, not about what routes
recordings.

`scripts/score_taxonomy_ruleset.py` **does** score the shipped ruleset, and it already carried
most of the branch-level half of this work: `score_branches` gives each branch's 2x2 against its
reference family set, and `tally_families` gives per-family counts. Two things it did not have, and
this work adds:

1. Its `families.parquet` is family x **branch**, not family x **gate**, and carries no gate
   readings at all.
2. `Confusion` had sensitivity, specificity and a false-positive rate but no **precision**, so
   the over-routing side
   of the ruleset could not be read off it.

## Why the logic sits in `gate_matrix.py` and not in `report.py` or `ruleset.py`

- `report.py` (1366 lines) is the candidate-detector sweep: `REFERENCE_STANDARDS`, `score_detector`,
  `recall_at_budgets`, `bucket_*`. Its subject is a detector that might ship. Ours is the ruleset
  that did. Adding a second subject to that module makes "which of these two things am I scoring"
  a question the reader has to keep answering.
- `ruleset.py` (751 lines) is the **per-recording** contract: load the ruleset, evaluate one
  recording, and the two corpus reductions (`tally_families`, `score_branches`) that the branch
  axis needs. Ours is a different reduction over a second axis, plus the value recomputation that
  `RouteEvaluation` deliberately does not carry.
- A new module keeps both contracts intact and gives the aggregation its own test surface. It
  imports from `ruleset.py` and adds nothing to it except the `precision` property, which belongs
  on `Confusion` because every other rate already does.

`gate_plot.py` is separate again, because a plot module that can be imported without the analysis
is a plot module that cannot smuggle an analysis knob into a figure.

## Why the values must be recomputed

`route_attributes` (`live_evidence.py:171-190`) records `gate_outcomes` — the fired/silent/
unavailable verdict per gate — and **not** the number behind it. Verified against the code and
against a run's `summary.json`, whose `routing.<BRANCH>` carries `route_state`, `unavailable_gates`
and `flags` and no reading. So the matrix re-reduces finished stores through `extract_features`,
then `gate_value` / `evaluate_gate`, rather than reading values back.

## `unavailable` is its own category

Three counts per cell, never two: `fired`, `silent`, `unavailable`. Two fired rates, because the
denominator is a real choice:

- `fired_rate` = `fired / n` — over every recording of the family, so it falls when evidence goes
  missing.
- `fired_rate_evaluable` = `fired / (fired + silent)` — over the recordings the gate could read.
  **`None` when no recording of the family could evaluate the gate**, and never `0.0` there.

The heatmap draws `fired_rate_evaluable`, so a `None` cell is hatched and unannotated rather than
painted at the bottom of the colour scale. "No recording of this family had this gate evaluable"
and "the gate never fired" are different facts and must look different.

This distinction is load-bearing and not hypothetical: on the 13-recording sample of 2026-09-15,
`airway.cough` was unavailable on 12 of 13, and readable on exactly one (43.10 dB against a
threshold of 50.0, so silent).

## Declared family is not ground truth

Nobody annotated which recordings contain speech. The declared task family says what the
participant was **asked** for. `specs/.../branch-voice.md` opens its task table with "Declared
families, not ground truth", and `specs/20260910-taxonomy-routing-evidence/measurements.md` says
"There are no branch labels."

So every rate in `branch_agreement.json` is an **agreement rate with the declaration**, and the
output says so in `framing`. A recording declared `prolonged-vowel` that routes to SPEECH because
the participant talked through the vowel is routing being right and the declaration being
uninformative. The column names carry `agreed` / `extra` / `missed` / `silent` rather than
correct/error/false-positive, and the disagreement table's two directions are `missed` (declared,
not routed) and `extra` (routed, not declared).

`Confusion`'s field names stay `tp/fp/tn/fn` because they are pre-existing and shared with the
candidate-detector path; the report layer never surfaces them under those names.

The family -> branch reference is read from the config's `taxonomy.ruleset.reference_family_set`,
which resolves through `FAMILY_SETS` into `families.py`'s own sets — no second mapping is written.
As shipped that is `AIRWAY: airway`, `SPEECH: lexical_speech`, `VOICE: voice`,
`DDK: syllable_repetition`. Note that `families.py` also defines `sustained`, `glide`, `cough`,
`breath` and `speech`, and **no branch references any of them**; the `"sustained"` entry in
`DECLARED_KIND` is not the VOICE reference set, `voice` (= `VOICE_ELICITING`) is.
`unassigned_families` reports every family in no branch's reference set on its own row, because
such a family is in no denominator on either side and must not sit silently in a negative.

## The three findings, and the deciding gate

A disagreement is qualified by the gate it turns on:

- for a **miss**, the branch gate that came closest to firing;
- for an **extra**, the firing gate that cleared its threshold by least.

and classified as one of:

| finding | what it means | what question it raises |
| --- | --- | --- |
| `unavailable` | every gate of the branch was unreadable | a missing-feature question |
| `near_threshold` | the deciding gate read within the band of its cut | a threshold question |
| `far` | it read outside the band | an evidence question |

Exhaustive and non-overlapping. An extra can never be `unavailable`, because something fired; the
fact that some *other* gate of that branch was unreadable while the branch routed anyway is
reported beside the finding as `n_unavailable_gates` (and per group as
`n_with_unavailable_gate`), not folded into it. That keeps "routed on partial evidence" visible
without making the three classes overlap.

The margin is relative: `(value - threshold) / |threshold|`, signed so positive is firing whichever
way the comparison points, and scaled so gates in seconds, dB, counts and probabilities are
comparable on one band. A gate with a zero threshold is scaled by 1 instead of dividing by zero.

## `near_threshold_band` = 0.25 is a convention, not a fit

Lives in `data/disagreement_profile/2026-09-15.yaml` (dated and versioned, per the
`classifier_ontology` idiom), refused on load if absent or non-positive, and its value is stamped
into every output's header. Nothing in the pipeline reads it.

**What was measured.** Every deciding gate on the 13-recording b2ai sample of 2026-09-15 (3
subjects; `runs/<tag>/{nohint,hinted}/`, gate readings identical across the two hint conditions, so
26 stores are 13 distinct recordings). That sample carried **0 missed branches** and **13 extra
branches**. The deciding gate's absolute relative margin over those 13, sorted:

```
0.0000  0.1023  0.3333  0.5000  0.5606  0.6667  0.7681
1.3776  1.4857  1.5214  1.6667  4.5000  9.0000
```

**Why nothing was fitted to it.** The only candidate empirical cut is the gap between 0.7681 and
1.3776. It is one gap in 13 points from 3 subjects, it separates no two named findings, and a band
placed in it would report 7 of 13 extras as threshold questions — a statement about three people.
Fitting to it would be exactly the defect CLAUDE.md names: an unmeasured decision wearing a
measured number's clothes.

**Why 0.25.** It is the reading of "just under threshold" that needs no sample: a gate within a
quarter of its own cut flips its call if the cut moves by a quarter. It is stated as a convention so
a reader can move it and watch the table move, rather than inheriting it as a finding.

**What would replace it.** The corpus run measures the same margin over ~62.5k recordings and every
family. If that distribution is bimodal, the trough is a derivation and the profile gets a new dated
version carrying it.

## Validation against the local sample

All four briefed figures reproduced exactly on the 26-store local sample:

| check | expected | measured |
| --- | --- | --- |
| `ddk.ppg_segment_rate_per_s`, two real DDK recordings | 12.33, 14.70 | 12.33 (`diadochokinesis-buttercup`), 14.70 (`diadochokinesis-v2-puhtuhkuh`) |
| same gate, every speech recording | <= 8.89 | max 8.89 (`rainbow-passage`); then 8.643, 7.441, 7.096 |
| `voice.sustained`, MPT recording | 15.89 s | 15.89 (`maximum-phonation-time-v2`) |
| `airway.cough` availability | unavailable 12 of 13 | unavailable 24 of 26 stores = 12 of 13 recordings |
| `airway.cough`, the one readable | 43.10 dB, threshold 50 -> silent | 43.10, silent |

### What the sample also showed, against expectation

Recall was 1.000 on all four branches (13 of 13 declared branches routed) and precision 0.400 to
0.571, so every disagreement on this sample is an over-route. The deciding gates of those 13
extras, smallest relative margin first:

| deciding gate | margin | recording | branch |
| --- | --- | --- | --- |
| `airway.bracketed_event` | +0.000 | `picture-description` | AIRWAY |
| `voice.chant` | +0.102 | `rainbow-passage` | VOICE |
| `ddk.lexical_repetition` | +0.333 | `story-recall` | DDK |
| `speech.lexical` | +0.500 | `prolonged-vowel` | SPEECH |
| `voice.sustained` | +0.561 | `picture-description` | VOICE |
| `ddk.lexical_repetition` | +0.667 | `rainbow-passage` | DDK |
| `voice.glide` | +0.768 | `story-recall` | VOICE |
| `airway.breath` | +1.378 / +1.486 / +1.521 | MPT, `prolonged-vowel`, `story-recall` | AIRWAY |
| `ddk.lexical_repetition` | +1.667 | `picture-description` | DDK |
| `speech.lexical` | +4.500 / +9.000 | the two DDK recordings | SPEECH |

**The VOICE over-routes are not a `voice.sustained` story.** All three fired `voice.chant`, two of
three fired `voice.glide`, and `voice.sustained` fired on only one of the three while reading
*silent* on the other two (2.568 s and 2.714 s against a 3.0 s cut). `rainbow-passage` entered
VOICE on `voice.chant` = 0.02205 against a cut of 0.02, the second-narrowest margin in the sample.
Pointing threshold work at `voice.sustained` on the strength of this sample would aim at the wrong
gate.

**One over-route came from `speech.lexical`:** `prolonged-vowel` routed to SPEECH on 3 lexical words
against a cut of 2. That is the case the framing above describes — routing arguably reading the
recording correctly against a declaration the participant did not follow.

13 recordings over 3 subjects is not a basis for any of this. The corpus run is the measurement.

## Outputs

`scripts/gate_family_matrix.py <features> <out_dir>` — two positional arguments, the config and the
profile overridable, no per-knob flags for anything measured.

| file | one row per |
| --- | --- |
| `gate_matrix.parquet` | (task family, gate): counts, three rates, reading distribution |
| `branch_agreement.json` | per branch: 2x2, recall/precision/specificity, unassigned families |
| `disagreements.parquet` | (recording, branch): deciding gate, its reading, margin, finding |
| `disagreement_groups.parquet` | (task family, branch, direction): findings, deciding gates, margins |
| `gate_matrix_fired.png` | the fired-rate panel |
| `gate_matrix_unavailable.png` | the unavailable-rate panel |
