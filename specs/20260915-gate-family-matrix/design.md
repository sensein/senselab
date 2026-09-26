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
2. Its `families.parquet` carries per-family counts but no per-family **routing** breakdown with
   the deciding gate, which is the question "for each task family, how many are routed where and
   what are the decision criteria". `family_routing.parquet` is that breakdown.

Note: an earlier revision of this section said `Confusion` needed a **precision** so "the
over-routing side of the ruleset could not be read off it". That was wrong twice over — the
over-routing side was always readable as `false_positive_rate`, which is the budget a threshold was
selected against, and a precision is not comparable across populations. The property added was
renamed `positive_share_of_fired`; see the 2026-09-15 section.

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
  imports from `ruleset.py` and adds nothing to it except `positive_share_of_fired`, which belongs
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
As shipped that is `AIRWAY: airway`, `SPEECH: speech`, `VOICE: voice`, `DDK: syllable_repetition`,
and it is **multi-label**: `speech` and `syllable_repetition` share the ten diadochokinesis
families, so those declare both SPEECH and DDK. It was single-label (`SPEECH: lexical_speech`,
with `syllable_repetition` held out of SPEECH's population) until 2026-09-15. Note that `families.py` also defines `sustained`, `glide`, `cough`,
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

Recall was 1.000 on all four branches (13 of 13 declared branches routed), so every disagreement on
this sample is an over-route — that is, budget spent, which is the axis a router is allowed to spend
on. **The per-branch precisions this section first reported (0.400 to 0.571) are superseded**: see
"the precision figures were not a contradiction" below for why that quantity is not comparable to
anything and what replaced it. The deciding gates of those 13 extras, smallest relative margin
first:

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
| `family_routing.parquet` | (task family, branch): how many routed, the firing and sole-firing gates, margins, declared or beyond |
| `family_states.parquet` | task family: route states, and how many branches each recording routed to |
| `gate_matrix.parquet` | (task family, gate): counts, three rates, reading distribution |
| `branch_agreement.json` | per branch: the raw 2x2, recall, the over-routing rate and the budget it sits inside, unassigned families |
| `disagreements.parquet` | (recording, branch): deciding gate, its reading, margin, finding |
| `disagreement_groups.parquet` | (task family, branch, direction): findings, deciding gates, margins |
| `gate_matrix_fired.png` | the fired-rate panel |
| `gate_matrix_unavailable.png` | the unavailable-rate panel |

---

## 2026-09-15: the precision figures were not a contradiction, and the naming was the defect

The owner challenged § "What the sample also showed" above — per-branch precision of 0.500 AIRWAY /
0.571 SPEECH / 0.500 VOICE / 0.400 DDK — as *"completely off from the work we had done with
rulesets"*, which reports 0.93–0.95 for the same gates over 62,547 recordings.

**Both sets of numbers are right, and they are not the same quantity.** This section establishes
what each measurement is over, because that is where the contradiction dissolved.

### The 0.93–0.95 figures are sensitivity. Nothing in the ruleset work ever reported a precision

`dag.md:1035-1038`'s branch table, over the 62,547-recording corpus:

| branch | sens | spec | spec-excl |
| --- | --- | --- | --- |
| AIRWAY | 0.972 | 0.779 | 0.779 |
| SPEECH | 0.954 | 0.663 | **0.895** |
| VOICE | 0.957 | 0.736 | 0.736 |
| DDK | 0.937 | 0.727 | 0.727 |

The remembered 0.93–0.95 band is the **sens** column. The corpus **specificities are 0.663 to
0.779** — the ruleset work's own record of substantial over-routing. `design.md:22` states outright
why no precision was there to contradict: *"`Confusion` had sensitivity, specificity and a
false-positive rate but no **precision**"*. Precision was added by this work, the day the challenge
was raised.

### The corpus already implied the challenged figures, and three of four are worse there

Deriving `tp/(tp+fp)` from the recorded 2x2 in
[`../20260817-triage-workflow-dag/runs/ruleset-score-20260912/ruleset_score.json`](../20260817-triage-workflow-dag/runs/ruleset-score-20260912/ruleset_score.json)
— arithmetic on counts already in the record, not a new measurement:

| branch | corpus tp | corpus fp | **corpus tp/(tp+fp)** | the challenged 13-file figure |
| --- | --- | --- | --- | --- |
| AIRWAY | 12,655 | 10,951 | **0.536** | 0.500 |
| SPEECH | 31,693 | 9,872 | **0.762** | 0.571 |
| VOICE | 7,945 | 14,332 | **0.357** | 0.500 |
| DDK | 7,485 | 14,878 | **0.335** | 0.400 |

The 13-recording figures are **consistent with the corpus, and for AIRWAY, VOICE and DDK they are
better than it.** There was never a contradiction to explain. What there was is a quantity nobody
had computed, printed beside a sensitivity, under a name that reads as accuracy.

### Why the number looked alarming: precision moves with prevalence and a budget does not

The 13-recording sample was assembled to exercise all four branches, so each branch's prevalence in
it is roughly a quarter. Corpus prevalence is 0.208 / 0.531 / 0.133 / 0.128. Precision is a function
of prevalence; specificity and the false-positive rate are not. So a precision measured on a
deliberately balanced sample of 13 is not comparable to anything, including itself on another
corpus — and comparing it to a sensitivity is comparing two different axes.

**This is the defect, and it is in the framing and the naming, not in the arithmetic.**
`Confusion.precision` was plain `tp/(tp+fp)` and correct; presenting it as the branch's quality
score was not.

### What replaced it

The router framing the design already had, and which the reporting had dropped. `dag.md:186-196`
and `:246-256`: *"a router's errors are asymmetric: an over-routed recording costs a branch some
discarded work, an under-routed one is never seen by anything that could interpret it"*, thresholds
chosen as *"the loosest cut inside a budget"* against `OVER_ROUTING_BUDGETS` = 2%, 5%, 10%, 20%.
`family-taxonomy-ruleset.md:886-888` states the same. `dag.md:250-256` records a threshold
deliberately loosened for *"1,929 additional branch invocations the branches discard"*.

Under that framing the over-routing side is measured as **budget spent** — the false-positive rate
over the undeclared recordings, which is what a threshold was selected against — and not as a
precision. `print_agreement` now reports, per branch: the raw 2x2, `recall` (sensitivity),
`over-rt` (the false-positive rate), the tightest declared budget that rate sits inside, and
`decl/routed` last, labelled a raw derived count ratio that is explicitly neither a precision nor
an accuracy.

`Confusion.precision` is renamed **`positive_share_of_fired`**. The docstring always said what it
was — *"fraction of firings the reference calls positive"* — and only the name claimed more. Every
raw count is preserved; `as_json()` carries `tp`/`fp`/`tn`/`fn` unchanged and no key named
`precision`.

### The reference is now multi-label, and it costs the record nothing

`reference_family_set.SPEECH` moves from `lexical_speech` to `speech`
(= `lexical_speech | syllable_repetition`, already defined in `families.py` and already in
`FAMILY_SETS`), and `excluded_by_construction` is null for every branch.

The owner directed this on 2026-09-15: *"there is a declared branch, which means evaluation of
rulesets should include it"*, and *"the reference is single-label when the content is not. A
diadochokinesis recording genuinely contains lexical speech and is genuinely DDK material. Routing
it to both SPEECH and DDK is correct on both counts."*

**Only SPEECH is affected, and only one family set had to change** — the four reference sets were
pairwise disjoint, so `reference_branches` returned exactly one branch for every family. It now
returns `("SPEECH", "DDK")` for the ten diadochokinesis families and one branch for every other.
No family was assigned a second branch by judgement; the one multi-label case falls out of a set
`families.py` already defined.

**The two treatments give identical specificity.** Holding `syllable_repetition` out of SPEECH's
population and making it positive both leave the negative side untouched, because the held-out
families were all positives:

| SPEECH, 62,547 recordings | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| `against_reference`, single-label | 31,693 | 9,872 | 19,440 | 1,542 | 0.954 | 0.663 |
| `excluding_construction` (what shipped) | 31,693 | 2,246 | 19,077 | 1,542 | 0.954 | **0.895** |
| multi-label (derived; the corpus re-run will confirm) | 39,319 | 2,246 | 19,077 | 1,905 | 0.954 | **0.895** |

So **every conclusion the design drew from `spec-excl` 0.895 stands unchanged**, as does the gate
figure 0.855 and every threshold selected against a budget. What multi-label adds is the 7,989
recordings the exclusion discarded: 7,626 correct SPEECH routings that earned no credit, and **363
DDK recordings SPEECH did not route — a genuine miss of speech content that holding them out made
invisible.** That is the information the exclusion cost, and it is why the owner asked for the
change.

`excluded_by_construction` is kept as a mechanism: the code, its two validation errors and its
tests are untouched, and `load_ruleset` still refuses a branch that holds out families its own
reference set calls positive — which is precisely what would have caught a half-done version of
this change. No branch uses it.

### An owed decision, not decided here

Two families arguably declare a second branch and were **not** assigned one, because the case is not
plain enough to settle without the owner:

- **`cape-v-sentences` / `cape-v-sentences-v2`** — the CAPE-V protocol is a *voice* assessment
  delivered through sentences. Lexical speech certainly; whether it declares VOICE is a judgement.
- **`loudness` / `loudness-v2`** — asks for words at varying vocal effort, which is a voice task
  carried by speech.

Both stay SPEECH-only. Anything routing them to VOICE reads as `beyond_declaration`, which is the
honest state rather than a quiet decision.

### DDK's missing node is not a fact about the ruleset

DDK is scored like any other branch, with no asterisk on its routing figures. *"The rulesets guide
which branch to go to, independent of whether that branch is designed/implemented"* (owner,
2026-09-15). A DDK route producing `SKIPPED — no node implements this branch` (`run.py:303-305`,
`branch-ddk.md:12`) is a fact about the graph's completeness and belongs with verdicts and flags; it
must not qualify a routing figure, and the ruleset's accuracy must not be read as evidence about
DDK's implementation status in either direction.

### Where each measurement is over, in one table

| figure | population | reference | denominator |
| --- | --- | --- | --- |
| `dag.md:1035-1038` sens/spec | 62,547 recordings, 48 families | declared family, single-label | per branch, whole corpus |
| `family-taxonomy-ruleset.md:193-198` | same corpus, one gate swept | `LEXICAL_SPEECH`, DDK out of negatives | per threshold, one gate |
| the challenged precisions | 13 recordings, 3 subjects | declared family, single-label | per branch, tp+fp |
| `family_routing.parquet` (new) | whatever shard is reduced | declared family, multi-label | per (family, branch), n of the family |
