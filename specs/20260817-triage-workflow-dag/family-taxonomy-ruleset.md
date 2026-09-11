# The family taxonomy ruleset

For each elicitation family: which branch a recording is routed to, which measurement confirms that
route, and what counts as a recording that lands in no bucket at all. The ruleset's data is
`taxonomy.ruleset` in `src/senselab/audio/workflows/triage/data/config/default.yaml`; the evaluator
over it is `routing_analysis/ruleset.py`.

**Nothing here is wired into TAXONOMY yet.** `taxonomy.presence_floor` is still `null` in every
line, `nodes/taxonomy.py` is untouched, and no node reads `taxonomy.ruleset`. This document is the
ruleset and its scoring; adopting it is a later step.

## The 48 families are not restated here

`routing_analysis/families.py` already holds them, as `DECLARED_KIND` (`speech`, `lexical_speech`,
`airway`, `voice`, `cough`, `breath`, `glide`, `sustained`) plus `SYLLABLE_REPETITION`. The ruleset
names a **set**, never a family list: `taxonomy.ruleset.declared_family_set` maps each branch to one
of those names, and `Ruleset.declared_branches` resolves a family through it. A family added to
`families.py` is routed by that alone.

## Four branches, because diadochokinesis was in none of three

`BRANCHES` is now `("AIRWAY", "SPEECH", "VOICE", "DDK")`.

Diadochokinesis is syllable repetition: `pa pa pa pa`, `puhtuhkuh` at rate. It is speech production
whose target is a syllable rather than a word, so it satisfies no existing branch's gate cleanly —
it holds no lexical content for SPEECH to be the authority on, it is not sustained or glided
phonation for VOICE, and it is not an airway manoeuvre. What it *does* do is fire the lexical
detectors: over the corpus sweep the ten DDK families fire the speech detectors at **92–99%**,
because a recognizer asked for words on `pa pa pa pa` returns words. Routing DDK through SPEECH
therefore does not fail loudly; it succeeds for the wrong reason and hands a branch a subject it
cannot assess. A fourth branch is the way to keep that firing rate from being read as lexical
evidence.

`families.py` was not changed to accommodate this. `SYLLABLE_REPETITION` already existed and DDK
families are already inside `DECLARED_KIND["speech"]`; the ruleset reads the narrower set by name
and the wider one is left as it is, so `declared_kinds("diadochokinesis-ka")` still returns
`{"speech"}` and the existing sweep's reference standards are unchanged.

## Two kinds of route

**Declared** — from the `task-` entity: what the participant was instructed to produce. One branch
per family set, confirmed by that branch's gates.

**Discovered** — content present regardless of instruction. Exactly one is declared today:
`SPEECH`, via `speech.intrusion`. A discovered route is an **addition**. A prolonged-vowel recording
with an intruding voice stays routed to VOICE for the vowel and is *additionally* routed to SPEECH
for the intrusion; the discovery never replaces, never suppresses, and never relaxes a declared
route's own gate.

## The gates

Each gate is one feature path, one comparison and one threshold. Every threshold below is the
**max-Youden operating point** from the 62,547-recording corpus sweep, except `ddk.declared`.

| gate | feature | op | threshold | corpus J | max-family firing | median-family firing |
| --- | --- | --- | --- | --- | --- | --- |
| `speech.declared` | `words.agreement` | >= | 4 words | 0.74 | 0.99 | 0.15 |
| `speech.intrusion` | `words.agreement` | >= | 2 words | 0.82 | 0.99 | 0.45 |
| `voice.sustained` | `span_longest_s.amplitude` | >= | 3.0 s | 0.65 | 0.93 | 0.25 |
| `voice.glide` | YAMNet singing-union peak, plain stream | >= | 0.05 | 0.75 | 0.93 | 0.10 |
| `airway.breath` | `residual.energy_fraction` | >= | 0.10 | 0.66 | 0.98 | 0.16 |
| `airway.cough` | `span_stats["all.peak_over_floor_db_max"]` | >= | 50.0 dB | 0.79 | 0.96 | 0.07 |
| `ddk.declared` | max token repetition in `transcript` | >= | 3 | **not measured** | — | — |

"Max-family firing" is the highest per-family firing rate at that threshold, "median-family firing"
the median across all 48. The pair is the spread the operating point was chosen against, and it is
what says whether a J is carried by separation or by prevalence.

The singing union is the AudioSet singing subtree as `LABEL_SETS["singing"]` in
`routing_analysis/labels.py` defines it — `A capella`, `Chant`, `Child singing`, `Choir`, `Humming`,
`Mantra`, `Singing`, `Synthetic singing`, `Vocal music`, `Yodeling` and the two gendered variants,
which YAMNet's 521-label grid does not carry — read as a peak over the union, the same construction
`_singing_detectors` in `detectors.py` sweeps.

### `ddk.declared` is a starting value, not a measurement

The DDK gate reads a feature no extraction writes: the largest number of times any single
normalised token repeats in the recording's consensus transcript. Normalisation is lowercasing and
splitting on punctuation as well as whitespace, so `Pa, pa. PA!` is three instances and
`pa-pa-pa-pa` is four. `RecordingFeatures.transcript` is capped at 300 characters
(`TRANSCRIPT_CAP`), which is ample for a repetition count and truncates mid-token; the fragment
counts as its own token rather than as another instance of the token it came from, which biases the
count down by at most one.

**Threshold 3 has not been swept.** No sweep exists over this feature, so there is no J, no firing
spread, and no operating point — it is a value chosen so that three repetitions of a syllable carry
the gate, and it sits in config exactly like the measured ones so that fitting it later is a config
change. Until it is swept, a DDK route's confirmation rate is uninterpretable.

## Open question: `speech.intrusion` at a median-family firing of 0.45

This is the ruleset's biggest open question and it is deliberately left as measured.

`speech.intrusion` at `words.agreement >= 2` carries the highest J of any gate (0.82), but its
median-family firing rate is **0.45**. Half of all families — most of which never ask for a word —
have a recording fire it at better than even odds. Two agreed words on a prolonged vowel is not a
plausible rate of genuine bystander speech in a supervised elicitation protocol; the likely
mechanism is **ASR hallucination on sustained phonation**, where a recognizer asked to transcribe a
held vowel emits short function words and a second recognizer agrees with it, producing an
`agreement` outcome that no human said.

It is implemented as measured, at 2, and **not quietly raised**. Raising it here would trade a
measured operating point for an unmeasured one and hide the defect in a threshold. What would settle
it, in order of cost:

1. Score `speech.intrusion` restricted to non-bracketed, non-function-word tokens — `words.lexical`
   and `words.agreement_lexical` are already extracted and already swept.
2. Listen to a stratified sample of the prolonged-vowel and maximum-phonation-time recordings that
   fire it, and count how many carry an actual second voice.
3. If the mechanism is confirmed, the fix is upstream of this ruleset — a consensus rule that will
   not promote a word to `agreement` when both recognizers hallucinated it — not a higher cut here.

Until then every discovered-SPEECH count from this ruleset is an upper bound.

## The evaluator

`evaluate_routes(features, ruleset) -> RouteEvaluation`, a pure function over one
`RecordingFeatures`. The frozen result carries the declared branches, which of them a gate
confirmed, which were left unconfirmed, which were unreadable, the discovered branches, the union
actually routed, whether the recording routed to nothing, and every gate's own outcome.

**`unavailable` is not `silent`.** `GateOutcome` has three members — `FIRED`, `SILENT`,
`UNAVAILABLE` — and a gate whose feature was never written to the store reads `UNAVAILABLE`. An
absent residual measurement is not a residual energy fraction of zero; a run with no YAMNet summary
is not a run with no singing; a recording with no consensus transcript is not a recording with no
repeated syllable. Collapsing the three into a boolean would report a broken or partial store as a
negative measurement, which is the failure mode that makes a scored ruleset look better than it is.
The distinction is carried in `RouteEvaluation.unavailable` and in `FamilyTally.unavailable`, per
branch, not summed away.

**Fall-through is a first-class output.** `RouteEvaluation.fell_through` is true when `routed` is
empty, and `FamilyTally.fell_through` counts it per family. A recording falls through when it was
declared to a branch whose gates were all silent or all unreadable, when its task family is in no
declared set at all, or both — and no discovery gate fired. The count answers the question directly:
how many recordings pass through the whole ruleset without landing in any bucket.

`tally_families(evaluations) -> dict[str, FamilyTally]` aggregates a stream of evaluations into the
per-family counts — declared, confirmed, discovered, unavailable (all four per branch) and
fell-through — so the ruleset can be scored over the corpus without holding it.

## What still needs a DDK arm

Adding `DDK` to `BRANCHES` is consistent with the graph as it stands — `nodes/routing.py` selects
branches through `BRANCH_FOR_KIND`, which has no `ddk` kind, so `DDK` is never selected and
`run.py`'s loop marks it `SKIPPED`; `run.py` then filters its node outcomes through `GRAPH_ORDER`,
which does not name `DDK`, so nothing is reported for it. The whole triage suite passes unchanged.

What a DDK arm would still need, none of it done here:

- `GRAPH_ORDER` in `vocabulary.py` does not include `DDK`, and `run.py`'s `branches` dispatch table
  has no `DDK` entry. A `DDK` that were ever selected would raise a `KeyError` there.
- `nodes/routing.py`'s `BRANCH_FOR_KIND` has no `ddk` kind, so nothing can select the branch.
- `nodes/report.py` builds `_EVIDENCE_BRANCHES = (*BRANCHES, "REDACT")`, so `_branch_evidence` now
  emits an always-empty `"DDK"` key in the report JSON. Its `source_spans` table has no `DDK` entry,
  which is why this is empty rather than an error.
- There is no `nodes/ddk.py`, and `specs/20260817-triage-workflow-dag/dag.md` still documents
  `BRANCHES` as the three.
