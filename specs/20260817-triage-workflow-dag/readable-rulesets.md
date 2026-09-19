# Readable rulesets: every gate reads, or says why it could not

> "there shouldn't be anything in rulesets that cannot be read. the rulesets + hints determine
> which branches are selected."
>
> "if a residual is required for a gate, then residual.enabled cannot be false. or even exist. it
> should just be true or not even a flag."
>
> — the owner, 2026-09-18

The ruleset declares nine gates over four branches. Every one of them reads one number out of
`RecordingFeatures` and compares it against one threshold. This document records what was found
when each of the nine was traced back to the code that writes the number it reads, and what was
changed.

The governing distinction throughout is between three different things that had been collapsed
into two outcomes:

- **a measurement that did not clear the threshold** — `SILENT`, a fact about the recording;
- **a measurement that was never taken** — `UNAVAILABLE`, a fact about the run;
- **a measurement that was taken and found nothing** — which is a fact about the recording, and
  read as `UNAVAILABLE` in one place and as a fabricated zero in another.

## 1. `residual.enabled` is gone

`data/config/default.yaml` shipped `residual:\n  enabled: true`. Its only reader was in
`preprocess.py::_residual`, two lines that raised `ValueError("residual.enabled is false")`.

Four of the nine gates depend on that block:

| gate | reads | needs |
| --- | --- | --- |
| `airway.breath` | `[residual, energy_fraction]` | the `residual` measurement |
| `airway.ppg_silent_fraction` | `[ppg, silent_fraction]` | the PPG, which runs on `enhanced` |
| — | — | `enhanced` and `residual` are both written by the one block |

`_residual` writes both streams: the lag-aligned FRCRN output as `enhanced`, and
`plain - g*enhanced` as `residual`. The PPG block reads `enhanced`. So setting the flag false made
two of AIRWAY's four gates unreadable at once, and the recording could then only reach AIRWAY
through `airway.cough` or `airway.bracketed_event`.

A flag whose `false` value makes declared gates unreadable is not an option a configuration may
offer. The key and its reader are both removed. The block now always runs; when FRCRN is genuinely
unavailable it raises, and that is recorded in the verdict's `absent` map like any other block
failure — which is the honest path, because it names a reason rather than a setting.

The configuration loader already refuses an unknown override key, so a stale
`residual:\n  enabled: false` in someone's variant YAML now fails loudly rather than silently
doing nothing.

Test cost: the flag had been doing double duty as a test-speed switch — several fixtures set it
false so that PREPROCESS would not reach the real FRCRN. `_stub_models` now always replaces
`enhance_audios` (defaulting to a pass-through), which is what those fixtures actually wanted. Two
tests that used the flag to produce a missing `enhanced` stream now make FRCRN raise instead,
which is the one remaining way that stream can be absent.

## 2. The `words` arm had no consensus guard

`detectors.py`'s `words` arm read `float(features.words.get(arguments[0], 0))` with no check on
`features.consensus_present`. The `bracketed_set` and `onomatopoeic` arms two lines below both
guard on it.

`features.words` is built from the live word entities. When both ASR blocks fail and no consensus
transcript is written, there are no word entities, so every count is zero — and `speech.lexical`
(`[words, lexical] at_least 2`) read `0.0` and recorded SPEECH as **declined**: "this recording
carries fewer than two lexical words". No recogniser had run. That is a false claim about the
recording, and it is worse than an admitted absence, because a declined branch is not revisited
and an unavailable one is named in `unavailable_gates`.

The guard is added. It affects `speech.lexical` and the `speech.transcript_agreement` flag.

## 3. Shape A: an empty label set produced no key at all

`features.py::_label_span_statistics` only ever created a `by_set` entry for a set some live span
carried:

```python
if members and any(label in members for label in labels):
    by_set.setdefault(set_name, []).append(span)
```

and only sets present in `by_set` reached `_distribution`. So on a recording with no cough-labelled
span, `yamnet.cough_labels.peak_over_floor_db_max` was never written,
`detector_value`'s `_optional` returned `None`, and `airway.cough` read `UNAVAILABLE`.

`airway.cough` was the only one of the nine gates that read `unavailable` when every instrument ran
and found nothing. The other eight read `silent`.

"No live span carries a cough label" is a definite measurement with a definite consequence: the
gate must not fire.

### The encoding chosen, and why

The precedent for the count was already in the tree: `_distribution` writes `<prefix>.span_count`
unconditionally, so a set that reaches it with an empty selection already yields a defined count of
zero. `_label_span_statistics` now seeds `by_set` with an entry for **every** `LABEL_SETS` name
that the classifier declares members for, so `<classifier>.<set>.span_count` is written for every
set on every recording, `0.0` where nothing carried it.

The dB keys are a different matter. `peak_over_floor_db_max` over an empty sample has no value, and
`value_stats({})` correctly returns `{}`. Two encodings were available:

- **a numeric sentinel written into the feature table.** Rejected. There is no dB that means "no
  span". A zero reads as a *loud* span against an `at_least 50.0` gate's units — the gate would not
  fire, but `peak_over_floor_db_min` under an `at_most` gate would fire on the same zero. Worse,
  the sentinel would then be in the features shard, indistinguishable from a measurement, and every
  downstream consumer (the detector profile's quantile ladder, the gate matrix, the tables) would
  have to learn to drop it.
- **an explicit non-firing outcome produced at read time.** Chosen. The feature table stays
  honest: no measurement, no dB key. The reader — `detector_value`'s `span_label_stat` and
  `span_label_set_stat` arms — consults the companion `<prefix>.span_count`. When that count is
  present and zero, the sample is empty *by measurement*, and the arm returns the module's existing
  `GATE_CLOSED` / `GATE_CLOSED_BELOW` sentinel according to the reader's polarity — the same pair
  the `gated` source already returns when its corroborator did not fire, defined as "below every
  threshold" and "above every threshold" respectively. When the count key is absent altogether, the
  classifier never ran and the arm still returns `None`.

This is what makes the two cases distinguishable, which is the whole point: `span_count` present
and zero is "ran, found none"; `span_count` absent is "never written". The polarity comes from
`ruleset._POLARITY`, which already maps `at_least`/`at_most` onto `above`/`below`; `gate_value`
now passes it into the `Detector` it builds, where it previously left the field at its default.

## 4. The `residual` arm could return NaN

`detectors.py`'s `residual` arm returned `float(value)` directly — the one gate-reachable table
read that did not go through `_optional`. `residual_energy_fraction` is `nan` when
`input_energy == 0` (`tasks/speech_enhancement/residual.py`), and `nan >= 0.10` is `False`, so a
non-finite value read as a quiet non-fire: `airway.breath` recorded SILENT on a number that does
not exist.

The arm now goes through `_optional`, which already rejects a non-finite value. The redundant
`if not features.residual` guard is removed with it: `_optional` on an empty table returns `None`
for the same reason.

## 5. `unavailable_gates` carries the reason

For the genuine "instrument did not run" cases — `voice.glide` and `voice.chant` when no
`plain|yamnet` summary was written, `airway.bracketed_event` and `speech.lexical` with no consensus
— no value is invented. What is missing is a reason, and PREPROCESS already records one per block
in its verdict's `absent` map.

`RecordingFeatures` now carries that map (`absent`, read off the PREPROCESS verdict entity the
extractor already visits). `RouteEvaluation.unavailable` changes from
`Mapping[str, tuple[str, ...]]` to `Mapping[str, Mapping[str, str]]` — branch to gate to reason —
and `branch_decision.unavailable_gates` follows it. The reason is the `absent` entry for the block
that would have written the evidence, resolved through a declared source-to-block mapping
(`features.EVIDENCE_BLOCKS`); where no block is named absent, the reason states the evidence that
was not in the store.

This turns "never judged" into an accountable statement: a reader of the store can tell
`airway.breath` unavailable *because FRCRN timed out* from `airway.breath` unavailable *because
nothing wrote a residual at all*.

## Reported, not fixed

### Shape B: the cough was found, on a span class that carries no dB

`peak_over_floor_db` is written only for `measure == "amplitude"` spans. ASR spans carry
`float("nan")`; gap and continuity spans carry no such attribute at all. So a cough label landing
on a gap span yields `yamnet.cough_labels.span_count = 1` — the cough **was** found — while
`peak_over_floor_db_max` is still absent from the finite sample, and under the fix above the gate
reads a definite non-fire on a recording where a cough was detected.

Measured directly, one cough-labelled span per run, `airway.cough` = `at_least 50.0` dB:

| span `measure` | `…cough_labels.span_count` | `…peak_over_floor_db_max` | gate |
| --- | --- | --- | --- |
| `amplitude` | 1.0 | 60.0 | fired |
| `gap` | 1.0 | absent | unavailable |
| `asr` | 1.0 | absent | unavailable |
| `continuity` | 1.0 | absent | unavailable |

So it is three of the four span classes, not two of three: `_measure_fields` writes
`peak_over_floor_db` only for `amplitude`, and the `float("nan")` an ASR span's in-memory `Span`
carries never becomes an entity attribute at all. The fix above does not mask this — the count is
one, not zero, so the empty-sample path is not taken and the gate still admits it could not read.

Recommendation, with the evidence, is in the report accompanying this change. The short form:
the gate's stated intent is "loudest cough-labelled span", and that intent is only true across
span classes if `peak_over_floor_db` is measured on every span class. Anything short of that is a
gate that reads a quantity three of four span classes do not have. This changes what the gate
*means*, so it is not a repair and is not made here.

### Shape C: a span explicitly recorded as not measured is folded into the negatives

`preprocess.py::_mark_unmeasured` records a span it could not score as an `assertion` carrying an
`unmeasured` reason. It is reached from eight call sites across `_span_hear` and `_span_yamnet`,
with reasons including `yamnet_scores_absent`, `no_covering_window`, `no_native_window` and the
exception type of whatever raised. `extract_features` absorbs assertions only when
`name == "squim"`. So for every other measurement, a span explicitly recorded as *not measured*
leaves no trace in the features at all, and the span is simply one of the negatives.

This is the item that bears directly on the fix above. Shape A now turns a zero count into a
definite non-fire, and that non-fire is only as strong as the denominator behind it. If YAMNet
scored eleven of fourteen spans and the other three are `_mark_unmeasured` assertions nobody reads,
`yamnet.cough_labels.span_count = 0` states "nothing carried the cough set" on eleven spans while
reading as though it were fourteen.

The precedent for the repair is already in the same file: `_squim_statistics` writes
`<population>.n` beside `<population>.unmeasured`, counting the assertions that carry an
`unmeasured` key. Recommended shape: absorb `span_yamnet` and `span_hear` assertions the same way,
so `RecordingFeatures` carries how many spans each per-span classifier could not score, and a
set-conditioned gate's zero count can be qualified by it.

### Two adjacent items

`RouteState` has no unavailable member. When the emptiness bypass itself is unreadable,
`evaluate_emptiness` returns `UNAVAILABLE` and `evaluate_routes` falls through to `UNEXPLAINED`,
which that enum's own docstring calls "a charge against the ruleset" — for something the ruleset
could not read. Recommended: a fourth member. `ROUTE_STATES` is derived from the enum, so the
tallies pick it up; what moves is the `states` key set the `FamilyTally` contract promises, and
the tests that assert all keys are present and sum to `recordings`.

`vocabulary.py` defaults `route_state` to `UNAVAILABLE` for any branch with no `branch_decision`
entity. That is a different fact — "no decision was recorded for this branch" — sharing one token
with "the gates could not be read". A reader of `file_verdict.routes` cannot currently tell a
branch ROUTING judged unreadable from one it never judged, which is the same "never judged"
problem item 5 addresses, one layer up. Pre-alpha allows the outright fix: a separate member for
the defaulted case.

---

## 6. The three items above, resolved

Written 2026-09-18, against `931ea104`. Items 5's Shape C and the two adjacent items are no longer
recommendations: each is implemented, and this section records what shape it took and what was
decided where the recommendation left a choice open.

### 6.1 The denominator: `span_coverage`

`_mark_unmeasured` now writes `span_id` into the assertion's attributes, the way a scored
`span_<classifier>` window already does. Without it the assertion can only be joined to its span by
extent, and extent is not a key: `_squim_statistics` gets away with an extent join because it
collapses to one measure per extent and never counts spans.

`extract_features` absorbs an assertion whose `name` is a per-span classifier and that carries an
`unmeasured` key, and `_span_coverage` reduces the result to two numbers per classifier, keyed the
way `_squim_statistics` keys its own:

```
yamnet.n            spans the classifier reached, scored and unread together
yamnet.unmeasured   how many of those it could not score
```

Both are restricted to live spans, so a retired span is out of the numerator and the denominator
alike. A classifier that reached no live span at all is keyed with neither, which keeps
`7afe18e2`'s distinction intact one level up: an absent pair is "this classifier never ran", a
present `unmeasured: 0.0` is "it read every span".

`detector_value` gained a `span_coverage` source and `evidence_blocks` routes it to the block that
would have written it, so the denominator is readable by the same machinery that reads the count.
No detector and no gate reads it. That is deliberate — see below.

### 6.2 Does an unmeasured span make a gate unreadable?

**No, on the evidence available.** The counts are carried and the gate stays readable.

The alternative is a rule of the form "a set count is unreadable when more than *p* of the
classifier's spans were unread". Every value of *p* is a decision about how much missing evidence
makes a negative untrustworthy, and nobody has measured that. This repository already has two
defects from literals that were never fitted (a silhouette coefficient read as a probability; a
2→10 dB HNR ramp under which ordinary voiced speech read as partly voiced), and a proportion cut
here would be a third of the same kind — worse, on the routing path, where it would silently
convert definite non-fires into unavailable gates on real recordings.

There is also a reason to prefer carrying the counts even once *p* is measured. A gate that reports
`UNAVAILABLE` on a partly-read classifier destroys the count it did read; a gate that fires or
stays silent with the coverage recorded beside it keeps both facts, and the reader that wants to
discount the negative can. `UNAVAILABLE` is the right answer when nothing was read; it is a lossy
answer when eleven of fourteen spans were.

What would change this: a measured relationship between unread proportion and the error rate of the
resulting routing decision, over a corpus with known content. That is a sweep, not a judgement, and
the mechanism to run it now exists — `span_coverage` is in every features shard.

**What this changes on real recordings: nothing.** No gate reads the new keys, and no existing key
changed value. The features shard gains one field.

### 6.3 `RouteState.UNREADABLE`

`evaluate_emptiness` has always distinguished three outcomes — `FIRED` (every named stream under
the floor), `SILENT` (at least one at or over it) and `UNAVAILABLE` (a named stream's summary is not
in the store). `evaluate_routes` tested only `is GateOutcome.FIRED`, so the third collapsed into the
second and the recording was recorded `UNEXPLAINED`, which the enum's own docstring calls a charge
against the ruleset. The ruleset was never shown anything to be charged for.

The enum now carries a fourth member and the bypass is read once into a three-way branch. The three
non-routed states are told apart on one axis, the bypass reading, with the branch gates silent in
all three:

| bypass | state | whose fault |
| --- | --- | --- |
| `FIRED` | `EMPTY` | the recording's — it carried nothing |
| `SILENT` | `UNEXPLAINED` | the ruleset's — content no gate accounted for |
| `UNAVAILABLE` | `UNREADABLE` | the run's — the evidence was never written |

`ROUTE_STATES` is derived from the enum, so `tally_families` and `gate_matrix`'s family rows carry
the fourth key without a change; what moved is the key set `FamilyTally.states` promises, and the
tests asserting those keys were updated rather than loosened — `test_every_tally_carries_all_four_
states_and_they_sum` still asserts the exact key list and the exact sum.

**What this changes on real recordings.** A recording that used to be recorded `unexplained` because
its `enhanced|yamnet` or `residual|yamnet` summary was missing is now recorded `unreadable`. The
**triage axis does not move**: `fold_file_verdict` flags on the new state too, under
`UNREADABLE_EMPTINESS` rather than `UNEXPLAINED_CONTENT`. What changes is which reason is recorded
and which corpus column the recording lands in — and that is the point, because the `unexplained`
count is read as the ruleset's error rate, and recordings whose evidence was never written were
inflating it.

Three test fixtures turned out to be in this class already: `routing_ruleset_test._features`
carries `classifier_streams=["plain|yamnet"]` and neither bypass stream, so every record it builds
with nothing routed was `UNEXPLAINED` under the old code purely because the bypass was unreadable.
The tests that meant "unexplained" now build their records through `_classified`, which carries both
streams; the ones that only incidentally asserted the state now assert `UNREADABLE`, which is what
their fixture actually describes.

### 6.4 `UNJUDGED`: a branch ROUTING never judged

`fold_file_verdict` defaulted `routes[branch]` to `UNAVAILABLE` for any branch with no
`branch_decision` entity. `UNAVAILABLE` is a value ROUTING writes, and it means something specific:
ROUTING looked, and every gate of that branch was unreadable. "No decision was recorded at all" is a
different fact, and a reader of `file_verdict.routes` could not tell the two apart.

The defaulted case is now `UNJUDGED`, a fifth member of `BRANCH_ROUTE_STATES` and the only one that
is never written by ROUTING — it is the fold's own reading of a decision that is absent.

**What this changes on real recordings.** `file_verdict.routes` reports `unjudged` where it
reported `unavailable`, for branches with no decision entity. `_agreement` resolves both to
`resolved`, as it did before and for the same reason — neither made a claim to agree or disagree
with — so no agreement row, no flag, no triage and no release outcome moves.
