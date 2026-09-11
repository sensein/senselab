# The family taxonomy ruleset

For each recording: which branches its own content routes it to, which family set each branch is
scored against, and what counts as a recording that lands in no bucket at all. The ruleset's data is
`taxonomy.ruleset` in `src/senselab/audio/workflows/triage/data/config/default.yaml`; the evaluator
over it is `routing_analysis/ruleset.py`.

**Nothing here is wired into TAXONOMY yet.** `taxonomy.presence_floor` is still `null` in every
line, `nodes/taxonomy.py` is untouched, and no node reads `taxonomy.ruleset`. This document is the
ruleset and its scoring; adopting it is a later step.

## Content-first, not instruction-first

The ruleset used to start from the instruction: `evaluate_routes` read the task family, resolved it
to the branches that family *declares*, and then evaluated only those branches' gates. A branch the
task had not declared could be reached only through a separate `discovery_gates` exception, of which
exactly one existed (`SPEECH`, via `speech.intrusion`).

That is now reversed, and the reversal is the whole change:

1. **Branches are decided purely from outputs.** Every branch's gates are evaluated on every
   recording. These are recordings, not trials — they carry whatever the participant produced, which
   is not restricted to what the protocol asked for.
2. **No detector output is treated as hallucination or error.** A reading is evidence. It is
   acknowledged and used in the context of the other readings; it is never discarded because it
   failed one cut. Where two readings of the same phenomenon disagree, both gates stay and the
   branch routes on either — it does not route on the worse of the two.
3. **The declared family is a reference standard, and a hint.** It scores the routed set
   (`agreed` / `missed` / `extra`, and the per-branch 2x2 in `score_branches`) and it will later
   refine the routed set as a hint. It never filters which gates run.

`confirming_gates` and `discovery_gates` are deleted; `branch_gates` replaces both, one mapping
whose gates all run on everything. `declared_family_set` is renamed `reference_family_set` to say
what it now is. Pre-alpha: no aliases, no compatibility path, no flag that restores instruction-first
routing.

### The evidence that forced it

74 `harvard-sentences-list` recordings routed to no branch on their own declared gate and were
rescued only by the `speech.intrusion` discovery gate — the exception, not the rule that was
supposed to cover them. Every one of the 74 is unambiguous read speech. Four measured examples:

| transcript | `words.lexical` | `words.agreement` |
| --- | --- | --- |
| `Pure-braid poodles have curls.` | 4 | 3 |
| `Dots of life betrayed black cat.` | 6 | 3 |
| `The alley is drew for one want of slates.` | 9 | 3 |
| `The first th- the first thing is show the the the hook and up in the b- in the big reaches.` | 21 | 3 |

`speech.declared` gated on `words.agreement >= 4`. Every row has ample lexical content and was
discarded because one recogniser disagreed. Reading the same recording twice and keeping only the
recordings where both readings match is not a measurement of speech; it is a measurement of
recogniser concordance. That is the concrete defect this restructure fixes, and it is why the two
old speech gates — two cuts on the *same* number, one for declared families and one for everything
else — were replaced by two gates on *different* numbers, either of which routes.

## The 48 families are not restated here

`routing_analysis/families.py` already holds them, as `DECLARED_KIND` (`speech`, `lexical_speech`,
`airway`, `voice`, `cough`, `breath`, `glide`, `sustained`) plus `SYLLABLE_REPETITION`. The ruleset
names a **set**, never a family list: `taxonomy.ruleset.reference_family_set` maps each branch to one
of those names, and `Ruleset.reference_branches` resolves a family through it. A family added to
`families.py` is scored by that alone.

## Four branches, because diadochokinesis was in none of three

`BRANCHES` is `("AIRWAY", "SPEECH", "VOICE", "DDK")`.

Diadochokinesis is syllable repetition: `pa pa pa pa`, `puhtuhkuh` at rate. It is speech production
whose target is a syllable rather than a word, so it satisfies no other branch's gate cleanly — it
holds no lexical content for SPEECH to be the authority on, it is not sustained or glided phonation
for VOICE, and it is not an airway manoeuvre. What it *does* do is fire the lexical detectors: over
the corpus sweep the ten DDK families fire the speech detectors at **92–99%**, because a recogniser
asked for words on `pa pa pa pa` returns words. Routing DDK through SPEECH therefore does not fail
loudly; it succeeds for the wrong reason and hands a branch a subject it cannot assess. A fourth
branch is the way to keep that firing rate from being read as lexical evidence.

`families.py` was not changed to accommodate this. `SYLLABLE_REPETITION` already existed and DDK
families are already inside `DECLARED_KIND["speech"]`; the ruleset reads the narrower set by name
and the wider one is left as it is, so `declared_kinds("diadochokinesis-ka")` still returns
`{"speech"}` and the existing sweep's reference standards are unchanged.

## The gates

Each gate is one feature path, one comparison and one threshold. A branch routes when **any** of its
gates fires.

| branch | gate | feature | op | threshold | corpus J | max-family firing | median-family firing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SPEECH | `speech.lexical` | `words.lexical` | >= | 4 words | **provisional** | — | — |
| SPEECH | `speech.agreement` | `words.agreement` | >= | 3 words | **provisional** | — | — |
| VOICE | `voice.sustained` | `span_longest_s.amplitude` | >= | 3.0 s | 0.65 | 0.93 | 0.25 |
| VOICE | `voice.glide` | YAMNet singing-union peak, plain stream | >= | 0.05 | 0.75 | 0.93 | 0.10 |
| VOICE | `voice.chant` | YAMNet `Chant` peak, plain stream | >= | 0.02 | 0.83 | 0.94 | 0.14 |
| AIRWAY | `airway.breath` | `residual.energy_fraction` | >= | 0.10 | 0.66 | 0.98 | 0.16 |
| AIRWAY | `airway.cough` | `span_stats["all.peak_over_floor_db_max"]` | >= | 50.0 dB | 0.79 | 0.96 | 0.07 |
| DDK | `ddk.lexical_repetition` | max token repetition in `transcript` | >= | 3 | **not measured** | — | — |

"Max-family firing" is the highest per-family firing rate at that threshold, "median-family firing"
the median across all 48. The pair is the spread the operating point was chosen against, and it is
what says whether a J is carried by separation or by prevalence. Every measured row above is the
max-Youden operating point from the 62,547-recording corpus sweep.

`voice.sustained`, `voice.glide`, `airway.breath` and `airway.cough` are unchanged: they sit at
measured operating points and this restructure had no evidence against any of them.

### `voice.chant`, added: the best voice separator in the sweep, previously unused

`voice.yamnet_chant_peak.plain` — the bare `Chant` label on the plain stream, not the singing union
that contains it — carries **corpus J 0.83** at 0.02, with max-family firing 0.94 and median-family
firing 0.14. That is the highest J of any voice-family detector in the sweep, above
`voice.glide`'s 0.75, and it was sitting in `PRIMARY_DETECTORS` unrouted. It is added as a third
VOICE gate rather than replacing `voice.glide`, because the two read different constructions of the
same evidence (one label against a twelve-label union) and either firing is enough — the content-first
rule is that a reading is used, not that the readings must agree.

### The SPEECH thresholds are provisional and unswept

`speech.lexical >= 4` and `speech.agreement >= 3` are **starting values, not operating points.** No
sweep exists over the pair, and they are deliberately not the old 4-and-2: keeping those would have
carried an instruction-first cut into a content-first ruleset while reporting it as measured.

What they are chosen to do: route every row of the harvard table above. `speech.lexical >= 4`
carries all four on lexical content alone, at the smallest of the four counts.
`speech.agreement >= 3` carries them again on the other reading, at the value all four share, and it
is what routes a short utterance — three words both recognisers returned — that the lexical cut is
above. Since a lexical word count is at least its agreement count, the agreement gate's only
independent contribution is at exactly three lexical words, all of them agreed.

**The sweep that would settle them**, and it is a scoring change rather than an extraction one,
because `words.lexical` and `words.agreement` are both already extracted and both already carry a
single-gate detector in `detectors.py` (`speech.words_lexical`, `speech.words_agreement`):

1. Score the **disjunction** over the 2-D grid `COUNT_GRID x COUNT_GRID` (100 points) against the
   `lexical_speech` reference set, on the full 62,547-recording corpus. The OR's operating point is
   not the pair of each gate's own max-J point, so sweeping the two gates separately — which is all
   that has ever been done — does not answer this.
2. Report max-family and median-family firing at each of the 100 points beside J, so a J carried by
   prevalence is visible.
3. Take the max-J point subject to the constraint that all 74 harvard fall-throughs route, and state
   the constraint as a constraint rather than folding it into the objective.

Until that runs, every SPEECH count out of this ruleset is provisional in both directions.

### `ddk.lexical_repetition` is lexical, and the acoustic gate is missing

The DDK gate reads a feature no extraction writes: the largest number of times any single normalised
token repeats in the recording's consensus transcript. Normalisation is lowercasing and splitting on
punctuation as well as whitespace, so `Pa, pa. PA!` is three instances and `pa-pa-pa-pa` is four.
`RecordingFeatures.transcript` is capped at 300 characters (`TRANSCRIPT_CAP`), which is ample for a
repetition count and truncates mid-token; the fragment counts as its own token rather than as
another instance of the token it came from, which biases the count down by at most one.

It was called `ddk.declared` and is renamed `ddk.lexical_repetition`, because measurement shows it
only works when the elicited unit is a dictionary word:

| family | elicited unit | instruction wording | fall-through |
| --- | --- | --- | --- |
| `diadochokinesis-buttercup` | word | "repeat the **word** /buttercup/" | 1.6% |
| `diadochokinesis-v2-puh` | syllable | "repeat the **syllable** /PA/" | 24.6% |

A recogniser transcribing a repeated dictionary word emits the same token each time and the counter
sees the repetition. A recogniser transcribing a repeated non-lexical syllable emits whatever
lexical neighbours it can find, and they differ between repetitions, so the count collapses. The
gate is therefore a **lexical** repetition gate and the name now says so — a 24.6% fall-through on
`-v2-puh` is not a measurement about that recording's content, it is the gate reading a transcript
of something that has no transcript.

**Threshold 3 has not been swept.** No sweep exists over this feature, so there is no J, no firing
spread, and no operating point.

**The missing gate is acoustic, and this document deliberately does not invent it.** What DDK needs
is a detector of periodic syllabic repetition in the signal — envelope periodicity, or a
rate-of-repetition estimate — which is independent of whether any recogniser can spell the unit. No
such feature is extracted today and no sweep exists for one. Adding an unfitted acoustic heuristic
here would repeat exactly the defect that threshold-3 already is, so it is recorded as an open
item instead.

## What `speech.intrusion` on prolonged vowels actually was

An earlier version of this document read the `speech.intrusion` firing rate on non-speech families
as **ASR hallucination on sustained phonation**, and called every discovered-SPEECH count an upper
bound. **That reading is wrong and is corrected here.**

`prolonged-vowel`'s protocol instruction is: *"repeating the sentence '1, 2, 3 aah' … hold the sound
'aah' until the timer runs out"*. The recording therefore contains a spoken preamble by design. The
1,226 prolonged-vowel recordings the intrusion gate found SPEECH in carry **real participant
speech** — the instructed count-in — not invented words.

The control is `maximum-phonation-time`, whose instruction asks for held phonation with no spoken
preamble. It fires at **3.2%**. Two families whose acoustics are near-identical, differing in
whether the protocol puts words at the front, differ by more than an order of magnitude in the rate
the gate fires. That is the gate reading content, correctly, in both cases.

This is the general principle in the one place it had been violated: a detector output is evidence.
`prolonged-vowel` routing to SPEECH *and* VOICE is the right answer — it is a recording of a spoken
count-in followed by a held vowel — and the correct handling is to route both and let the
per-branch scoring show it as an `extra` against the reference set, not to suppress the reading as
an error.

## The evaluator

`evaluate_routes(features, ruleset) -> RouteEvaluation`, a pure function over one
`RecordingFeatures`. Every branch's gates run. The frozen result carries:

- `routed` — the branches a gate fired for, from content alone, in `BRANCHES` order.
- `declared` — the branches the family is a reference positive for, **for comparison only**.
- `agreed` / `missed` / `extra` — routed and declared, declared and not routed, routed and not
  declared.
- `unavailable` — per branch, the *names* of the gates whose feature could not be read. A branch is
  keyed only when it has such a gate, and it is keyed whether or not another gate routed the branch
  anyway: an unreadable feature is a fact about the store, not about the recording.
- `fell_through` — `routed` is empty.
- `gate_outcomes` — every gate's own outcome, each gate evaluated once.

**`unavailable` is not `silent`.** `GateOutcome` has three members — `FIRED`, `SILENT`,
`UNAVAILABLE` — and a gate whose feature was never written to the store reads `UNAVAILABLE`. An
absent residual measurement is not a residual energy fraction of zero; a run with no YAMNet summary
is not a run with no chant; a recording with no consensus transcript is not a recording with no
repeated syllable. Collapsing the three into a boolean would report a broken or partial store as a
negative measurement, which is the failure mode that makes a scored ruleset look better than it is.

**Fall-through is a first-class output.** A recording falls through when no gate of any branch fired
for it. Under instruction-first routing a recording could fall through while carrying content a
non-declared branch's gate would have fired on — the 74 harvard recordings did exactly that — and
that class of fall-through no longer exists by construction.

`tally_families(evaluations) -> dict[str, FamilyTally]` aggregates a stream of evaluations into the
per-family counts, per branch on each of `routed`, `declared`, `agreed`, `missed`, `extra` and
`unavailable`, plus the fall-through count, so the ruleset can be scored over the corpus without
holding it.

`score_branches(evaluations) -> dict[str, Confusion]` answers the question this restructure exists
to ask: **can content-only routing recover the overarching families?** One 2x2 per branch — a
recording is a reference positive when its family is in that branch's `reference_family_set` entry,
and a prediction positive when the branch is in `routed` — carrying raw `tp`/`fp`/`tn`/`fn` and the
sensitivity and specificity derived from them. `Confusion` is the same class `report.py` scores
individual detectors with, so a branch's operating point and a candidate detector's are read on the
same scale. Sensitivity is `None` when a corpus holds no reference positive for a branch; that is
not a sensitivity of zero.

`scripts/score_taxonomy_ruleset.py` is the thin driver: it composes `load_ruleset`,
`evaluate_routes`, `tally_families` and `score_branches`, prints the per-branch counts, the
per-branch 2x2 with sensitivity and specificity, and the worst fall-through families, and persists
all of it to `ruleset_score.json`. It holds no rule logic.

## Next: the hint layer, which refines and never suppresses

Not implemented here, deliberately. The declared family is currently used only to score. The next
piece is to use it as a **hint** over an already-routed set:

- A hint names **which of the routed branches is the target** of the elicitation — the branch whose
  assessment the protocol was designed to support — leaving the others routed as context.
- A hint **flags a contradiction**: a family that declares a branch nothing routed to (`missed`) is
  a recording that did not do what it was asked, or a store that failed to measure it, and either
  is worth surfacing. A branch routed that the family never declared (`extra`) is content the
  protocol did not ask for, which is exactly what triage is looking for.
- A hint **never suppresses a routed branch and never lowers a gate.** It cannot remove a branch
  from `routed`, and no gate's threshold may depend on the family. Any such coupling would put the
  instruction back in the routing path through a side door, which is the failure this restructure
  removed.

## What still needs a DDK arm

Adding `DDK` to `BRANCHES` is consistent with the graph as it stands — `nodes/routing.py` selects
branches through `BRANCH_FOR_KIND`, which has no `ddk` kind, so `DDK` is never selected and
`run.py`'s loop marks it `SKIPPED`; `run.py` then filters its node outcomes through `GRAPH_ORDER`,
which does not name `DDK`, so nothing is reported for it.

What a DDK arm would still need, none of it done here:

- An acoustic syllable-repetition feature, as above. Without it DDK's gate is a transcript gate.
- `GRAPH_ORDER` in `vocabulary.py` does not include `DDK`, and `run.py`'s `branches` dispatch table
  has no `DDK` entry. A `DDK` that were ever selected would raise a `KeyError` there.
- `nodes/routing.py`'s `BRANCH_FOR_KIND` has no `ddk` kind, so nothing can select the branch.
- `nodes/report.py` builds `_EVIDENCE_BRANCHES = (*BRANCHES, "REDACT")`, so `_branch_evidence` emits
  an always-empty `"DDK"` key in the report JSON. Its `source_spans` table has no `DDK` entry, which
  is why this is empty rather than an error.
- There is no `nodes/ddk.py`, and `specs/20260817-triage-workflow-dag/dag.md` still documents
  `BRANCHES` as the three.
