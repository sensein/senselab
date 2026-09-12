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
recogniser concordance. That is the concrete defect this restructure fixes. The first pass replaced
the two old speech gates — two cuts on the *same* number, one for declared families and one for
everything else — with two gates on *different* numbers, either of which routed. That was still one
gate too many: concordance had been demoted from the only reading to one of two, when it is not a
reading of whether speech occurred at all. SPEECH is now one gate on lexical content, and
concordance is a flag. See "SPEECH routes on ASR words alone" below.

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
| SPEECH | `speech.lexical` | `words.lexical` | >= | 2 words | 0.809 | — | — |
| VOICE | `voice.sustained` | `span_longest_s.amplitude` | >= | 3.0 s | 0.65 | 0.93 | 0.25 |
| VOICE | `voice.glide` | YAMNet singing-union peak, plain stream | >= | 0.05 | 0.75 | 0.93 | 0.10 |
| VOICE | `voice.chant` | YAMNet `Chant` peak, plain stream | >= | 0.02 | 0.83 | 0.94 | 0.14 |
| AIRWAY | `airway.breath` | `residual.energy_fraction` | >= | 0.10 | 0.66 | 0.98 | 0.16 |
| AIRWAY | `airway.cough` | `span_label_set_stats["yamnet.cough_labels.peak_over_floor_db_max"]` | >= | 50.0 dB | **not re-measured** | — | — |
| AIRWAY | `airway.bracketed_event` | `bracketed_types` over `taxonomy.airway_bracket_tokens` | >= | 1 token | ~0.710 | — | — |
| AIRWAY | `airway.ppg_silent_fraction` | `ppg.silent_fraction` | >= | 0.90 | recall-first, not J | — | — |
| DDK | `ddk.lexical_repetition` | max token repetition in `transcript` | >= | 3 | **not measured** | — | — |
| DDK | `ddk.ppg_segment_rate_per_s` | `ppg.segment_rate_per_s` | >= | 10 /s | recall-first, not J | — | — |

The two posteriorgram gates were chosen by recall at an over-routing budget rather than by J, and
their measurements are in `specs/20260911-praat-ppg-detectors/design.md` rather than here.

Beside the gates, and never among them:

| branch | flag | feature | op | threshold |
| --- | --- | --- | --- | --- |
| SPEECH | `speech.transcript_agreement` | `words.agreement` | >= | 3 words |

And behind all of them, consulted only where none fired, one bypass:

| bypass | feature | rule |
| --- | --- | --- |
| `emptiness` | max tracked-label peak of `enhanced\|yamnet` and of `residual\|yamnet` | both `< 0.2` |

And after all of them, `QUALITY`: a node every recording reaches, gated by nothing. See "QUALITY is
a graph edge, not a route" below.

The SPEECH J above is `sens 0.954 + spec-excluding-DDK 0.855 - 1`; `airway.bracketed_event`'s comes
off the capped transcript and is a lower bound. Both are derived in their own sections below, and
neither is a max-Youden point over a swept grid — SPEECH's threshold sits on an artifact boundary
and the bracket gate has not been re-measured from the extracted counts yet.

"Max-family firing" is the highest per-family firing rate at that threshold, "median-family firing"
the median across all 48. The pair is the spread the operating point was chosen against, and it is
what says whether a J is carried by separation or by prevalence. Every VOICE and residual/span
AIRWAY row above is the max-Youden operating point from the 62,547-recording corpus sweep.

`voice.sustained`, `voice.glide` and `airway.breath` are unchanged: they sit at measured operating
points and this restructure had no evidence against any of them. `airway.cough` no longer reads the
blanket span statistic; see "The cough gate reads cough-labelled spans" below. Its 50 dB cut is
carried over onto the conditioned population **unswept**, because the conditioned population is
smaller and louder than the blanket one and no sweep over it exists. Re-fitting it from the corpus
run is the first thing to do with that run.

### `voice.chant`, added: the best voice separator in the sweep, previously unused

`voice.yamnet_chant_peak.plain` — the bare `Chant` label on the plain stream, not the singing union
that contains it — carries **corpus J 0.83** at 0.02, with max-family firing 0.94 and median-family
firing 0.14. That is the highest J of any voice-family detector in the sweep, above
`voice.glide`'s 0.75, and it was sitting in `PRIMARY_DETECTORS` unrouted. It is added as a third
VOICE gate rather than replacing `voice.glide`, because the two read different constructions of the
same evidence (one label against a twelve-label union) and either firing is enough — the content-first
rule is that a reading is used, not that the readings must agree.

### SPEECH routes on ASR words alone, at two of them

**The gate for speech is the ASR.** `branch_gates.SPEECH` is one entry, `speech.lexical`, reading
`words.lexical`. `speech.agreement` is **deleted from `branch_gates`** — it was never a statement
that speech occurred. Agreement is a statement about *what* was said, and it is now a flag; see
"Agreement is a SPEECH-branch flag" below.

**Why 2 and not 1.** With `LEXICAL_SPEECH` as the reference set, `lexical >= 1` fires on 0.569 of
glide families and 0.403 of sustained families — a single spurious token on a held vowel, which is
an ASR artifact rather than speech. At `>= 2` the glide rate collapses to 0.065. Two is where that
artifact dies, and it is chosen on that artifact boundary rather than on a swept J.

**Why not 4.** Going further does not remove noise, it removes speech. `prolonged-vowel` drops from
0.786 at `>= 3` to 0.293 at `>= 4`, and its word-count histogram says exactly why: a spike of 791
recordings at *exactly three* lexical words, which is the protocol's spoken preamble
*"1, 2, 3 aah"*. Threshold 4 discards the whole spike.

| threshold | sens | spec (excluding DDK) | `prolonged-vowel` firing | glide-family firing |
| --- | --- | --- | --- | --- |
| >= 1 | — | — | — | 0.569 |
| >= 2 | 0.954 | 0.855 | — | 0.065 |
| >= 3 | 0.932 | 0.877 | 0.786 | — |
| >= 4 | — | 0.933 | 0.293 | — |

**2 is chosen and still unswept in the J sense**: it is an artifact boundary, not a max-Youden
point over a grid. **3 is the defensible alternative** — sens 0.932 and spec-excluding-DDK 0.877,
against 0.954 / 0.855 at 2 — and choosing between them is a judgement about which error costs more,
which no number here settles.

**The headline specificity understates the gate badly, and the reason is the reference set.** At
`>= 4`, 7,468 of the 8,899 false positives (83.9%) are diadochokinesis families. Those recordings
are speech. They sit in the negative set only because `LEXICAL_SPEECH` excludes DDK by
construction. Excluding DDK from the negatives, specificity at `>= 4` is **0.933** rather than
0.696. At `>= 2` the same correction runs 71.3% DDK and **0.855** excluding DDK. Every
specificity quoted in this section is the DDK-excluded one unless it says otherwise.

**`words.total` is the wrong field to read.** `total >= 1` scores specificity **0.075**, because
bracketed tokens fire almost everywhere — nearly every recording carries a `[breath]` or an `[uh]`.
`lexical`, the non-bracketed count, is the field that separates. The bracketed half is not noise
either; it is airway evidence, and it has its own gate now.

**What a sweep would still add**, unchanged from the earlier pass and no longer blocking: score
`speech.lexical` over `COUNT_GRID` against `LEXICAL_SPEECH` with DDK held out of the negatives, and
report max-family and median-family firing at each point beside J, so a J carried by prevalence is
visible.

### Agreement is a SPEECH-branch flag

`speech.transcript_agreement` (`words.agreement >= 3`, renamed from `speech.agreement`) is declared
in `taxonomy.ruleset.branch_flags`, not in `branch_gates`. It is evaluated on every recording, its
outcome is recorded in `gate_outcomes` like any gate's, and a branch it fired on is named in
`RouteEvaluation.flags` and counted in `FamilyTally.flagged`. It contributes to nothing else:
`routed`, `agreed`, `missed`, `extra` and `fell_through` are all computed without it, and
`load_ruleset` refuses a configuration that lists one gate as both a gate and a flag of the same
branch.

The distinction it encodes: **agreement is not the hallmark of whether speech was said, it is about
whether there was doubt about what was said.** Three recognisers converging on the same three words
says the transcript is trustworthy, not that the recording contains speech — the lexical count
already said that. Three recognisers diverging says the transcript is doubtful, which is a fact a
downstream consumer of the SPEECH branch needs and a fact the router must not act on.

**No downstream consumer is implemented.** The flag is carried and reported; nothing reads it yet.

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

**The acoustic gate beside it is `ddk.ppg_segment_rate_per_s`**, added 2026-09-12: the
posteriorgram's argmax-segment rate, which is a rate of articulatory change and needs no recogniser
to spell the unit. It was fitted rather than assumed, and its measurements are in
`specs/20260911-praat-ppg-detectors/design.md`. Threshold 3 on the lexical gate is still unswept.

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
- `flags` — per branch, the *names* of the flag gates that fired. Keyed only when one did.
- `state` — one of `routed`, `empty` and `unexplained`; see "Three outcomes, and only one of them
  indicts the ruleset" below. It replaces the `empty` and `fell_through` booleans, which could both
  be false and said nothing about which of the three had happened.
- `gate_outcomes` — every gate's own outcome, each gate evaluated once. Never empty now: every gate
  runs on every recording, including one the emptiness rule goes on to call empty.

**`unavailable` is not `silent`.** `GateOutcome` has three members — `FIRED`, `SILENT`,
`UNAVAILABLE` — and a gate whose feature was never written to the store reads `UNAVAILABLE`. An
absent residual measurement is not a residual energy fraction of zero; a run with no YAMNet summary
is not a run with no chant; a recording with no consensus transcript is not a recording with no
repeated syllable. Collapsing the three into a boolean would report a broken or partial store as a
negative measurement, which is the failure mode that makes a scored ruleset look better than it is.

**The unexplained count is a first-class output.** A recording is unexplained when no gate of any
branch fired for it and the audio does not say why. Under instruction-first routing a recording could
land there while carrying content a non-declared branch's gate would have fired on — the 74 harvard
recordings did exactly that — and that class no longer exists by construction.

`tally_families(evaluations) -> dict[str, FamilyTally]` aggregates a stream of evaluations into the
per-family counts, per branch on each of `routed`, `declared`, `agreed`, `missed`, `extra`,
`unavailable` and `flagged`, plus `states`, which carries **all three** route states whether or not
the family reached one and sums to `recordings`. Two booleans could not: `empty=False,
fell_through=False` was the routed case and read as an absence of both facts rather than as a third
one.

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

## Span statistics conditioned on the label the span carries

`RecordingFeatures.span_label_stats` was added because the span/label join was being thrown away at
extraction, and the whole-file measurements that survived it are the weaker half of the evidence.

### What the whole-corpus cough gateway measured

Reference = the cough-eliciting families, 62,547 recordings.

| detector | best operating point | sens | spec | J |
| --- | --- | --- | --- | --- |
| `cough.yamnet_cough_minus_breath` | `>= 0` | 0.731 | 0.902 | 0.632 |
| `cough.amplitude_peak_over_floor_db_max` | `>= 55 dB` | 0.767 | 0.827 | 0.594 |

The amplitude detector's own sweep grid stopped at 50 dB and reported J 0.534, so the grid ceiling
was hiding its optimum by 0.06 of J. `DB_OVER_FLOOR_GRID` now runs to 80 dB (45, 55, 60, 70 and 80
added), which is the only reason the 55 dB point is visible at all. A grid whose best threshold is
its own last entry has not been swept; it has been truncated.

The two rows are the motivation for conditioning. The classifier detector is the stronger of the
two and reads no amplitude; the amplitude detector is the weaker and reads no label. A loud
amplitude span is weak evidence of a cough. A loud amplitude span *that YAMNet labels Cough* is a
different and much stronger quantity, and nothing in the extracted record could express it, because
the extractor kept spans and labels in separate tables joined by nothing.

### `all.peak_over_floor_db_*` and `amplitude.peak_over_floor_db_*` are the same numbers

Only amplitude spans carry a finite `peak_db`; continuity and gap spans carry none, and
`_span_statistics` drops a non-finite peak from the sample before summarising it. So the `all`
population of the `peak_over_floor_db` distribution *is* the `amplitude` population, and every
statistic of the two is numerically identical on every recording. This is not a defect and neither
key is redundant — the same is not true of the duration or SQUIM distributions, where `all` is a
genuinely wider sample — but a sweep that reports both as independent detectors is reporting one
detector twice.

### Argmax was the first pass, and it starved the feature

The first pass attributed each span to the **argmax** of its per-label scores: top-1, no floor, one
label. Measured over the whole corpus, that is what it cost.

| measurement | value |
| --- | --- |
| recordings with **no** Cough-labelled span at all | 60,092 of 62,547 |
| recall ceiling that imposes on any `Cough`-conditioned gate | **0.624** |
| `max`, `p75` and `p90` conditioned on `Cough` | **identical curves** |

The three statistics coincide because a recording with a Cough span almost always has exactly one,
so its distribution has one point and every percentile of it is that point. `p75` and `p90` were
added to separate a recording with one loud cough from one with a loud cough and a louder door slam;
with one span each they cannot.

And the refinement still worked. At a **2% over-routing budget**:

| gate | recall at 2% over-routing |
| --- | --- |
| `airway.cough` conditioned on the `Cough` label | **0.624** (its ceiling) |
| `cough.yamnet_cough_minus_breath` | 0.499 |
| blanket `amplitude_peak_over_floor_db_max` | 0.176 |

Equal-or-better recall at roughly a quarter of the cost of the blanket reading. The conditioning is
not the problem; the attribution feeding it is. **This table is the baseline the change below is
measured against.**

### Membership is top-K ∩ floor, and it is the same rule PREPROCESS writes

A span carries a label when the classifier puts it in **the span's top K** *and* its score is **at or
above the floor**. Both numbers are configuration — `windows.<classifier>.label_top_k` and
`windows.<classifier>.default_threshold`, shipped at **4** and **0.2** — and neither appears as a
literal in code. `LabelMembership` in `senselab/audio/workflows/triage/label_membership.py` is the
one implementation of the rule; `_confident_labels` in `nodes/preprocess.py` and
`_label_span_statistics` in `routing_analysis/features.py` are its two callers.

Consequences, each of them a test:

- A span may carry several labels and contributes to **each** of their distributions. Argmax kept
  one and threw the rest away.
- A label over the floor but ranked fifth carries nothing. Membership is a conjunction.
- A span whose best label is under the floor carries nothing at all, which is different from a span
  the classifier never saw.
- Pooling across the windows of one span happens **before** the rank: a label's score on a span is
  its maximum over every window the classifier placed there, and the top-K is taken over those
  maxima. Taking each window's top-K and unioning them would rank a label on whichever window it
  happened to survive, and the two are not the same set.

### The label sets are read by name, never restated

`routing_analysis/labels.py` owns `LABEL_SETS` — `cough_labels`, `breath_labels`, `singing`,
`whistle` — per classifier. `features.py` iterates that mapping and emits one distribution per set
beside the per-label ones, into `RecordingFeatures.span_label_set_stats`, keyed
`"<classifier>.<set>.…"`. A span counts toward a set when **any** member of it is one of the labels
the span carries, and it counts **once** however many members qualify: the set's distribution is over
spans, not over (span, label) pairs, so it is not the union of its members' distributions.

Those tuples are being regenerated from the AudioSet ontology. Nothing outside `labels.py` restates
a member of them — not this document's rules, not the config, not the detectors — so the
regeneration lands without a second edit.

`span_label_set_stats` is a field of its own rather than more keys in `span_label_stats`, and a
reader source of its own (`span_label_set_stat`). A set name and a label name would otherwise share
one namespace, where `singing` and `Singing` are one casefold apart.

### The cough gate reads cough-labelled spans

`branch_gates.AIRWAY`'s `airway.cough` was `span_stat all.peak_over_floor_db_max >= 50`: the loudest
span in the file, whatever anything called it. As a whole-corpus door that scores **J −0.141** and
fires on **38.7% of non-airway recordings** — it is a loudness detector, and loudness is not an
airway event.

It now reads `span_label_set_stat yamnet.cough_labels.peak_over_floor_db_max`, at the same 50 dB.

The blanket detector **stays in the catalogue**. Within the airway families it is the best
cough-vs-breath discriminator there is, at **J 0.787** — but that is a question the AIRWAY branch
asks after it has the recording, not a question the router asks to decide whether to hand it over.
The two jobs wanted the same number at different operating points, and only one of them is the
router's.

The 50 dB cut is **carried over unswept** onto a population that is smaller and louder than the one
it was fitted on. It is the first thing the corpus run should re-fit.

### The detectors added

Three per-label, from the earlier pass, reading `("span_label_stat", "yamnet.Cough.…")`:

- `cough.yamnet_cough_span_peak_over_floor_db_max`
- `cough.yamnet_cough_span_peak_over_floor_db_p75`
- `cough.yamnet_cough_span_peak_over_floor_db_p90`

Three set-conditioned, reading `("span_label_set_stat", "yamnet.cough_labels.…")`:

- `cough.yamnet_cough_set_span_peak_over_floor_db_max`
- `cough.yamnet_cough_set_span_peak_over_floor_db_p75`
- `cough.yamnet_cough_set_span_peak_over_floor_db_p90`

All six over `DB_OVER_FLOOR_GRID`, which runs to 80 dB. Six and not the cross-product of every set
and every statistic: the sets other than `cough_labels` are emitted into the shard so a later sweep
can read them, and no detector is catalogued for a question nobody has asked yet.

Whether `p75` and `p90` separate from `max` is now a live question rather than a foregone one. Under
argmax they were the same curve because the population had one member; under top-4 ∩ 0.2 a recording
can carry several cough-set spans and the three can differ.

### Size

The shard is **1,207 MB over 62,547 recordings, 19.3 kB each** — the number after the argmax pass,
which took it from 983 MB (+23%) against an estimate of +6–11%. That estimate was wrong because it
assumed 2–4 emitted labels per recording; the count came in higher.

Each emitted group costs 9 keys — `span_count` plus the eight statistics of `_stats` — at **430–520
bytes**. Two things grow the group count:

| source | groups added | bytes |
| --- | --- | --- |
| up to 4 labels per span instead of 1 | realistically ~2x the distinct **tracked** labels, since most of a span's top 4 (`Speech`, `Silence`, `Music`, room tone) are untracked and dropped | +0.9–1.8 kB |
| the named sets, per classifier | at most 6 (3 sets x 2 span classifiers); in practice `cough_labels` and `breath_labels` on the classifiers that fired | +0.9 kB |

**Estimate: +10% to +15%, so roughly 1.33–1.39 GB.** That is under the ~2 GB ceiling, so emission
stays over all of `TRACKED_LABELS` plus the named sets rather than being restricted to the sets.

The absolute worst case is 36 tracked labels (29 YAMNet, 7 HeAR) plus 6 sets, 42 groups at ~470 B =
~20 kB, which would double a recording. It needs 36 distinct tracked labels each inside some span's
top 4 and over 0.2 in one recording, and it does not occur. If the corpus run comes back over 2 GB,
the fix is to emit only the sets and drop the per-label groups — the sets are what the gates read.

Extraction parses the `span_yamnet` records it used to skip by raw-string prefilter. Measured at
17.3 kB and 99 µs per record for a 521-label dump on this laptop, at a few tens of spans per
recording that is a few milliseconds per store and a few minutes over the corpus. Top-K adds a sort
of that dump per span rather than a max over it, which is the same order of work; pooling per-label
maxima across a span's windows holds one float per label per live span for the length of one store,
a few MB at the corpus's span counts and freed per recording.

### What the store's own `labels` list now says

PREPROCESS writes two things per span classifier window: `raw_scores`, the model's full output, and
`labels`, the membership over it. Under the shipped config the analysis could not read `labels`:
`windows.yamnet.default_threshold` and `windows.hear.default_threshold` were both `null`, so
`_confident_labels` never ran and **every** per-span window in the 62,547-recording extraction is
stamped `labelled: false` with no `labels` key. That is the whole reason the analysis re-derives
membership from `raw_scores`.

Both floors are now **0.2** and both classifiers carry `label_top_k: 4`, so a future run stamps the
same membership the analysis re-derives.

**This changes `config_hash`, and it changes PREPROCESS.** Plainly:

- `span_yamnet` and `span_hear` windows now arrive with `labelled: true`, a `labels` list, a
  `scores` mapping, and the `default_threshold` and `label_top_k` that produced them. They already
  carried `raw_scores` and still do — the model's output is a measurement and is never a decision's
  to discard.
- `nodes/taxonomy.py`'s `_span_label_evidence` reads that `labels` list, so it has evidence where it
  had none. `taxonomy.presence_floor` is still null in every line, so every line still reads
  unavailable and no state changes yet.
- The whole-file `<classifier>_windows` fold is a **different** block. It reads
  `windows.<classifier>.label_thresholds` through `require`, that key is still null, and so
  `yamnet_windows`, `ast_windows` and `hear_windows` stay `absent` exactly as before. Store size per
  recording is unchanged; the fold would add one entity carrying all 521 raw scores per window.
- **The existing corpus is unaffected.** It was produced with the nulls, its span windows carry
  `labelled: false`, and the analysis re-derives from the stored `raw_scores` either way. Nothing
  needs re-running to read it under the new rule.

`analyze_routing_evidence.py` takes `--config` and passes the same resolved pair into every worker,
so an override moves the extraction and a future PREPROCESS run together. Extraction refuses to run
when a floor is null rather than defaulting one.

## Emptiness is a bypass, not a precondition

Owner's model, verbatim: *"everything reaches quality after other branches, but emptiness reaches
quality directly"*, and *"emptiness will only be evaluated if the other rules have not found a
route."*

The previous pass had it the other way round. `evaluate_routes` asked the emptiness rule first and
returned early when it fired, so **no gate ran on an empty recording** and `gate_outcomes` came back
empty. That is now inverted:

1. Every branch gate and every branch flag is evaluated on every recording. Nothing short-circuits.
2. Emptiness is consulted only where no gate fired, as the explanation for that.

### Three outcomes, and only one of them indicts the ruleset

`RouteEvaluation.state` is a `RouteState`, and it has exactly three members:

| state | what happened | what it is a charge against |
| --- | --- | --- |
| `routed` | one or more branches claimed the recording | nothing |
| `empty` | nothing routed, and the audio explains why | the recording |
| `unexplained` | nothing routed, and the audio does not | **the ruleset** |

The names are the point. `fell_through` did not say whether the recording was blank or whether the
gates had missed something, and it was reported beside an `empty` count that a reader had to subtract
by hand. `unexplained` is the number to drive the gates from: it is content the ruleset could not
account for. `FamilyTally.states` counts all three per family and
`scripts/score_taxonomy_ruleset.py` prints all three as a corpus block, so no reader has to derive
one from the other two.

### What inverting the order costs, and why it is the right cost

Because gates now always run, an empty recording where a gate fires is **routed**, not empty. That
is intended: a gate firing is evidence the file is not empty, and it is the same content-first rule
that governs every other reading here. Two facts follow, both of them signals rather than bugs:

- The disagreement — the emptiness rule says blank, a gate says otherwise — is a **measurement of
  the emptiness rule**, and the corpus run is where it becomes visible. Do not suppress it, and do
  not add a rule that lets emptiness veto a gate.
- The previous ordering depressed branch sensitivity in `score_branches` for a reason unrelated to
  the gates: an empty recording stayed a reference positive for whatever its family declared, that
  branch landed in `missed`, and the gate that might have recovered it was never asked. Every gate
  now gets its chance first, so a sensitivity change between the two passes is partly this and not
  a change in any gate.

`taxonomy.ruleset.emptiness` is unchanged as data: both stream names and the 0.2 floor stay exactly
as they were. Only *when* it is consulted has changed.

### QUALITY is a graph edge, not a route

`QUALITY` is in `vocabulary.GRAPH_ORDER`, after `VOICE` and before `REDACT`. It is **not** in
`BRANCHES`, it has no entry in `taxonomy.ruleset.branch_gates` or `branch_flags`, and it is not in
`reference_family_set`. Every recording reaches it whatever routed, which is precisely why it cannot
be a route: a gate that fires on everything selects nothing.

What that costs elsewhere in the graph, all of it checked:

- `run.py` skips `GRAPH_ORDER` slices on the ADMIT-fail and PREPROCESS-fail paths, so `QUALITY`
  would have been recorded `SKIPPED` there and absent on the happy path. `_drive_branches` now
  records it `SKIPPED` after the branch loop as well, so every path agrees. There is no
  `nodes/quality.py` and the runner has no dispatch entry for it.
- `verdict.py`'s `_GRAPH_ORDER` is `GRAPH_ORDER[:-1]`, so `FileVerdict.ran` already carried
  `QUALITY: skipped` from the store's own reading, with or without the runner's line.
- `nodes/report.py` orders node rows by `GRAPH_ORDER`, so a `QUALITY` row reads `skipped`.
- The `quality:` config section — `stoi_floor`, `pesq_floor`, the disruption tolerances — is still
  null and still read by nothing. Declaring the node does not adopt those values.

### What the two streams measure on short recordings

Of the 1,615 recordings under 0.5 s:

| stream | median max peak | fraction below 0.2 |
| --- | --- | --- |
| `enhanced\|yamnet` | 0.003 | 0.83 |
| `residual\|yamnet` | 0.000 | 0.98 |

Median silence fraction on that set is **1.000**. For the 60,932 recordings at 0.5 s or longer,
`enhanced|yamnet`'s median max peak is **0.999** and only 8% fall below 0.2. The separation is not
marginal: three orders of magnitude between the two populations on the same statistic.

Applying `enhanced < 0.2 AND residual < 0.2` marks **1,336 of the 1,615** short recordings empty,
of which only **53** carry any lexical word at all. The remaining 252 short recordings are real but
truncated — `Speech = 1.00`, one to three lexical words, in clips of 0.2 to 0.45 s — and the AND
correctly leaves them out of the empty set. **0.2** is the floor, one config value read for both
streams. Those numbers were taken under the old ordering, where the rule decided alone; under the
new one the 53 with lexical words route to SPEECH instead, and the empty count over the same
population is a corpus measurement this document does not yet have.

### Why both streams and not the enhanced one alone

The enhanced stream is FRCRN's output and the residual is what enhancement removed. A recording
where enhancement suppressed everything reads low on `enhanced` while the residual still carries
the removed content, so the enhanced stream on its own would call a noisy-but-non-empty recording
empty. Requiring both to be under the floor means the claim is that neither the cleaned signal nor
what was subtracted from it carried anything a classifier could name.

### This belongs in ADMIT eventually

Emptiness is **ADMIT-shaped**: it is a statement that the recording should not have entered the
graph at all, which is that node's question, not the ruleset's. It lives in the ruleset now for one
reason — that is where it can be measured against the routed set, over an extracted feature shard,
without re-running the graph. `nodes/admit.py` is deliberately untouched. Moving it there is an open
item, and the move should carry the floor and both stream names with it rather than restating them.
The move also has to carry the bypass ordering: ADMIT runs before any gate, so an ADMIT that
discarded on emptiness would restore exactly the short-circuit this pass removed.

### What the empty set does to the scores

An empty recording stays a reference positive for whatever branch its family declares, so its
`declared` branch lands in `missed` and it counts as a false negative in `score_branches`. That is
the honest reading — a branch genuinely was not recovered — but it means a corpus with many empties
reports a depressed sensitivity for a reason that has nothing to do with the gates. `FamilyTally`
counts `empty` in `states` so that reason is visible rather than inferred.

## Bracketed tokens are AIRWAY evidence, not speech

`[breath]` is a detection with a timing. The ruleset was discarding it as "not lexical", which is
true and is not a reason to throw it away.

### Bracket types are strongly separable

Counts over the corpus, taken from the consensus transcript — which is capped at
`TRANSCRIPT_CAP = 300` characters, so **every number in this table is a floor**:

| token | total | airway families | lexical-speech families | other |
| --- | --- | --- | --- | --- |
| `[uh]` | 15,746 | 354 | 10,991 | 4,401 |
| `[breath]` | 10,896 | **9,037** | 980 | 879 |
| `[um]` | 10,117 | 292 | 9,663 | 162 |
| `[cough]` | 2,045 | **2,001** | 24 | 20 |
| `[laughter]` | 1,924 | 777 | 569 | 578 |
| `[throatclearing]` | 542 | 360 | 172 | 10 |
| `[sniff]` | 100 | 82 | 17 | 1 |

Breath, cough, throat-clearing and sniff brackets fire on **0.734** of airway families against
**0.024** of lexical-speech ones: roughly sens 0.734 / spec 0.976, **J ~0.710**, which is better
than the current best AIRWAY entry (`airway.cough` at J 0.79 is on a different reference set;
against the same one this is the strongest bracket-derived reading available). `[uh]` and `[um]` go
the other way entirely — they are fillers, they are speech, and they are **not** in the airway set.
`[laughter]` splits 777/569 and is in neither set.

### Typed counts, extracted rather than re-parsed

`words.bracketed` was a bare count with no type, and the transcript it could have been recovered
from is capped. `RecordingFeatures.bracketed_types` is therefore extracted from the **word
entities**: every live consensus word whose `bracketed` attribute is true contributes its
`bracket_type(text)` — the text between the brackets, casefolded, with every non-alphanumeric
character removed, so `[Throat Clearing]`, `[throat-clearing]` and `[THROATCLEARING]` are one type.
A word invalidated in the store contributes nothing, as everywhere else. The field is
`dict[str, int]`, keyed only by types the recording carries, sorted, and it costs nothing on a
recording with no brackets.

Because it comes off the entities, it is not subject to the 300-character cap and it is not subject
to whatever the transcript renderer did to the tokens. **The table above came from the capped
transcript and is a lower bound; the gate must be re-measured from the extracted counts before its
J is quoted as settled.** `detectors.py` carries the candidates that sweep does:
`airway.bracketed_breath`, `_cough`, `_throatclearing`, `_sniff`, `_laughter`, the union
`airway.bracketed_airway_union`, and `speech.bracketed_uh`, `_um` and `_filler_union` on the other
side, each over `COUNT_GRID`.

### The gate

`airway.bracketed_event` reads `("bracketed_set", "airway")` at `>= 1` and is a member of
`branch_gates.AIRWAY`. **Which bracket types are airway is config**, at
`taxonomy.airway_bracket_tokens` beside `taxonomy.airway_ontology_roots`, currently
`[breath, cough, throatclearing, sniff]`. `load_ruleset`
resolves the set name into the gate's feature tuple at load, so the gate carries its members and
the reader stays a plain read over `RecordingFeatures`. A gate naming a set `BRACKET_SET_PATHS`
does not have raises rather than reading an empty union.

A store that never ran consensus reads `UNAVAILABLE` on this gate rather than zero, matching the
rule everywhere else: an unwritten measurement is not a measurement of nothing.

## Onomatopoeic renderings are the same events, spelled as words

`[cough]` is a bracket the previous section can count. `Cough` is the same event, in the same
recording, spelled by the recognizer as an ordinary word — and `words.lexical` counts it as speech.
The corpus carries **1,176** instances of `cough` as a lexical token, plus the Chinese renderings
`咳` and `呵`.

`words.onomatopoeic_tokens` has existed for exactly this since the v2 configuration was drawn, and
was `null` in every shipped release. It is populated now.

### What the null key was costing

**545** `respiration-and-cough-*` recordings carry `words.lexical >= 4`. **523** of those also trip
the `ddk.lexical_repetition` gate, because `cough cough cough cough` is four repeats of one token —
which is what that gate counts, and it is counting a cough. Those recordings route to AIRWAY,
SPEECH **and** DDK. AIRWAY is right; the other two are handed a subject they cannot assess, which is
the failure mode "Four branches, because diadochokinesis was in none of three" describes for DDK
firing the speech detectors, arriving here from the opposite direction.

Three from the corpus:

```
lex=  6 agr=  1  '[cough] Cough Cough Cough Cough Cough Cough'
lex=  5 agr=  0  '[cough] 咳 咳 咳 咳 咳'
lex=  4 agr=  0  '[throatclearing] cough cough cough cough'
```

**Agreement does not catch it, and cannot.** Mean `words.agreement` over those 545 is **0.39**, and
only **20 of 545** reach 2. The two recognizers pick *different* onomatopoeia — English `cough`
against Chinese `咳` — so they never agree on the token even though both heard the same event.
This is the case the agreement flag was separated from the speech gate for (see "Agreement is a
SPEECH-branch flag"): concordance is a statement about what was said, and here both readings are
right about the event and disagree about its spelling.

Note also the first and third rows: a bracketed `[cough]` and a lexical `Cough` in the *same*
recording. One recognizer's bracket does not suppress the other's word, so the bracket gate already
fires on some of these and the lexical count still routes them to SPEECH.

### The lexicon, measured

A strict lexicon — `{cough, coughs, coughing, 咳, 呵, ahem, hack, khh, kof, cof}` — firing at
**at least 2 tokens**, over all 62,547 recordings:

| population | fires |
| --- | --- |
| cough families | **0.361** (1,015 of 2,813) |
| breath families | 0.001 |
| DDK families | 0.005 |
| lexical speech | 0.001 |

As a COUGH gate: **sens 0.361, spec 0.999**, tp 1,015, fp 81 of 59,734.

Low recall, near-perfect precision. Under this document's router framing — over-routing costs a
branch some discarded work, under-routing loses the recording entirely — that is nearly free recall,
and it is recall on a population the acoustic cough gate reaches by a different route.

**Two tokens, not one, and the reason is `hack` and `cough`.** Both are ordinary English words. One
of either inside read speech is a mention, not a manoeuvre; two of them in one recording is not how
anyone discusses a cough. The 0.001 lexical-speech rate above is what that cut buys.

### One flat list, and no invented breath set

The key's comment says "cough/breath-like". Only the cough set has been measured, so only the cough
set is shipped. There is no `breath:` group holding an empty list waiting for someone to fill it
from intuition.

It is **one flat list rather than a mapping of event to tokens**, for a mechanical reason:
`nodes/preprocess.py` already reads this key as a token sequence, one `vocabulary_key` per entry. A
mapping would make that reader take the group names as the vocabulary and bracket the literal word
`cough` while ignoring every token under it — a silent, total inversion of what the key means. When
a breath set is measured, the shape to reach for is a second key beside this one, not a grouping
inside it.

### Populating the key changes PREPROCESS, at one token rather than two

This is the same key `consensus.bracketed_form` reads, and that reader has **no threshold**: from
now on, one `cough` in a recording is rendered `[COUGH]`, leaves `words.lexical`, and lands in
`bracketed_types["cough"]`, where `airway.bracketed_event` reads it at `>= 1`. That fixes the 545 at
their source and is why the key exists.

It is not the operating point measured above. **The firing rate of this lexicon at `>= 1` token was
not measured**, so the cost of the difference — a genuine spoken mention of a cough or a hack, now
rendered non-lexical — is stated here rather than quantified. The 81 lexical-speech false positives
at `>= 2` are a *lower* bound on it and nothing more: every recording that fires at two tokens fires
at one, and an unknown number more join them. Measuring the `>= 1` rate is the next thing to do with
this key, and it is a question about PREPROCESS's reader, not about the detector below.

### The feature counts word entities, and the detector is staged, not gating

`RecordingFeatures.onomatopoeic_types` is `dict[str, int]`, keyed by vocabulary key, counting
**live, unbracketed consensus word entities** whose key is in the vocabulary. It is exactly the
construction `bracketed_types` uses in the previous section, and for the same reason: the transcript
is capped at `TRANSCRIPT_CAP = 300` characters, so a count taken from it is a **floor** — every
number in the bracket-type table above is one — while a count taken off the entities is **exact**.
On these particular recordings the cap would cost little, since fifty `cough`s fit inside 300
characters; but "costs little on this corpus" is a property of the corpus, and the entity count is
exact on every corpus. It also survives whatever the transcript renderer did to the tokens.

The two fields never count the same word. A bracketed token goes to `bracketed_types` and stops
there; an unbracketed one goes to `onomatopoeic_types`. Counting `[cough]` in both would inflate
both, and the pair would no longer add up to the recording's word stream.

**The vocabulary the extractor reads is the reader's own, not the run's.** A store produced while
the key was null carries `cough` as a lexical word; extracting it under the populated vocabulary is
what makes the 62,547-recording sweep readable at all. `extract_features` therefore takes the
vocabulary as an argument — from `onomatopoeic_vocabulary(config)`, beside `span_label_memberships`
— rather than reaching for a packaged default.

`cough.words_onomatopoeic` reads `("onomatopoeic",)`, the sum over every key, and returns `None`
where no consensus measurement was written — an unwritten measurement is not a measurement of
nothing, as everywhere else. It is **not in `branch_gates`** and **not in `DETECTORS`**. The threshold above is a
hand-measurement, not a recall-first operating point chosen over a swept grid, and this document's
rule is that a gate's cut comes off a sweep.

### A detector the profile has never seen

Since `specs/20260912-detector-grids/design.md`, a detector's grid is derived from that detector's
own distribution in `data/detector_profile/`, and a detector the profile does not carry raises at
`build_catalogue`. That is the intended way to learn the corpus needs re-sweeping — and it is a
deadlock for a detector that has never been swept, because the sweep that would profile it cannot
run until it is declared, and declaring it in the catalogue breaks the import.

It is resolved by declaring it **outside** the catalogue. `UNPROFILED_DETECTORS` holds candidates
with no grid at all: `detector_value` reads one exactly like a catalogue detector, which is all a
profiling sweep needs, and `sweep_points` returns nothing, so no threshold is ever scored. Nothing
is defaulted, interpolated or borrowed from a sibling.

Two checks keep the staging area from becoming a place things are forgotten, both at import, both
modelled on `_check_pins`:

| check | raises when |
| --- | --- |
| name collision | a staged id is already in the catalogue — one id, one grid |
| now profiled | the newest profile carries a staged id, so its grid is derivable and it belongs in the catalogue |

The second is the one that matters. The next corpus sweep profiles `cough.words_onomatopoeic`, the
new profile ships, and the import raises until the candidate is moved into `_CANDIDATES` — where it
gets its integer grid and is scored at every count from 0 upward, including the 2 measured here.
A staged detector that quietly stayed staged after its profile arrived would be a detector reported
as unmeasured while its measurement sat in the file, which is the same failure as a silent grid
fallback.

Rejected, and why:

| candidate | why not |
| --- | --- |
| declare it in `_CANDIDATES` and ship a hand-written grid | the grid constants were removed for cause; a hand grid is exactly what `specs/20260912-detector-grids/design.md` deleted |
| let `build_catalogue` skip a detector the profile lacks | a silent skip is how three detectors scored 0.000 everywhere for three sweeps |
| derive the grid from a sibling reading `words.lexical` | the siblings rule is for a *gated* detector reading the same primary feature; a different feature's distribution is not this one's |
| a second onomatopoeia key, so `words.onomatopoeic_tokens` could stay null | pre-alpha: one lexicon, not two that drift |
