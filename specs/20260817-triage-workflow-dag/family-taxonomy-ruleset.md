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
| AIRWAY | `airway.cough` | `span_stats["all.peak_over_floor_db_max"]` | >= | 50.0 dB | 0.79 | 0.96 | 0.07 |
| AIRWAY | `airway.bracketed_event` | `bracketed_types` over `taxonomy.airway_bracket_tokens` | >= | 1 token | ~0.710 | — | — |
| DDK | `ddk.lexical_repetition` | max token repetition in `transcript` | >= | 3 | **not measured** | — | — |

Beside the gates, and never among them:

| branch | flag | feature | op | threshold |
| --- | --- | --- | --- | --- |
| SPEECH | `speech.transcript_agreement` | `words.agreement` | >= | 3 words |

And ahead of all of them, one precondition:

| precondition | feature | rule |
| --- | --- | --- |
| `emptiness` | max tracked-label peak of `enhanced\|yamnet` and of `residual\|yamnet` | both `< 0.2` |

The SPEECH J above is `sens 0.954 + spec-excluding-DDK 0.855 - 1`; `airway.bracketed_event`'s comes
off the capped transcript and is a lower bound. Both are derived in their own sections below, and
neither is a max-Youden point over a swept grid — SPEECH's threshold sits on an artifact boundary
and the bracket gate has not been re-measured from the extracted counts yet.

"Max-family firing" is the highest per-family firing rate at that threshold, "median-family firing"
the median across all 48. The pair is the spread the operating point was chosen against, and it is
what says whether a J is carried by separation or by prevalence. Every VOICE and residual/span
AIRWAY row above is the max-Youden operating point from the 62,547-recording corpus sweep.

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
- `flags` — per branch, the *names* of the flag gates that fired. Keyed only when one did.
- `empty` — the emptiness precondition fired, so no branch gate was asked anything.
- `fell_through` — the recording carried content and still routed nowhere. **An empty recording is
  not a fall-through**, and the two are separate columns in the tally.
- `gate_outcomes` — every gate's own outcome, each gate evaluated once; empty when `empty`.

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
per-family counts, per branch on each of `routed`, `declared`, `agreed`, `missed`, `extra`,
`unavailable` and `flagged`, plus the empty count and the fall-through count, so the ruleset can be
scored over the corpus without holding it.

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

### Why the span's best-scoring label, and not the store's own `labels` list

PREPROCESS writes two things per span classifier window: `raw_scores`, the model's full output, and
`labels`, the subset clearing `windows.<classifier>.default_threshold`. `labels` is the natural
membership reader, and `nodes/taxonomy.py`'s `_span_label_evidence` uses exactly that.

It cannot be used here. `windows.yamnet.default_threshold` and `windows.hear.default_threshold` are
both `null` in the shipped config — no ROC over this corpus exists to fit them from — so every
per-span window in the 62,547-recording extraction carries `labelled: false` and no `labels` key at
all. Conditioning on `labels` would emit nothing for the entire corpus.

The reduction used instead is the span's **best-scoring label**: the maximum `(label, score)` pair
over every window the classifier placed on that span, which is `_per_span_label_scores`'s
per-label-max followed by `top_label`'s argmax, and is the same number either order is taken in. It
introduces no threshold, so it adds no unfitted literal to the code, and the thresholding stays
where it belongs — in the detector's own swept grid. A span whose best label is outside
`TRACKED_LABELS` carries no tracked label and contributes to nothing.

### Size

`span_label_stats` costs 430-520 bytes per emitted label: nine keys (`span_count` plus the eight
statistics of `_stats`), one label at a time. A recording emits a label only when a live span
carries it, so the count per recording is bounded by the number of live spans and, in practice,
concentrates on 2-4 distinct labels — argmax over 521 AudioSet labels is not diverse across the
spans of one recording. That is 0.9-1.8 kB against a current 15.7 kB per recording (983 MB /
62,547), so **+6% to +11% on the shard**, well short of the doubling that would have forced
emission down to `taxonomy.audioset_airway_labels` and `taxonomy.hear_airway_labels`. All of
`TRACKED_LABELS` is therefore emitted. The absolute worst case is 36 labels (29 tracked for YAMNet,
7 for HeAR) on a recording with at least 36 spans each argmaxing differently, which is a doubling
and does not occur.

Extraction now parses the `span_yamnet` records it used to skip by raw-string prefilter. Measured at
17.3 kB and 99 µs per record for a 521-label dump on this laptop, at a few tens of spans per
recording that is a few milliseconds per store and a few minutes over the corpus — the prefilter
was worth having while no detector read those bytes, and is not once one does.

### The detectors added

Three, not the cross-product. The feature is general; these are the ones about to be tested against
the two rows above:

- `cough.yamnet_cough_span_peak_over_floor_db_max`
- `cough.yamnet_cough_span_peak_over_floor_db_p75`
- `cough.yamnet_cough_span_peak_over_floor_db_p90`

Each reads `("span_label_stat", "yamnet.Cough.peak_over_floor_db_<stat>")` over
`DB_OVER_FLOOR_GRID`. `max` is the direct conditioned analogue of
`cough.amplitude_peak_over_floor_db_max`; `p75` and `p90` are there because a recording with one
loud cough and one louder door slam has the same `max` and a lower `p90`, and which of the three
separates best is a measurement, not a guess.

## Emptiness is its own check, ahead of routing

A recording that contains nothing and a recording whose content matched no gate are two different
outcomes, and `fell_through` conflated them. It no longer does.

`taxonomy.ruleset.emptiness` is a precondition, evaluated before any branch gate: a recording is
**empty** when the highest tracked-label peak of **both** `enhanced|yamnet` and `residual|yamnet`
falls below `peak_floor`. When it fires, no branch gate is asked anything, `routed` is empty,
`gate_outcomes` is empty, and `empty` is true while `fell_through` is false. A named stream whose
summary is absent from the store reads unavailable and the recording is **not** called empty — an
absent classifier summary is not a stream that scored zero.

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
streams.

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

### What the empty set does to the scores

An empty recording stays a reference positive for whatever branch its family declares, so its
`declared` branch lands in `missed` and it counts as a false negative in `score_branches`. That is
the honest reading — a branch genuinely was not recovered — but it means a corpus with many empties
reports a depressed sensitivity for a reason that has nothing to do with the gates. `FamilyTally`
counts `empty` separately so that reason is visible rather than inferred, and
`scripts/score_taxonomy_ruleset.py` prints the empty count on its own line above the fall-through
count.

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
`taxonomy.airway_bracket_tokens` beside `taxonomy.audioset_airway_labels` and
`taxonomy.hear_airway_labels`, currently `[breath, cough, throatclearing, sniff]`. `load_ruleset`
resolves the set name into the gate's feature tuple at load, so the gate carries its members and
the reader stays a plain read over `RecordingFeatures`. A gate naming a set `BRACKET_SET_PATHS`
does not have raises rather than reading an empty union.

A store that never ran consensus reads `UNAVAILABLE` on this gate rather than zero, matching the
rule everywhere else: an unwritten measurement is not a measurement of nothing.
