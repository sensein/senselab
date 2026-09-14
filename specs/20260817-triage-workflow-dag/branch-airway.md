# AIRWAY branch

What the branch answers: **is there airway content in this recording, where is it, and what kind?**

This document is the branch's own design. The frame it sits in — the four stages, the five verbs,
what a declaration is and how deviations are recorded — is
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).
Nothing here restates it; where a capability depends on it, it is named.

## The tasks this branch serves

Counts are declared families over the 62,547 recordings the ruleset was scored on. **A declared
family is a declaration, not ground truth** — it is what the protocol says the recording is, not
what a listener confirmed it to be. Every count below is a count of declarations.

| family | n |
| --- | --- |
| `respiration-and-cough-fivebreaths` | 3,576 |
| `respiration-and-cough-breath` | 1,788 |
| `respiration-and-cough-cough` | 1,788 |
| `respiration-and-cough-threequickbreaths` | 1,718 |
| `respiration-and-cough-v2-breath` | 699 |
| `respiration-and-cough-v2-threebreaths` | 699 |
| `respiration-and-cough-v2-threebreathsmouth` | 699 |
| `respiration-and-cough-v2-threebreathsnose` | 699 |
| `respiration-and-cough-v2-hardcough` | 698 |

Two task shapes, and they want different measurements. A **cough** task asks for one or a few
forced expulsive events and the clinical question is about the event itself — its abruptness, its
voicing, whether it was produced at all. A **breath** task asks for a *count* of respiratory cycles
(`fivebreaths`, `threebreaths`, `threequickbreaths`) and the clinical question is about the cycle:
how many, how regular, and — for `threebreathsmouth` versus `threebreathsnose` — through which
route. The v2 `hardcough` variant asks for effort, which no capability here measures.

### A recording routed here whose declared task is not airway

The branch runs on content, not on declaration. AIRWAY routes on gates over measured evidence, so
most recordings reaching it are not respiration tasks at all — at the shipped cut it routed 23,606
recordings where 13,017 declare an airway family
(`runs/ruleset-score-20260912/ruleset_score.json`). On such a recording the branch does exactly what
it does anywhere: it looks for airway content and marks, refines, contests or proposes it. A cough
inside a Harvard-sentence reading is a real cough and is recorded as one. **It is not a deviation
from the sentence task** — that judgement belongs to SPEECH, which holds that task's declaration,
and to whoever reads the deviations. AIRWAY's findings never carry a penalty for the recording not
matching its label.

## Capabilities

### A1 — Label an airway event (**built**)

**Question.** Does this span carry cough or breath?

**Reads.** PREPROCESS's general spans — every live `span` whose `family` is absent (`airway.py:198`)
— and PREPROCESS's per-span HeAR measurements, bucketed by the `span_id` attribute
(`airway.py:241-245`). A span overlapped by a lexical consensus word is skipped as transcribed
content (`airway.py:253`, `_is_transcribed` at `:69-82`); a span overlapped only by bracketed words
stays eligible, because `[COUGH]` and `[BREATH]` are the events being looked for.

**Computes.** The intersection of the span's HeAR labels with `airway.labels_of_interest`, which
ships `[Cough, Breathe]` (`default.yaml:136`). A span with no member of interest is left alone
(`airway.py:260-261`).

**Emits.** One `label` assertion per member, `wasDerivedFrom` the span, carrying `label`,
`hear_window_ids`, `in_certified_silence` and `merged_proposals` (`airway.py:265-277`).

**Serves.** All nine families, and every recording routed here on content.

**Parameter-free?** No, but its one parameter is a **label set**, not a threshold —
`labels_of_interest` names which HeAR classes count as airway. It is a vocabulary decision, not a
fitted cut, so the no-refits rule does not reach it.

### A2 — Corroborate a label against AudioSet (**built**)

**Question.** Does a second classifier, on its own grid, agree?

**Reads.** PREPROCESS's whole-file `yamnet_window` measurements overlapping the HeAR window the
label came from (`_windows_covering`, `airway.py:51-66`; called at `:295`). The corroboration set is
each HeAR label's mapped AudioSet node and its descendants, read from the classifier-ontology
profile rather than a hand list (`_corroboration`, `airway.py:95-125`) — which is why `Cough` is
corroborated by `Throat clearing` without either being written twice.

**Emits.** A `confirm` assertion when an overlapping YAMNet label is in the closure, a `contest`
when it is in `airway.contest_labels`, and an `abstain` carrying `colocated_windows_n` when neither
fired (`airway.py:305-338`).

**Parameter-free?** The corroboration side is — it is derived from the ontology. The contest side is
not, and is dead (A3).

### A3 — Contest a label (**gated behind null config, structurally dead**)

`airway.contest_labels` is `null` (`default.yaml:138`), so `_contest_labels` returns the empty set
(`airway.py:141`) and no YAMNet label can ever contest. `contested_n` is therefore structurally zero
and the "contested by" flag at `airway.py:323-326` is unreachable. In practice the only flag AIRWAY
can raise today is `lexical_contamination`.

**Owed ground truth.** A declared set of YAMNet labels that may contest a HeAR airway label, disjoint
from the airway evidence closure — the disjointness is already enforced (`airway.py:143-149`). It
cannot be fitted against the corpus: a contest set fitted against declared families would encode
which labels co-occur with the declaration, not which labels genuinely deny a cough.

**A parameter-free alternative exists and is preferred.** The branch contract defines `contest` as
"the span does not carry what was proposed". Under that definition AIRWAY contests a span for which
an airway label was proposed and no evidence of any kind survives within its extent — no label set,
no threshold. That formulation needs no fitted list and should replace A3 rather than waiting on
one. Its firing rate is **unmeasured** and must be counted before it ships, because
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)
part (a) enlarges the set of spans with no admissible evidence.

### A4 — Lexical contamination (**built**)

**Question.** Did the participant speak during the airway task?

**Reads.** Every live lexical word against the hull of the airway-labelled spans
(`airway.py:343-373`). Bracketed words are not lexical and do not contaminate.

**Emits.** An `interval` entity `airway_labelled_interval` and, when any lexical word intersects it,
a `flag` assertion with `reason: "lexical_contamination"` and the offending `word_ids`.

**Under the contract this is a deviation, not a flag.** A lexical word inside a breath task is a
located observation — `off_task_extent` — and whether it disqualifies the recording is someone
else's call. Migrating it is behaviour-preserving in evidence and strictly more informative in form:
today it yields one file-level flag, tomorrow one deviation per contaminating extent.

### A5 — Respiratory cycle count (**not built**)

**Question.** How many breaths are there, and does that match what the task asked for?

This is the capability the breath families actually need and the one AIRWAY most conspicuously
lacks. `fivebreaths` (3,576), `threequickbreaths` (1,718) and `v2-threebreaths` (699) each declare a
count; nothing in the branch counts anything. `labelled_n` is a count of *spans carrying a label*,
which is not a count of breaths: PREPROCESS merges adjacent proposals (`merged_proposals`,
`airway.py:262-263`), so one span can cover several cycles, and a quiet inhalation may clear no gate
at all.

**What it would read.** The span set, the per-span HeAR labels, and the energy envelope PREPROCESS
already writes. A respiratory cycle is an inhale–exhale alternation, and the two differ in the
envelope: exhalation is longer and higher-amplitude than the inhalation preceding it. Detecting the
alternation rather than the events independently is what makes a *cycle* count rather than an event
count.

**What it emits.** `propose` spans for cycles PREPROCESS did not find — the owner's standard case,
"estimate inhalation and exhalation even though the initial spans may not have generated all of
them" — each `family: "AIRWAY"`, plus a `counts` measurement entry `expected_event_count` carrying
`found` and the `declared` half copied from the declaration.

**Owed ground truth.** Whether an envelope alternation is a respiratory cycle is exactly the
judgement nobody has verified by listening. The capability can be *specified* now and cannot be
*fitted* now. It should ship as: propose the alternations, count them, record the count beside the
declared count, and assert no discrepancy.

### A6 — Nasal versus oral route (**not built, and not obviously measurable**)

`v2-threebreathsmouth` (699) and `v2-threebreathsnose` (699) differ only in route. Distinguishing
them acoustically is a real research question — nasal breathing is lower-amplitude with reduced
high-frequency energy — and senselab has the spectral machinery
(`extract_spectral_moments`, `praat_parselmouth.py:945`). **Whether it separates these two families
is unmeasured**, and measuring it against the declared families would fit the declaration. Recorded
here so the capability is not silently assumed.

## Deviations

| type | evidence |
| --- | --- |
| `off_task_extent` | a lexical word inside the airway-labelled interval (today's `lexical_contamination`, re-formed per-extent) |

`expected_event_count` is a `counts` entry, not a deviation — it has no extent. It requires A5.

AIRWAY emits no `stimulus_mismatch` (no airway task carries a stimulus text) and no `filler`.

## What exists today

| capability | status |
| --- | --- |
| A1 label | built |
| A2 corroborate | built |
| A3 contest | **gated behind null config; structurally dead** |
| A4 lexical contamination | built, as a file-level flag rather than a deviation |
| A5 cycle count | **not built** |
| A6 route | **not built** |

**The branch runs no model.** It reads PREPROCESS's `span_hear` and `yamnet_window` measurements and
decides from them (`airway.py` module docstring; `:166-176`). An earlier design had it re-run HeAR
per span; that was removed as a redundant second pass over what PREPROCESS already computed.

**There is no airway-specific span gate.** `airway.k_db`, `airway.k_db_by_task` and
`airway.k_margin_db` are retired — none appears in `default.yaml` and `airway.py` reads no `k_db`.
The prior version of this document described all three as live; it was wrong. Matching a
separately-configured airway threshold against the spans' own `k_db` attribute silently excluded
every continuity and ASR span, neither of which carries one.

**A gap span can carry the whole verdict.** `airway.py:198` selects `family is None`, and gap spans
are written with no `family` key, so background regions are full members of this branch's evidence
set. Since `labelled_n` is what separates `pass` from `fail` (`airway.py:375-382`), a gap span whose
HeAR windows name `Cough` can decide an airway verdict alone. How often that happens is
**unmeasured**. The branch-contract spec's part (b) proposes typing gaps as background, which would
remove them from this selector; [`branch-quality.md`](branch-quality.md) says what then concludes on
them.

## What the branch emits

```
spans        none today; A5 would propose family: "AIRWAY" cycle spans
assertions   label, confirm | contest | abstain, flag(lexical_contamination)
interval     airway_labelled_interval, the hull of the labelled spans
counts       none today; A5 would add expected_event_count {found, declared}
verdict      { labelled_n, by_label, contested_n, merged_n, flags }
```

**The verdict's basis, exactly** (`airway.py:375-382`):

- `FAIL` when no span was proposed at all, or when spans exist and none carries a label of interest.
- `FLAG` when any flag accumulated — today only `lexical_contamination`, since the contest path is
  dead.
- `PASS` otherwise.

The verdict detail is `{labelled_n, by_label, contested_n, merged_n, flags}` (`airway.py:392-398`).
The prior version of this document listed `near_gate_n` and `k_db` in that block; neither is
written.

## Out of scope

Severity, effort, any conclusion about a kind that is not airway, and any refit of `spans.k_db` —
the corpus is scored against declarations, so fitting the gate against it fits the declaration.
