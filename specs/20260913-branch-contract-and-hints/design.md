# The branch contract, and the declaration that informs it — 2026-09-13

## What this is

Four stages, one contract for all four branches, and a declaration read from the BIDS sidecars
rather than guessed from a filename.

Three things forced this design at once. The branches do not share a job description: SPEECH
proposes its own spans, VOICE reads spans nothing writes, AIRWAY reads the general set including
the gaps. Hints are extractable and inert — the packaged map is `null` and the only real one raises
on load. And the spans everything reads are permissive by construction, which is correct for a
recall-first router and wrong for anything that has to say what happened.

This spec settles what a branch is *for*. It does not build one.

---

## The pipeline shape

```
PREPROCESS   Shared derivatives, extend work included. Content-first, hint-blind.
             Emits a clean span set: defensible, not exhaustive.

SCREEN       TAXONOMY ⊕ ROUTING, merged. Consolidates the classifier evidence, resolves
             the declaration, applies the ruleset and the hints, emits the routing decision.

BRANCHES     Receive every upstream output, their route, and the declaration. Annotate
             PREPROCESS's spans (label, contest, refine, trim) and propose the ones it
             missed as `family: "<branch>"` spans. Emit typed deviations. Conclude.

VERDICT      Admit/reject, flag for human review, durable description.
```

### The nodes this does not place

`GRAPH_ORDER` has ten entries (`vocabulary.py:14-25`) and **contains neither REPORT nor FIGURE** —
`REPORT_NODE` is defined at `run.py:44` and concatenated onto the outcome list at `:453`; FIGURE's
name exists only at `figure.py:44`. Of the non-branch nodes it does contain: ADMIT runs before
PREPROCESS; **QUALITY is not terminal** — REDACT runs after it (`run.py:311-317`); and **REDACT is
its own `GRAPH_ORDER` entry gated on SPEECH's result**, not a step of SPEECH. REPORT and FIGURE
re-read a finished store. All keep their current positions.

**QUALITY is the sharp case, and it decides the scope of the verbs.** QUALITY writes `contest`
assertions over PREPROCESS's clip spans (`quality.py`, the clip-consistency check) — one of the
contract's own verbs, performed by a node that is not a branch. So **the verbs are store-wide, not branch-only**:
any node may label, contest, refine, trim or propose, and the contract below says how, regardless of
who performs it. What is branch-specific is the *family* the findings are written under and the
question the node concludes on.

### The merge reverses a decision made the same day

`specs/20260912-ruleset-in-pipeline/design.md:367-370` — landed hours before this spec — kept
TAXONOMY and ROUTING as two nodes, on the argument that TAXONOMY emits measurements and ROUTING
emits decisions, and that the L1-measures/L2-decides boundary is worth a node boundary.

The owner reversed it: **the middle stage's product is the routing decision.** The consolidation
exists to serve it. A node whose output nothing acts on is a seam, not a layer.

That premise was checked against the tree and holds more strongly than earlier drafts of this spec
claimed. **`consensus_taxonomy` has exactly one production consumer: routing's own reduction**
(`features.py:755`). The only other reader is `extend.py:330 rewrite_consensus_taxonomy`, which
recomputes it rather than consuming it. `figure.py:603-617` reads `<classifier>_label_summary` — a
*different* TAXONOMY measurement — and never the consensus. `report.py:425,484-487` reads
PREPROCESS's `<classifier>_windows` and neither the consensus nor the summaries.

**One argument previously made here is withdrawn as unsound.** That `evaluate_live_routes` serialises
the whole store (`live_evidence.py:124-141`, called from `:162-168`) proves too much — VERDICT and
REPORT read the whole store too, and nobody proposes merging them — and it rests on stage 1
scaffolding already scheduled for replacement by `reduce_records` over `required_sources(ruleset)`.
The merge stands on one product, one stage, and the single verified consumer.

### What the merge costs, and what must be budgeted

**Two node names retire, and node name is a join key.** `GRAPH_ORDER` contains `"TAXONOMY"` and
`"routing"` (`vocabulary.py:14-25`); verdict entities, `run.json`, the figure and the report all join
on it, and every finished store carries activities with `node: "TAXONOMY"`.

**`"routing"` is load-bearing beyond `GRAPH_ORDER`.** `vocabulary.py:113` defines
`_ROUTING = "routing"`, read by the flag grounds at `:373` and `:384`, and `verdict.py:36,202` joins
on it. Dropping it from the vocabulary stops the "routing failed" FLAG firing on pre-merge stores,
silently — the same failure shape as the demoted consolidation flag above.

Stage 2's own audit states the governing rule — **the writer's vocabulary may shrink, the reader's
may not** — and this is the same shape as the `kind` prov-type fix of 2026-09-13. So: SCREEN is the
only name written; `"TAXONOMY"` and `"routing"` remain readable, and a reader encountering either
folds it as SCREEN's predecessor rather than raising. A regression test parametrised over the
readable set, as `prov_store_test.py::test_every_readable_entity_type_round_trips` is for prov types.

**An ordering constraint demotes back to intra-node.** Stage 2 promoted the
`voice.glide` / `voice.chant` → `yamnet_label_summary` dependency from step ordering inside TAXONOMY
to an edge in `GRAPH_ORDER` (`taxonomy.md:60-63`, ruleset `design.md:372-375`). Merging demotes it
again. **The intra-node ordering test must return with the merge** — without it VOICE silently stops
routing, which has already happened once.

**Two verdicts become one, and it must fold both conclusions.** TAXONOMY and ROUTING each write one;
`store.md` requires a node's verdict be attributed to its last step. An earlier revision demoted the
consolidation's conclusion to a `detail` field — **that deletes a live FLAG ground.**
`taxonomy.py:360` writes `Outcome.FLAG`, with the string at `:361`: "no per-span classifier produced scores; there was
nothing to consolidate"; `vocabulary.py:423` folds on `outcome` and reads no `detail`; and
`routing.py:263` writes `PASS` unconditionally. Demoting it would silently stop that flag firing.

**SCREEN's verdict flags if either conclusion flags**, and is attributed to the routing step as the
stage's last.

**Note the compounding with rule (a).** Excluding covering-window scores makes "no per-span
classifier produced scores" *more* common, so rule (a) grows the population this flag fires on. Both
pieces must name the interaction.

**Four documents assert the split and must be updated**, not only the one cited above:
`taxonomy.md:27-29` ("a node that both measures content and decides what runs on it cannot be checked
against itself"), `taxonomy.md:109-110`, `routing.md:16`, `dag.md:751-753`.

---

## Span cleanliness — three parts

A span shorter than a classifier's window cannot be classified by its own content. That is the
defect, and it is not a threshold problem: a 200 ms cough is a real event, so proposing it is right.
What is wrong is the label attached to it afterwards.

### (a) A label belongs to a span only if a window lies inside it

**This applies to YAMNet's per-span measurements and not to HeAR's, and the difference is
load-bearing.**

`_span_yamnet` gives a span shorter than the native window the overlap-weighted scores of the
whole-file windows that *cover* it, marked `attribution: "covering_windows"` with
`covering_windows_n` and `covering_seconds` (`preprocess.py:1981-1985`); a long span gets
`attribution: "native"` (`preprocess.py:2026`). A covering-window label is a statement about up to a
second of audio attributed to a fifth of it.

`_span_hear` does something different. `span_hear_input` places a span shorter than
`HEAR_WINDOW_SECONDS` in a silent 2 s buffer, "so its only detector result describes the span
itself" (`hear.py:426-443`); a longer span is passed through (`hear.py:444-446`) and its native
windows are mapped back by `hear_window_extent`. HeAR's short-span label is about the span.

**The rule: a covering-window label is recorded, and is not eligible as evidence.** The flag already
exists and is already written; nothing filters on it today.

**Which readers the rule applies to**, stated exactly because this is not a cosmetic change:

| reader | what changes |
| --- | --- |
| `consensus_taxonomy` rows (`taxonomy.py:192`) | a covering-window score does not contribute a row or a `peak_by_classifier` entry |
| `RecordingFeatures` span-label stats (`features.py:1137 _label_span_statistics`, via `_absorb_span_window`) | covering-window scores excluded from `span_label_stats` and `span_label_set_stats` |
| figure, report | rendered as attributed-from-outside, not as the span's own label |

**AIRWAY is not on that list, and an earlier revision wrongly put it there.** AIRWAY reads
**`span_hear`** (`airway.py:242`, and its docstring says so at `:171`), and `attribution` is written
only by `_span_yamnet`. With `_windows_covering` deferred below, **AIRWAY's own reads are untouched
by rule (a)** — but `airway.cough` is a *routing gate*, so rule (a) still changes whether AIRWAY runs
at all. The two senses must not be conflated: the rule cannot alter what AIRWAY sees, and can alter
whether it is asked to look.

**So rule (a) either moves routing or does nothing.** `airway.cough`'s feature is
`span_label_set_stat` over `yamnet.cough_labels.peak_over_floor_db_max` (`default.yaml:245-248`),
built by `_label_span_statistics` from exactly these per-span measurements with no attribution check,
and coughs are usually shorter than 0.96 s. There is no scoping that keeps the rule's benefit without
changing the router: branch evidence is empty, and `consensus_taxonomy`'s only production consumer is
`features.py:755` — routing's own reduction. An earlier revision offered "scope away from
`RecordingFeatures`, apply to the consensus and to branch evidence" as a fallback; that fallback is
vacuous and is withdrawn.

Worse, the gate becomes **structurally unable to fire on the class it names**: rule (a) makes
covering-window YAMNet labels ineligible, `yamnet.py:257-258` makes a native window impossible under
0.96 s, and coughs are usually shorter than that.

**So the corpus count must compare candidates, not measure one change.** A single before/after has
only one legal reading and decides nothing. The three candidates:

1. **Rule (a) as written**, and `airway.cough` accepted as near-silent.
2. **Rule (a) not shipped**, and covering-window labels kept as routing evidence.
3. **Rule (a) shipped and `airway.cough` re-sourced onto HeAR evidence** — AIRWAY already reads
   `span_hear`, and HeAR classifies a short span on its own content (`hear.py:426-443`), so the
   gate's subject is available from a classifier that rule (a) does not touch.

Candidate 3 is the one that keeps both the rule and the gate, and it is the reason the count is a
comparison. **This is the decision procedure for whether rule (a) ships at all**, not a precaution
attached to shipping it.

No PREPROCESS re-run is needed; existing stores need `rewrite_consensus_taxonomy` re-run to pick up
the consensus change.

**The count itself needs a fix first.** `features.py:1079-1091` appends every live span to
`live_spans` with no `family` filter, and `_span_statistics`'s `all.*` bucket includes them. That is
harmless in-run, because routing precedes the branches and no family span exists yet — but **not in
an offline recompute over finished stores** (`scripts/analyze_routing_evidence.py:158`), which is how
a gate count would be produced. Branch-proposed spans would enter the router's own statistics. The
family filter lands before the count does.

**The remedy is incomplete, and the gap is named rather than closed.** AIRWAY has a *second*
covering-window mechanism the flag does not touch: `_windows_covering(store, "yamnet", hear_extent)`
(helper `airway.py:51-66`, called at `:295`) takes every **whole-file** `yamnet_window` measurement
merely overlapping a HeAR window and lets its labels confirm or contest. Those are whole-file windows
and carry no `attribution` key at all — the flag is a property of `span_yamnet`, which these are not.
**Rule (a) is therefore scoped to `span_yamnet` measurements.** AIRWAY's whole-file corroboration is
a separate mechanism with the same underlying weakness, and closing it is deferred to the AIRWAY
piece of the branch contract, where the confirm/contest logic is being rewritten anyway.

HeAR's isolation has its own caveat, and it is a different one: a 200 ms event centred in 1.8 s of
digital silence is not what the model saw in training. Whether that distorts its scores is
**unmeasured**. It is not addressed here and must not be conflated with the covering-window problem.

### (b) Gap spans are background, never events

Gap spans are the complement of the **kept** spans — `covered` is built from `combined`, the four
sources' surviving proposals (`preprocess.py:1572`) — not the complement of everything proposed. A
gap shorter than `min_duration_ms` is not emitted at all. They carry `measure: "gap"` and
`merged_proposals: 0` (`preprocess.py:1586-1591`) and, being written with no `family`
(`preprocess.py:1583-1592`), they are selected by AIRWAY's `family is None` filter (`airway.py:198`)
as ordinary evidence.

Type them as background. They stay measured and visible; they stop being eligible to be something
that happened.

**One interaction to resolve later.** Once branches propose spans, a branch-proposed event can sit
inside an extent typed background. The smallest defensible rule: a background span is a statement
about what PREPROCESS found, not a claim of emptiness, so a branch-proposed span inside one neither
retires nor contradicts it — the background typing is superseded for that extent and the reconciling
record is the branch's own span. Whether background spans should instead be split around
branch-proposed events is **unresolved**.

### (c) Boundaries reconciled, not first-writer-wins

`_novel` (`preprocess.py:1467-1492`) appends a record to `corroborated_by` on every span a later
candidate overlaps, and keeps the earlier proposer's extent unchanged. Four sources agreeing on an
event is precisely when its boundary can be stated well, and that is the moment the current code
discards the information.

**This piece has no rule yet, and that is the blocker.** Union, intersection, proposer-weighted
median and highest-peak-corroborator are all defensible, and the ground-truth rule below forbids
choosing between them by measurement on this corpus. A reconciliation rule must therefore be a
**parameter-free definition** justified by what it means, not by what it scores — or the piece waits
until something has been listened to. **Unresolved**, deliberately, and it is why (c) is sequenced
last.

The one part of (c) that is not blocked is a plain bug: continuity spans are `wasDerivedFrom` the
energy envelope rather than the continuity trace (`preprocess.py:1533` — `state["envelope_id"]`),
even though the trace is in the activity's `reads` (`:1440-1441`). The provenance edge names the
wrong source. Small, independent, and split out of (c) below.

### Refitting `spans.k_db` was considered and rejected

`spans.k_db: 6.0` (`default.yaml:40`) is documented in `config-derivations.md:106-120` as
"provisional and expected to be refit — a deliberately permissive placeholder". Refitting it against
the corpus is the obvious fourth part of this section, and it is not being done.

**The corpus is scored against declared families — the BIDS `task-` id — which is a declaration and
not ground truth.** A threshold fitted against it would fit the declaration. Nothing in this corpus
has been validated by listening. No threshold in the triage graph gets refit until something has.

This applies to every fitted value, not only `k_db`.

---

## The declaration

### Where it comes from

Two JSON sidecars per recording, at different grains, read off the corpus at
`/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/`.

**`<stem>_recording-metadata.json`** — per recording:

```
language, instructions, record_id, recording_id, recording_acoustic_task_id, session_id,
recording_duration, recording_size, recording_profile_name, recording_profile_version,
recording_input_gain, recording_microphone, task_name, stimulus_text, speech_type,
stimulus_source, audio_channel_count, audio_sample_rate
```

A real one: `task_name: "harvard-sentences-list-34-6"`, `stimulus_text: "Try to trace the fine lines
of the painting."`, `speech_type: "read"`, `recording_duration: "6.1"`.

**`<stem>_task-<task>_acoustictask-metadata.json`** — per acoustic task:

```
language, instructions, record_id, acoustic_task_id, session_id, acoustic_task_name,
acoustic_task_cohort, acoustic_task_status, acoustic_task_duration, stimulus_text,
speech_type, audio_channel_count, audio_sample_rate
```

A real one: `acoustic_task_name: "respiration-and-cough-(v2)"`, `acoustic_task_duration: "151.0"`,
`speech_type: "non-lexical"`, and `instructions` carrying the protocol's own words — *"tap the
record button and cough HARD as if something were stuck in your throat."*

### Three things are broken today

**The hint is parsed from the filename while the sidecar goes unread.**
`runs/b2ai-v2/make_hints.py` extracts the BIDS `task-` token from the stem and emits `may_contain`
plus `metadata.{task_token, speech_type, task_id, registry}`. Every recording carries the protocol's
own `task_name` and `speech_type` beside it. `stimulus_text` — which `AudioHints.expected_speech`
exists to hold — is never carried at all.

**The only populated map raises on load.** `runs/b2ai-v2/override.yaml:261` is keyed
`routing.hint_kind_map`. Stage 2 renamed the packaged key to `hint_branch_map`
(`default.yaml:127`), and `_merge` refuses any key the packaged config lacks (`config.py:147-148`).

**Its values would all be rejected even after renaming.** `_map_tags` casefolds the map's *keys*
against the declared tags (`routing.py:102`) but tests the *value* against `BRANCHES` unchanged
(`:107-108`). The override's values are lowercase kinds — `cough: airway` — and `BRANCHES` is
`("AIRWAY", "SPEECH", "VOICE", "DDK")`.

The mechanism matters for the fix. `bad_map_values` is keyed **by declared tag on the recording**,
not by map entry (`routing.py:106-113`, recorded at `:234`), so a map entry no file declares never
surfaces. The conclusion still holds — checked against every rule in `make_hints.py:136-182`, every
corpus recording declares at least one tag the override maps — so **every file in the run** would
take an `Outcome.FLAG` against ROUTING (`vocabulary.py:382-384`). This is a **value-case fix**;
`routing.md:76-78` documents testing the value unchanged as deliberate, and that test is not the
defect.

**A fourth, found by measurement rather than by reading: the null map makes the file verdict assert
the opposite of the declaration.** Because every tag is unmapped, no decision carries `hint_tags`, so
`_hint_claims` returns `{}` rather than `None` (`nodes/verdict.py:182-184`) and `FileVerdict.hints`
reads `found_unclaimed` / `no_claim` for every branch — including `AIRWAY: found_unclaimed` on runs
whose `may_contain` declared `[cough, airway]`. It renders: `summary.json`'s
`recording.declared_hints` (`report.py:1097`) and the PDF header (`report.py:1426-1427`, `:1467`).
The `UNREAD_DECLARATION` flag built for this case fires only when no decision survived
(`vocabulary.py:385-386`). Recorded, with the vocabulary decision it is blocked on, at
[`../20260817-triage-workflow-dag/verdict.md`](../20260817-triage-workflow-dag/verdict.md) and
measured in
[`../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md`](../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md)
§ B. **Owed a code change.** The first three items above are what a hint-blind run costs; this one is
what it *says*, and it is worse, because a silent inertness misleads nobody.

### What SCREEN resolves

One `declaration` measurement per recording, carrying task identity, expected content and structure,
the stimulus text where one exists, and the acquisition facts.

**Two table grains, because one key cannot carry both.** `acoustic_task_name` has ~28 values;
`task_name` is per recording and has hundreds (`harvard-sentences-list-34-6`). Expected *kind* of
content is knowable at the acoustic-task grain; a specific stimulus, and counts like "five breaths",
are knowable only at the recording grain. So:

- `data/task_expectations/<version>.yaml`, two top-level maps: `by_acoustic_task` keyed by
  `acoustic_task_name`, and `by_task` keyed by `task_name`, the latter taking precedence where both
  match. Each entry quotes the sidecar's `instructions` verbatim as its derivation.
- **An entry's keys are**: `expected_content` (which branch's subject the task asks for),
  `expected_event_class` and `expected_event_count` (what `expected_event_count`'s `declared` half
  reads, absent for tasks with no countable event — **since 2026-09-21 the field is two,
  `required_count` and `typical_count`, and lives on the `Expectation` row rather than in a YAML
  map; see `specs/20260921-required-and-typical-counts/design.md`**), `targeted_speaker_count`
  (what `speaker_count`'s `declared` half reads, and the "single target" claim a branch conditions
  on), and `instructions`.
  Nothing else in this spec reads a field of an entry.
- **Versioned by filename**, as the detector profile is, and named in the `declaration` measurement
  so a run says which table it read.
- **Absent key is VERDICT-visible**: the declaration records `expectations: null` with the key it
  looked for. A recording whose task has no entry gets no expectations and no deviations, and that
  absence is reported rather than read as conformance.
- **Two tests**: that each entry's quoted instruction is byte-identical to the sidecar's, and that
  every `acoustic_task_name` present in the corpus has an entry.

`instructions` is prose — *"repeating the syllables 'puhtuhkuh' as quickly and consistently as
possible"* — and must not be parsed. The table is read from it by a human and checkable against it.

**Where the declaration lives in the data structures.** `AudioHints` has a typed home for the
stimulus (`expected_speech` / `ExpectedSpeech`) and none for `task_name`, `acoustic_task_name`,
`speech_type`, declared duration, microphone, channel count or sample rate. The smallest defensible
choice: **the stimulus goes into `expected_speech`, and everything else goes into `metadata` under a
contract written in this spec** — keys `task_name`, `acoustic_task_name`, `speech_type`,
`declared_duration_s`, `microphone`, `channels`, `sample_rate` — rather than adding typed fields to
a data structure outside the triage module, which would need its own consumers and tests. Promoting
them to typed fields is **unresolved** and deferred.

**The contract and the live readers name different keys — owed a code change, and the contract is the
side to change.** Checked against the tree 2026-09-15: `nodes/voice.py` reads
`hint.metadata["population"]` (`voice.py:66`) and `hint.metadata["task"]` (`voice.py:146`), and
**neither key is in the seven above**, so a hint written to this contract reaches neither consumer. Of
the three `metadata` keys anything in `src/senselab` reads, only ROUTING's `speech_type`
(`routing.py:42`, read at `:80`) is covered. The contract is the side to move: `task_name` and
`acoustic_task_name` are the sidecar's own names and separate the per-recording grain from the
per-family one, which is the distinction a single `task` cannot carry, and `population` is not a field
the sidecars hold at all — so deciding what `voice.py:146` should read is deciding which grain a
declared duration range is keyed at, which is this spec's question rather than the branch's. V7's and
V2's reading of the same mismatch from the branch's side is in
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
*Unresolved*. Both paths are inert today — `voice.f0_range_by_population` and
`voice.task_duration_ranges` are both null (`default.yaml:183`, `:185`) — so this costs nothing until
either is populated, which is the moment it becomes silent.

**When the two grains disagree** — both carry `speech_type` and `stimulus_text` — the recording-grain
file wins, and the disagreement is recorded on the declaration as a field rather than resolved
silently.

**Measured, across all 112 recordings of the local corpus copy, 2026-09-15.** The grains disagree on
exactly the field this spec exists to carry, and agree on the ones it could have guessed:

- `speech_type` and `language`: **never disagree** between the two sidecars.
- `stimulus_text`: **disagrees on 44 of 112** — *empty* in the acoustictask JSON for Harvard, Cape-V,
  Productive-Vocabulary and Stroop, *family-generic* for Free-speech, while the recording JSON carries
  the actual prompt.
- the corpus's `speech_type` distribution: `non-lexical` 59, `read` 30, `elicited` 19, `recall` 4 —
  the same four values
  [`../20260817-triage-workflow-dag/benchmarks/open.md`](../20260817-triage-workflow-dag/benchmarks/open.md)
  § *Hints* enumerates.
- `recording_profile_name`: `Speech` 87, `Breathe` 20, `Cough` 5.

Two consequences for whatever builds the declaration. The `*_acoustictask-metadata.json` is **per task
family, not per recording** — one file serves all 11 breath/cough wavs of a session — so it alone
cannot build a per-recording hint, and the precedence rule above is not a tie-break for rare cases but
the only path to the prompt on 39% of the corpus. And a builder that reads the family sidecar only
drops `expected_speech` precisely where the prompt exists.

**When no hint is supplied**, SCREEN writes the `declaration` measurement anyway, carrying
`expectations: null` and naming why. An absent declaration and an unread one must stay
distinguishable, which is the same rule the graph already applies to absent measurements.

### A ruleset that fires may write or refine a span's label — owner decision, 2026-09-15

The owner: *"rulesets could update span labels if relevant."*

Today the ruleset decides routing and nothing else. This widens it: **when a rule fires, it may stamp
or refine a label on the span whose evidence fired it** — `label` where the span carried none,
`refine` where it carried one the rule sharpens, under the same verb table as any other annotation.
SCREEN's product becomes the routing decision *and* the labels its own evidence supports.

**It removes VOICE's span-source problem rather than relocating it.** The phonation label is applied
at SCREEN time by the rule that fired, so VOICE receives a *labelled* subject and no retired detector
has to be resurrected. **This changes the plan
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
currently implies**, and it retires this spec's own claim below — that the branch becoming the
proposer "is the whole fix". **The owner has since settled the source**: VOICE's subject is the spans
the ruleset labelled and VOICE refines them, at that document's § *The subject is the spans the
ruleset labelled, and VOICE refines them*. What stays open there is the family a label on an
`amplitude` span puts it in, and the two prerequisites below.

**A ruleset-written label is an assertion about a span and needs the same provenance as any other** —
which rule wrote it, from which evidence, at what value. Without those three it is indistinguishable
from a classifier's own label, which is precisely the distinction the store exists to keep.

**Two prerequisites, both owed a code change, because the evaluation carries neither today.**

- **The value is not recorded at all.** `evaluate_gate` computes `value = gate_value(features, gate)`
  (`routing_analysis/ruleset.py:405`) and returns only a three-member enum (`:406-409`);
  `RouteEvaluation.gate_outcomes` is `Mapping[str, GateOutcome]` (`:210`, filled at `:467`) and
  `route_attributes` serialises exactly that and no number (`live_evidence.py:172-190`, the key at
  `:185`). So a rule-written label could not today cite the number that produced it. Registered
  against the ruleset at [`../20260817-triage-workflow-dag/routing.md`](../20260817-triage-workflow-dag/routing.md)
  § *Open derivations*.
- **The span identity is discarded one step before the gate sees it.** `live_spans` rows carry
  `"id"` (`routing_analysis/features.py:1084`) and the reduction
  `span_longest_s[measure] = max(durations)` (`:1134`) keeps only the scalar. A `Gate` is "one
  threshold rule over one number" (`ruleset.py:92-106`) with no span in it, so **"the span whose
  evidence fired it" is not addressable from a fired gate.** Which span a reduction attributes to is
  a contract question rather than a threshold: `max` has a unique argument only until two spans tie.

**Which evidence the phonation label should come from, on the measurement we have.** A **directed
recommendation with its evidence, not a settled rule.** Drive the phonation label from the **measured
gate**, not from the classifier label. On the 20 s held vowel of the 2026-09-15 run the consensus
taxonomy read `Chant` 0.937, `Music` 0.930, `Mantra` 0.899 and `Brass instrument` 0.661 — all
outranking anything voice-specific — while `voice.sustained` measured 15.89 s off the amplitude
envelope and was right (benchmark § G and § D). A label sourced from the consensus would name that
vowel music. **One recording is one recording**: this is the direction the evidence points, not a
fitted rule.

### Two standing rules

**A declaration may add and inform. It may never suppress.** Routing is the union:
`will_run = by_ruleset or forced_by_declaration`. The declaration is the recording's own task,
read off its BIDS stem and resolved through `taxonomy.ruleset.reference_family_set`, with a hint tag
as an optional second source; see [`../20260817-triage-workflow-dag/routing.md`](../20260817-triage-workflow-dag/routing.md).
A branch the content routed cannot be un-routed by a declaration that disagrees with it.

**Content the task did not ask for is content, not error.** A breathing recording that carries speech
routes SPEECH on the ruleset's evidence and has that speech recorded.

---

## The branch contract

> A branch receives the shared derivative state, its route, and the declaration. It **labels** what it
> recognises, **contests** what was proposed but is not there, **refines** a boundary the proposer got
> wrong, and **trims** a span to the extent that serves the task — all four as assertions about spans
> that keep their own identity. It **proposes** what the proposer missed but the declaration says to
> expect, and that alone mints a `family: "<branch>"` span. It emits typed deviations and concludes on
> its own question.

The owner's example is the specification: *a breathing task routing through AIRWAY will try to
estimate inhalation and exhalation even though the initial spans may not have generated all of them;
and a non-airway task routing through AIRWAY will do its best to find breathing or other airway
information and mark, refine or refute it.*

Quoted as given; "mark" and "refute" are the store's `label` and `contest`, for the reason below.

The consequence worth stating plainly: **PREPROCESS's span set is clean but not complete.** Clean is
PREPROCESS's job — the three parts above. Complete is the branch's, and only the branch can do it,
because only the branch knows what it is looking for.

### A branch refines and reviews, and a correct subject is not a precondition — owner decision, 2026-09-15

The owner: *"voice would still need to improve/update/review phonation spans and other tasks that are
assigned to it."*

**A branch does not wait for a correct subject to be handed to it.** It receives spans and *improves*
them, which in the five verbs is **`refine`** — assert a corrected extent, corrected metadata, or
both on an existing span, which keeps its id, its `family` and its measurements. Reviewing a span is
`label`, `contest` or `refine`. **Only `propose` mints**, and it mints only what nothing proposed at
all. (Until the owner decision of 2026-09-15, `refine` was extent-only and this sentence read
"assert a tightened extent on an existing span"; § *`refine` covers metadata as well as extent*
records the widening.)

**So `FAIL` for "no span of my family" is the wrong shape of answer.** Not because the recording must
hold the branch's subject — it may not — but because a branch that returns before reading the spans it
was given has reviewed nothing. Where the branch's own family is absent, the answer is the annotations
it made on what was there.

**VOICE is where this is visible, and the inconsistency is worth stating plainly.** **VOICE is
routed by a measurement on `amplitude` spans and then fails for want of a `phonation` span.**
`voice.sustained`'s feature is `[span_longest, amplitude]` — the longest live amplitude span in
seconds, cut 3.0 s (`default.yaml:252-255`) — while the selector at `voice.py:230` admits only
`_PHONATION_FAMILY` (`:40`). The amplitude spans the route was decided on are live in the store,
unexamined, when the branch takes the no-span return at `voice.py:235-264`. Measured on 13 real b2ai
recordings, **6 of 6 VOICE-routed recordings returned `Outcome.FAIL`** there, including a
maximum-phonation-time recording whose `voice.sustained` read **15.89 s** and a glide at **12.27 s**
([`../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md`](../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md)
§ D).

**And the second half of the owner's sentence carries as much as the first.** A branch must also do
its best on *other tasks assigned to it* — a recording routed to it whose declared family is none of
its own. That is the owner's standing rule for branches, already stated for AIRWAY in the owner's
example quoted at the head of this section, and the decision generalises it: **a branch receiving a
task outside its declared families does its best to find, mark, refine or refute evidence of its own
kind, rather than failing for want of a declared subject.** For VOICE that is the common case rather
than the edge case — it routed 22,277 recordings against 8,306 declaring a voice family.

**This is also what `voice.py:333-346` does wrongly today.** It mints a *second* span from an input
span, re-keyed by period-aligned onset and carrying `onset_kind` — the re-minting the annotating verbs
replace with a `refine` assertion, and the reason `_spans_of_family` (`report.py:289-306`) must split
one family into two populations on `("onset_kind" in attributes)`. Once the minting becomes a
`refine`, that split has nothing to separate.
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
§ *The state of this branch* already states that; what the decision adds is that improving those spans
**is VOICE's work**, not a precondition it is entitled to wait for.

### What each verb writes — four annotate, one mints

**This reverses the previous revision of this spec**, which had `refine` and `trim` mint new span
entities and invalidate or shadow the original. That was wrong, and the reason is that the store has
**two** addressing mechanisms and re-minting breaks both.

**Per-span measurements are addressed by the `span_id` attribute, not by traversal.**
`airway.py:242-245` builds `span_hear_by_span` from `window.attributes.get("span_id")` and looks up
`span_hear_by_span.get(span.id, [])` at `:256`; `features.py:350` and `:886-926`, `taxonomy.py:103-122`
and `figure.py:1834-1835` all key the same way. **A re-minted span carries zero labels under all
four, silently**, while `features.py:1074` drops the invalidated original. So a `refine` would
*delete* the span's classifier evidence from `span_label_set_stats` — including the feature
`airway.cough` is built on. "Readers traverse `wasDerivedFrom`" is not a contract that can be
adopted: it is four rewrites plus a reverse index `ProvStore` does not expose, since `derived_from`
resolves one way only.

**And carrying a measurement forward is wrong on its own terms.** `attribution: "native"`
(`preprocess.py:2026`) and `isolated_span: True` (`:344`) are claims about how the **old** extent was
fed to a model. After a narrowing refine, a traversing reader would get a label flagged `native` for
an extent containing no native window — rule (a)'s premise inverted by rule (a)'s own spec.

**So four of the five verbs annotate. They never re-mint a span. The span keeps its id, its
`family`, its measurements and its liveness; nothing is invalidated and nothing is orphaned.**

**The contract adopts the store's existing spellings rather than inventing new ones.** An earlier
revision renamed `label`→`mark` and `contest`→`refute`; that is withdrawn, and the rename is not
merely unnecessary but unsafe. `verb: "label"` is written by **SPEECH as well as AIRWAY** —
`speech.py:961` writes `{"verb": "label", "label": "pii", ...}` and `redact.py:237` selects on
exactly that pair to decide what gets redacted. Renaming `label` stops PII redaction firing,
silently. The real readers of `verb` are `redact.py:237`, `report.py:196`, `report.py:327` and
`extend.py:542`; it appears in neither `verdict.py` nor `figure.py`.

| verb | `verb:` value | writes | carries | mints |
| --- | --- | --- | --- | --- |
| label | `"label"` | `assertion`, `wasDerivedFrom` the span | what the span carries | no |
| contest | `"contest"` | `assertion`, `wasDerivedFrom` the span | that it does not carry what was proposed | no |
| refine | `"refine"` | `assertion`, `wasDerivedFrom` the span | `corrected_extent: [start, end]` and/or `corrected_attributes: {key: value}`, at least one | no |
| trim | `"trim"` | `assertion`, `wasDerivedFrom` the span | `task_extent: [start, end]`, plus the `off_task_extent` finding | no |
| propose | — | `span`, `family: "<branch>"`, `wasDerivedFrom` its evidence | a region PREPROCESS did not find | **yes** |

`label` and `contest` already exist and keep their meanings. **`refine`, `trim` and `propose` are
written by nothing in the graph** — swept over `nodes/` and `extend.py` on 2026-09-15, the full
inventory of `verb:` values any node writes is:

| verb written | by | site |
| --- | --- | --- |
| `label` | AIRWAY, SPEECH | `airway.py:266`; `speech.py:961`, carrying `label: "pii"` |
| `confirm` | AIRWAY | the loop pair at `airway.py:303`, written at `:310` |
| `contest` | AIRWAY, QUALITY | the same loop pair at `airway.py:303`, written at `:310`; `quality.py:291` via `CONTEST_VERB` (`:62`) |
| `abstain` | AIRWAY | `airway.py:332` |
| `flag` | AIRWAY | `airway.py:368`, `reason: lexical_contamination` |
| `measure` | PREPROCESS | `preprocess.py:1835`, `:1842`, `:1860` |
| `withdraw` | PREPROCESS | `preprocess.py:599` via `WITHDRAW_VERB` (`:136`) |
| `attribute` | SPEECH | `speech.py:793`, one per word |
| `refine`, `trim`, `propose` | — | **nothing writes them** |

So three of the contract's five verbs are unwritten, and six verbs are written that the table does
not carry. Of those six, `abstain` and `flag` are sanctioned below as keeping their meanings outside
the table, and `confirm` has a settled migration (§ *Where each branch already stands* — `confirm`
becomes a `label` carrying its corroborating window ids). **`attribute`, `measure` and `withdraw`
have no such standing.** `attribute` acquires one in § *`refine` covers metadata as well as extent*
below; `measure` and `withdraw` are PREPROCESS's and are untouched by these decisions.

**`abstain` and `flag` also keep their current meanings** and are not folded into this table:
`abstain` records that colocated evidence existed and decided nothing, and `flag` is a branch-level
finding rather than a span-level one.

**The corrected extent goes in a named attribute, never in `assertion.extent`.** Every existing
assertion sets `extent=span.extent` (`airway.py:272`, `:308`, `:330`) and `report.py:1144`'s
`_timing` relies on that convention, so a `refine` that moved `assertion.extent` would be read as
timing the span itself. Hence `corrected_extent` and `task_extent` as attributes, with
`assertion.extent` continuing to name the span being annotated.

**The precedent for reading these already exists.** `figure.py:577-585` consumes SQUIM *assertions*
by walking `store.derived_from(entity.id)` — assertion → span, the one direction the store resolves.
That is exactly the traversal the annotating verbs need, already in production.

`refine` and `trim` still emit their deviations; they simply do not rewrite the store's spans. A
corrected extent is a claim beside the original — attributable, reversible, and never in competition
with it for a reader keying on `span_id`.

**But SPEECH's, VOICE's and DDK's *assertions* currently reach no reader.** `report.py:1127` drops
every assertion whose branch is not AIRWAY, and `figure.py:577-578` reads assertions only where
`name == "squim"`. Their **spans** are read — `report.py:1130-1139` has explicit arms for them, and
`propose` mints spans — so it is precisely the four annotating verbs whose output is invisible.
Widening REPORT's assertion read is a piece of work in its own right, listed below, and not something
the existing readers absorb for free.

**This is the store's own idiom, not an invention.** `corroborated_by` already records rival
proposals as an attribute without minting entities, and `withdraw_contradicted_clips` with
`_retire_quality_findings` is the module's one invalidation cascade *precisely because* minting forces
one. Annotation needs no cascade, so there is none to enumerate.

It also makes `trim` non-destructive by construction, which is what the no-suppression rule wanted
and what the previous revision's "retires nothing" could not deliver while still double-counting in
`features.py:1080-1093`.

**A branch's own output is invisible to AIRWAY**, which selects `family is None` (`airway.py:198`).
Anything written as `family: "<branch>"` — including AIRWAY's own proposals — is outside that filter.
Under the annotation model this affects only `propose`, and it is a constraint the AIRWAY piece must
address rather than a defect in the contract.

**Nothing measures a proposed span.** No node runs a model at branch time, and `airway.py:171-173`
makes not re-running HeAR an explicit design point. **A proposed span therefore carries its branch's
own evidence and no per-span classifier measurement at all**; adding a branch-time classifier pass is
out of scope for this design and is listed as unresolved.

Even if one were added, **YAMNet could not classify a short proposed span**: `span_yamnet_input`
raises `SpanTooShortForYAMNet` below 0.96 s (`yamnet.py:257-258`), so rule (a)'s own motivating case —
a 200 ms cough — can never acquire a native YAMNet window, whatever proposes it. HeAR has the
capability, via the silent buffer (`hear.py:426-443`), but no mechanism invokes it at branch time.

### `refine` covers metadata as well as extent — owner decision, 2026-09-15

The owner, on speaker attribution and SPEECH: *"speech may need to resolve across multiple speakers,
so it could fall under refinement (adding/adjusting span metadata, or creating an aggregated span)."*

Attribution maps onto the verbs already in the table and **no sixth verb is added**:

- **adding or adjusting a span's metadata → `refine`**
- **creating an aggregated span → `propose`**, which remains the only minting verb

**This widens `refine`, and the old definition is recorded rather than overwritten.** Until
2026-09-15 `refine` was **extent-only**: the table's `carries` column read `corrected_extent:
[start, end]` and nothing else, and § *A branch refines and reviews* glossed the verb as *"assert a
tightened extent on an existing span"*. A store written before this date therefore holds `refine`
assertions that are all extent corrections, and a reader of one must not infer from their shape that
metadata refinement was declined — it was not yet available. **From 2026-09-15 `refine` means
improving a span in either way: its extent, its metadata, or both.** The verb's object is unchanged —
an existing span, which keeps its id, its `family`, its measurements and its liveness.

**The contract was already using the widened sense before it admitted to it, which is the reason the
change is a correction and not only an addition.** § *A ruleset that fires may write or refine a
span's label*, decided earlier the same day, says a fired rule writes `label` where the span carried
none and **`refine` where it carried one the rule sharpens** — a *label*, which is metadata, not an
extent. [`../20260817-triage-workflow-dag/family-taxonomy-ruleset.md`](../20260817-triage-workflow-dag/family-taxonomy-ruleset.md)
§ *A fired rule may write or refine a span's label* restates it in the same words. Under the
extent-only table neither statement was writable.

#### The attribute shape

A `refine` assertion carries **one or both** of two named attributes, and must carry at least one:

| attribute | what it corrects | shape |
| --- | --- | --- |
| `corrected_extent` | the span's extent | `[start, end]` |
| `corrected_attributes` | the span's metadata | a mapping from the span attribute's own key to the corrected value |

`assertion.extent` continues to name the span being annotated, for the reason given above: every
existing assertion sets `extent=span.extent` (`airway.py:272`, `:308`, `:330`) and `report.py:1144`'s
`_timing` reads it as that span's timing. A metadata refinement gets a named attribute for exactly
the same reason an extent correction does.

**Presence is the discriminator, for a reader and for code.** `"corrected_extent" in attributes` says
an extent is being corrected; `"corrected_attributes" in attributes` says metadata is; both present
is one act of improvement that did both, which is the ordinary case when a boundary fix changes what
the span should be called. A `refine` carrying neither is malformed — a testable condition rather
than a convention, and the one shape a reader may reject.

**Why a nested mapping rather than a flat `corrected_<key>` per field.** Because the store has
readers that key on an assertion's **top-level** attribute names without testing `verb` at all:
`figure.py:578` and `routing_analysis/features.py:1092` both select assertions on
`attributes.get("name") == "squim"`, and `report.py:1129` builds an entity's rendered description
from `attributes.get("name") or attributes.get("family")` for every branch entity it lists. A
corrected value written at the top level under the span's own key — `name`, `family`, `label` — is
therefore one field name away from being read as the assertion's own property. Nesting puts every
corrected value out of reach of every existing top-level test, and it lets one assertion correct more
than one field without a reader having to diff the assertion against the span to work out which keys
were meant.

**The prior value is not copied into the assertion, and does not need to be.** For the extent,
`assertion.extent` already carries it. For the metadata, the span carries it: nothing rewrites a
span's attributes, the module's only `was_invalidated_by` call is `extend.py:327`, and `quality.py:27`
states the rule outright — *"PREPROCESS's spans are never invalidated here: the store is
append-only"*. So the span reached through `store.derived_from(assertion.id)` still holds what the
refiner read. An absent key on that span additionally distinguishes *correcting* a value from
*supplying* one, with no extra attribute.

#### An aggregated span is a `propose`, and three questions it raises

**What family it carries: `family: "<branch>"`, unchanged.** Aggregation changes nothing about the
minting rule, and [`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *`propose` versus `refine` — scoped by family* already works this exact case: *"A phonation broken
by two 400 ms gaps is three PREPROCESS spans under `spans.min_separation_ms: 30`; the branch's single
attempt span simply covers all three. The branch is not refining them, so they neither need merging
nor conflict with it."* An aggregate covers ground no live span *of the branch's own family* covers,
so it mints under the rule as written.

**What it is `wasDerivedFrom`: every span it aggregates, plus the evidence that built it.** This is
production practice rather than a new rule. SPEECH's `family: "speech"` span is already
`wasDerivedFrom` every live non-SPEECH span it overlaps (`speech.py:893-895`, over the `prior_spans`
set built at `:577-578`), and `ProvStore.derived_from` returns a **list** (`utils/prov_store.py:525`),
so many-to-one derivation is a capability of the store and not a tolerated irregularity. The one
production reader that walks the relation iterates it — `for span_id in store.derived_from(entity.id)`
at `figure.py:580` — so an aggregate does not break it.

**Whether the spans it aggregates are left untouched: yes, and it is a property of the code, not only
of the design.** Nothing under `nodes/` invalidates anything; the module's single `was_invalidated_by`
call is `extend.py:327`, inside `withdraw_contradicted_clips`. An aggregate neither retires nor
contradicts the spans beneath it — the same shape as § *(b) Gap spans are background, never events*,
where a branch-proposed span inside a background extent supersedes the typing for that extent and
retires nothing.

**None of the three is undetermined, so none is recorded as owed.** The adjacent question that *is*
open — whose family a **ruleset-written label** puts a span in, and therefore whether the labelled
span is refinable by the branch at all — is untouched by this decision and stays where it was, at
[`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *The two owner decisions of 2026-09-15 leave the minting rule alone and open one question*.

#### SPEECH's `attribute` writes become `refine` or `propose` — owed a code change

`speech.py:793` writes one `verb: "attribute"` assertion per word — the highest-volume verb in the
store and not one of the contract's five. Under this decision it has two destinations and no third:

- **`refine`**, where the claim is *this span's speaker is X*: adding or adjusting the span's speaker
  metadata, carried in `corrected_attributes`. The span-level home already exists — SPEECH computes
  `attributed_to` from its words (`speech.py:877-878`), writes it on the span (`:885`) and derives
  `nontarget` from it (`:886-888`) — so what the migration moves is the per-word assertion, not the
  claim itself.
- **`propose`**, where resolving across speakers produces an **aggregated span**: one span per
  contiguous run attributed to the same speaker, minted `family: "speech"` under the three answers
  above.

Which words land in which is SPEECH's own piece to settle, not this section's; what is settled is
that `attribute` has no standing as a fifth branch verb and that these two are its destinations.
Recorded against the branch at
[`../20260817-triage-workflow-dag/branch-speech.md`](../20260817-triage-workflow-dag/branch-speech.md)
§ *S6 — Word→speaker attribution*.

#### What this does not resolve

**Who may attribute a speaker to a span is still open.** The owner's sentence is about SPEECH, a
branch. Whether PREPROCESS's coming whole-file diarization derivative may itself attribute speakers
to spans, or must emit segments for consumers to intersect, is a different question and is not
decided here. The division recorded today stands unchanged —
[`../20260817-triage-workflow-dag/dag.md`](../20260817-triage-workflow-dag/dag.md)
§ *5e. QUALITY — a graph edge every recording reaches, and now a node*: **PREPROCESS measures, the
branches refine, QUALITY judges**, with a speaker *identity* question staying with SPEECH.

**One document disagrees about whether `refine` is a verb at all.**
[`../20260817-triage-workflow-dag/store.md`](../20260817-triage-workflow-dag/store.md) `:40` and
`:72` state that *"`refine` and `withdraw` are gone as verbs"*, replaced by `wasDerivedFrom` and
`wasInvalidatedBy`, with `wasDerivedFrom` glossed as *"a narrower extent **or a better value**"* —
which is this decision's widened sense, reached by a different route. This contract reinstates
`refine` as a `verb:` value, so the two documents conflict, and they conflicted before this decision
rather than because of it. Not reconciled here: `store.md` is out of this spec's scope for the same
reason § *Explicitly unresolved* gives for the writer/reader vocabulary rule.

### Where each branch already stands

**SPEECH already does it, unnamed.** It takes no subject from the general span set, groups lexical
word runs into its own spans, and writes them with `family: "speech"` (`speech.py:879-883`). It is
the working instance of the contract. It does not *ignore* the general set, and an earlier revision
said so: it `used`s every live non-SPEECH span (`speech.py:577-578`, the edges at `:600-601`) and
makes each of its own spans `wasDerivedFrom` the ones it overlaps (`:893-895`) — which is the
provenance shape § *An aggregated span is a `propose`* generalises.

**VOICE was designed to and cannot.** Its subject is every live span whose `family` is `phonation`
(`voice.py:230`, `_PHONATION_FAMILY` at `:40`). **Nothing reachable proposes one.** The detector that
did was retired 2026-09-04; VOICE itself writes the family at `voice.py:337`, but that code sits
downstream of the no-span path it always takes, so it never runs. The branch returns `Outcome.FAIL`
on every recording.

Under this contract **the branch is the proposer**, and that is the whole fix — **with the selector
changed to the branch family.** `propose` writes `family: "voice"`, so VOICE's input filter and its
output family coincide, which is what the contract wants.

**"The whole fix" is now the ruleset's label plus VOICE's `refine`, on the owner decisions of
2026-09-15 above.** A ruleset-written phonation label gives VOICE a labelled amplitude span to
`refine` without any proposer, and that is the settled flow:
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
§ *The subject is the spans the ruleset labelled, and VOICE refines them*. Neither the
`consensus_taxonomy` rework nor a replacement detector is the span source any more; V1 stays as the
specification of the refiner. Everything in the rest of
this section — the selector change, the family rename, what it costs REPORT — applies to whichever
source mints or labels the span, so none of it is withdrawn.

**That rename is not costless, and an earlier revision wrongly called it so.** `phonation` is the
family VOICE **writes today** (`voice.py:337`), and REPORT reads it: `_spans_of_family(store,
"phonation", voice=False)` at `report.py:673`, the `voice=True` reads at `:715` and `:1154`, the
descriptions at `:1134` and `:1173`, and VOICE's summary keyed on `phonation_s` at `:104`.
`_spans_of_family` (`report.py:289-306`) exists precisely to separate the detector-proposed
population from VOICE's own, by `onset_kind`. So this is a **writer-vocabulary change on a family
REPORT reads**, and it is settled by the rule this spec already invokes twice: **the writer's
vocabulary may shrink, the reader's may not.**

`phonation` becomes a **historical family** — VOICE writes `voice`, and REPORT keeps reading
`phonation` forever, because finished stores carry those spans forever. This is the same shape as the
`kind` entity type in `PROV_TYPE` (`prov_store.py`, retired 2026-09-13 and kept readable for exactly
this reason), and the `kind` fix is the precedent to follow rather than re-derive.

Concretely: `report.py:104`'s `phonation_s` **stays**. It names seconds of phonation, not a family,
and it sums spans of both — `phonation` from stores written before the change, `voice` after.
`_spans_of_family` (`report.py:289-306`) grows to read both families; it currently takes a single
`family: str`, so it needs either a sequence parameter or two calls merged.

**One consequence to record rather than paper over.** `_spans_of_family`'s `voice` parameter splits on
`("onset_kind" in span.attributes)`, and `onset_kind` is written only by the second minting at
`voice.py:333-346` — the very minting this contract replaces with a `refine` assertion. So for stores
written under the contract **nothing carries `onset_kind`, and the `voice=True` reads at
`report.py:715` and `:1154` return empty.** The split remains correct for historical stores and
becomes vacuous for new ones; what replaces it is the distinction between a span and a `refine`
assertion over it. REPORT's VOICE arms need that substitution, not just the family widening.

**VOICE is the worked example of the contract, and of what it forbids.** `voice.py:333-346` mints a
second, period-aligned span from an input span, carrying `onset_kind` and `offset_kind` — **that is
exactly the re-mint the annotating verbs replace.** Under this contract that minting becomes a
`refine` assertion carrying `corrected_extent`, the original span keeps its id and its measurements,
and `_spans_of_family`'s two-population problem dissolves with it.

**AIRWAY has half the verbs already**, writing `label`, `confirm`, `contest`, `abstain` and `flag`
assertions across its first three steps (`airway.py:230` through `:373`).

Two of those are the contract's own verbs already, which is why the contract adopts their spellings
rather than renaming them. `confirm` becomes a `label` carrying its corroborating window ids;
`abstain` and `flag` keep their current meanings.

**`contest`'s contract meaning is already settled; what AIRWAY needs is a criterion for applying
it.** The contract's definition is the one in the verb table — *the span does not carry what was
proposed* — and QUALITY's clip contradiction (`quality.py:285-295`) is an instance of it under a
completely different criterion. The two must not be conflated.

**AIRWAY's criterion needs to be threshold-free, and gets one — with the same budget rule (a) gets.**
`airway.contest_labels` is `null` (`default.yaml:138`), so `_contest_labels` returns the empty set
(`airway.py:150`) and `contested_n` is structurally zero; the ground-truth rule forbids fitting the
list. So AIRWAY contests without one: **it contests a span when it finds no evidence of any kind
within that span's extent.** That is AIRWAY's particular test, not the contract's, and it leaves the
fitted list as a later refinement letting AIRWAY contest on *contrary* evidence rather than only on
absent evidence.

**That definition interacts badly with rule (a) and must be counted before it ships.** Rule (a) makes
covering-window labels ineligible as evidence, and coughs are usually sub-0.96 s, so "no evidence of
any kind" would contest most short spans. `contested_n` would go from structurally zero to
near-total. **The same before-and-after count rule (a) requires is required here**, and for the same
reason: the two changes compound.

**Every contest flags the file**, by one of two chains. The usual one is short: contests → AIRWAY's
`flags` → AIRWAY `Outcome.FLAG` (`airway.py:379-380`) → the any-FLAG fold at `vocabulary.py:423` →
`Triage.FLAG`. The mismatch ground at `vocabulary.py:393` plays no part in it, because `_agreement`
runs over `findings` and `_resolved` (`:209-220`) maps every non-FAIL outcome to `present`.

But a recording whose spans are *all* contested on absence takes `not labels_by_span`
(`airway.py:376-378`) and returns **FAIL**, which resolves `absent` and therefore does reach the
mismatch ground — and that is exactly the population a contest-on-absence targets. Both chains end in
a flagged file; the budget is required either way.

`contested_n` also needs redefining before it can be counted: it increments inside the per-label loop
(`airway.py:289`, `:323-324`), and a span with no members never enters that loop (`:260-261`) — which
is exactly the population a contest-on-absence targets.

**Background-typed gap spans are not contestable, and the reason is the verb's object, not the
evidence.** A gap span carries real branch-readable evidence — gaps are appended to `span_ids`
(`preprocess.py:1596`) and the per-span classifiers run over them, which is why 143 of 384 consensus
rows traced to gaps in the 2026-09-07 measurement — a figure that predates its own fix, per
Corrections below, and is cited here only for the mechanism it demonstrates. So "no evidence within the extent" is not automatically true of a gap. What is
true is that `contest` carries *that a span does not carry what was proposed*, and **a gap proposes
nothing**, so a contest over one has no object.

**DDK has no node.** `nodes/ddk.py` does not exist. Since stage 2 the ruleset routes DDK, so a
recording with DDK content receives a `branch_decision` with `will_run` true and is recorded
`SKIPPED` with `NO_NODE` (`run.py:303-305`), which flags the file.

---

## Deviations are facts, not judgements

A branch enumerates typed, located deviations, each with its evidence. **Downstream nodes, humans or
machines determine whether a deviation is legitimate.** The pipeline does not assert that a task was
not performed.

This rules out the conformance state vocabulary that was drafted and rejected —
`conformed` / `deviated` / `not_performed` / `not_declared`. A state is a judgement. A deviation is
an observation with an extent.

**Deviations are recorded and are not folded into VERDICT's flag grounds until ground truth exists.**
`filler` and `stimulus_mismatch` are expected on ordinary read speech; routing them into the flag
column would flag the corpus — the exact failure stage 1 refused when it declined to let a
default-uncertain kind line flag every recording.

**That argument was derived over three types and now governs eleven, and it does not fit all of
them equally.** `filler`, `stimulus_mismatch`, `repeated_item` and `lexical_content` are ordinary on
spontaneous or read speech and would flag the corpus. `truncation`, `omission`,
`sweep_direction_mismatch` and `syllable_sequence_mismatch` are **not** ordinary — each says the
production departed from what the instruction asked for, and each is a candidate flag ground once
ground truth exists to set a rate against. `verdict.deviation_flags` stays `false` for all eleven,
because no ground truth exists for any of them and a rule fitted to none is worse than a rule that
flags none.

A per-type folding policy is the natural extension and is **deliberately not built**: every entry
would read `false` today, which is a mechanism carrying no decision, and this graph does not ship an
unmeasured decision. Build it with the first measured rate, not before.

A deviation is an observation **with an extent**. Eleven qualify, declared in
`nodes/branches.py`'s `DEVIATION_TYPES` and enforced at the write by `write_findings`:

| type | what it says | branch |
| --- | --- | --- |
| `stimulus_mismatch` | a lexical word that is not the word the stimulus expected | SPEECH |
| `filler` | a disfluency or non-speech token where the task expected lexical content | SPEECH |
| `off_task_extent` | a region of the recording that does not serve the declared task | AIRWAY, SPEECH, shared |
| `lexical_content` | a lexical word where the task expected none | VOICE |
| `omission` | an expected token the recording does not realise | SPEECH, VOICE |
| `repeat_attempt` | a further carrier where the task expected one production | VOICE |
| `repeat_reading` | an alignment covering the expected sequence more than once | SPEECH |
| `repeated_item` | an item repeated where the task expected each once | SPEECH |
| `sweep_direction_mismatch` | a pitch sweep running against its declared direction | VOICE |
| `syllable_sequence_mismatch` | a syllable whose place is not the one its cycle position expects | DDK |
| `truncation` | a production the recording does not contain the end of | AIRWAY, DDK, SPEECH, VOICE |

**This table said "three" until 2026-09-16, while the branches emitted eleven.** Nothing validated
the name at the write, so the vocabulary and the code drifted apart silently — and the three it
named were a subset, not a mistake, which is why nothing ever failed. `write_findings` now refuses
an undeclared name and `branches_test.py`'s AST sweep refuses a declared name nobody emits, so the
two cannot part again in either direction.

The three named above are the three the folding argument below was actually derived over. The other
eight inherited a justification written without them in view — see the note under that rule.

**`speaker_count` and `expected_event_count` are not deviations** — they are file-level counts with
no extent, and the section's own definition excludes them.

They are written as a **per-branch `counts` measurement**, each entry carrying `found` and
`declared`, the latter copied from the declaration. Not "beside the declaration", which an earlier
draft said and which **cannot be built**: the declaration is SCREEN's, written at stage 2, while
`expected_event_count`'s `found` half can only come from a branch at stage 3. (`speaker_count` would
fit there, since its `found` half comes from PREPROCESS's diarization at stage 1 — but splitting the
two placements is worse than one rule.) A measurement rather than verdict `detail` because these are
facts, not conclusions.

Earlier drafts called them `extra_speaker` and `missing_expected_event` and put them in the deviation
table; both the names and the placement encoded a comparison the branch is not entitled to make.

### The line the ground-truth rule draws through this table

A declaration used as a **condition for what to look for** is the contract's whole point. A
declaration used as a **reference to score against** is what the ground-truth rule forbids, because
the declaration is not verified.

The three deviations are on the right side of it: each reports something observed in the audio,
located, with the declaration only saying where to look.

**The two counts are what forced the line to be drawn.** As `missing_expected_event` and
`extra_speaker` they scored the recording against a count the declaration asserts and nobody verified
— and "expected five breaths, found three" is as likely to mean the table is wrong as that the
participant under-performed. Recorded instead as **`found` and `declared` on the per-branch `counts`
measurement, with no discrepancy asserted**, they stay observations. Whoever reads them may compare; the
branch does not, and both **owe ground truth** before any consumer treats the difference as a
finding.

### Read speech has the sharpest case

`stimulus_text` is a reference transcript. Lexical words align against it — and the aligner exists:
`consensus.py::align_sources` over `harmonize._align_pair`, specified in `transcript-alignment.md`.
**Bracketed tokens are their own channel** — disfluency and non-speech, not transcript errors — so
`[uh]` and `[breath]` never read as misread words.

"Read the wrong sentence" and "read it with six fillers" are different rows, not one score.

---

## What PREPROCESS owes

### Whole-file diarization as a shared derivative

Today SPEECH runs pyannote itself, over `(min word start, max word end)` — the lexical word hull
(`speech.py:642-646`). It cannot see a speaker outside that hull, which is where an interrupting
voice or a background talker lives.

Move it to PREPROCESS, whole-file, in the same shape as the PPG extension:

- **Every branch gets it.** A cough from a second person in a respiration recording is an airway
  finding AIRWAY has no way to notice today.
- **`speaker_count` becomes available for every task.**
- **SPEECH's diarize step becomes a read — owner decision, 2026-09-15**: *"speech should not have
  to rerun pyannote to do this, just take the output and use it."* SPEECH does not run pyannote; it
  reads the shared derivative. `speech.second_diarizer` (`data/config/default.yaml:189`, null, so
  `second_record` stays `"not_consulted"` at `speech.py:684-685`) becomes a question about that
  derivative rather than about SPEECH's own pass. This is a behaviour change to a shipped branch, not
  a pure addition, and it is **owed a code change**: removing SPEECH's diarizer run and replacing it
  with a read.

**The migration gains coverage rather than trading it, which is why the decision costs nothing to
take.** SPEECH's own run is scoped to `(min word start, max word end)` — the lexical word hull,
computed at `speech.py:642-645` under the step comment at `:641` that says as much — so a speaker who
talks before the participant starts, after they stop, or inside a pause falls outside the interval
and cannot be found at all. A whole-file derivative has no hull. The branch's own account of the
defect is at [`../20260817-triage-workflow-dag/branch-speech.md`](../20260817-triage-workflow-dag/branch-speech.md)
§ *S5 — Speaker count*.

**It lands together with the `attribute` → `refine`/`propose` migration** of § *`refine` covers
metadata as well as extent*: both touch SPEECH's speaker handling and the same assertions — step 6's
word→speaker attribution (`speech.py:774-798`) consumes the `speaker_segments` the shared derivative
would now supply.

**The streams are `enhanced` and `residual`, decided by the owner on 2026-09-15. No pilot is owed,
and the question that would have needed one does not arise.**

The objection that held this open was that the derivative exists to catch a quiet background talker
and enhancement suppresses exactly that. The owner's resolution: **enhancement does not destroy that
talker, it partitions the recording.** If the background voice is suppressed, then `enhanced` holds a
single speaker and the suppressed voice is in `residual` — where it can be analysed directly. Nothing
is lost, so there is nothing to trade off and nothing to pilot.

This costs no new plumbing, because `residual` is already first-class: `residual = plain - g*enhanced`,
lag-aligned and gain-fitted (`../../src/senselab/audio/workflows/triage/nodes/preprocess.py:2323`),
written with its own stream entity whenever that block runs (`:2393-2402`), already classified as
`residual_yamnet_scores` / `residual_ast_scores` / `residual_hear_scores`, and already read by the
ruleset — `airway.breath` is `[residual, energy_fraction]` (`data/config/default.yaml:270`) and
`residual|yamnet` sits in `peak_streams` at `:246`.

The two halves answer different questions and both are wanted:

| stream | what its speaker count answers |
| --- | --- |
| `enhanced` | how many voices survive enhancement — one, for a single-participant protocol |
| `residual` | whether a voice was *removed*, which is the evidence a background talker was there |

So the config key is a **list** of streams, not one value, and the measurement records a count per
stream rather than a single number. They must not be summed: a disagreement between the two halves is
itself the finding. An empty `residual` diarization is the expected, ordinary result and reads as a
typed absence, not a failure.

This also settles why `enhanced` rather than raw for the surviving-voice half: it matches the PPG,
which reads `enhanced` by the same owner's decision and for the same reason — many of these
recordings carry background noise that a model trained on clean speech handles worse on the
unprocessed stream. The `audio_analysis` conclusion that off-target speaker detection runs on raw
belongs to a different question and does not carry here.

**Raw or enhanced: resolved 2026-09-15 by measuring both halves, not by picking a side.** The
reason this was open stands: the derivative's headline justification is catching a quiet background
talker, enhancement suppresses exactly that, and this project's standing conclusion for off-target
speaker detection is that it runs on raw for that reason. The owner's resolution:

> the derivative exists to catch a quiet background talker and enhancement suppresses exactly
> that → then the enhanced audio has a single speaker, and the residual can be analyzed. there is
> no problem with this.

Enhancement **partitions** the recording rather than destroying what it removes, and the residual is
already first-class here — computed, lag-aligned, gain-fitted, written as its own stream, already
classified beside `enhanced`, already read by the ruleset. So `diarization.streams` is
`[enhanced, residual]`: the enhanced half says how many voices survived enhancement, the residual
half says whether one was removed, **the two counts are never summed**, and a disagreement between
them is itself the finding. **No pilot is owed.** The derivation is in
[`../20260817-triage-workflow-dag/config-derivations.md`](../20260817-triage-workflow-dag/config-derivations.md)'s
`diarization` section.

The claim that pyannote is reliable at the single/multi-speaker distinction is **uncited** and is
still not relied on here; `benchmarks/diarization.md` and `benchmarks/glides-diarization.md` are in
this tree and the corpus pass should be read against them.

**Cost:** two model passes over the corpus, delivered as an extend driver.

**Landed 2026-09-15.** PREPROCESS's `diarization` block writes one measurement per stream, and
`scripts/extend_diarization.py` adds them to the stores that already exist.

**What this section still asks for and the change deliberately does not do.** The owner has since
directed that *"speech should not have to rerun pyannote to do this, just take the output and use
it"* — so SPEECH's own pass over the word hull (`speech.py:642-646`) is going away and this
derivative becomes the only diarization in the graph. That read-swap is **not** in this change,
because it is a behaviour change to a shipped branch: exactly what SPEECH reads today, what the
migration gains, and the three consequences it must handle are enumerated in
[`../20260915-preprocess-diarization/design.md`](../20260915-preprocess-diarization/design.md)'s
"What SPEECH must read". `speech.second_diarizer` is untouched and becomes a question about the
shared derivative. The multi-voice *judgement* is likewise absent by design — PREPROCESS measures,
branches refine their task spans, QUALITY judges against those refined spans after every branch.

### An extend driver runs a stage's work late

Not "an extend driver is PREPROCESS" — that is wrong twice over. `extend_quality.py` runs QUALITY,
and `rewrite_consensus_taxonomy` recomputes what will be a SCREEN product. The correct statement:
**an extend driver runs a stage's work against a finished store, under the same contract as the
original pass**; what it appends is a derivative every downstream reader treats identically, and the
only difference is when the store was open.

`extend.py:3-6` already holds this position — "the layout is the workflow's, not any one driver's" —
so this spec does not correct that docstring; an earlier draft mis-attributed the opposite view to it.

---

## VERDICT

Three jobs:

| job | source |
| --- | --- |
| **admit / reject** | ADMIT's failure (`unmeasurable`); the ruleset's `empty` state (`acoustically_empty`); branch findings that the content is unusable |
| **flag for human review** | route-vs-finding mismatch; `unexplained` route state; the existing grounds at `vocabulary.py:364-405` |
| **describe for the record** | branch findings, `consensus_taxonomy`, the declaration |

**The existing precedence is ADMIT-fail → *any* FLAG → `empty` → pass** (`vocabulary.py:419-429`), so
`acoustically_empty` discards only when nothing flagged; `verdict.md:135-140` records the additional
condition that no hint claims otherwise. The table above is not a replacement for that ordering.

**The description job requires VERDICT to read more of the store than it does.** Its docstring says it
holds every node's `verdict` entity, ROUTING's `branch_decision`s and its `ruleset_routing`, and that
"this node reads nothing else" (`verdict.py:218-220`). Reading branch findings and the consensus
taxonomy is a deliberate widening of that contract. **The smallest defensible choice: VERDICT does
not widen.** The description is assembled by REPORT, and VERDICT's `detail` carries the identifiers a
reader follows.

**REPORT does not read the whole store either, and an earlier revision claimed it did.** It reads
spans and measurements across branches — including SPEECH's and VOICE's spans, which have their own
arms at `report.py:1130-1139` — but `report.py:1127` drops every assertion whose branch is not AIRWAY,
and `figure.py:577-578` reads assertions only where `name == "squim"`. Under the annotating contract
that means **the four annotating verbs' output reaches no reader** outside AIRWAY. Widening REPORT's
assertion read is a prerequisite for the description job, and is listed as its own piece below. Whether the durable description should
instead be its own entity written by VERDICT is **unresolved**.

**A deviation does not drive a reject**, and for now does not drive a flag either. Whether a
deviation is disqualifying is task-specific, needs ground truth, and is not decided here.

---

## This is several implementation plans, not one

Dependency order, with the pieces that are genuinely independent marked:

1. **The three hint breakages** — read the sidecars, rename the override key, fix the value casing.
   **Independent.** Note `make_hints.py` also reads `routing.hint_kind_map` and raises when it is
   absent (`:414-416`, validated at `:20-23` and `:429-439`), so renaming only `override.yaml` breaks
   hint generation; both move together.
2. **The `preprocess.py:1533` continuity derivation bug.** **Independent, small**, split out of piece 9.
3. **Span cleanliness (a) and (b)** — the attribution filter and background typing, scoped to
   `span_yamnet`. Includes the before-and-after gate-firing count on the corpus and a
   `rewrite_consensus_taxonomy` pass over existing stores.
4. **The merge** — TAXONOMY ⊕ ROUTING into SCREEN, with the retired node names kept readable, the
   intra-node ordering test restored, the single verdict decided, and the four documents updated.
5. **The declaration** — the two-grain table, SCREEN's `declaration` measurement, the `metadata` key
   contract. Depends on 1. **Its dependency on 4 is optional** — the measurement can be written by
   TAXONOMY pre-merge — and decoupling it removes the only blocking dependency on the largest piece.
6. **Whole-file diarization** — the extend driver, then SPEECH's diarize step becoming a read. The
   streams and the SPEECH read are both settled by the owner decisions of 2026-09-15 in § *Whole-file
   diarization as a shared derivative*, so **no pilot is owed**; an earlier revision of this list put
   one first. **Not independent of 5**: `speaker_count` needs the declaration's single-target claim.
7. **Widen REPORT's assertion read — by verb, not by branch.** `report.py:1123` already filters to
   `_EVIDENCE_BRANCHES` (defined at `:56`), so `:1127` is purely the AIRWAY-on-assertions
   restriction, and lifting it branch-wise would admit SPEECH's `verb: "attribute"` assertions —
   **one per word** (`speech.py:790-793`). Admit the contract's five verbs plus `abstain` and `flag`,
   and `deviate` with them
   ([`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
   § *Deviations and counts are stored*); leave `measure` (`preprocess.py:1835`, `:1842`, `:1860`) and
   `withdraw` (`:599`) out. **`attribute` is on the leave-out list only until its migration lands** —
   under § *`refine` covers metadata as well as extent* it becomes `refine` or `propose`, both
   admitted. Either order works; doing 7 first makes the migrated assertions visible the day they are
   written.
   **Independent of the branch work and a prerequisite for it**; without it a branch can be built and
   its annotations be invisible.
8. **The branch contract, per branch** — SPEECH first (closest to it), then AIRWAY (which also closes
   the `_windows_covering` gap), then VOICE (which needs a proposer before it has a subject), then
   DDK (which needs a node). Depends on 5, on 6 for `speaker_count`, and on 7 to be visible.

   **The owner decisions of 2026-09-15 change VOICE's place in this piece**: its subject is a
   ruleset-written label, which lands in SCREEN rather than here, and VOICE's own work is the
   `refine` over it. The settled flow is at
   [`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
   § *The subject is the spans the ruleset labelled, and VOICE refines them*.
9. **Boundary reconciliation** — blocked on a parameter-free definition, per (c).

**One ordering inversion is accepted and budgeted.** Piece 9 changes every span extent, hence which
spans clear 0.96 s — the whole basis of piece 3 — and which need refining in piece 8. Since 9 is
blocked on a definition that does not exist, 3 and 8 proceed first and **are re-measured after 9
lands**; that re-measurement is part of 9's cost, not a surprise.

---

## Explicitly unresolved

**Every fitted threshold awaits ground truth.** `spans.k_db`, `airway.contest_labels`, the speech
quality floors, the deviation thresholds. The corpus is labelled by declaration, not by
verification. Nothing is refit until something has been listened to.

**The boundary reconciliation rule** (c) — no parameter-free definition yet.

**Which source gives VOICE its subject is settled, 2026-09-15**: amplitude spans with a
ruleset-written label, refined by VOICE. The flow and the evidence behind it are at
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
§ *The subject is the spans the ruleset labelled, and VOICE refines them*. **What remains unresolved is whose family a
ruleset-written label puts a span in**, since a branch `refine`s only a span of the family it
proposes into — at
[`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *The two owner decisions of 2026-09-15 leave the minting rule alone and open one question*.

**What a ruleset-written label records, exactly** — the rule, the evidence path and the value are
settled as required; the entity shape, and which span a reduced feature attributes its firing to, are
not.

**The writer/reader vocabulary rule deserves stating once.** This design retires three
writer-vocabulary values with live readers: the `kind` entity type (done 2026-09-13), the node names
`TAXONOMY` and `routing` (the merge), and the `phonation` span family (VOICE). Three instances of one
rule, currently written as three special cases. **Recommendation: state it once in `store.md` as a
general contract** — a writer may stop emitting a value, a reader may never stop accepting one — with
the three instances as its examples. Not edited here; `store.md` is out of this spec's scope.

**Whether a branch may run a classifier pass at branch time.** Nothing does today, and
`airway.py:171-173` makes not re-running HeAR an explicit design point, so a proposed span carries no
per-span classifier evidence. Adding such a pass is out of scope here.

**DDK has a contract, which is not the same as being specified.** What DDK concludes about a segment
rate, and what its findings are, is undecided. Until the node exists, recordings with DDK content
flag.

**Whether a deviation is disqualifying**, per task.

**`expected_event_count` and `speaker_count` owe ground truth** — recorded as observations, no
discrepancy asserted.

**Promoting the declaration's fields from `metadata` to typed `AudioHints` fields.**

**Whether background spans split around branch-proposed events**, or are merely superseded for that
extent.

**Whether VERDICT should write the durable description as its own entity**, rather than REPORT
assembling it.

**Raw vs enhanced for diarization is resolved, 2026-09-15** — the streams are `enhanced` and
`residual`, and no pilot is owed (§ *Whole-file diarization as a shared derivative*). It was listed
here as unresolved until that date, and the line is kept so a reader of the older text finds where it
went.

**The corpus size disagrees with itself by 28.** `20260910-taxonomy-routing-evidence/measurements.md:1,18-19`
and two other documents attest **62,550 recordings / 62,547 stores read**; a
`find -name store.jsonl | wc -l` over the run tree on 2026-09-13 returned **62,578**. Both numbers are
real. This spec quotes no corpus count in prose for that reason; anything that needs one should use
the attested 62,550 until the 28 are accounted for.

**HeAR's silent-buffer isolation of short spans is unmeasured.**

---

## Corrections to things already written down

**Three documents wrongly assert that FIGURE and REPORT read `consensus_taxonomy`**: `taxonomy.md:64-65`,
`20260912-ruleset-in-pipeline/design.md:367-369`, and `taxonomy.py:6`. They do not — `figure.py:603-617`
reads `<classifier>_label_summary`, `report.py:425,484-487` reads PREPROCESS's `<classifier>_windows`,
and the only production consumer is `features.py:755`. `dag.md:358` is the one that has it right.
**All three are to be corrected**; the error was inherited into an earlier draft of this spec and
removed.

**`dag.md:1111-1116` is wrong about gap spans.** It states their background content reaches no
decision; they are written with no `family` (`preprocess.py:1583-1592`) and `airway.py:198` selects
`family is None`, so they *are* AIRWAY evidence. `dag.md:1210` contradicts `dag.md:1116` within the
same document. Fix dag.md.

**`preprocess.md` is stale in three places**: `:67-68` and `:160-195` document `phonation_spans` as
live and feeding VOICE, which has not been true since 2026-09-04; `:152` says AIRWAY reads and may
adjust `spans.k_db`, which contradicts `config-derivations.md:106,116-119` and the code (`airway.py`
reads no `k_db`); `:141-148`'s span algorithm contradicts `dag.md:643-656` and `default.yaml:37-40`.

**The 94%-of-taxonomy-rows figure predates its own fix.**
`benchmarks/taxonomy-vs-task-2026-09-07.md:203-288` measured that 360 of 384 `consensus_taxonomy`
rows (94%) trace to spans shorter than 0.96 s, 293 (76%) to spans shorter than 0.3 s, and 143 to gap
spans. The periodic fill that caused it was replaced with covering-window attribution on 2026-09-08
(`benchmarks/span-fill-recovery-2026-09-08.md:283`). **The current rate is unmeasured.** The
structural argument here does not depend on the number; the number must not be quoted as current.

**`recording_input_gain` does not predict clipping.** A gain can be set anywhere and the signal can
still clip; a low gain does not mean a clean file. It is an acquisition fact, not a clipping
predictor.

**`acoustic_task_status` is out of scope** — the protocol's own record of whether the task completed,
which is upstream or downstream human evaluation.
