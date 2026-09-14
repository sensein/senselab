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

BRANCHES     Receive every upstream output, their route, and the declaration. Mark, refine,
             refute, propose and trim into `family: "<branch>"` spans. Emit typed deviations.
             Conclude on their own question.

VERDICT      Admit/reject, flag for human review, durable description.
```

### The nodes this does not place

ADMIT, QUALITY, REDACT, REPORT and FIGURE are in `GRAPH_ORDER` and are not branches. They keep their
current positions: ADMIT before PREPROCESS, QUALITY terminal after the branch loop, REDACT a step of
SPEECH, REPORT and FIGURE re-reading a finished store.

**QUALITY is the sharp case, and it decides the scope of the verbs.** QUALITY writes `contest`
assertions over PREPROCESS's clip spans (`quality.py`, the clip-consistency check) — that is
`refute`, performed by a node that is not a branch. So **the verbs are store-wide, not branch-only**:
any node may mark, refine, refute, propose or trim, and the contract below says how, regardless of
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
on it, and every finished store carries activities with `node: "TAXONOMY"`. Stage 2's own audit
states the governing rule — **the writer's vocabulary may shrink, the reader's may not** — and this
is the same shape as the `kind` prov-type fix of 2026-09-13. So: SCREEN is the only name written;
`"TAXONOMY"` and `"routing"` remain readable, and a reader encountering either folds it as SCREEN's
predecessor rather than raising. A regression test parametrised over the readable set, as
`prov_store_test.py::test_every_readable_entity_type_round_trips` is for prov types.

**An ordering constraint demotes back to intra-node.** Stage 2 promoted the
`voice.glide` / `voice.chant` → `yamnet_label_summary` dependency from step ordering inside TAXONOMY
to an edge in `GRAPH_ORDER` (`taxonomy.md:60-63`, ruleset `design.md:372-375`). Merging demotes it
again. **The intra-node ordering test must return with the merge** — without it VOICE silently stops
routing, which has already happened once.

**Two verdicts become one.** TAXONOMY and ROUTING each write one; `store.md` requires a node's
verdict be attributed to its last step. SCREEN's verdict is the **routing** conclusion, attributed to
the routing step, because that is the stage's product. The consolidation's conclusion — whether there
was anything to consolidate — becomes a field in that verdict's `detail`, not a verdict of its own. A
consolidation that had nothing to work from is a fact about the run that the routing verdict reports;
it is not a second judgement.

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
`covering_windows_n` and `covering_seconds` (`preprocess.py:1952-1972`); a long span gets
`attribution: "native"` (`preprocess.py:2013`). A covering-window label is a statement about up to a
second of audio attributed to a fifth of it.

`_span_hear` does something different. `span_hear_input` places a span shorter than
`HEAR_WINDOW_SECONDS` in a silent 2 s buffer, "so its only detector result describes the span
itself" (`hear.py:426-443`); a longer span is passed through (`hear.py:466-468`) and its native
windows are mapped back by `hear_window_extent`. HeAR's short-span label is about the span.

**The rule: a covering-window label is recorded, and is not eligible as evidence.** The flag already
exists and is already written; nothing filters on it today.

**Which readers the rule applies to**, stated exactly because this is not a cosmetic change:

| reader | what changes |
| --- | --- |
| `consensus_taxonomy` rows (`taxonomy.py:192`) | a covering-window score does not contribute a row or a `peak_by_classifier` entry |
| `RecordingFeatures` span-label stats (`features.py:1137 _label_span_statistics`, via `_absorb_span_window`) | covering-window scores excluded from `span_label_stats` and `span_label_set_stats` |
| branch evidence (AIRWAY's per-span read, `airway.py:242-245`) | covering-window labels not actionable |
| figure, report | rendered as attributed-from-outside, not as the span's own label |

**This is not consumer-side-only and it moves routing.** `airway.cough`'s feature is
`span_label_set_stat` over `yamnet.cough_labels.peak_over_floor_db_max` (`default.yaml:245-248`),
built by `_label_span_statistics` from exactly these per-span measurements with no attribution check.
Coughs are usually shorter than 0.96 s. **A before-and-after gate-firing count over the corpus is part
of this piece, not a follow-up.** If the shift is unacceptable the rule is scoped away from
`RecordingFeatures` and applied only to the consensus and to branch evidence — but that is a decision
to make against the measurement, not before it.

No PREPROCESS re-run is needed; existing stores need `rewrite_consensus_taxonomy` re-run to pick up
the consensus change.

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
sources' surviving proposals (`preprocess.py:1559`) — not the complement of everything proposed. A
gap shorter than `min_duration_ms` is not emitted at all. They carry `measure: "gap"` and
`merged_proposals: 0` (`preprocess.py:1568-1580`) and, being written with no `family`
(`preprocess.py:1570-1578`), they are selected by AIRWAY's `family is None` filter (`airway.py:198`)
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

`_novel` (`preprocess.py:1454-1479`) appends a record to `corroborated_by` on every span a later
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
energy envelope rather than the continuity trace (`preprocess.py:1520` — `state["envelope_id"]`),
even though the trace is in the activity's `reads` (`:1427-1428`). The provenance edge names the
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

**When the two grains disagree** — both carry `speech_type` and `stimulus_text` — the recording-grain
file wins, and the disagreement is recorded on the declaration as a field rather than resolved
silently.

**When no hint is supplied**, SCREEN writes the `declaration` measurement anyway, carrying
`expectations: null` and naming why. An absent declaration and an unread one must stay
distinguishable, which is the same rule the graph already applies to absent measurements.

### Two standing rules

**A hint may add and inform. It may never suppress.** Routing is the union:
`will_run = by_ruleset or forced_by_hint` (`routing.py:215-220`). A branch the content routed cannot
be un-routed by a declaration that disagrees with it.

**Content the task did not ask for is content, not error.** A breathing recording that carries speech
routes SPEECH on the ruleset's evidence and has that speech recorded.

---

## The branch contract

> A branch receives the shared derivative state, its route, and the declaration. It **marks** what it
> recognises, **refines** boundaries the proposer got wrong, **refutes** what was proposed but is not
> there, **proposes** what the proposer missed but the declaration says to expect, and **trims** a
> span to the extent that serves the task. It writes its findings as `family: "<branch>"` spans, emits
> typed deviations, and concludes on its own question.

The owner's example is the specification: *a breathing task routing through AIRWAY will try to
estimate inhalation and exhalation even though the initial spans may not have generated all of them;
and a non-airway task routing through AIRWAY will do its best to find breathing or other airway
information and mark, refine or refute it.*

The consequence worth stating plainly: **PREPROCESS's span set is clean but not complete.** Clean is
PREPROCESS's job — the three parts above. Complete is the branch's, and only the branch can do it,
because only the branch knows what it is looking for.

### What each verb writes

Stated per verb, because "refine a boundary" under an append-only PROV store is not an edit.

| verb | writes | edges | invalidates | cascade |
| --- | --- | --- | --- | --- |
| `mark` | `assertion`, `verb: "mark"` | `wasDerivedFrom` the span | nothing | none |
| `refute` | `assertion`, `verb: "refute"` | `wasDerivedFrom` the span | nothing | none — the span stays live |
| `propose` | `span`, `family: "<branch>"` | `wasDerivedFrom` the evidence that proposed it | nothing | none |
| `refine` | new `span`, `family: "<branch>"`, new extent | `wasDerivedFrom` the original span | the original | measurements carry forward |
| `trim` | new `span`, `family: "<branch>"`, narrower extent | `wasDerivedFrom` the original span | nothing | plus an `off_task_extent` deviation |

**Three decisions the re-minting verbs force, made here:**

**Measurements carry forward; they are not re-measured.** A refined span's per-span measurements
remain attached to the original entity, and a reader reaches them by traversing `wasDerivedFrom`.
Re-measuring would mean a model pass inside every branch, and the measurement would in any case be of
a different extent, so it would not be the same quantity. The cost is that readers must traverse;
that is stated here as part of the contract rather than discovered per reader.

**`refine` invalidates the original; `trim` does not.** Refinement asserts the original extent was
wrong, so leaving it live double-counts. Trimming asserts the original extent was right and part of
it is off-task, so the original stands and the narrowing is an additional, narrower finding. This is
what keeps `trim` on the correct side of the no-suppression rule: **trimming writes a narrower
entity, retires nothing, and records the off-task extent as its own finding.** Built any other way it
becomes a destructive crop.

**A finished store no longer reproduces the routing that ran on it**, because `refine` retires spans
the ruleset read. That is accepted and must be recorded: the routing decision is itself an entity in
the store with its own inputs, so what the ruleset saw is recoverable from the decision rather than by
re-running the gates. `store.md` makes which fold is current the reader's choice; the rule here is
that **the current fold is the one reached by following `wasDerivedFrom` forward from a live span**,
and retired spans are reachable but not current.

`extend.withdraw_contradicted_clips` with `_retire_quality_findings` is the only existing precedent
for an invalidation cascade in this module, and each re-minting verb needs its own enumeration of
what it retires written the same way, before it is built.

**A contradiction resolved: branch-proposed short spans.** A span a branch proposes has no per-span
classifier measurement — PREPROCESS ran before it existed — so under rule (a) it could only ever
acquire a covering-window label, which rule (a) declares ineligible. Resolution: **rule (a) is scoped
to spans PREPROCESS proposed.** A branch that proposes a span and wants it classified runs its own
classifier pass over that span, which produces a native in-span window and is therefore eligible on
rule (a)'s own terms. A branch that proposes without classifying gets a span carrying its own
evidence and no classifier label, which is a coherent thing to be.

### Where each branch already stands

**SPEECH already does it, unnamed.** It ignores the general span set, groups lexical word runs into
its own spans, and writes them with `family: "speech"` (`speech.py:879-883`). It is the working
instance of the contract.

**VOICE was designed to and cannot.** Its subject is every live span whose `family` is `phonation`
(`voice.py:239-240`, `_PHONATION_FAMILY` at `:39`). Nothing in the tree proposes one — the detector
was retired 2026-09-04 — so the branch takes its no-span path and returns `Outcome.FAIL` on every
recording. Under this contract **the branch is the proposer**, and that is the whole fix.

**AIRWAY has half the verbs already**, writing `label`, `confirm`, `contest`, `abstain` and `flag`
assertions across its first three steps (`airway.py:230` through `:373`).

`mark` ≈ `label` and `refute` ≈ `contest` is a **writer-vocabulary rename on a closed set that
verdict, figure, report and extend read**, so the rule from the merge applies unchanged: SCREEN-era
code writes the new spellings, readers keep both. `confirm` becomes a `mark` carrying its
corroborating window ids. **`abstain` and `flag` map to no verb and stay as they are** — `abstain`
records that colocated evidence existed and decided nothing, which is neither a mark nor a refutation,
and `flag` is a branch-level finding rather than a span-level one.

**`refute` needs a threshold-free definition, and gets one.** `airway.contest_labels` is `null`
(`default.yaml:138`), so `_contest_labels` returns the empty set (`airway.py:150`) and `contested_n`
is structurally zero; the ground-truth rule forbids fitting the list. So `refute` is defined without
one: **a branch refutes a span when it finds no evidence of any kind within that span's extent.**
That is a statement about absence the branch can make from what it already reads, needs no fitted
label list, and leaves the fitted list as a later refinement that would let a branch refute on
*contrary* evidence rather than only on absent evidence.

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

| type | what it says |
| --- | --- |
| `extra_speaker` | more than one speaker where the declaration claims a single target |
| `stimulus_mismatch` | a lexical word that is not the word the stimulus expected |
| `filler` | a disfluency or non-speech token where the task expected lexical content |
| `missing_expected_event` | the declaration expects an event class the branch did not find |
| `off_task_extent` | a region of the recording that does not serve the declared task |

### The line the ground-truth rule draws through this table

A declaration used as a **condition for what to look for** is the contract's whole point. A
declaration used as a **reference to score against** is what the ground-truth rule forbids, because
the declaration is not verified.

`stimulus_mismatch`, `filler` and `off_task_extent` are on the right side: each reports something
observed in the audio, located, with the declaration only saying where to look.

**`missing_expected_event` is a fit in all but name** — it scores the recording against a count the
declaration asserts and nobody verified, and "expected five breaths, found three" is as likely to
mean the table is wrong as that the participant under-performed. It is retained as a **recorded
observation owing ground truth**: the branch reports what it found and what the table expected, as
two numbers, and asserts no discrepancy. The same applies to **`extra_speaker` conditioning on
`targeted_speaker_count`** — the speaker count found is an observation; that it exceeds a declared
target is a comparison against an unverified claim.

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
- **`extra_speaker` becomes available for every task.**
- **SPEECH's diarize step becomes a read**, and `speech.second_diarizer` (`default.yaml:166`, null,
  so `not_consulted`) becomes a question about the shared derivative. This is a behaviour change to a
  shipped branch, not a pure addition.

**Raw or enhanced is undecided and must be piloted first.** The derivative's headline justification
is catching a quiet background talker, and enhancement suppresses exactly that; this project's
standing conclusion for off-target speaker detection is that it runs on raw for that reason. A few
dozen files scored on both streams decides it, and that pilot precedes the corpus pass.

The claim that pyannote is reliable at the single/multi-speaker distinction is **uncited** and is not
relied on here; `benchmarks/diarization.md` and `benchmarks/glides-diarization.md` are in this tree
and the pilot should be read against them.

**Cost:** a model pass over the corpus, comparable to the PPG extension, delivered as an extend
driver.

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
not widen.** The description is assembled by REPORT, which already reads the whole store, and
VERDICT's `detail` carries the identifiers a reader follows. Whether the durable description should
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
2. **The `preprocess.py:1520` continuity derivation bug.** **Independent, small**, split out of piece 7.
3. **Span cleanliness (a) and (b)** — the attribution filter and background typing, scoped to
   `span_yamnet`. Includes the before-and-after gate-firing count on the corpus and a
   `rewrite_consensus_taxonomy` pass over existing stores.
4. **The merge** — TAXONOMY ⊕ ROUTING into SCREEN, with the retired node names kept readable, the
   intra-node ordering test restored, the single verdict decided, and the four documents updated.
5. **The declaration** — the two-grain table, SCREEN's `declaration` measurement, the `metadata` key
   contract. Depends on 1. **Its dependency on 4 is optional** — the measurement can be written by
   TAXONOMY pre-merge — and decoupling it removes the only blocking dependency on the largest piece.
6. **Whole-file diarization** — the raw-vs-enhanced pilot first, then the extend driver, then SPEECH's
   diarize step becoming a read. **Not independent of 5**: `extra_speaker` needs the declaration's
   single-target claim.
7. **The branch contract, per branch** — SPEECH first (closest to it), then AIRWAY (which also closes
   the `_windows_covering` gap), then VOICE (which needs a proposer before it has a subject), then
   DDK (which needs a node). Depends on 5, and on 6 for `extra_speaker`.
8. **Boundary reconciliation** — blocked on a parameter-free definition, per (c).

**One ordering inversion is accepted and budgeted.** Piece 8 changes every span extent, hence which
spans clear 0.96 s — the whole basis of piece 3 — and which need refining in piece 7. Since 8 is
blocked on a definition that does not exist, 3 and 7 proceed first and **are re-measured after 8
lands**; that re-measurement is part of 8's cost, not a surprise.

---

## Explicitly unresolved

**Every fitted threshold awaits ground truth.** `spans.k_db`, `airway.contest_labels`, the speech
quality floors, the deviation thresholds. The corpus is labelled by declaration, not by
verification. Nothing is refit until something has been listened to.

**The boundary reconciliation rule** (c) — no parameter-free definition yet.

**DDK has a contract, which is not the same as being specified.** What DDK concludes about a segment
rate, and what its findings are, is undecided. Until the node exists, recordings with DDK content
flag.

**Whether a deviation is disqualifying**, per task.

**`missing_expected_event` and `extra_speaker` owe ground truth** — recorded as observations, no
discrepancy asserted.

**Promoting the declaration's fields from `metadata` to typed `AudioHints` fields.**

**Whether background spans split around branch-proposed events**, or are merely superseded for that
extent.

**Whether VERDICT should write the durable description as its own entity**, rather than REPORT
assembling it.

**Raw vs enhanced for diarization** — pending the pilot.

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
decision; they are written with no `family` (`preprocess.py:1570-1578`) and `airway.py:198` selects
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
