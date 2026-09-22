# The branch contract, and the declaration that informs it — 2026-09-13

## What this is

Four stages, one contract for every branch, and a declaration read from the BIDS sidecars
rather than guessed from a filename.

Three things forced this design at once. The branches did not share a job description: SPEECH
proposed its own spans, VOICE read spans nothing wrote, AIRWAY read the general set including
the gaps. Hints are extractable and inert — the packaged map is `null` and the only real one raises
on load. And the spans everything reads are permissive by construction, which is correct for a
recall-first router and wrong for anything that has to say what happened.

This spec settles what a branch is *for*. It does not build one.

**Read as of 2026-09-22.** The graph moved under this document between its writing and that date,
and this pass re-read every claim in it against the tree. Four changes carry through the whole
text and are worth knowing before the first section: **DDK is not a branch** — it is SPEECH's
syllable-train instrument, and `BRANCHES` is three; **a branch writes by `propose` only** and
carries no outcome, reporting findings that `write_findings` turns into `deviate` and `contest`
assertions; **every gate lives in VERDICT**, resolved family → group → default, key by key; and
**VOICE was rebuilt** onto the amplitude spans this document said it should read. Where a passage
describes the older shape, it now says which half of it still holds.

---

## The pipeline shape

```
PREPROCESS   Shared derivatives, extend work included. Content-first, hint-blind.
             Emits a clean span set: defensible, not exhaustive.

SCREEN       TAXONOMY ⊕ ROUTING, merged. Consolidates the classifier evidence, resolves
             the declaration, applies the ruleset and the hints, emits the routing decision.

BRANCHES     Receive every upstream output, their route, and the declaration. Annotate
             PREPROCESS's spans (label, contest, refine, trim) and propose the ones it
             missed as `family: "<branch>"` spans. Emit typed deviations. Report.

VERDICT      Admit/reject, decide the declared task's conformance against the task group's
             gates, flag for human review, durable description.
```

Of the branch line, `propose`, `contest` and the typed deviations are built; `label`, `refine` and
`trim` are not. The branch reports and does not conclude: VERDICT decides.

### The nodes this does not place

`GRAPH_ORDER` has ten entries (`vocabulary.py:16-27`) and **contains neither REPORT nor FIGURE** —
`REPORT_NODE` is defined at `run.py:49` and concatenated onto the outcome list at `:432`; FIGURE's
name exists only at `figure.py:52`. Of the non-branch nodes it does contain: ADMIT runs before
PREPROCESS; **QUALITY is not terminal** — REDACT runs after it (`run.py:301-309`); and **REDACT is
its own `GRAPH_ORDER` entry gated on SPEECH's result** (`run.py:302`), not a step of SPEECH. REPORT
and FIGURE re-read a finished store. All keep their current positions.

**QUALITY is the sharp case, and it decides the scope of the verbs.** QUALITY writes `contest`
assertions over PREPROCESS's clip spans (`quality.py:273-290`, via `CONTEST_VERB` at `:46`) — one of
the contract's own verbs, performed by a node that is not a branch. So **the verbs are store-wide,
not branch-only**:
any node may label, contest, refine, trim or propose, and the contract below says how, regardless of
who performs it. What is branch-specific is the *family* the findings are written under and the
question the node reports on.

**Since 2026-09-21 a branch no longer concludes at all**, which moves the second half of that
sentence. A branch returns a `BranchReport` carrying no outcome and no conformance
(`vocabulary.py:201-228`), and VERDICT applies the declared task group's gates to the branch's own
`measure` findings to reach one (`verdict.py:10-14`, `gate_conformance` at `:433`). Everything
below that places a threshold, a gate or an `Outcome` inside a branch is describing the graph
before that change; where the difference matters the passage says so.

### The merge reverses a decision made the same day

`specs/20260912-ruleset-in-pipeline/design.md:379-391` — landed hours before this spec — kept
TAXONOMY and ROUTING as two nodes, on the argument that TAXONOMY emits measurements and ROUTING
emits decisions, and that the L1-measures/L2-decides boundary is worth a node boundary. That
document still records the merge as unimplemented and the two nodes as standing (`:390-391`), which
the tree confirms: `GRAPH_ORDER` carries both names and no `SCREEN`.

The owner reversed it: **the middle stage's product is the routing decision.** The consolidation
exists to serve it. A node whose output nothing acts on is a seam, not a layer.

That premise was checked against the tree and holds more strongly than earlier drafts of this spec
claimed. **`consensus_taxonomy` has exactly one production consumer: routing's own reduction**
(`features.py:756`). The only other reader is `extend.py:326 rewrite_consensus_taxonomy`, which
recomputes it rather than consuming it. `figure.py:619` reads `<classifier>_label_summary` — a
*different* TAXONOMY measurement — and never the consensus. `report.py:510,528` reads
PREPROCESS's `<classifier>_windows` and neither the consensus nor the summaries.

**One argument previously made here is withdrawn as unsound.** That `evaluate_live_routes` serialises
the whole store (`live_evidence.py:135-153`, called from `:173-179`) proves too much — VERDICT and
REPORT read the whole store too, and nobody proposes merging them — and it rests on stage 1
scaffolding already scheduled for replacement by `reduce_records` over `required_sources(ruleset)`
(`live_evidence.py:58`). The merge stands on one product, one stage, and the single verified
consumer.

### What the merge costs, and what must be budgeted

**Two node names retire, and node name is a join key.** `GRAPH_ORDER` contains `"TAXONOMY"` and
`"routing"` (`vocabulary.py:16-27`); verdict entities, `run.json`, the figure and the report all join
on it, and every finished store carries activities with `node: "TAXONOMY"`.

**`"routing"` is load-bearing beyond `GRAPH_ORDER`.** `vocabulary.py:148` defines
`_ROUTING = "routing"`, and it is the node the fold attributes five of its flag grounds to —
"routing failed" (`:601-609`), the bad map values (`:610-612`), `unexplained` (`:615-616`),
`unreadable` (`:617-618`) and the critical absence (`:619-625`). `verdict.py:163-181` orders the
node verdicts it reads by `GRAPH_ORDER` and puts a name outside it last. Dropping `"routing"` from
the vocabulary stops the "routing failed" FLAG firing on pre-merge stores, silently — the same
failure shape as the demoted consolidation flag above.

Stage 2's own audit states the governing rule — **the writer's vocabulary may shrink, the reader's
may not** — and this is the same shape as the `kind` prov-type fix of 2026-09-13. So: SCREEN is the
only name written; `"TAXONOMY"` and `"routing"` remain readable, and a reader encountering either
folds it as SCREEN's predecessor rather than raising. A regression test parametrised over the
readable set, as `prov_store_test.py::test_every_readable_entity_type_round_trips` is for prov types.

**An ordering constraint demotes back to intra-node.** Stage 2 promoted the
`voice.glide` / `voice.chant` → `yamnet_label_summary` dependency from step ordering inside TAXONOMY
to an edge in `GRAPH_ORDER` (`taxonomy.md:85`, ruleset `design.md:395`); the two gates are still
configured (`default.yaml:429`). Merging demotes it
again. **The intra-node ordering test must return with the merge** — without it VOICE silently stops
routing, which has already happened once.

**Two verdicts become one, and it must fold both conclusions.** TAXONOMY and ROUTING each write one;
`store.md` requires a node's verdict be attributed to its last step. An earlier revision demoted the
consolidation's conclusion to a `detail` field — **that deletes a live FLAG ground.**
`taxonomy.py:350` writes `Outcome.FLAG`, with the string at `:351`: "no per-span classifier produced scores; there was
nothing to consolidate"; `vocabulary.py:682` folds on `outcome` and reads no `detail`; and
`routing.py:317` writes `PASS` unconditionally. Demoting it would silently stop that flag firing.

**SCREEN's verdict flags if either conclusion flags**, and is attributed to the routing step as the
stage's last.

**Note the compounding with rule (a).** Excluding covering-window scores makes "no per-span
classifier produced scores" *more* common, so rule (a) grows the population this flag fires on. Both
pieces must name the interaction.

**Several documents assert the split and must be updated**, not only the one cited above:
`taxonomy.md:27-29` ("a node that both measures content and decides what runs on it cannot be checked
against itself"), `routing.md:16` ("It measures nothing and classifies nothing"), and `dag.md:53`,
`:73-75`, `:89`, `:102`, `:154`, `:164`, which name TAXONOMY and `routing` as separate nodes in the
runner's order, the diagram and the skip rules.

---

## Span cleanliness — three parts

A span shorter than a classifier's window cannot be classified by its own content. That is the
defect, and it is not a threshold problem: a 200 ms cough is a real event, so proposing it is right.
What is wrong is the label attached to it afterwards.

### (a) A label belongs to a span only if a window lies inside it

**This applies to YAMNet's per-span measurements and not to HeAR's, and the difference is
load-bearing.**

`_span_yamnet` gives a span shorter than the native window the overlap-weighted scores of the
whole-file windows that *cover* it (`_covering_window_attribution`, `preprocess.py:365-397`), marked
`attribution: "covering_windows"` with
`covering_windows_n` and `covering_seconds` (`preprocess.py:2465-2467`); a long span gets
`attribution: "native"` (`preprocess.py:2509`). A covering-window label is a statement about up to a
second of audio attributed to a fifth of it.

`_span_hear` does something different. `span_hear_input` places a span shorter than
`HEAR_WINDOW_SECONDS` in a silent 2 s buffer, "so its only detector result describes the span
itself" (`audio/tasks/health_acoustics/hear.py:426-443`); a longer span is passed through (`:444-446`)
and its native windows are mapped back by `hear_window_extent` (`:449`). HeAR's short-span label is
about the span.

**The rule: a covering-window label is recorded, and is not eligible as evidence.** The flag already
exists and is already written; nothing filters on it today.

**Which readers the rule applies to**, stated exactly because this is not a cosmetic change:

| reader | what changes |
| --- | --- |
| `consensus_taxonomy` rows (`taxonomy.py:191`) | a covering-window score does not contribute a row or a `peak_by_classifier` entry |
| `RecordingFeatures` span-label stats (`features.py:898 _label_span_statistics`, via `_absorb_span_window` at `:358`) | covering-window scores excluded from `span_label_stats` and `span_label_set_stats` |
| figure, report | rendered as attributed-from-outside, not as the span's own label |

**AIRWAY's own reads changed after this was written, and the conclusion does not.** AIRWAY read
`span_hear` alone when this section was drafted; `classifier_windows` (`airway.py:213-222`) now
returns every `span_hear` **and** `span_yamnet` measurement together, so a covering-window YAMNet
label is inside what AIRWAY sees. Rule (a) therefore reaches AIRWAY's own reads as well as the
routing gate — the earlier "AIRWAY's reads are untouched" scoping is gone. The two senses still
must not be conflated: one is what AIRWAY sees, the other is whether it is asked to look.

**So rule (a) moves routing.** `airway.cough`'s feature is
`span_label_set_stat` over `yamnet.cough_labels.peak_over_floor_db_max` (`default.yaml:462-463`),
built by `_label_span_statistics` from exactly these per-span measurements with no attribution check,
and coughs are usually shorter than 0.96 s. There is no scoping that keeps the rule's benefit without
changing the router: `consensus_taxonomy`'s only production consumer is
`features.py:756` — routing's own reduction. An earlier revision offered "scope away from
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

**The count itself needs a fix first.** `features.py:1123-1135` appends every live span to
`live_spans` with no `family` filter, and `_span_statistics`'s (`:977`) `all.*` bucket includes them.
That is
harmless in-run, because routing precedes the branches and no family span exists yet — but **not in
an offline recompute over finished stores** (`scripts/analyze_routing_evidence.py:158`), which is how
a gate count would be produced. Branch-proposed spans would enter the router's own statistics. The
family filter lands before the count does, and is still unwritten.

**The second covering-window mechanism this section deferred has gone with AIRWAY's rewrite.**
`_windows_covering(store, "yamnet", hear_extent)` took every **whole-file** `yamnet_window`
measurement merely overlapping a HeAR window and let its labels confirm or contest. No such helper
is in `airway.py` any more, and the confirm/contest loop it fed went with it; what AIRWAY reads now
is `classifier_windows` (`:213-222`), which is per-span measurements only. **Rule (a) stays scoped
to `span_yamnet` measurements**, and that scoping now leaves nothing of AIRWAY's outside it.

HeAR's isolation has its own caveat, and it is a different one: a 200 ms event centred in 1.8 s of
digital silence is not what the model saw in training. Whether that distorts its scores is
**unmeasured**. It is not addressed here and must not be conflated with the covering-window problem.

### (b) Gap spans are background, never events

Gap spans are the complement of the **kept** spans — `covered` is built from `combined`, the four
sources' surviving proposals (`preprocess.py:2007`) — not the complement of everything proposed. A
gap shorter than `min_duration_ms` is not emitted at all (`:2011`, `:2014`). They carry
`measure: "gap"` and `merged_proposals: 0` and are written with no `family`
(`preprocess.py:2016-2031`), so they are selected by `candidate_spans`'s
`family in (None, "airway")` filter (`airway.py:196-210`) as ordinary AIRWAY evidence.

Type them as background. They stay measured and visible; they stop being eligible to be something
that happened.

**One interaction to resolve later.** Once branches propose spans, a branch-proposed event can sit
inside an extent typed background. The smallest defensible rule: a background span is a statement
about what PREPROCESS found, not a claim of emptiness, so a branch-proposed span inside one neither
retires nor contradicts it — the background typing is superseded for that extent and the reconciling
record is the branch's own span. Whether background spans should instead be split around
branch-proposed events is **unresolved**.

### (c) Boundaries reconciled, not first-writer-wins

`_novel` (`preprocess.py:1906-1929`, written onto the span at `:1997-1999`) appends a record to
`corroborated_by` on every span a later
candidate overlaps, and keeps the earlier proposer's extent unchanged. Four sources agreeing on an
event is precisely when its boundary can be stated well, and that is the moment the current code
discards the information.

**This piece has no rule yet, and that is the blocker.** Union, intersection, proposer-weighted
median and highest-peak-corroborator are all defensible, and the ground-truth rule below forbids
choosing between them by measurement on this corpus. A reconciliation rule must therefore be a
**parameter-free definition** justified by what it means, not by what it scores — or the piece waits
until something has been listened to. **Unresolved**, deliberately, and it is why (c) is sequenced
last.

The one part of (c) that is not blocked is a plain bug, and it is still there: continuity spans are
`wasDerivedFrom` the
energy envelope rather than the continuity trace (`preprocess.py:1970` — `state["envelope_id"]`,
carried to the edge at `:2003`),
even though the trace is in the activity's `reads` (`:1879-1880`). The provenance edge names the
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
`specs/20260817-triage-workflow-dag/runs/b2ai-v2/make_hints.py` extracts the BIDS `task-` token from
the stem (`task_token` at `:323`) and emits `may_contain`
plus `metadata.{task_token, speech_type, task_id, registry}` (`:381-384`). Every recording carries
the protocol's
own `task_name` and `speech_type` beside it. `stimulus_text` — which `AudioHints.expected_speech`
exists to hold — is never carried at all.

**The only populated map raises on load.** That campaign's `override.yaml:261` is keyed
`routing.hint_kind_map`. Stage 2 renamed the packaged key to `hint_branch_map`
(`default.yaml:150`), and `_merge` refuses any key the packaged config lacks (`config.py:162`).

**Its values would all be rejected even after renaming.** `_map_tags` casefolds the map's *keys*
against the declared tags (`routing.py:115`) but tests the *value* against `BRANCHES` unchanged
(`:121`). The override's values are lowercase kinds — `cough: airway`
(`specs/20260817-triage-workflow-dag/runs/b2ai-v2/override.yaml:262`) — and `BRANCHES` is
`("AIRWAY", "SPEECH", "VOICE")` (`vocabulary.py:33`).

The mechanism matters for the fix. `bad_map_values` is keyed **by declared tag on the recording**,
not by map entry (`routing.py:119-126`, recorded at `:278`), so a map entry no file declares never
surfaces. The conclusion still holds — checked against every rule in `make_hints.py:136-182`, every
corpus recording declares at least one tag the override maps — so **every file in the run** would
take an `Outcome.FLAG` against ROUTING (`vocabulary.py:610-612`, the ground at `:150`). This is a
**value-case fix**; `routing.md:76-78` documents testing the value unchanged as deliberate, and
that test is not the defect.

**A fourth was found by measurement rather than by reading, and it has since been closed.** With
every tag unmapped, no decision carried a claim, so `_hint_claims` returned `{}` rather than `None`
and `FileVerdict.hints` read `found_unclaimed` / `no_claim` for every branch — including
`AIRWAY: found_unclaimed` on runs whose `may_contain` declared `[cough, airway]`. It rendered:
`summary.json`'s `recording.declared_hints` (`report.py:1173`) and the PDF header (`report.py:1543`).

**What closed it is that a declaration no longer has to come through the hint map.** A branch
decision's `declared` is `branch in by_family or bool(hint_tags)` (`routing.py:257`, written at
`:272`), and `by_family` is the ruleset's reading of the recording's own BIDS stem
(`routing.py:229`). `_hint_claims` reads exactly that field (`verdict.py:330`), so a cough
recording's `AIRWAY` claim survives an inert map. What remains true is the shape of the remaining
hole: `_hint_claims` returns `None` — and the fold raises `UNREAD_DECLARATION` (`vocabulary.py:160-163`,
fired at `:613-614`) — only when a declaration existed and **no decision survived at all**
(`verdict.py:328-329`). Recorded, with the vocabulary decision it was blocked on, at
[`../20260817-triage-workflow-dag/verdict.md`](../20260817-triage-workflow-dag/verdict.md) and
measured as of 2026-09-15 in
[`../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md`](../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md)
§ B, which predates the fix. The first three items above stand and are what a hint-blind run costs.

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

**The mismatch this section recorded has resolved itself, in the direction it predicted.** Checked
against the tree 2026-09-15, `nodes/voice.py` read `hint.metadata["population"]` and
`hint.metadata["task"]`, neither of them in the seven keys above, so a hint written to this contract
would have reached neither consumer. VOICE's rewrite removed both reads. **Exactly one
`hint.metadata` key is read anywhere in `src/senselab` today** — ROUTING's `speech_type`
(`routing.py:45`, read at `:93`) — and it is in the seven. `voice.f0_range_by_population` and
`voice.task_duration_ranges` are still null (`default.yaml:230`, `:232`) and nothing reads them.

So the contract is now the only claimant on those keys, and the reason it should stay the contract is
unchanged: `task_name` and `acoustic_task_name` are the sidecar's own names and separate the
per-recording grain from the per-family one, which is the distinction a single `task` cannot carry,
and `population` is not a field the sidecars hold at all. V7's and V2's reading of the mismatch from
the branch's side is in
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
*Unresolved*; the branch it described no longer exists in that form.

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

- **The value is not recorded at all.** `evaluate_gate` returns `gate_outcome_of(gate_value(features,
  gate), gate)` (`routing_analysis/ruleset.py:426-436`), and `gate_outcome_of` (`:385-398`) reduces
  the number it was handed to a three-member enum;
  `RouteEvaluation.gate_outcomes` is `Mapping[str, GateOutcome]` (`:199`, filled at `:536`) and
  `route_attributes` serialises exactly that and no number (`live_evidence.py:183-208`, the key at
  `:202`). So a rule-written label could not today cite the number that produced it. Registered
  against the ruleset at [`../20260817-triage-workflow-dag/routing.md`](../20260817-triage-workflow-dag/routing.md)
  § *Open derivations*.
- **The span identity is discarded one step before the gate sees it.** `live_spans` rows carry
  `"id"` (`routing_analysis/features.py:1128`) and the reduction
  `span_longest_s[measure] = max(durations)` (`:1188`) keeps only the scalar. A `Gate` is "one
  threshold rule over one number" (`ruleset.py:85`) with no span in it, so **"the span whose
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

**A declaration may add and inform. It may never suppress.** This is the rule the code implements:
`will_run = not critical and (by_ruleset or by_declaration or by_default)` (`routing.py:260`), a
union in which a declaration only adds. The declaration is the recording's own task,
read off its BIDS stem and resolved through `taxonomy.ruleset.reference_family_set`
(`branches.py:953-962`), with a hint tag
as an optional second source that "adds routes and removes none" (`default.yaml:143-148`); see
[`../20260817-triage-workflow-dag/routing.md`](../20260817-triage-workflow-dag/routing.md).
A branch the content routed cannot be un-routed by a declaration that disagrees with it.

**Content the task did not ask for is content, not error.** A breathing recording that carries speech
routes SPEECH on the ruleset's evidence and has that speech recorded.

---

## The branch contract

> A branch receives the shared derivative state, its route, and the declaration. It **labels** what it
> recognises, **contests** what was proposed but is not there, **refines** a boundary the proposer got
> wrong, and **trims** a span to the extent that serves the task — all four as assertions about spans
> that keep their own identity. It **proposes** what the proposer missed but the declaration says to
> expect, and that alone mints a `family: "<branch>"` span. It emits typed deviations and reports
> what it read.

The last clause read *"concludes on its own question"* when this was written. It does not: a branch
returns a `BranchReport` carrying no outcome and no conformance, and VERDICT decides
(`vocabulary.py:201-228`).

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

**VOICE was where this was visible, and the inconsistency has since been closed by taking exactly
this decision.** VOICE was routed by a measurement on `amplitude` spans and then failed for want of
a `phonation` span: `voice.sustained`'s feature is `[span_longest, amplitude]` — the longest live
amplitude span in seconds (`default.yaml:446`) — while its selector admitted only the `phonation`
family, and the amplitude spans the route was decided on sat live in the store, unexamined.
Measured on 13 real b2ai
recordings on 2026-09-15, **6 of 6 VOICE-routed recordings returned `Outcome.FAIL`** there,
including a
maximum-phonation-time recording whose `voice.sustained` read **15.89 s** and a glide at **12.27 s**
([`../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md`](../20260817-triage-workflow-dag/benchmarks/hints-and-routing-2026-09-15.md)
§ D). VOICE now reads those same amplitude spans as its subject (`voice.py:8-11`), so the
measurement is a record of the defect rather than of the branch.

**And the second half of the owner's sentence carries as much as the first.** A branch must also do
its best on *other tasks assigned to it* — a recording routed to it whose declared family is none of
its own. That is the owner's standing rule for branches, already stated for AIRWAY in the owner's
example quoted at the head of this section, and the decision generalises it: **a branch receiving a
task outside its declared families does its best to find, mark, refine or refute evidence of its own
kind, rather than failing for want of a declared subject.** For VOICE that is the common case rather
than the edge case — it routed 22,277 recordings against 8,306 declaring a voice family.

**That generalisation is now the shape of every branch.** `dispatch` (`branches.py:1048-1078`)
picks between two entry points from the declared task family: `align_<branch>` for a task of the
branch's own kind, `detect_<branch>` for a task of any other, which "evaluates nothing" and returns
`UNDETERMINED`. No branch fails for want of a declared subject, because the out-of-family mode is a
first-class entry point rather than an error path.

**The second minting this section called out has gone.** VOICE minted a *second* span from an input
span, re-keyed by period-aligned onset and carrying `onset_kind` — the re-minting the annotating
verbs were to replace with a `refine` assertion, and the reason `_spans_of_family` had to split one
family into two populations. VOICE proposes once, into `voice`, and the split went with it.
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
§ *The state of this branch* states the old shape; what the decision added is that improving those
spans **is VOICE's work**, not a precondition it is entitled to wait for, and that is what shipped.

### What each verb writes — four annotate, one mints

**This reverses the previous revision of this spec**, which had `refine` and `trim` mint new span
entities and invalidate or shadow the original. That was wrong, and the reason is that the store has
**two** addressing mechanisms and re-minting breaks both.

**Per-span measurements are addressed by the `span_id` attribute, not by traversal.**
PREPROCESS stamps `span_id` on every per-span window (`preprocess.py:350`), and every reader keys on
it: `airway.py:430-454 decided_label_sets`, `features.py:358` and `:898-963`,
`taxonomy.py:110-122` and `figure.py:441-448`. **A re-minted span carries zero labels under all
of them, silently**, while `features.py:1117-1118` drops the invalidated original. So a `refine`
would *delete* the span's classifier evidence from `span_label_set_stats` — including the feature
`airway.cough` is built on. "Readers traverse `wasDerivedFrom`" is not a contract that can be
adopted: it is four rewrites plus a reverse index `ProvStore` does not expose, since `derived_from`
resolves one way only.

**And carrying a measurement forward is wrong on its own terms.** `attribution: "native"`
(`preprocess.py:2509`) and `isolated_span: True` (`:355`) are claims about how the **old** extent was
fed to a model. After a narrowing refine, a traversing reader would get a label flagged `native` for
an extent containing no native window — rule (a)'s premise inverted by rule (a)'s own spec.

**So four of the five verbs annotate. They never re-mint a span. The span keeps its id, its
`family`, its measurements and its liveness; nothing is invalidated and nothing is orphaned.**

**The contract adopts the store's existing spellings rather than inventing new ones.** An earlier
revision renamed `label`→`mark` and `contest`→`refute`; that is withdrawn, and the rename is not
merely unnecessary but unsafe. `verb: "label"` is SPEECH's PII mark —
`speech.py:1981` writes `{"verb": "label", "label": "pii", ...}` and `redact.py:259` selects on
exactly that pair to decide what gets redacted. Renaming `label` stops PII redaction firing,
silently. The readers of `verb` are `redact.py:259`, `report.py:208`, `report.py:1209` and
`extend.py:517`; it appears in neither `verdict.py` nor `figure.py`.

| verb | `verb:` value | writes | carries | mints |
| --- | --- | --- | --- | --- |
| label | `"label"` | `assertion`, `wasDerivedFrom` the span | what the span carries | no |
| contest | `"contest"` | `assertion`, `wasDerivedFrom` the span | that it does not carry what was proposed | no |
| refine | `"refine"` | `assertion`, `wasDerivedFrom` the span | `corrected_extent: [start, end]` and/or `corrected_attributes: {key: value}`, at least one | no |
| trim | `"trim"` | `assertion`, `wasDerivedFrom` the span | `task_extent: [start, end]`, plus the `off_task_extent` finding | no |
| propose | — | `span`, `family: "<branch>"`, `wasDerivedFrom` its evidence | a region PREPROCESS did not find | **yes** |

`label` and `contest` already exist and keep their meanings. **`refine` and `trim` are
written by nothing in the graph, and `propose` is now the only verb a branch has.** Swept over
`nodes/` and `extend.py`, the full inventory of `verb:` values any node writes is:

| verb written | by | site |
| --- | --- | --- |
| `deviate` | AIRWAY, SPEECH, VOICE | `branches.py:478-481`, from every `deviation()` finding |
| `contest` | AIRWAY, SPEECH, VOICE, QUALITY | `branches.py:478-481`, from every `contest()` finding — `airway.py:1089`, `speech.py:1358`, `voice.py:802`; `quality.py:279` via `CONTEST_VERB` (`:46`) |
| `label` | SPEECH | `speech.py:1981`, carrying `label: "pii"` |
| `measure` | PREPROCESS | `preprocess.py:2324`, `:2331`, `:2349` |
| `withdraw` | PREPROCESS | `preprocess.py:601` via `WITHDRAW_VERB` |
| `exempt` | REDACT | `redact.py:951` |
| `refine`, `trim` | — | **nothing writes them** |

**The branch verb set collapsed rather than grew, and that is the change this section most needs to
record.** Both entry points of every branch "write by `propose` only" (`branches.py:6`,
`speech.py:6`, `voice.py:8`): a branch mints spans in its own family through `propose_span`
(`branches.py:378-407`), which writes no `verb` at all, and everything else it has to say is a
`Finding` of one of four kinds — `deviation`, `count`, `measure`, `contest` (`FINDING_KINDS`,
`branches.py:112`). `write_findings` (`:425-497`) turns a `deviation` into a `verb: "deviate"`
assertion keyed `deviation_type`, a `contest` into a `verb: "contest"` assertion keyed `claim`, a
`measure` into its own measurement, and folds every `count` into one `counts` measurement carrying
`found` beside `declared`.

So `label` is no longer a branch verb — AIRWAY's `label`, `confirm`, `abstain` and `flag` assertions
all went with its rewrite, and SPEECH's PII mark is the only `label` left in the graph. `attribute`
is gone too; SPEECH writes no per-word assertion. `deviate` is the verb the contract's typed
deviations actually carry, and the contract's table below does not name it.

**Of the contract's five verbs, `propose` and `contest` are live, `label` survives only as
REDACT's input, and `refine` and `trim` remain unwritten.**

**The corrected extent goes in a named attribute, never in `assertion.extent`.** A `contest`
assertion is written with the contested span's own extent (`branches.py:260-272`, the write at
`:480-482`), as is QUALITY's (`quality.py:275-277`), and `report.py:1176`'s
`_timing` relies on that convention, so a `refine` that moved `assertion.extent` would be read as
timing the span itself. Hence `corrected_extent` and `task_extent` as attributes, with
`assertion.extent` continuing to name the span being annotated.

**The precedent for reading these already exists.** `figure.py:572-590` consumes SQUIM *assertions*
by walking `store.derived_from(entity.id)` (`:585`) — assertion → span, the one direction the store
resolves. That is exactly the traversal the annotating verbs need, already in production.

`refine` and `trim` still emit their deviations; they simply do not rewrite the store's spans. A
corrected extent is a claim beside the original — attributable, reversible, and never in competition
with it for a reader keying on `span_id`.

**But SPEECH's and VOICE's *assertions* still reach no reader.** `report.py:1202` drops
every assertion whose branch is not AIRWAY, and `figure.py:583` reads assertions only where
`name == "squim"`. Their **spans** are read — `report.py:1205-1215` has explicit arms for them, and
`propose` mints spans — so it is precisely the annotating verbs whose output is invisible, and that
now includes every `deviate` assertion SPEECH and VOICE write.
Widening REPORT's assertion read is a piece of work in its own right, listed below, and not something
the existing readers absorb for free.

**This is the store's own idiom, not an invention.** `corroborated_by` already records rival
proposals as an attribute without minting entities, and `withdraw_contradicted_clips` with
`_retire_quality_findings` is the module's one invalidation cascade *precisely because* minting forces
one. Annotation needs no cascade, so there is none to enumerate.

It also makes `trim` non-destructive by construction, which is what the no-suppression rule wanted
and what the previous revision's "retires nothing" could not deliver while still double-counting in
`features.py:1123-1135`.

**AIRWAY now sees its own family and no other branch's.** `candidate_spans` selects
`family in (None, "airway")` (`airway.py:196-210`), a widening of the `family is None` filter this
section was written against. Anything written as `family: "speech"` or `family: "voice"` is still
outside it, and it is a constraint the AIRWAY piece must address rather than a defect in the
contract.

**Nothing measures a proposed span.** No node runs a model at branch time: AIRWAY's docstring states
that both its modes read PREPROCESS's own derivatives and run nothing (`airway.py:1-9`), and SPEECH
"runs no ASR and no diarizer" (`speech.py:8`). **A proposed span therefore carries its branch's
own evidence and no per-span classifier measurement at all**; adding a branch-time classifier pass is
out of scope for this design and is listed as unresolved.

Even if one were added, **YAMNet could not classify a short proposed span**: `span_yamnet_input`
raises `SpanTooShortForYAMNet` below 0.96 s (`yamnet.py:257-258`), so rule (a)'s own motivating case —
a 200 ms cough — can never acquire a native YAMNet window, whatever proposes it. HeAR has the
capability, via the silent buffer (`health_acoustics/hear.py:441-443`), but no mechanism invokes it
at branch time.

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

`assertion.extent` continues to name the span being annotated, for the reason given above: a
`contest` assertion is written with the contested span's own extent (`branches.py:260-272`) and
`report.py:1176`'s
`_timing` reads it as that span's timing. A metadata refinement gets a named attribute for exactly
the same reason an extent correction does.

**Presence is the discriminator, for a reader and for code.** `"corrected_extent" in attributes` says
an extent is being corrected; `"corrected_attributes" in attributes` says metadata is; both present
is one act of improvement that did both, which is the ordinary case when a boundary fix changes what
the span should be called. A `refine` carrying neither is malformed — a testable condition rather
than a convention, and the one shape a reader may reject.

**Why a nested mapping rather than a flat `corrected_<key>` per field.** Because the store has
readers that key on an assertion's **top-level** attribute names without testing `verb` at all:
`figure.py:583` and `routing_analysis/features.py:1140` both select assertions on
`attributes.get("name") == "squim"`, and `report.py:1204` builds an entity's rendered description
from `attributes.get("name") or attributes.get("family")` for every branch entity it lists. A
corrected value written at the top level under the span's own key — `name`, `family`, `label` — is
therefore one field name away from being read as the assertion's own property. Nesting puts every
corrected value out of reach of every existing top-level test, and it lets one assertion correct more
than one field without a reader having to diff the assertion against the span to work out which keys
were meant.

**The prior value is not copied into the assertion, and does not need to be.** For the extent,
`assertion.extent` already carries it. For the metadata, the span carries it: nothing rewrites a
span's attributes, the module's only `was_invalidated_by` call is `extend.py:322`, and `quality.py:11`
states the rule outright — *"It withdraws no span: the store is
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
`wasDerivedFrom` every live non-SPEECH span it overlaps (`speech.py:1872-1881`, over the
`prior_spans` set built at `:1509-1511`), and `ProvStore.derived_from` returns a **list**
(`utils/prov_store.py:530-532`),
so many-to-one derivation is a capability of the store and not a tolerated irregularity. The one
production reader that walks the relation iterates it — `for span_id in store.derived_from(entity.id)`
at `figure.py:585` — so an aggregate does not break it.

**Whether the spans it aggregates are left untouched: yes, and it is a property of the code, not only
of the design.** Nothing under `nodes/` invalidates anything; the module's single `was_invalidated_by`
call is `extend.py:322`, inside the clip-contest supersession. An aggregate neither retires nor
contradicts the spans beneath it — the same shape as § *(b) Gap spans are background, never events*,
where a branch-proposed span inside a background extent supersedes the typing for that extent and
retires nothing.

**None of the three is undetermined, so none is recorded as owed.** The adjacent question that *is*
open — whose family a **ruleset-written label** puts a span in, and therefore whether the labelled
span is refinable by the branch at all — is untouched by this decision and stays where it was, at
[`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *The two owner decisions of 2026-09-15 leave the minting rule alone and open one question*.

#### SPEECH's `attribute` writes became `propose` — landed

`speech.py` wrote one `verb: "attribute"` assertion per word when this was decided: the
highest-volume verb in the store and not one of the contract's five. It is gone. SPEECH writes no
per-word assertion at all, and the claim it carried is now on the span: `attributed_to` is computed
from the run's words and stamped on the proposed span, with `nontarget` derived beside it
(`speech.py:1869-1884`). The destination the decision named as `propose` — *one span per contiguous
run attributed to the same speaker, minted `family: "speech"`* — is what the code does
(`_SpeakerRun` at `speech.py:609`, `MINT` at `:646-647`).

The `refine` half was never needed: because the span is minted carrying the attribution rather than
having it added afterwards, there is no prior value to correct. What stays settled is that
`attribute` has no standing as a branch verb. Recorded against the branch at
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
word runs into its own spans, and mints them with `family: "speech"` (`MINT` at `speech.py:646-647`,
the runs at `:1865-1884`). It is
the working instance of the contract. It does not *ignore* the general set, and an earlier revision
said so: it `used`s every live non-SPEECH span (`speech.py:1509-1511`, the edges at `:1532-1533`) and
makes each of its own spans `wasDerivedFrom` the ones it overlaps (`:1872-1881`) — which is the
provenance shape § *An aggregated span is a `propose`* generalises.

**VOICE was designed to and could not; it has since been rebuilt, and the fix taken was this
section's.** Its subject was every live span whose `family` was `phonation`, nothing reachable
proposed one, and the branch returned `Outcome.FAIL` on every recording. That is no longer the
shape. VOICE's subject is now *"PREPROCESS's `amplitude` spans qualified by `phonation_tracks` and
`continuity_trace`, over which VOICE mints its own `family: "voice"` spans. It waits for no
`phonation` span and edits none"* (`voice.py:8-11`). `_PHONATION_FAMILY` is gone, and so is the
no-span return.

So **the branch became the proposer, with its output family as its own**, which is what this section
called the whole fix and what the contract wants. The alternative route the owner decisions of
2026-09-15 opened — a ruleset-written phonation label that VOICE `refine`s — was not the one taken:
no rule writes a span label today, and VOICE reads the amplitude spans directly. The settled flow at
[`../20260817-triage-workflow-dag/branch-voice.md`](../20260817-triage-workflow-dag/branch-voice.md)
§ *The subject is the spans the ruleset labelled, and VOICE refines them* describes a source the code
does not use; the amplitude-span reading beneath it is what shipped.

**The family rename landed, and it cost REPORT less than this section budgeted for.** `phonation`
is no longer a span family anywhere: `report.py` reads `_spans_of_family(store, "voice")` (`:1229`,
`:778`) and holds no `phonation` family read and no `voice=True`
parameter. `_spans_of_family` (`report.py:299`) now takes one `family: str` and nothing else,
because there is only one population to read. The **historical** half of the rule this section
invoked — *the writer's vocabulary may shrink, the reader's may not* — was **not** honoured here:
a finished store written before the rename carries `phonation` spans that nothing in REPORT will
ever read again. Whether that is acceptable, or whether `_spans_of_family` should read both, is
worth deciding rather than discovering.

`phonation_s` survived as a name and changed owner: it is VOICE's own `_detail` field
(`voice.py:878`, `:910`, declared for the report at `common.py:637`) rather than a REPORT summary
key, and it still names seconds of phonation rather than a family.

The `onset_kind` two-population split went with the rename. `onset_kind` and `offset_kind` are
written nowhere, the second period-aligned minting that wrote them is gone, and nothing needs the
distinction between the detector's spans and VOICE's own — VOICE's are the only ones.

**AIRWAY's five assertion verbs are gone.** It wrote `label`, `confirm`, `contest`, `abstain` and
`flag` when this was written. It now writes by `propose` and by `Finding` alone, like every other
branch; the only assertion verbs its activity generates are `deviate` and `contest`, both through
`write_findings`.

**`contest`'s contract meaning is settled, and AIRWAY now has a criterion.** The contract's
definition is the one in the verb table — *the span does not carry what was proposed* — and
QUALITY's clip contradiction (`quality.py:273-290`) is an instance of it under a completely
different criterion. The two must not be conflated.

**The criterion AIRWAY took is not the one this section proposed, and it is threshold-free by a
different route.** This section proposed *contest a span when no evidence of any kind is found
within its extent*. What `detect_airway` does instead is contest a span the classifier windows
decided a label set for, where the event walk marked no event overlapping it — the reason string is
`no_raw_score_over_p_score_min` (`airway.py:1084-1089`). It contests on a label that did not survive
the raw-score reading, not on absent evidence. `airway.contest_labels`, whose nullity was the
problem this section was solving, is no longer a config key at all, and `_contest_labels` is gone
with it.

**The compounding with rule (a) is still real and still uncounted.** Rule (a) makes covering-window
labels ineligible as evidence and coughs are usually sub-0.96 s, so it moves which spans have a
decided label set at all — which is the input to AIRWAY's contest test. **The same before-and-after
count rule (a) requires is required here.**

**A contest no longer flags the file by itself.** The two chains this section traced both ran
through an `Outcome` a branch no longer returns. AIRWAY returns a `BranchReport` carrying no
outcome, and `contested_n` is a `_detail` field counted off the findings
(`airway.py:1205`) that reaches no flag ground. What a contest can still do is arrive at VERDICT as
a `conformance` of False through the declared family's gates, or as a `mismatch` when the branch
proposed no span and routing routed it (`vocabulary.py:652-655`). Whether AIRWAY's contests *should*
flag is a question this contract reopens rather than answers.

**Background-typed gap spans are not contestable, and the reason is the verb's object, not the
evidence.** A gap span carries real branch-readable evidence — gaps are appended to `span_ids`
(`preprocess.py:2031`) and the per-span classifiers run over them, which is why 143 of 384 consensus
rows traced to gaps in the 2026-09-07 measurement — a figure that predates its own fix, per
Corrections below, and is cited here only for the mechanism it demonstrates. So "no evidence within the extent" is not automatically true of a gap. What is
true is that `contest` carries *that a span does not carry what was proposed*, and **a gap proposes
nothing**, so a contest over one has no object.

AIRWAY's shipped criterion respects that without being told to: `decided_label_sets`
(`airway.py:430-454`) is what a contest needs, and a gap carrying no decided label set never
reaches `contest` (`:1088-1089`). A gap **is** read, though — as an `off_task_extent` deviation
when no proposed span covers it (`branches.py:1535-1554`), which is the "measured and visible" half
of the typing this section asks for, already written.

**DDK is not a branch: it was dissolved into SPEECH, and a DDK recording now routes SPEECH.** This
spec was written while DDK was a fourth branch with no node; that is no longer the shape.
`BRANCHES` is `("AIRWAY", "SPEECH", "VOICE")` in both `vocabulary.py:33` and `branches.py:58`, and
the ten `diadochokinesis-*` families are `SYLLABLE_REPETITION` (`families.py:18-31`), which
`families.py:61` unions into `SPEECH_ELICITING`. So SPEECH holds 31 in-family rows —
`LEXICAL_SPEECH`'s 21 plus those ten (`branches.py:759-856`) — and a recording declaring a DDK task
is in-family for SPEECH, which takes its align mode.

`nodes/ddk.py` does exist, and it is an instrument rather than a branch: its own docstring says so
(`ddk.py:1-11`), and `speech.py:98-105` imports `align_ddk`, `read_ddk`, `syllable_detail` and the
two absence constants from it. `align_speech` serves the ten families through `align_ddk`; every
measurement it writes is a SPEECH measurement.

**Nothing is `SKIPPED` with `NO_NODE` any more.** `NO_NODE` (`run.py:51`) is still written at
`run.py:296`, but only for a branch the runner has no callable for, and the callable map at
`run.py:288-292` covers all three of `BRANCHES` — so the arm is unreachable on the shipped graph.
The claim that DDK routing flags the file no longer holds: there is no DDK route to flag.

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

**That argument was derived over three types and now governs ten, and it does not fit all of
them equally.** `filler`, `stimulus_mismatch`, `repeated_item` and `lexical_content` are ordinary on
spontaneous or read speech and would flag the corpus. `truncation`, `omission` and
`sweep_direction_mismatch` are **not** ordinary — each says the
production departed from what the instruction asked for, and each is a candidate flag ground once
ground truth exists to set a rate against. `verdict.deviation_flags` stays `false`
(`default.yaml:303`) for all ten,
because no ground truth exists for any of them and a rule fitted to none is worse than a rule that
flags none.

A per-type folding policy is the natural extension and is **deliberately not built**: every entry
would read `false` today, which is a mechanism carrying no decision, and this graph does not ship an
unmeasured decision. Build it with the first measured rate, not before.

A deviation is an observation **with an extent**. Ten qualify, declared in
`nodes/branches.py`'s `DEVIATION_TYPES` (`branches.py:115-126`) and enforced at the write by
`write_findings` (`branches.py:453-458`). The branch column is the node whose activity generates
the assertion, read off the emitting call sites:

| type | what it says | branch |
| --- | --- | --- |
| `filler` | a disfluency or non-speech token where the task expected lexical content | SPEECH (`speech.py:1072`) |
| `lexical_content` | a lexical word where the task expected none | VOICE (`voice.py:567`) |
| `off_task_extent` | a region of the recording that does not serve the declared task | AIRWAY (`airway.py:563`, `:812`, and `off_task_findings` at `:821`, `:908`, `:995`) |
| `omission` | an expected token the recording does not realise | SPEECH (`speech.py:1066`), VOICE (`voice.py:476`) |
| `repeat_attempt` | a further carrier where the task expected one production | VOICE (`voice.py:560`) |
| `repeat_reading` | an alignment covering the expected sequence more than once | SPEECH (`speech.py:1025`) |
| `repeated_item` | an item repeated where the task expected each once | SPEECH (`speech.py:1244`) |
| `stimulus_mismatch` | a lexical word that is not the word the stimulus expected | SPEECH (`speech.py:1040`, `:1054`, `:1177`) |
| `sweep_direction_mismatch` | a pitch sweep running against its declared direction | VOICE (`voice.py:731`) |
| `truncation` | a production the recording does not contain the end of | AIRWAY (`airway.py:809`), SPEECH (`speech.py:1016`, and `ddk.py:987` under SPEECH), VOICE (`voice.py:556`, `:742`) |

No row names DDK, because DDK is not a branch: `ddk.py` runs inside SPEECH's activity, so the
deviation it writes is SPEECH's. `off_task_extent`'s helper `off_task` lives in `branches.py:1535`
and is shared by construction, but AIRWAY is the only caller, so the row names AIRWAY alone rather
than "shared".

**`syllable_sequence_mismatch` is no longer declared.** It was the eleventh row until its only
writer went: a threshold on per-position posterior mass had no derivation, so the decode reports
mass per position instead of asserting a mismatch. `ddk_test.py:1155-1157` asserts the absence
deliberately, and `specs/20260817-triage-workflow-dag/ddk-template-decode.md:638` records why. It
was also this section's fourth example of a "not ordinary" deviation, and the sentence above now
carries three.

**This table said "three" until 2026-09-16, while the branches emitted more.** Nothing validated
the name at the write, so the vocabulary and the code drifted apart silently — and the three it
named were a subset, not a mistake, which is why nothing ever failed. `write_findings` now refuses
an undeclared name (`branches.py:453-458`) and `branches_test.py`'s AST sweep refuses a declared
name nobody emits (`branches_test.py:1436-1444`), so the two cannot part again in either direction.

The three named above are the three the folding argument below was actually derived over. The other
seven inherited a justification written without them in view — see the note under that rule.

**`speaker_count` and `expected_event_count` are not deviations** — they are file-level counts with
no extent, and the section's own definition excludes them.

They are written as a **per-branch `counts` measurement**, each entry carrying `found` and
`declared`, the latter copied from the declaration. Not "beside the declaration", which an earlier
draft said and which **cannot be built**: the declaration is SCREEN's, written at stage 2, while
a declared count's `found` half can only come from a branch at stage 3. (`speaker_count` would
fit there, since its `found` half comes from PREPROCESS's diarization at stage 1 — but splitting the
two placements is worse than one rule.) A measurement rather than verdict `detail` because these are
facts, not conclusions.

**This is built as written.** `write_findings` folds every `count` finding into one `counts`
measurement per branch, `{name: {found, declared, ...}}`, derived from the union of its entries'
sources (`branches.py:470-473`, `:488-496`). The `declared` half is spelled by the count's own kind:
`count` carries `declared` (`:275-287`), `count_against_instruction` carries `required` and the
unit the instruction spoke it in (`:290-302`), and `count_beside_typical` carries `typical` with the
spec path its median was measured in (`:305-322`). None of the three asserts a discrepancy.

Earlier drafts called them `extra_speaker` and `missing_expected_event` and put them in the deviation
table; both the names and the placement encoded a comparison the branch is not entitled to make.

### The line the ground-truth rule draws through this table

A declaration used as a **condition for what to look for** is the contract's whole point. A
declaration used as a **reference to score against** is what the ground-truth rule forbids, because
the declaration is not verified.

All ten declared types are on the right side of it: each reports something observed in the audio,
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
`consensus.py::align_sources` (`:276`) over `harmonize.align_pair`
(`audio_analysis/harmonize.py:446`), specified in `transcript-alignment.md`; SPEECH reads it
through `stimulus.align_stimulus` (`speech.py:107`).
**Bracketed tokens are their own channel** — disfluency and non-speech, not transcript errors — so
`[uh]` and `[breath]` never read as misread words.

"Read the wrong sentence" and "read it with six fillers" are different rows, not one score.

---

## What PREPROCESS owes

### Whole-file diarization as a shared derivative

When this was written, SPEECH ran pyannote itself over `(min word start, max word end)` — the
lexical word hull. It could not see a speaker outside that hull, which is where an interrupting
voice or a background talker lives. **Both halves of this section have since landed**; what each
one settled is marked below.

Move it to PREPROCESS, whole-file, in the same shape as the PPG extension:

- **Every branch gets it.** A cough from a second person in a respiration recording is an airway
  finding AIRWAY has no way to notice today.
- **`speaker_count` becomes available for every task.**
- **SPEECH's diarize step becomes a read — owner decision, 2026-09-15; landed.** *"speech should not
  have to rerun pyannote to do this, just take the output and use it."* SPEECH runs no diarizer of
  its own: its docstring says so (`speech.py:8-10`) and `_read_diarization` (`speech.py:164-192`)
  takes the first configured stream that has a `<stream>_diarization` measurement.
  `speech.second_diarizer` (`data/config/default.yaml:236`) is still null, so the corroborator at
  `speech.py:1649` never runs and `second_record` stays `"not_consulted"` (`speech.py:1631`,
  `:1482`); it is now a question about the shared derivative rather than about a pass of SPEECH's
  own.

**The migration gained coverage rather than trading it, which is why the decision cost nothing to
take.** SPEECH's own run was scoped to `(min word start, max word end)` — the lexical word hull — so
a speaker who
talks before the participant starts, after they stop, or inside a pause fell outside the interval
and could not be found at all. A whole-file derivative has no hull. The branch's own account of the
defect is at [`../20260817-triage-workflow-dag/branch-speech.md`](../20260817-triage-workflow-dag/branch-speech.md)
§ *S5 — Speaker count*.

**It landed together with the `attribute` migration** of § *`refine` covers
metadata as well as extent*: both touched SPEECH's speaker handling, and the word→speaker
attribution that consumed the segments is now stamped on the proposed span rather than written as
a per-word assertion.

**The streams were `enhanced` and `residual`, decided by the owner on 2026-09-15; the packaged
config narrowed to `[enhanced]` on 2026-09-20.** No pilot was owed, and the question that would
have needed one did not arise. What narrowed it is a reader, not a reconsideration: `diarization.streams`
is `[enhanced]` (`default.yaml:220`) because *"SPEECH reads the first configured stream that has one
and nothing else in the graph reads any, so a second stream is computed and never read"*
(`default.yaml:216-219`). The argument below for why both halves are wanted is unchanged and
unimplemented; what it now needs is a consumer for the residual count, not a decision.

The objection that held this open was that the derivative exists to catch a quiet background talker
and enhancement suppresses exactly that. The owner's resolution: **enhancement does not destroy that
talker, it partitions the recording.** If the background voice is suppressed, then `enhanced` holds a
single speaker and the suppressed voice is in `residual` — where it can be analysed directly. Nothing
is lost, so there is nothing to trade off and nothing to pilot.

This costs no new plumbing, because `residual` is already first-class: `residual = plain - g*enhanced`,
lag-aligned and gain-fitted (`../../src/senselab/audio/workflows/triage/nodes/preprocess.py:2822-2829`),
written with its own stream entity whenever that block runs (`:2884-2897`), already classified
beside `enhanced`, and already read by the
ruleset — `airway.breath` is `[residual, energy_fraction]` (`data/config/default.yaml:459`) and
`residual|yamnet` sits in `peak_streams` at `:435`.

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
classified beside `enhanced`, already read by the ruleset. So `diarization.streams` was set to
`[enhanced, residual]`: the enhanced half says how many voices survived enhancement, the residual
half says whether one was removed, **the two counts are never summed**, and a disagreement between
them is itself the finding. **No pilot is owed.** The derivation is in
[`../20260817-triage-workflow-dag/config-derivations.md`](../20260817-triage-workflow-dag/config-derivations.md)'s
`diarization` section.

**Narrowed to `[enhanced]` on 2026-09-20**, after the corpus run showed the residual pass had no
reader (`default.yaml:216-220`). The reasoning above is not withdrawn — it says what a residual
count would be *for* — but the finding it describes is not being taken today, because nothing
compares the two counts. Restoring the second stream is a reader's job, not a config change.

The claim that pyannote is reliable at the single/multi-speaker distinction is **uncited** and is
still not relied on here; `benchmarks/diarization.md` and `benchmarks/glides-diarization.md` are in
this tree and the corpus pass should be read against them.

**Cost:** two model passes over the corpus, delivered as an extend driver.

**Landed 2026-09-15.** PREPROCESS's `diarization` block writes one measurement per stream, and
`scripts/extend_diarization.py` adds them to the stores that already exist.

**The read-swap this section then owed has also landed.** SPEECH's own pass over the word hull is
gone; `_read_diarization` (`speech.py:164-192`) takes the first configured stream that carries a
`<stream>_diarization` measurement, and this derivative is the only diarization in the graph. What
SPEECH must read, and the consequences the migration had to handle, are enumerated in
[`../20260915-preprocess-diarization/design.md`](../20260915-preprocess-diarization/design.md)'s
"What SPEECH must read". `speech.second_diarizer` is untouched and is now a question about the
shared derivative. The multi-voice *judgement* is still absent by design — PREPROCESS measures
only, takes no multi-voice decision and gates nothing, and the recording-level judgement is
QUALITY's, taken after the branches have run (`default.yaml:208-212`).

### An extend driver runs a stage's work late

Not "an extend driver is PREPROCESS" — that is wrong twice over. `extend_quality.py` runs QUALITY,
and `rewrite_consensus_taxonomy` recomputes what will be a SCREEN product. The correct statement:
**an extend driver runs a stage's work against a finished store, under the same contract as the
original pass**; what it appends is a derivative every downstream reader treats identically, and the
only difference is when the store was open.

`extend.py:1-6` already holds this position — the module carries the operations "every extend driver
needs over that layout", the layout being the run root's rather than any one driver's — so this spec
does not correct that docstring; an earlier draft mis-attributed the opposite view to it.

---

## VERDICT

Three jobs:

| job | source |
| --- | --- |
| **admit / reject** | ADMIT's failure (`unmeasurable`); the ruleset's `empty` state (`acoustically_empty`); branch findings that the content is unusable |
| **flag for human review** | route-vs-finding mismatch; `unexplained` route state; the existing grounds at `vocabulary.py:592-663` |
| **describe for the record** | branch findings, `consensus_taxonomy`, the declaration |

**The existing precedence is ADMIT-fail → *any* FLAG → `empty` → pass** (`vocabulary.py:676-688`), so
`acoustically_empty` discards only when nothing flagged; `verdict.md:250` records the additional
condition that no hint claims otherwise. The table above is not a replacement for that ordering.

**A fourth job arrived after this table was written, and it is the largest.** VERDICT decides the
declared task's conformance: `gate_conformance` (`verdict.py:433`) resolves the declared family's
task group from `verdict.gates`, reads each gate's input off the **branch's own `measure` findings**,
and substitutes the answer onto the report of the branch that owns the family and reported
`in_family` (`verdict.py:10-14`). Bounds resolve most-specific-first — `by_family`, then `by_group`,
then `default`, a family overriding its group **key by key** (`default.yaml:317-326`) — and a layer
naming no value for a gate does not apply it. Every gate applied is recorded on the verdict, in
`FileVerdict.gates` (`vocabulary.py:366-367`, `:389`), so the conformance can be read backwards.
This is why a branch carries no outcome: the threshold that turns a reading into a judgement about
the recording is here and nowhere else (`default.yaml:297-299`).

**The description job requires VERDICT to read more of the store than it did.** Its docstring said
it held every node's `verdict` entity, ROUTING's `branch_decision`s and its `ruleset_routing`, and
that "this node reads nothing else". That contract is already widened, though by the gates rather
than by the description: VERDICT now also reads the branch reports (`verdict.py:184-196`), the spans
each branch proposed (`:199`), the declared expectation (`:333`) and each gate's reading off the
branch's `measure` findings (`gate_readings`, `:350`). **The smallest defensible choice stands: the
*description* is assembled by REPORT**, and VERDICT's `detail` carries the identifiers a reader
follows.

**REPORT does not read the whole store either, and an earlier revision claimed it did.** It reads
spans and measurements across branches — including SPEECH's and VOICE's spans, which have their own
arms at `report.py:1205-1215` — but `report.py:1202` drops every assertion whose branch is not
AIRWAY, and `figure.py:583` reads assertions only where `name == "squim"`. That means **every
`deviate` and `contest` assertion SPEECH and VOICE write reaches no reader**. Widening REPORT's
assertion read is a prerequisite for the description job, and is listed as its own piece below.
Whether the durable description should
instead be its own entity written by VERDICT is **unresolved**.

**A deviation does not drive a reject**, and for now does not drive a flag either. Whether a
deviation is disqualifying is task-specific, needs ground truth, and is not decided here.

---

## This is several implementation plans, not one

Dependency order, with the pieces that are genuinely independent marked:

1. **The three hint breakages** — read the sidecars, rename the override key, fix the value casing.
   **Independent, and still owed.** Note `make_hints.py` also reads `routing.hint_kind_map` and
   raises when it is absent (`:414-416`, validated at `:429-441`), so renaming only `override.yaml`
   breaks hint generation; both move together.
2. **The `preprocess.py:1970` continuity derivation bug.** **Independent, small**, split out of piece 9. Still open.
3. **Span cleanliness (a) and (b)** — the attribution filter and background typing, scoped to
   `span_yamnet`. Includes the before-and-after gate-firing count on the corpus and a
   `rewrite_consensus_taxonomy` pass over existing stores.
4. **The merge** — TAXONOMY ⊕ ROUTING into SCREEN, with the retired node names kept readable, the
   intra-node ordering test restored, the single verdict decided, and the four documents updated.
5. **The declaration** — the two-grain table, SCREEN's `declaration` measurement, the `metadata` key
   contract. Depends on 1. **Its dependency on 4 is optional** — the measurement can be written by
   TAXONOMY pre-merge — and decoupling it removes the only blocking dependency on the largest piece.
6. **Whole-file diarization** — the extend driver, then SPEECH's diarize step becoming a read.
   **Done.** Both halves landed; `diarization.streams` narrowed to `[enhanced]` on 2026-09-20.
7. **Widen REPORT's assertion read — by verb, not by branch.** **Still owed.** `report.py:1189`
   already filters to
   `_EVIDENCE_BRANCHES` (defined at `:83`), so `:1202` is purely the AIRWAY-on-assertions
   restriction. The per-word `attribute` assertions that made a branch-wise lift unsafe are gone, so
   the objection to lifting it branch-wise has weakened — but the verb-wise rule is still the right
   one. Admit `deviate` and `contest`, the two verbs `write_findings` emits
   ([`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
   § *Deviations and counts are stored*), and any of the contract's five a branch comes to write;
   leave `measure` (`preprocess.py:2324`, `:2331`, `:2349`), `withdraw` (`:601`) and REDACT's
   `exempt` (`redact.py:951`) out.
   **Independent of the branch work and a prerequisite for it**; without it a branch can be built and
   its annotations be invisible. Today that is not hypothetical: every `deviate` SPEECH and VOICE
   write is already invisible.
8. **The branch contract, per branch** — **largely done, by a different route than this list
   planned.** All three branches are built on the propose-only foundation with the two-mode
   dispatch. What is not done is the annotating half: `refine` and `trim` are written by nothing,
   so a branch improving a span it did not propose has no way to say so. DDK is not a piece of this
   at all — it is SPEECH's syllable-train instrument, not a branch.
9. **Boundary reconciliation** — blocked on a parameter-free definition, per (c). Still open.

**One ordering inversion is accepted and budgeted.** Piece 9 changes every span extent, hence which
spans clear 0.96 s — the whole basis of piece 3 — and which need refining in piece 8. Since 9 is
blocked on a definition that does not exist, 3 and 8 proceed first and **are re-measured after 9
lands**; that re-measurement is part of 9's cost, not a surprise.

---

## Explicitly unresolved

**Every fitted threshold awaits ground truth.** `spans.k_db`, the speech quality floors
(`speech.speech_test_stoi_floor` and `speech.speech_test_si_sdr_floor`, both null at
`default.yaml:238-239`), the deviation thresholds, and every numeric key under `verdict.gates`. The
corpus is labelled by declaration, not by
verification. Nothing is refit until something has been listened to. `airway.contest_labels` has
left the list by leaving the config: AIRWAY contests on its own criterion now and reads no fitted
label list.

**The boundary reconciliation rule** (c) — no parameter-free definition yet.

**Which source gives VOICE its subject is settled, and the code took the simpler half of the
settlement**: PREPROCESS's `amplitude` spans, qualified by `phonation_tracks` and
`continuity_trace`, over which VOICE mints its own `family: "voice"` spans (`voice.py:8-11`). No
ruleset-written label is involved, because no rule writes one. **What remains unresolved is whose
family a ruleset-written label would put a span in**, and it is now hypothetical rather than
blocking — at
[`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *The two owner decisions of 2026-09-15 leave the minting rule alone and open one question*.

**What a ruleset-written label records, exactly** — the rule, the evidence path and the value are
settled as required; the entity shape, and which span a reduced feature attributes its firing to, are
not.

**The writer/reader vocabulary rule deserves stating once, and the case for stating it has
strengthened.** This design retires three
writer-vocabulary values with live readers: the `kind` entity type (done 2026-09-13), the node names
`TAXONOMY` and `routing` (the merge, unbuilt), and the `phonation` span family (VOICE). The third
has since happened, and the rule was not applied: VOICE writes `voice` and REPORT reads only
`voice`, so a `phonation` span in a store written before the change reaches no reader. That is the
failure this rule exists to prevent, arrived at by not having the rule written down anywhere a
reader of `report.py` would find it. **Recommendation, unchanged: state it once in `store.md` as a
general contract** — a writer may stop emitting a value, a reader may never stop accepting one — with
the three instances as its examples. Not edited here; `store.md` is out of this spec's scope.

**Whether a branch may run a classifier pass at branch time.** Nothing does today —
`airway.py:1-9` and `speech.py:8` both make running no model an explicit design point — so a
proposed span carries no
per-span classifier evidence. Adding such a pass is out of scope here.

**DDK is specified and built, as SPEECH's instrument rather than as a branch.** The ten
`diadochokinesis-*` families carry their instruction as a phoneme sequence
(`branches.py:608-621`, the rows at `:815-848`), `nodes/ddk.py` reads the train off the envelope's
modulation peak and off a cyclic template decode over a posteriorgram, and what it measures is
SPEECH's. No recording with DDK content flags for want of a node.

**Rate and regularity are reported rather than judged.** Each syllable-repetition row carries a
`typical_count` — a corpus median with its derivation (`branches.py:553-601`, the medians'
provenance at `:604-605`) — and `count_beside_typical` (`:305-322`) writes the count beside it
under an explicit rule that nothing may be judged against a median — enforced at import by
`UNGATEABLE_READINGS` (`gates.py:97`), which makes a gate table naming `typical_count` raise. What
a *required* count may be gated at, once a tolerance is derived, is still open.

**Whether a deviation is disqualifying**, per task.

**The declared counts and `speaker_count` owe ground truth** — recorded as observations, no
discrepancy asserted. `expected_event_count` is two fields now, `required_count` and
`typical_count` (`branches.py:511-601`), and the second may never be gated at all; the first owes a
tolerance before it can be.

**Promoting the declaration's remaining fields from `metadata` to typed `AudioHints` fields.**
`AudioHints` carries `targeted_speaker_count` and `expected_speech` as typed fields already
(`audio_hints.py:149-154`); the seven acquisition and task-identity keys this spec's contract names
have no typed home.

**Whether background spans split around branch-proposed events**, or are merely superseded for that
extent.

**Whether VERDICT should write the durable description as its own entity**, rather than REPORT
assembling it.

**Raw vs enhanced for diarization is resolved, 2026-09-15** — the streams were `enhanced` and
`residual`, and no pilot was owed (§ *Whole-file diarization as a shared derivative*); the config
narrowed to `[enhanced]` alone on 2026-09-20 for want of a reader. It was listed
here as unresolved until 2026-09-15, and the line is kept so a reader of the older text finds where
it went.

**The corpus size disagrees with itself by 28.** `20260910-taxonomy-routing-evidence/measurements.md:1,18-19`
and two other documents attest **62,550 recordings / 62,547 stores read**; a
`find -name store.jsonl | wc -l` over the run tree on 2026-09-13 returned **62,578**. Both numbers are
real. This spec quotes no corpus count in prose for that reason; anything that needs one should use
the attested 62,550 until the 28 are accounted for.

**HeAR's silent-buffer isolation of short spans is unmeasured.**

---

## Corrections to things already written down

**One document still wrongly asserts that FIGURE and REPORT read `consensus_taxonomy`**:
`taxonomy.py:6`, whose docstring says "ROUTING, FIGURE and REPORT read them". They do not —
`figure.py:619` reads `<classifier>_label_summary`, `report.py:510,528` reads PREPROCESS's
`<classifier>_windows`, and the only production consumer is `features.py:756`. The two other
documents named here have since been corrected: `taxonomy.md:90-94` now states the same thing this
spec does, and `20260912-ruleset-in-pipeline/design.md:384` with it. **`taxonomy.py:6` is the one
left**; the error was inherited into an earlier draft of this spec and removed.

**`dag.md`'s gap-span claim is no longer locatable.** This spec cited `dag.md:1111-1116` and
`:1210` as stating that a gap span's background content reaches no decision, and as contradicting
each other. `dag.md` is 388 lines and carries no such passage, so there is nothing left to fix
there. The underlying facts hold: gaps are written with no `family` (`preprocess.py:2016-2031`),
`candidate_spans` selects `family in (None, "airway")` (`airway.py:196-210`), so they *are* AIRWAY
evidence, and `off_task` reads them as `off_task_extent` deviations (`branches.py:1535-1554`).

**`preprocess.md`'s three stale places have gone two different ways.** `:67-68` now marks
`phonation_spans` as retired 2026-09-04 and names the section below it as "a design for a detector
that does not exist", and `:171` states outright that `airway.py` reads no `k_db` and that there is
no `airway.k_db` key — both corrections landed. What remains is `:179-221`, which is still that
non-existent detector's design, kept deliberately under the flag at `:67`. The span-algorithm
sketch is at `:160-167`, and it should be read against `default.yaml:37-42` rather than against
`dag.md`, whose corresponding passage is gone.

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
