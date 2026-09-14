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
PREPROCESS   Shared derivatives, extend drivers included. Content-first, hint-blind.
             Emits a clean span set: defensible, not exhaustive.

SCREEN       TAXONOMY ⊕ ROUTING, merged. Consolidates the classifier evidence, resolves
             the declaration, applies the ruleset and the hints, emits the routing decision.

BRANCHES     Receive every upstream output, their route, and the declaration. Mark, refine,
             refute, propose and trim into `family: "<branch>"` spans. Emit typed deviations.
             Conclude on their own question.

VERDICT      Admit/reject, flag for human review, durable description.
```

### The merge reverses a decision made the same day

`specs/20260912-ruleset-in-pipeline/design.md` stage 2 — landed hours before this spec — kept
TAXONOMY and ROUTING as two nodes, on the argument that TAXONOMY emits measurements and ROUTING
emits decisions, and that the L1-measures/L2-decides boundary is worth a node boundary.

The owner reversed it, and the reason is better than the argument it replaced: **the middle stage's
product is the routing decision.** The consolidation exists to serve it. A node whose output nothing
acts on is a seam, not a layer. The measure/decide distinction survives inside the merged node as
the distinction between what it records and what it decides; it does not need two nodes to hold it.

The counter-argument that was raised and answered: `consensus_taxonomy` has consumers other than
routing (`figure.py`, `report.py`, and the voice rework `voice.py:1-9` is waiting on), so merging
makes a shared descriptive artifact a byproduct of a decision node. The answer is that a byproduct
with three consumers is still an output; what makes an artifact second-class is nobody reading it,
which was the condition the merge fixes rather than creates.

A second, weaker argument for the merge that only appeared on inspection: `evaluate_live_routes`
serialises the **whole store** and reduces it (`live_evidence.py:63-77`), so the router's dependency
was never a narrow TAXONOMY→ROUTING handoff in the first place. Splitting the two nodes did not
narrow an interface; it put a boundary in the middle of one read.

`SCREEN` is the proposed name. `TAXONOMY` understates the routing half; `ROUTING` understates the
consolidation. `SCREEN` matches vocabulary already in the tree (`screening` in the report document,
`SCREENED_KINDS` in the deleted fold).

---

## Span cleanliness — three parts

A span shorter than a classifier's window cannot be classified by its own content. That is the
defect, and it is not a threshold problem: a 200 ms cough is a real event, so proposing it is right.
What is wrong is the label attached to it afterwards.

### (a) A label belongs to a span only if a window lies inside it

**This applies to YAMNet and not to HeAR, and the difference is load-bearing.**

`_span_yamnet` gives a span shorter than the native window the scores of the whole-file windows that
*cover* it, overlap-weighted, and marks the result `attribution: "covering_windows"` with
`covering_windows_n` and `covering_seconds` (`preprocess.py:1952-1972`); a long span gets
`attribution: "native"` (`preprocess.py:2013`). A covering-window label is therefore a statement
about up to a second of audio attributed to a fifth of it.

`_span_hear` does something different. `span_hear_input` places a span shorter than
`HEAR_WINDOW_SECONDS` in a silent 2 s buffer, "so its only detector result describes the span
itself" (`hear.py:426-446`); a longer span is passed through and its native windows are mapped back
onto the recording's timeline by `hear_window_extent` (`hear.py:449`). HeAR's short-span label is
about the span.

So the rule: **a covering-window label is recorded, and is not eligible to become a
`consensus_taxonomy` row or branch evidence.** The flag that expresses this already exists and is
already written; today nothing filters on it. Making it load-bearing is the cheapest high-value
change in this spec.

HeAR's isolation has its own caveat, and it is a different one: a 200 ms event centred in 1.8 s of
digital silence is not what the model saw in training. Whether that distorts its scores is
**unmeasured**. It is not addressed here and must not be conflated with the covering-window problem.

### (b) Gap spans are background, never events

Gap spans are the complement of the union of the four proposal sources — by construction, "where the
recording's background lives" (`preprocess.py:1555-1557`). They carry `measure: "gap"` and
`merged_proposals: 0` (`preprocess.py:1568-1580`). They are also full members of AIRWAY's evidence
set, which selects every live span whose `family` is `None` (`airway.py:198`), and they become
`consensus_taxonomy` rows like any other span.

Type them as background. They stay measured and visible; they stop being eligible to be something
that happened.

### (c) Boundaries reconciled, not first-writer-wins

`_novel` (`preprocess.py:1454-1487`) appends a record to `corroborated_by` on every span a later
candidate overlaps, and keeps the earlier proposer's extent unchanged. Four sources agreeing on an
event is precisely when its boundary can be stated well, and that is the moment the current code
discards the information.

Alongside it, a plain bug: continuity spans are `wasDerivedFrom` the energy envelope rather than the
continuity trace (`preprocess.py:1520` — `state["envelope_id"]`), even though the trace is in the
activity's `reads` (`preprocess.py:1427-1428`). The provenance edge names the wrong source.

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

Two JSON sidecars per recording, at different grains. Read off the corpus at
`/orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/`.

**`<stem>_recording-metadata.json`** — per recording:

```
language, instructions, record_id, recording_id, recording_acoustic_task_id, session_id,
recording_duration, recording_size, recording_profile_name, recording_profile_version,
recording_input_gain, recording_microphone, task_name, stimulus_text, speech_type,
stimulus_source, audio_channel_count, audio_sample_rate
```

A real one: `task_name: "harvard-sentences-list-34-6"`, `stimulus_text: "Try to trace the fine lines
of the painting."`, `speech_type: "read"`, `recording_duration: "6.1"`,
`recording_microphone: "Headphones"`, `recording_input_gain: "0.758"`.

**`<stem>_task-<task>_acoustictask-metadata.json`** — per acoustic task:

```
language, instructions, record_id, acoustic_task_id, session_id, acoustic_task_name,
acoustic_task_cohort, acoustic_task_status, acoustic_task_duration, stimulus_text,
speech_type, audio_channel_count, audio_sample_rate
```

A real one: `acoustic_task_name: "respiration-and-cough-(v2)"`, `acoustic_task_duration: "151.0"`,
`stimulus_text: ""`, `speech_type: "non-lexical"`, and `instructions` carrying the protocol's own
words — *"tap the record button and cough HARD as if something were stuck in your throat."*

### Three things are broken today

**The hint is parsed from the filename while the sidecar goes unread.**
`specs/20260817-triage-workflow-dag/runs/b2ai-v2/make_hints.py` extracts the BIDS `task-` token from
the stem and emits `may_contain` plus `metadata.{task_token, speech_type, task_id, registry}`. Every
recording carries the protocol's own `task_name` and `speech_type` beside it. `stimulus_text` — the
field `AudioHints.expected_speech` exists to hold — is never carried at all.

**The only populated map raises on load.** `runs/b2ai-v2/override.yaml:261` is keyed
`routing.hint_kind_map`. Stage 2 renamed the packaged key to `hint_branch_map`
(`default.yaml:127`), and `_merge` refuses any key the packaged config lacks (`config.py:147-148`).

**Its values would all be rejected even after renaming.** `_map_tags` casefolds the map's *keys*
against the declared tags, but tests the *value* against `BRANCHES` unchanged
(`routing.py:103-108`). The override's values are lowercase kinds — `cough: airway` — and `BRANCHES`
is `("AIRWAY", "SPEECH", "VOICE", "DDK")`. Every entry would land in `bad_map_values`, which appends
an `Outcome.FLAG` reason against ROUTING (`vocabulary.py:382-384`) — on every file in the run.

### What SCREEN resolves

One `declaration` measurement per recording, carrying:

- **task identity** — `task_name`, `speech_type`, and the acoustic task's name and duration.
- **expected content and structure** — from a per-task table in `data/`, keyed by `task_name`, each
  entry carrying the instruction text quoted beside it as its derivation. `instructions` is prose —
  *"repeating the syllables 'puhtuhkuh' as quickly and consistently as possible"* — and must not be
  parsed. There are ~28 tasks; a table read by a human from the instructions, and checkable against
  them, is the honest mechanism.
- **the stimulus text**, where the task has one.
- **the acquisition facts** — declared duration, microphone, channel count, sample rate.

### Two standing rules

**A hint may add and inform. It may never suppress.** Routing is the union:
`will_run = by_ruleset or forced_by_hint` (`routing.py:215-220`), which the code already does. A
branch the content routed cannot be un-routed by a declaration that disagrees with it.

**Content the task did not ask for is content, not error.** A breathing recording that carries speech
routes SPEECH on the ruleset's evidence and has that speech recorded. The recording not matching its
own label is a finding, never a reason to discard what was found.

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

### Where each branch already stands against this contract

**SPEECH already does it, unnamed.** It ignores the general span set, groups lexical word runs into
its own spans, and writes them with `family: "speech"` (`speech.py:880-883`). It is the working
instance of the contract.

**VOICE was designed to and cannot.** Its subject is every live span whose `family` is `phonation`
(`voice.py:1-9`, `_PHONATION_FAMILY` at `:39`). Nothing in the tree proposes one — the detector was
retired 2026-09-04 — so the branch takes its no-span path and returns `Outcome.FAIL` on every
recording. Its one `prov_type="span"` write (`voice.py:336`) re-mints an existing span inside
measurement machinery that is never reached. Under this contract **the branch is the proposer**, and
that is the whole fix.

**AIRWAY has half the verbs already.** It writes `label`, `confirm`, `contest`, `abstain` and `flag`
assertions across its first three steps (`airway.py:230` through `:373`). `mark` ≈ `label` and
`refute` ≈ `contest`, so most of this is
renaming toward a shared vocabulary; `refine`, `propose` and `trim` are genuinely new.

**`contest` is dead code.** `airway.contest_labels` is `null` (`default.yaml:138`), so
`_contest_labels` returns the empty set (`airway.py:141`) and `contested_n` is structurally zero.
Refutation needs that mechanism replaced or that config fitted — and per the ground-truth rule above,
fitting it is not available yet.

**DDK has no node.** `nodes/ddk.py` does not exist. Since stage 2 the ruleset routes DDK, so a
recording with DDK content now receives a `branch_decision` with `will_run` true and is recorded
`SKIPPED` with `NO_NODE` (`run.py:303-305`), which flags the file. This spec gives DDK a contract; it
does not build the node.

---

## Deviations are facts, not judgements

A branch enumerates typed, located deviations, each with its evidence. **Downstream nodes, humans or
machines determine whether a deviation is legitimate.** The pipeline does not assert that a task was
not performed.

This rules out the conformance state vocabulary that was drafted and rejected —
`conformed` / `deviated` / `not_performed` / `not_declared`. A state is a judgement. A deviation is
an observation with an extent.

A starting set of types:

| type | what it says |
| --- | --- |
| `extra_speaker` | more than one speaker in a single-target recording |
| `stimulus_mismatch` | a lexical word that is not the word the stimulus expected |
| `filler` | a disfluency or non-speech token where the task expected lexical content |
| `missing_expected_event` | the declaration expects an event class the branch did not find |
| `off_task_extent` | a region of the recording that does not serve the declared task |

Illustrative expectations from the per-task table:

| task | declared expectation | deviations it makes available |
| --- | --- | --- |
| harvard-sentences | `stimulus_text` | `stimulus_mismatch`, `filler`, `extra_speaker` |
| diadochokinesis | a sustained `puhtuhkuh` syllable train | `missing_expected_event`, `off_task_extent` |
| respiration-and-cough | cough events; N breaths | `missing_expected_event`, `extra_speaker` |
| maximum-phonation-time | one sustained vowel | `missing_expected_event`, `off_task_extent` |

### Read speech has the sharpest case

`stimulus_text` is a reference transcript. Lexical words align against it; **bracketed tokens are
their own channel** — disfluency and non-speech, not transcript errors. The bracketing work already
separates them, so `[uh]` and `[breath]` never read as misread words.

"Read the wrong sentence" and "read it with six fillers" are different rows, not one score.

---

## What PREPROCESS owes

### Whole-file diarization as a shared derivative

Today SPEECH runs pyannote itself, over `(min word start, max word end)` — the lexical word hull
(`speech.py:642-646`). It therefore cannot see a speaker outside that hull, which is exactly where an
interrupting voice or a background talker lives.

Move it to PREPROCESS, on the enhanced stream, whole-file, in the same shape as the PPG extension.
Three consequences:

- **Every branch gets it.** A cough from a second person in a respiration recording is an airway
  finding AIRWAY currently has no way to notice.
- **`extra_speaker` becomes available for every task**, single-target protocols included.
- **SPEECH's diarize step becomes a read**, and `speech.second_diarizer` (`default.yaml:166`, null,
  so `not_consulted`) becomes a question about the shared derivative rather than about SPEECH's own
  pass.

Pyannote is reliable at the single-speaker/multi-speaker distinction, which is the question this
derivative primarily has to answer.

**Cost:** a model pass over 62,578 recordings, comparable to the PPG extension, delivered as an
extend driver.

### An extend driver is PREPROCESS

The owner's framing, and it is better than the one currently written down. `extend.py`'s module
docstring treats extend drivers as their own category with their own layout concerns. They are
PREPROCESS reaching a finished store late. Everything a driver appends is a shared derivative that
every branch reads, under the same contract as one written in the original pass; the only difference
is when the store was open.

---

## VERDICT

Three jobs, from distinguishable sources:

| job | source |
| --- | --- |
| **admit / reject** | ADMIT's own failure (`unmeasurable`); the ruleset's `empty` route state (`acoustically_empty`); branch findings that the content is unusable |
| **flag for human review** | deviations; route-vs-finding mismatch; `unexplained` route state |
| **describe for the record** | branch findings, `consensus_taxonomy`, the declaration |

The first two discard grounds exist today (`vocabulary.py:417-429`). What this design adds to the
reject column is branch findings; what it adds to the flag column is deviations.

The description stands whether or not a declaration existed — it is the durable, provenance-backed
statement of what is in the recording, and it does not depend on the recording having a task.

**A deviation does not drive a reject.** Someone who read the sentence with three fillers performed
the task. Whether a deviation is disqualifying is task-specific, needs ground truth, and is not
decided here.

---

## This is several implementation plans, not one

Stated so nobody tries to land it in one pass. The dependency order:

1. **The three hint breakages** — read the sidecars, rename the override key, fix the value casing.
   Independent of everything else here, and fixes a config that currently raises.
2. **Span cleanliness (a) and (b)** — make `attribution: "covering_windows"` load-bearing, type gap
   spans as background. Consumer-side only; no proposer change, no corpus run.
3. **The merge** — TAXONOMY ⊕ ROUTING into SCREEN. Touches the graph, VERDICT, REPORT and FIGURE, and
   partially reverts stage 2. Independent of 1 and 2 but large.
4. **The declaration** — the per-task table and SCREEN's `declaration` measurement. Depends on 1 and
   3.
5. **Whole-file diarization** — the extend driver and SPEECH's diarize step becoming a read. A model
   pass over the corpus; independent of 1–4 but expensive.
6. **The branch contract, per branch** — SPEECH first (it is closest), then AIRWAY, then VOICE (which
   needs a proposer before it has a subject), then DDK (which needs a node). Depends on 4, and on 5
   for `extra_speaker`.
7. **Span cleanliness (c)** — boundary reconciliation in `_novel`, plus the `preprocess.py:1520`
   derivation bug. The bug fix is independent and small; the reconciliation changes every span
   extent in every future run and should land alone.

## Explicitly unresolved

**Every fitted threshold awaits ground truth.** `spans.k_db`, `airway.contest_labels`, the speech
quality floors, the deviation thresholds. The corpus is labelled by declaration, not by
verification. Nothing is refit until something has been listened to.

**DDK has a contract and no node.** This spec specifies it. Building it is separate, and until it is
built, recordings with DDK content flag.

**Whether a deviation is disqualifying is undecided**, per task, deliberately.

**The per-task expectation table does not exist yet.** Its shape is specified here; its ~28 entries
are to be read from the `instructions` fields by a human and recorded with those instructions quoted
as derivation.

**HeAR's silent-buffer isolation of short spans is unmeasured** — see the caveat under (a).

---

## Corrections to things already written down

**The 94%-of-taxonomy-rows figure predates its own fix.**
`specs/20260817-triage-workflow-dag/benchmarks/taxonomy-vs-task-2026-09-07.md:203-288` measured that
360 of 384 `consensus_taxonomy` rows (94%) trace to spans shorter than 0.96 s, 293 (76%) to spans
shorter than 0.3 s, and 143 to gap spans. The periodic fill that caused it was replaced with
covering-window attribution on 2026-09-08
(`benchmarks/span-fill-recovery-2026-09-08.md:286-300`). **The current rate is unmeasured.** The
structural argument in this spec does not depend on the number; the number must not be quoted as
current.

**`recording_input_gain` does not predict clipping.** A gain can be set anywhere and the signal can
still clip; a low gain does not mean a clean file. It is an acquisition fact, not a clipping
predictor, and must not be used as one.

**`acoustic_task_status` is out of scope.** It is the protocol's own record of whether the task
completed — upstream or downstream human evaluation, not the triage graph's judgement to make or to
check.

**`speech.py`'s `family: "speech"` write is at `:880-883`**, not `:858` as stated in earlier
discussion; `runs/b2ai-v2/override.yaml`'s `hint_kind_map` is at `:261`, not `:260`.
