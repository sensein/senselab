# QUALITY

What this node answers: **does the store contradict itself, and is anything true of the recording
that is a property of the room rather than of the content?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).
This document is grounded in the code and in [`dag.md`](dag.md); where the two disagree the code
wins and the disagreement is named.

## QUALITY is not a branch, and the difference is load-bearing

Three properties separate it from AIRWAY, SPEECH, VOICE and DDK. Each is deliberate.

**It reads stored records only.** Entities and their attributes. It decodes no audio, opens no
sidecar and re-derives nothing — every amplitude it compares was measured by the node that held the
signal (`quality.py` module docstring, `:8-12`). This is what makes `scripts/extend_quality.py` able
to run it over a finished store years later, and it is a property to preserve rather than a
limitation to lift. A capability proposed for QUALITY that needs the waveform does not belong here;
it belongs in PREPROCESS, with QUALITY reading what PREPROCESS wrote.

**It runs on every path PREPROCESS completed.** `run.py:310` calls it unconditionally after the
branch loop — whatever routing selected, and whether or not routing itself raised, since `selected`
falls back to the empty set when `routed is None` (`run.py:296`). It is therefore the one node
positioned to see every branch's output at once. Note it is *not* terminal: REDACT runs after it
(`run.py:311-317`), and `dag.md` has been corrected on that point.

**It is not routed.** No gate selects it, and `KIND` is `None` (`quality.py:56`). So the question
every branch document must answer — what it does on a recording routed to it whose declared task
belongs elsewhere — does not arise. QUALITY sees everything.

### Where it sits in the four stages, and whether its `contest` is the contract's

The branch-contract spec's four stages are PREPROCESS → SCREEN → BRANCHES → VERDICT, and QUALITY is
in none of them. **It is a fifth position: after the branches, before VERDICT, reading the whole
store.** The spec's own review flagged that the taxonomy leaves it unplaced; the resolution is that
the four stages describe the *routing and measurement* path, and QUALITY is a cross-cutting audit
over the result of that path. ADMIT, REDACT, REPORT and FIGURE sit outside the four for the same
kind of reason.

**Its `contest` is the contract's `contest`.** The contract defines `contest` as *"the span does not
carry what was proposed"*. A clip span proposes that the signal reached its ceiling over that
extent; an unclipped sample louder than that ceiling says it did not. That is an instance of the
contract's definition under a different test — the test being an amplitude comparison rather than an
absence of evidence. `quality.py:291` writes `verb: CONTEST_VERB` with `CONTEST_VERB = "contest"`
(`:61`), the same wire value AIRWAY writes, and a reader keying on `verb` gets a consistent meaning.

**This is the evidence that the verbs are store-wide rather than branch-only.** QUALITY is not a
branch and it contests. What is branch-specific is the *family* a `propose` writes, not the verb
vocabulary.

## Capabilities

### Q1 — Clip consistency (**built**)

**Question.** Does any clip span assert a ceiling that a sample outside every clip span exceeds?

**Reads.** PREPROCESS's `clip` spans for their extents, and the `clip_amplitude` measurement for
every amplitude including per-span levels (`quality.py:8-12`). The spans say what was asserted; the
measurement says what the samples were.

**Computes.** For each clip span with a stored level, whether the whole-file unclipped peak exceeds
that level by more than `quality.clip_contradiction_margin` of it (`quality.py:271`).

**Emits.** One `contest` assertion per contradiction, `wasDerivedFrom` the span it contests
(`quality.py:291-302`). **PREPROCESS's spans are never invalidated here** — the store is
append-only and the span is PREPROCESS's reading, not QUALITY's to withdraw.

**Its expected count is zero, and that is the point.** `_clip_spans` now applies the same comparison
at detection and never writes a candidate it contradicts, so this check is the *audit* of that rule
(`quality.py:31-36`). It deliberately reads nothing about whether the rule ran: a store from the
completed corpus, or one whose spans came from anywhere but `_clip_spans`, carries spans nothing
filtered, and an audit that assumed compliance would measure nothing on a fresh store either.

**One trap, recorded in `dag.md`.** The margin is QUALITY's own `config.require`
(`quality.py:239`), not PREPROCESS's — only the edge guard comes off the measurement (`:246`). They
agree today because both read the same key, so an override changed between a PREPROCESS run and a
later `extend_quality.py` pass would audit spans against a margin they were never detected under.

**Parameter-free?** No — `quality.clip_contradiction_margin` is a threshold. But it is a
*tolerance on a physical comparison*, not a classification cut, and it is already derived and
recorded in `data/`. The no-refits rule does not reach it.

### Q2 — Background content (**not built; depends on gap spans becoming background**)

**Question.** What is in the regions no proposer claimed, and is any of it a property of the room?

**`dag.md` states the boundary**: the branches answer "is the content the protocol asked for
present"; a mains hum is a property of the room, not an event, and questions of that shape are
QUALITY's. But it also records that the mechanism cuts against the thesis — gap spans carry no
`family` key (`preprocess.py:1570-1578`), `airway.py:198` selects `family is None`, and **a gap span
can carry the whole AIRWAY verdict on its own** because `labelled_n` separates `pass` from `fail`
(`airway.py:376-383`, `:394`).

So background is not QUALITY's today; it is AIRWAY's, by accident of a missing attribute. If the
branch-contract spec's part (b) types gap spans as background, they leave AIRWAY's selector and
**QUALITY becomes where their content is concluded on**.

**What it would then read.** The gap spans' own per-span classifier measurements — they are in
`state["span_ids"]` and the per-span classifiers run over them (`preprocess.py:1583`, `:1587`;
`_span_hear` at `:1861-1865`), so each already carries HeAR windows and needs no new computation.

**What it would conclude.** That a background region carries a label, and which. Not a verdict on
the recording: a hum, a keyboard, a passing vehicle is an observation about the room. The natural
form is a `label` assertion over the background span, and — where the label is one a branch would
have acted on — a note that it was *not* treated as an event.

**What is unmeasured.** How often a gap span currently decides an AIRWAY verdict. `dag.md` records
that nobody has counted it. That count is a prerequisite for knowing what part (b) would change, and
it is a count over the corpus, not a fit against it — so the no-refits rule permits it.

### Q3 — Cross-branch contradiction (**not built; in remit**)

**Question.** Do two branches say incompatible things about the same extent?

QUALITY is the only node positioned to ask. It runs after every branch (`run.py:310`) and reads the
whole store, so both branches' assertions are in front of it. Nothing else in the graph has that
view: VERDICT reads only verdict entities, `branch_decision`s and `ruleset_routing`
(`verdict.py:218-220` — "This node reads nothing else").

**What it would read.** Assertions over overlapping extents from different branches.

**What it would emit.** A `contest`, under the same definition — one branch's span does not carry
what another branch proposed for the same region. Its `wasDerivedFrom` names both.

**What makes this hard, and why it is not specified further here.** Two branches labelling the same
extent differently is usually *not* a contradiction: a cough during a sentence is genuinely both
airway content and an interruption of speech, and the contract's whole premise is that content the
task did not ask for is still content. A real contradiction needs a pair of claims that cannot both
hold — and enumerating those pairs is a domain question nobody has answered. **In remit, not yet
specifiable.** Recording it as in remit is what stops the next QUALITY-shaped check being built
somewhere it does not belong.

### Q4 — Acquisition consistency (**not built; depends on the declaration**)

**Question.** Does the recording match what the protocol says was recorded?

**Reads.** The declaration's `recording_duration`, `audio_sample_rate`, `audio_channel_count` and
`recording_microphone`, against the `stream` entity ADMIT wrote — which carries `sampling_rate`,
`channels`, `size_bytes` and a `checksum_sha256`.

**Computes.** Equality, and for duration a difference. All three of sample rate, channel count and
duration are directly comparable; the comparison is arithmetic, not a judgement.

**Emits.** A `counts`-shaped measurement carrying `declared` and `found` per field. **Asserting no
discrepancy** — a duration that disagrees with the declaration is an observation, and whether it
invalidates the recording is not QUALITY's call.

**Parameter-free?** Sample rate and channel count are exact. Duration needs a tolerance, and a
tolerance is a number nobody has. State the difference and let a reader judge, rather than owe a cut.

**`recording_input_gain` is not in this capability.** A gain can be set anywhere and the signal can
still clip; it does not predict clipping and must not be used as if it did.

### Q5 — SQUIM (**measured by others; QUALITY should not conclude on it**)

Per-span SQUIM (`stoi`, `pesq`, `si_sdr`) is written by PREPROCESS over the general spans and again
by SPEECH over its own (`speech.py:959-1010`).

**Should QUALITY read it?** It may — the numbers are stored records, which is QUALITY's whole diet.
**It should not conclude on it.** `speech.speech_test_stoi_floor` and
`speech.speech_test_si_sdr_floor` are both null (`default.yaml:168-169`) and neither can be fitted:
a floor fitted against declared families encodes which recordings the protocol labelled, not which
are intelligible. So there is no cut, and inventing one here would be the same error SPEECH avoided
by leaving `squim_vote` at `not_evaluated` (`speech.py:631`).

**What QUALITY can do without a cut** is report the distribution — the per-span values and their
spread — as a measurement, so a reader has them in one place. That is a description, not a
judgement, and it needs no threshold.

**Owed ground truth.** Any intelligibility floor. It should be established by listening, not by
fitting.

## What QUALITY does not emit

No deviations. A deviation is a departure from what a *task* asked for, and QUALITY holds no
declaration about a task — it holds the store. Q4's acquisition comparison is the nearest thing, and
it is a `counts`-shaped measurement rather than a deviation because it compares the recording to its
own metadata, not the performance to its instruction.

No spans. QUALITY proposes nothing; it audits what others proposed.

## What exists today

| capability | status |
| --- | --- |
| Q1 clip consistency | **built**; expected count zero, it is the audit of a detection-time rule |
| Q2 background content | **not built**; blocked on gap spans being typed background |
| Q3 cross-branch contradiction | **not built**; in remit, not yet specifiable |
| Q4 acquisition consistency | **not built**; needs the declaration |
| Q5 SQUIM description | **not built**; must not conclude |

## What the node emits

```
assertions   contest, one per contradicted clip span
verdict      { signal, preceded_by, clip_spans_n, checked_n, unmeasurable_n,
               contradicted_n, unclipped_samples_n, unclipped_peak,
               unclipped_peak_time_s, clip_contradiction_margin,
               clip_edge_guard_samples, contradictions, flags }
```

**The verdict's basis, exactly** (`quality.py:312-317`):

- `FLAG` when any contradiction was found.
- `PASS` with *"no clip span over the recording; nothing to contradict"* when nothing was measurable.
- `PASS` with *"no clip span sits below an unclipped sample"* otherwise.

`preceded_by` names the nodes whose verdicts were live when QUALITY read the store
(`quality.py:115`, recorded at `:329`) — which on a fresh run is routing and the branches, and on an
`extend_quality.py` pass over a run that never routed is neither.

**What it refuses.** A dependency that is absent is an operational fact, not a finding: clip spans
with no clip-amplitude measurement beside them raise, and the runner records the node `ERRORED`
(`quality.py:14-16`). A contradiction QUALITY *can* measure is always a finding and never a raise.

## Out of scope

Anything needing the waveform — that belongs in PREPROCESS. Any normative quality judgement.
Withdrawing another node's reading. Any threshold fitted against declared families.
