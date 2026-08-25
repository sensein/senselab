# The triage DAG as a Nextflow workflow

> **SUPERSEDED — kept as a record of what was executed, not as a description of the graph.**
>
> This prototype implements the **v1** design, including the residual fold that the v2 specs
> retired: VOICE's subject here is what is left after SPEECH's spans and AIRWAY's labelled
> intervals are subtracted, and its outcome table raises an absence to a `flag` when a hint
> declares the kind. Neither is the graph any more — [`branch-voice.md`](../branch-voice.md)
> makes PREPROCESS's phonation spans the subject and opens no span in the branch, and
> [`verdict.md`](../verdict.md) puts the hint mismatch in VERDICT's fold, because a branch that
> flags resolves its kind `present` and a branch with no subject may not do that.
>
> The **v2 specs in [`../`](..) govern**. What survives here is the measurement: Nextflow 26.04.6
> was installed and the graph below was run, so the resumability, invalidation and stub-mode
> figures in "What was actually verified" are real numbers about a real execution. The design
> tensions section records what that execution taught, which is why the prototype is kept rather
> than rewritten: re-running it against v2 would cost the very thing it is retained for.

An executable expression of the design in [`../`](..) — nine processes over an append-only element
store, with `-resume`, a stub mode that runs the whole graph without a model, and a hard separation
between the sensitive store and the releasable derivative.

```
ADMIT ─→ PREPROCESS ─→ TAXONOMY ─→ AIRWAY ─→ SPEECH ─┬─→ REDACT ─┐
                                                     └─→ VOICE ──┴─→ VERDICT
                                                       (concurrent)
                        STORE_VIEW materialises the store as the design describes it
```

That is not the graph the brief drew, and the difference is the first thing this README has to
account for. See [Design tensions](#design-tensions).

---

## What was actually verified

Nextflow **26.04.6** (build 12646) on macOS 15 / OpenJDK 25, installed for this purpose. Everything
in this section was run, not reasoned about.

| check | command | result |
| --- | --- | --- |
| parses and lints | `nextflow lint .` | **11 files, 0 errors, 0 warnings** |
| whole graph, no models | `nextflow run . -stub-run -profile stub --input 'rec*.wav'` | 18 tasks (2 recordings × 9), 0 failed |
| resumability | same command `-resume` | `completed=0 cached=18` |
| selective invalidation | `-resume --models.hear.revision <sha>` | `completed=14 cached=4` — see [what `-resume` costs](#what--resume-costs) |
| every fold path | 7 stub scenarios | table below |
| ADMIT `fail` short-circuits | `--stub_scenario admit_fail` | **2 tasks ran** (ADMIT, VERDICT), triage `fail`, `kinds {}` |
| branch `fail` is not a file `fail` | `--stub_scenario cough` | SPEECH exits **0** with outcome `fail`; triage `pass` |
| provenance is mandatory | `nextflow run .` with no revisions | refused, naming all 9 unresolved model roles |
| publish roots cannot nest | `--release_dir <inside store_dir>` | refused |
| the release guard bites | `--redact.id_pattern 'rel-[0-9a-f]{12}'` | REDACT exit 3, pipeline aborted, **`release/` never published** |
| single-node execution | `--node AIRWAY --store_in <dir>` | 1 task, wrote its own segment and verdict |
| store merge is order-independent | 4 random segment permutations → `STORE_VIEW` | `elements.json` byte-identical |
| ids are stable across runs | fresh run vs fresh run | identical segment filenames, i.e. identical content hashes |

Stub scenarios, all through real Nextflow:

| `--stub_scenario` | triage | release | what it exercises |
| --- | --- | --- | --- |
| `pass_speech` | `flag` | `releasable` | the ordinary speech path. Flags because VOICE's gate is un-derived |
| `cough` | `pass` | `not_assessed` | **a branch `fail` that is normal.** SPEECH has no subject; REDACT never runs |
| `contradiction` | `flag` | `releasable` | TAXONOMY said speech absent, SPEECH passed → contradiction; plus a YAMNet `contest` |
| `undecided_min_families` | `flag` | `releasable` | the presence rule's own threshold decides the answer → `undecided` |
| `empty` | `fail` | `not_assessed` | triage **row 3** — measured, nothing in it. Not row 1 |
| `pii_survives` | `flag` | `withheld` | REDACT verification fails. Nothing published to `release/` |
| `admit_fail` | `fail` | `not_assessed` | triage **row 1** — could not measure. Nothing else runs |

`pass_speech` reaching `flag` rather than `pass` is not a bug and is worth dwelling on: with the
phonation gate's floors un-derived, any recording with unclaimed residual energy flags. A `pass` is
currently reachable only when the residual is empty. That is the honest consequence of
[`benchmarks/open.md`](../benchmarks/open.md) and it will change when the gate is fitted, not before.

### What is real code and what is a seam

The `bin/triage-node` script implements, for real: the store (ids, segments, provenance
enforcement, the append-only merge, the integrity sweep), the parameter discipline, VERDICT's
complete fold, `RELEASE_GUARD`, ADMIT (threshold-free, so it needs no model), and every stub.

Every measurement that needs a model goes through `Seam.need()`, which returns a value from
`--replay <json>` if one is supplied and otherwise **exits 4 naming exactly what was missing**. It
never returns a plausible number. Wiring senselab in means implementing that one method. Exit 4 is a
node error rather than a branch `fail`, on the same principle `branch-speech.md` applies to a PII
detector that did not run: "could not check" is not "clean".

So: the graph, the store, the folds and the safety properties are executable and tested. The
acoustics are not implemented, and this document does not claim they are.

---

## Running it

```bash
# topology, no models, nothing to download
nextflow run . -stub-run -profile stub --input 'data/*.wav'

# one node, standalone, against a directory of segments
nextflow run . --node AIRWAY --input rec.wav --store_in results/store/rec/ --derivatives path/to/derivatives

# the fold over an existing store
nextflow run . --node VERDICT --input rec.wav --store_in results/store/rec/
```

A real run needs every model role resolved to a 40-hex commit. There is no way round this and it is
deliberate:

```bash
cat > models.yaml <<'YAML'
models:
  yamnet:         { id: google/yamnet,   revision: <40-hex> }
  hear:           { id: <id>,            revision: <40-hex> }
  crisperwhisper: { id: nyralabs/CrisperWhisper2.0_turbo, revision: <40-hex> }
  # ... alignment, squim, diarizer, separation, pii, ast
YAML
nextflow run . --input 'data/*.wav' -params-file models.yaml
```

`--node <NAME>` replaces `-entry RUN_<NAME>`: Nextflow 26's strict parser dropped `-entry`, so the
dispatch is a parameter. Same capability, different flag.

### Output layout — read this before pointing anything at it

```
results/
├── store/                      SENSITIVE. Do not release. Do not sync.
│   ├── <rec>/segment.*.jsonl   the append-only store, one segment per author
│   ├── <rec>/verdict.*.json    each node's outcome + verdict + view (element ids)
│   ├── <rec>/figure.*.png      figures, which carry element ids by design
│   ├── <rec>/view/             STORE_VIEW's materialised fold + integrity report
│   └── _pipeline_info/         trace, report, timeline, DAG
└── release/                    releasable. REDACT's artifacts only.
    └── <rec>/{audio,transcript,figure}.redacted.*, manifest.json
```

**The work directory is as sensitive as `store/`.** It holds every intermediate, including
PREPROCESS's `derivatives/` with the unredacted transcript. Nothing publishes it, and nothing
cleans it either — `nextflow clean` is an operator decision, on the same footing as `redact.md`'s
point that removing the original recording is an authorised act and not a side effect.

---

## Design tensions

### The tension, stated precisely

Nextflow is dataflow. A process is meant to be a pure function of its declared inputs, so that its
task hash is computable, so that `-resume` is sound. `store.md` specifies the opposite: an
append-only store that any node may read *in full*, explicitly retiring the per-node input contracts
that [`archive/ports.md`](../archive/ports.md) made normative.

Two failure modes, and they are not symmetric:

* A **literal shared mutable store** — one file every process appends to — destroys the task hash.
  A node's result would depend on when it ran relative to its siblings, which is exactly the
  "execution order significant in a way the DAG does not declare" that makes `-resume` unsound
  rather than merely conservative.
* **Threading the whole store through channels as one value** restores purity, and reintroduces
  the port discipline through the back door: to hand a node its inputs you must know what they are.

### The reconciliation

**The store is an append-only log of single-author, content-addressed segments. Edges in the DAG
declare *visibility*, not content.**

Each node writes exactly one `segment.<node>.<sha256-16>.jsonl` and never touches another node's.
The store "as seen by node N" is the set of segments from N's ancestors, staged into `store_in/`.
The node reads whatever it likes in there and records what it read in its segment header.

The property that makes this sound is in `store.md` itself, in three sentences that were written for
other reasons:

1. "Nothing is deleted and nothing is overwritten."
2. An element id is "stable, assigned once by the node that first proposed the element" — so
   exactly one author writes any given element.
3. An assertion never mutates its target: `contest` leaves both claims standing, `withdraw` is not
   deletion, `refine` keeps the coarse extent.

Together those make the merged store a **grow-only set**. Union is commutative, associative and
idempotent, so concatenating segments in any order yields the same store. "The store as of node N"
is therefore a pure function of the *set* of upstream segments rather than of the *sequence* in
which they were written — which is precisely the property Nextflow needs, arrived at without giving
the port discipline back.

`STORE_VIEW` materialises the store the design describes, and its `integrity.json` reports what
would break the argument. One finding is **enforced** rather than reported: an element id claimed by
two authors exits non-zero, because it falsifies property 2 and with it the commutativity everything
else rests on. A `contest` is *not* an integrity problem — `store.md` says both survive and the
outcome is a `flag`, so the view records both and the fold does not choose.

I verified the commutativity claim rather than asserting it: four random permutations of one
recording's seven segments produce a byte-identical `elements.json`.

### What survived from the retired port discipline, and what did not

`archive/ports.md` §1.3 defined **two** kinds of input port: a *data port* wired to another task's
output, and a *parameter port* wired to one key of the versioned config. `store.md` retires the
first and says nothing about the second.

So the second stays, as `nodeConfig()` in `main.nf`: each node receives its own config slice, not an
ambient params object. That is not tidiness. It is the only lever left against over-invalidation
(below), and it is the rule whose violation `archive/ports.md` measured in the existing code —
`pass_summary: dict[str, Any]`, eight keys read from nine modules at 33 undeclared sites.

Model provenance rides the same mechanism. Every model reaches a process as a `val` input
`role=id@revision`, so **changing a revision changes the task hash**. senselab learned this
expensively: cache keys carried a bare `model_id`, an upstream push loaded new weights under an
unchanged key, and a result computed by the old commit was served as current. `validateParams()`
refuses any revision that is not 40 hex characters, and `triage-node` refuses again at run time,
eagerly, before any measurement is attempted.

### What breaks: `-resume`

**It works, and it over-invalidates. Measured, not estimated.**

Changing one model's revision — HeAR, used only by TAXONOMY and AIRWAY — invalidated **14 of 18
tasks**. Only ADMIT and PREPROCESS survived:

```
-resume --models.hear.revision <new sha>
  TAXONOMY   recomputed   correct: it loads HeAR
  AIRWAY     recomputed   correct: it loads HeAR
  SPEECH     recomputed   OVER-INVALIDATION: it loads no HeAR
  VOICE      recomputed   OVER-INVALIDATION: it loads no model at all
  REDACT     recomputed   OVER-INVALIDATION
  STORE_VIEW recomputed   correct: the store changed
  VERDICT    recomputed   correct: the verdicts changed
  ADMIT, PREPROCESS  cached
```

The mechanism: a node stages whole *segments*, and Nextflow hashes them. AIRWAY's segment hash moved,
so SPEECH's task hash moved, even though SPEECH reads only AIRWAY's `label` assertions and those did
not change. Under an element-level port discipline SPEECH would have declared `airway-labelled
spans` and could have stayed cached.

**This is the price of retiring the data ports, and it is the right side of the trade.** The failure
is over-invalidation: work is repeated, and a stale result is never served. The failure mode of
ports is *under*-invalidation — a node that reads more than it declared and gets a cached answer
computed from something that has since changed. In an append-only store that is unrecoverable,
because a wrong element cannot be deleted, only contested.

Two smaller notes:

* `process.cache = 'lenient'` hashes on size + timestamp rather than content, so a shared
  filesystem that rewrites timestamps does not invalidate the world. Switch to `'deep'` if you do
  not trust it.
* Segment names are content hashes and element ids are content-addressed, so a fresh run and a
  resumed run produce byte-identical segments. Verified. A counter-based id scheme would renumber
  everything downstream of any change and every cross-segment assertion would dangle.

### What breaks: two branches appending concurrently

Nothing, and the reason is worth being precise about, because "append-only" is doing less work here
than it looks.

There is no shared file. REDACT and VOICE — the only genuinely concurrent pair — write
`segment.redact.*.jsonl` and `segment.voice.*.jsonl` in separate task directories. There is no lock,
no last-writer-wins, no interleaving. The merge happens later, in a reader, over immutable files.

What *would* break is two concurrent nodes proposing the **same element id**, which is possible in
principle because ids are content hashes and two nodes could hash the same content. It cannot break
silently: `STORE_VIEW` exits non-zero on a duplicate id claimed by two authors. In practice the node
name is part of every id's preimage, so a collision requires a deliberate mistake.

The thing that genuinely does not work concurrently is a node **contesting an assertion made by a
sibling running at the same time**. `store.md`'s `confirm` and `contest` name a prior assertion, and
you cannot name what has not been written. Every `contest` in the design is within-node
(AIRWAY's YAMNet against AIRWAY's HeAR) or across a DAG edge, so nothing in the current design needs
this. A future node that wants to contest a *sibling's* claim needs a second pass, and the
round-based workflow this design deliberately retired is what a second pass turns into.

### What breaks: the three branches are not concurrent

**The brief's `{ AIRWAY, SPEECH → REDACT, VOICE }` fan-out is not realisable from the node documents
as written.** They describe a chain:

| edge | the sentence that puts it there |
| --- | --- |
| AIRWAY → SPEECH | `branch-speech.md` step 4: "A segment inside the interval that overlaps an `airway_spans` entry is **withdrawn**, not relabelled" |
| AIRWAY → VOICE | `branch-voice.md` step 1: the residual is "not covered by an airway-labelled span" |
| SPEECH → VOICE | `branch-voice.md` step 1: "not covered by a speech span" |

`branch-speech.md` marks its AIRWAY read "AIRWAY, if present", which reads like an optional input.
It is not one a scheduler can exploit. If SPEECH runs concurrently with AIRWAY, whether its step-4
withdrawals happen depends on which process finished first — a non-deterministic result from a
declared-deterministic graph. So the edge is drawn and the branches serialise.

**This is a cost of the store, not of Nextflow.** Under `archive/ports.md`, VOICE would have needed
a `residual_windows` port and there was no producer for it anywhere in the graph — which
`branch-voice.md` names as the defect the store fixed. The store gave the residual a producer by
letting VOICE read what the other branches asserted. The same freedom put two edges into the DAG
that the fan-out diagram does not show. **Read-anything is not free; it converts undeclared reads
into declared ordering.**

What is left is genuine: REDACT ∥ VOICE within a recording, and every recording in parallel with
every other. Recovering more would mean one of:

* **Splitting SPEECH step 8.** `branch-speech.md` calls SQUIM quality "a parallel branch of the
  graph" that "blocks nothing", and its outputs are `measurement` elements absent from SPEECH's
  verdict. It could be its own process running alongside REDACT and VOICE. Not done: it would put a
  second author on the same recording's quality claims and `store.md` gives no rule for that.
* **A two-phase graph** — branches on the store as of TAXONOMY, then a refinement pass. That is
  rounds, which this design retired.
* **Narrowing the reads.** If VOICE's residual subtracted only what it needs, some edges might go.
  That is the port discipline, and it is what `store.md` set out to remove.

### PII: why the two publish roots and the guard

After SPEECH the store holds an unredacted transcript, and being append-only it holds it forever.
REDACT produces a derivative and **cannot** make the store releasable. So:

* Two publish roots. `validateParams()` refuses a configuration where one is inside the other, in
  stub mode too, because a stub that teaches the wrong shape has taught the wrong shape.
* Only REDACT declares a `release/` output. Every other process publishes to `store/`.
* Figures, views and verdicts carry element ids for traceability. Correct in the store,
  **disqualifying** in a release artifact, exactly as `redact.md` says.

The release manifest deliberately carries less than `redact.md`'s verdict, and the reasoning goes
one step past what that document spells out. It omits element and assertion ids (the join key
`redact.md` names); extents (a position indexes the store's `pii` elements, which are keyed by
category + extent); `by_category` and `survived` (category plus position is most of a finding); and
the recording's content hash and any store-side run id — **a shared key is a join key whatever it is
called.** `redact.md` names element ids because they are the obvious case, not the only one. The
recording's *identity* does stay, in the release path: a redacted derivative is inherently of that
recording, and knowing which file it came from tells you nothing about where a name was removed.

The guard is part of the node, not a downstream concern, on the same principle `redact.md` applies to
verification. Before REDACT exits, `RELEASE_GUARD` sweeps every byte and every filename under
`release/` for the id pattern, for store-segment shapes, and for store-indexing JSON keys. A hit
exits 3, which aborts — and because `publishDir` runs only after a task succeeds, **nothing reaches
`release_dir`**. Verified by planting a pattern that matches: REDACT exit 3, no release directory.
There is a second, independent byte sweep in shell, so a bug in the Python is not the only thing
standing there.

SPEECH's verdict gets its own check: `CHECK_VERDICT_NO_PII_TEXT` refuses a `pii` block carrying
`text`, because a verdict that quotes the PII it found has published it into whatever reads the
verdict.

### Outcomes are values, never exit statuses

A branch `fail` is normal — a cough recording has no speech. Every node exits **0** when it ran,
whatever it concluded, and its outcome is a string in a verdict file. Non-zero means the node broke:

| exit | meaning |
| --- | --- |
| 0 | the node ran. `fail` / `flag` / `pass` is in the verdict file |
| 2 | usage error |
| 3 | `RELEASE_GUARD` refused, or a verdict quoted PII |
| 4 | a measurement has no source (no backend, no replay entry) |
| 5 | a parameter has no measured value and no admissible interval |
| 6 | a model revision is not a resolved 40-hex commit |
| 7 | store integrity: one element id, two authors |

Verified with `--stub_scenario cough`: SPEECH exits 0 with outcome `fail`, and the file verdict is
`triage: pass`.

The one place an outcome controls execution is ADMIT, because `verdict.md` triage row 1 requires it.
The mechanism is a **product**, not a verdict: ADMIT's `audio` output is `optional: true`, so a
`fail` emits no file and nothing downstream runs. VERDICT still runs, via `join(remainder: true)`,
which is what makes row 1 reachable at all. Verified: 2 tasks, `triage: fail`, `kinds: {}`.

The same trick handles "did REDACT run". SPEECH emits an optional `marker.transcript.json` — it
exists iff a recognizer returned a word. REDACT joins on it. A recording with no speech never
reaches REDACT and `verdict.md` gives `release: not_assessed`, which that document is careful to say
is **not** `releasable`, because the audio was never examined for content a transcript could not
carry. Reading SPEECH's `outcome` to decide this instead would put a verdict in the control flow,
which is the thing that breaks resumability.

**No branch is gated on TAXONOMY.** `verdict.md`'s "TAXONOMY said absent, the branch said pass →
flag" row is unreachable if a branch only runs when TAXONOMY admits its kind. Half the contradiction
table would be dead code. So every branch runs and TAXONOMY's outcome is advisory input to the fold.
See ambiguity **A-4**.

### Parameters with no measured value

The rule: **never a midpoint.** Two treatments, and which one applies is a property of the parameter,
not a convenience.

**Interval-valued.** Evaluate the rule at every admissible value. Unanimous → that answer.
Divergent → `undecided` / `flag`, because the measurement does not locate the boundary.

* VOICE's phonation gate. `branch-voice.md` gives periodicity anywhere in `(0.44, 0.933)` and RMS
  anywhere in `(0.0007, 0.0161)` — a factor of 2.1 and a factor of 23, on one recording, with the
  derivation slot deliberately empty. The gate is evaluated at both endpoints of each. A run that
  passes at both is voiced; fails at both is not; differs is recorded `gate_undetermined` and the
  branch flags. `branch-voice.md` already lists that flag — "the gate's parameters are still
  un-derived and a run sits near the interval's edge" — so this is its own rule made executable.
* TAXONOMY's `min_families`. Evaluated over `{2,3}` for airway and `{1,2}` for speech.
  `taxonomy.md` already defines `undecided` as "families disagree, **or any is unsure**", and a rule
  whose threshold is unlocated is unsure in the same sense. Exercised by
  `--stub_scenario undecided_min_families`.

**Unrunnable.** No interval, so nothing to evaluate over, so the node refuses.

* REDACT's padding margin must exceed the **worst** alignment edge error, which is unquantified.
  There is no admissible range, only an unquantified bound. So REDACT is **opt-in**
  (`redact.enabled = false` by default) and hard-errors with exit 5 if enabled with a null margin.
  Disabled, it does not run and `release` is `not_assessed` — `verdict.md`'s own vocabulary for
  exactly this. A median would leave a fragment of a name audible, and of the two edge failures
  that is the unrecoverable one.

**Neither.** SPEECH's SQUIM thresholds stay null and need no handling: `branch-speech.md` step 8 is
"reported, never gated", so the quality `fail` is unreachable by design and there is nothing to
resolve.

**The word-gap threshold** is a third case. It has no measured value, and rather than invent one or
refuse, the node falls back to the recognizer's own utterance segmentation — a measurement carrying
provenance, not a decision of ours — and records `span_grouping: "recognizer_native"` on every span
so a reader can tell which rule produced it. If the recognizer supplies no utterance boundaries
either, exit 5. This is a judgement call, and it is the one place I chose a behaviour the design does
not state; see **A-9**.

No config value in this repo is a number the design does not state. `models.hear.id`,
`models.alignment.id`, `models.squim.id`, `models.pii.id` and the second ASR/diarizer are `null`
because no design document names a repository, and `validateParams()` makes the operator supply
them.

---

## Ambiguities and under-specifications

Listed rather than guessed. Several are genuinely undecided by design and belong in
[`benchmarks/open.md`](../benchmarks/open.md); the rest are places two documents disagree, or a
document is silent on something an implementation must settle.

### Structural

**A-1. `store.md`'s element kinds are incomplete.** It names five — `span`, `word`, `speaker`,
`interval`, `measurement`. The node documents then use five more: `kind` (`taxonomy.md`), and
`stream`, `pii`, `target_match` (`branch-speech.md`'s view table); REDACT "writes new elements" of an
unnamed kind. Nothing reconciles the lists. *Implemented:* both sets accepted,
`integrity.json` reports `kinds_outside_store_md` rather than normalising them away.

**A-2. Assertions have no id, but `confirm`/`contest` must name one.** `store.md`'s assertion record
is `{ element_id, verb, value?, author, evidence }`, and its `confirm` verb is defined as agreeing
"with a **named prior assertion**". There is no field to name. *Implemented:* added `id` and
`target_assertion_id`.

**A-3. ADMIT's product is not an element of any kind.** A decoded waveform is not a span, word,
speaker, interval or measurement. *Implemented:* ADMIT writes no segment at all; the store begins at
PREPROCESS. Appending a level or clip element to have something to append is exactly the
accumulation of unread ports `admit.md` refuses.

**A-4. TAXONOMY both gates the branches and must not.** `taxonomy.md`: "so the graph knows which
branches to run", and `fail` means "every kind is absent — no branch would run". `verdict.md`
requires "absent + branch `pass` → flag", which cannot happen if absence skips the branch.
*Implemented:* every branch runs unconditionally; TAXONOMY's outcome is advisory. The alternative
makes two rows of the contradiction table dead code, which seems the worse reading — but this is a
real conflict and someone should decide it.

**A-5. The fan-out is a chain.** Covered above. The largest structural finding.

**A-6. VOICE's residual has no `K`.** `preprocess.md` says `K` is "per reader; AIRWAY reads at this
setting" — 18 dB. `branch-voice.md` step 1 wants "intervals carrying energy above the floor", and
the only such elements in the store are PREPROCESS's spans at AIRWAY's `K`. So VOICE silently
inherits an 18 dB contrast threshold it never chose, which is high for quiet phonation. Same for
`hangover`, declared "per consumer" with only one value stated. *Implemented:* VOICE reads the
K=18 dB spans, and this is flagged here rather than patched.

**A-7. `figure` is asymmetric.** AIRWAY and SPEECH both produce one; VOICE's product has none.
Unexplained. *Implemented:* as written.

**A-8. PREPROCESS has no outcome vocabulary** ("No `fail`, no `flag`") but `verdict.md` folds "every
node's verdict". *Implemented:* it always reports `pass` and puts its informative content —
`no_contrast`, `derivatives_written` — in the verdict body.

### Parameters with no measured value

**A-9. The word-gap threshold** (`benchmarks/open.md`: "unspecified. Any value is a claim about what
makes one utterance"). *Implemented:* recognizer-native utterance boundaries, recorded as such.
**This is my choice, not the design's.**

**A-10. `min_families` per kind.** Interval-valued over `{2,3}` / `{1,2}`. The admissible sets are
inferred: 1 is excluded for airway because one family is not "agreement". `taxonomy.md`'s remark
that two families is "near-unanimity for speech" reads like an argument for 2, but does not state it.

**A-11. The phonation gate's floors.** Interval-valued over the stated endpoints.

**A-12. The redaction margin.** Unrunnable; REDACT is opt-in.

**A-13. SQUIM thresholds.** Not needed — quality is never gated.

### Fold semantics

**A-14. `kinds` when ADMIT failed.** `verdict.md` says "nothing is claimed about the recording" and
also specifies a `kinds` map. *Implemented:* `kinds: {}`. A per-kind state word meaning "we never
looked" would be read as a measurement.

**A-15. REDACT `fail` conflates two findings.** `verdict.md` maps any REDACT `fail` to `withheld`.
"A PII finding survived verification" and "REDACT could not run" are different facts, distinguishable
only through `reasons`. *Implemented:* the second is exit 5 rather than a `fail`, so the conflation
does not arise; but if you enable REDACT with a margin you derived, and it then fails for a different
node-level reason, the distinction is only in `reasons`.

**A-16. Does REDACT run when SPEECH failed?** `verdict.md`'s `not_assessed` row implies not; nothing
says how the graph knows. *Implemented:* SPEECH's optional transcript marker.

**A-17. Fabrication candidates.** `branch-speech.md` lists "fabrication candidates survive" as a
`flag` condition; `benchmarks/open.md` says "SPEECH detects them and nothing acts on them". These
disagree. *Not implemented* — the detector is not written, so nothing flags on it. Flagged rather
than faked.

**A-18. VERDICT "writes one element".** It must also assert over TAXONOMY's `kind` elements to record
its resolution, so it writes one element and N assertions. Minor, but the count is wrong as stated.

**A-19. A branch `flag` against `TAXONOMY absent`.** `verdict.md`'s last row is "— | `flag` | flag,
whatever the screen said", which settles the triage axis but not the resolved *kind* state.
*Implemented:* the screen's state is kept if it was `present`/`absent`, else `undecided`.

**A-20. AIRWAY's hint rows overlap.** "`fail` when no span proposed **and no hint declares airway
content**" and "`flag` when a hint declares airway content not found". With a hint and no span, both
rows describe the situation. *Implemented:* `fail` is tested first and the hint negates it, so the
result is `flag` — consistent with "a hint changes only what an absence means" and "never promotes a
`fail` to a `pass`".

---

## Layout

```
main.nf              the graph, STORE_EDGES, validateParams(), nodeConfig()
nextflow.config      every parameter, each traceable to a design document or explicitly null
lib/Triage.groovy    config/model serialisation for the process command lines
modules/*.nf         one file per node; each header states what the node does and which edge it owes
bin/triage-node      the nodes: store, provenance, folds, guards, stubs, and the model seam
assets/              no-hints.json, an empty derivatives dir for standalone runs
```

Each module header names the sentence in the design that put its edges and its behaviour there. If
you change an edge, change the quote too, or the next reader cannot tell which one is authoritative.
