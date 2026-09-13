# The measured ruleset runs inside the pipeline — stage 1 of two, 2026-09-12

## What this is

One decision — which branches a recording routes to — has two mechanisms, and the one in production
is dead.

**Production.** PREPROCESS's evidence, TAXONOMY's kind lines, `kind_state`, `routing.BRANCH_FOR_KIND`,
branches. Every line is gated by a `taxonomy.presence_floor.*` value and every one of those values is
`null`, so `_line_state` reads `unavailable` for all of them, every kind folds `uncertain`, and
`routing` runs every branch on every recording. It has never routed anything on this corpus, because
it has never declined anything either.

**Measured.** `routing_analysis/ruleset.py`: eleven gates — ten that route a branch and one that
flags one — each a single feature path, one comparison and one threshold, all read from
`taxonomy.ruleset`. 0.97 / 0.95 / 0.96 / 0.94 sensitivity across AIRWAY / SPEECH / VOICE / DDK over
62,547 recordings, 333 of them (0.5%) reaching no branch. No node imports it, and `routing_analysis`
is by its own docstring a module that "runs no model and writes nothing into a run directory".

This stage makes the ruleset **run** inside the graph and **record** what it concluded. It changes
which branches no recording runs. `kind_state` still decides execution; the ruleset's selection is
written beside it so agreement can be counted on the corpus before the switch. Nothing is deleted.

## The problem: `extract_features` streams a file that does not exist yet

`evaluate_routes` takes a `RecordingFeatures`. `features.py::extract_features` builds one by
streaming a finished `run/store.jsonl` off disk. TAXONOMY holds the same store in memory, mid-run,
with no file.

## Decision 1 — the reader returns a whole `RecordingFeatures`

The alternative was a narrower structure carrying only what the configured gates read. It was
rejected, and not on grounds of effort.

`evaluate_gate` calls `gate_value`, which calls `detector_value(features, Detector(...))`. That
function is the single dispatch from a feature path to a number, it is typed on `RecordingFeatures`,
and it is twenty-two branches wide. A narrower structure does not avoid that dispatch — it needs a
second one, and a second one is a second definition of what `words.lexical`, `span_longest.amplitude`
and `ppg.silent_fraction` mean. That is precisely the drift this stage exists to prevent.

The economy the narrow option promised is also mostly illusory. The reduction is one pass over
records the store already holds; the per-field cost is bounded by what is in the store. The one
field that costs anything independent of the store is `ppg`, which opens a float16 npz sidecar and
runs `extract_ppg_segments` over it. Two of the eleven packaged gates read it, so the packaged ruleset
cannot skip it anyway.

The narrow option is not dead, it is deferred to where it can be had without a second definition.
`live_evidence.required_sources(ruleset)` already computes, from the loaded ruleset, the set of
feature sources the configured gates read — for the packaged ruleset:

```
bracketed_set, peak_label, peak_set, ppg, residual,
span_label_set_stat, span_longest, stream_peak_max, transcript_repeat, words
```

`stream_peak_max` is in that set and in no gate: it is what the emptiness bypass reads, and it is
consulted on every recording no gate fired on. Once stage 2 has factored the reduction (below), that
tuple is what a caller hands it to narrow the pass. It is recorded on the measurement now so the
corpus run can say what a given `config_hash` actually consumed.

## Decision 2 — one definition, kept by round-tripping the store rather than re-reading it

`extract_features` is one function: a `read_store(path)` generator, a sixty-line reduction over the
records it yields, and a post-pass over the live words and spans. The reduction is not separable
today without editing `features.py`, which another agent is in.

So the in-pipeline reader does the one thing that cannot drift: it hands the live store to the
store's **own** writer, `ProvStore.write_jsonl`, and the resulting file to `extract_features`. Both
paths then reduce the same bytes with the same code. The equality the tests assert is true by
construction in stage 1 — which is the point: it is pinned now so that stage 2's factoring cannot
quietly break it.

The file is written **under `run_dir`** and nowhere else. A measurement that names a sidecar names it
by a path relative to the store's own directory, and `_ppg_summary` resolves it against
`store_path.parent`. A serialisation in `/tmp` would silently turn every posteriorgram gate
`unavailable` — two of eleven — and the run would look healthy. It is a `NamedTemporaryFile` with the
`routing-evidence-` prefix and it is unlinked in a `finally`.

### What the round trip costs

A synthetic store at roughly PREPROCESS scale — 304 entities: 120 consensus words, 60 spans, 120
per-span classifier measurements carrying 30 raw scores each, a 521-label whole-file summary —
serialising to 115,371 bytes:

```
write_jsonl            : 3.5 ms
evaluate_live_routes   : 5.6 ms   (serialise + reduce + eleven gates + bypass)
```

Taken on a loaded laptop, so these are upper bounds. Against a PREPROCESS pass that runs CrisperWhisper,
Qwen, YAMNet, AST, HeAR, an enhancement model and a posteriorgram, 5.6 ms is not a cost that needs
defending. What would need defending is a second definition of eleven gates, and this buys it away.

## Decision 3 — absence stays absence

`GateOutcome` already separates `FIRED`, `SILENT` and `UNAVAILABLE`, and that separation is
load-bearing throughout the analysis: a gate whose measurement was never written is not a gate that
declined. Nothing here collapses it. The reader reproduces it because it reproduces the features,
and the recorded measurement carries every gate's outcome verbatim plus the per-branch `unavailable`
mapping `RouteEvaluation` already builds.

The same distinction is kept one level up. `RouteState` has three members and `routed` can be empty
for two different reasons — `EMPTY`, where the bypass found every tracked stream peak under
`emptiness.peak_floor`, and `UNEXPLAINED`, where it could not read one at all or read one over the
floor. And one level up again: `routing._ruleset_selection` returns `((), None)` when the
measurement is absent or its `state` is null, so a reading that was never made can never be counted
as a reading that routed nothing.

## Decision 4 — the evaluation carries no declaration

`RouteEvaluation` has a `declared` field: the branches the recording's task family is a reference
positive for. That is a reference standard, and it is read out of the BIDS stem's `task-` id, which
is a declaration.

TAXONOMY does not read declarations — `taxonomy()`'s own docstring says a classification that reads
the declaration cannot disagree with it. So the in-pipeline evaluation sets `task_id` and `family`
to `""`, `declared` comes back empty, and `agreed` / `missed` / `extra` with it. `routed`, `state`,
`unavailable`, `flags` and `gate_outcomes` are the fields this path fills, and they are content-only,
which is what the corpus comparison needs.

The `stem` is still filled, off the `recording` stream entity ADMIT wrote. That is an identifier for
joining a row back to a run, not evidence, and reading it needs no new argument threaded through the
node signature.

## Decision 5 — a reading that decides nothing must not be able to fail the node

TAXONOMY raising means `_drive_branches` skips routing, every branch and REDACT. A new,
non-authoritative, never-measured-in-pipeline reading with the power to do that would be a bad
trade. `_write_ruleset_routing` catches, and records the failure on the measurement:
`state: null`, `error: "<Class>: <first line>"`, through the same `describe_exception` the runner
uses. The failure is then countable over the corpus rather than invisible or fatal.

This is not error-swallowing dressed up. The discipline is the one `run._attempt` already applies at
the node boundary, applied one level down because the reading sits below the node's own contract. It
stops applying the moment the ruleset becomes authoritative: at that point a failure to route *is* a
failure to run the graph, and stage 2 must let it raise.

A test asserts the reading is actually made on a realistically seeded store — an evaluation that
always recorded an error would pass every other test in this stage and say nothing.

## What TAXONOMY writes

One `measurement` entity named `ruleset_routing`, on `taxonomy`'s own step
`ruleset_routing`, via the shared `write_measurement`. `vocabulary.RULESET_ROUTING` names it, so
`routing` reads it back without importing the reader or its pyarrow-bearing dependencies.

| attribute | |
| --- | --- |
| `authoritative` | `false`, for as long as `kind_state` decides execution |
| `error` | null, or the failure |
| `state` | `routed` \| `empty` \| `unexplained`, or null when `error` is set |
| `routed` | the branches a gate fired for, in `BRANCHES` order — DDK included |
| `gate_outcomes` | every gate's outcome by name; every gate is evaluated on every recording |
| `unavailable` | per branch, the gates whose feature could not be read |
| `flags` | per branch, the flag gates that fired; a flag annotates and never routes |
| `sources` | what `required_sources` says this configuration consumes |
| `stem` | the recording's stem, for the join |

Both shapes carry the same keys. A reader discriminates on `state`/`error`, never on a missing key.

**It is written after the label summaries and the consensus taxonomy, and that ordering is a
requirement, not an accident.** `voice.glide` and `voice.chant` read `plain|yamnet`, which
`extract_features` populates from the `yamnet_label_summary` measurement — which TAXONOMY itself
writes, in the step before. Evaluated earlier, both gates would read `unavailable` and VOICE would
never route. A test pins it.

### Where the in-pipeline reading and the analysis reading legitimately differ

They reduce the same code, but not the same store: TAXONOMY's store is a prefix of the final one.
Every gate's evidence is written by PREPROCESS or by TAXONOMY's own earlier steps, so no gate is
affected today. A future gate reading something a *branch* writes would read `unavailable` in the
pipeline and available in the analysis — the equivalence tests would not catch that, because they
compare one store both ways. Any gate added to `taxonomy.ruleset` must be checked against this.

## `routing` records both selections

`routing` reads `ruleset_routing` and stamps every `branch_decision` with `ruleset_will_run` beside
`will_run`, and `ruleset_route_state` beside `kind_state`. The two columns are on one entity, so
corpus agreement is a column pair rather than a join. `RoutingResult` gains `ruleset_runs` and
`ruleset_state` for the same reason at the runner level.

`will_run` is unchanged, and nothing reads `ruleset_will_run` to decide anything. `BranchDecision`
in `vocabulary.py` does not carry the new attributes and `fold_file_verdict` therefore cannot see
them — deliberately: the fold's `kinds`, its `agreement` table and its `acoustically_empty` discard
must not move under a reading that is not yet trusted.

## DDK: expressible, not runnable

`vocabulary.BRANCHES` has named DDK since it was written. `routing.BRANCH_FOR_KIND` has three
entries, `run._drive_branches`'s dispatch table has three, and `for branch in BRANCHES:
branches[branch]` therefore raises `KeyError` — and kills the whole run — the moment anything puts
DDK into the selected set. The ruleset routes DDK, so this stage had to make that safe.

**What was not done, and why.** Adding `"ddk": "DDK"` to `BRANCH_FOR_KIND` looks like the obvious
move and is wrong. TAXONOMY writes no `ddk` kind line; `routing` reads a missing line as
`uncertain`; `uncertain` runs the branch. DDK would then be selected on *every* recording, VERDICT
would flag every recording with "DDK was asked to run and never ran", and a fourth `branch_decision`
would enter the fold — carrying a `ddk` kind whose state is not `absent`, which alone would stop the
`acoustically_empty` discard from ever firing again. A branch with no node must not acquire a kind
line by default, and it must not acquire a kind line that asserts absence without evidence either.

**What was done.** The crash is a dispatch problem, so it is fixed at the dispatch. `run.py` looks
the branch up rather than indexing it; a branch with no node is recorded `SKIPPED` carrying the note
`NO_NODE`, selected or not, and the note reaches `run.json`. `routing.py` declares
`UNCLASSIFIED_BRANCHES` — the branches in `BRANCHES` that no kind line decides, which is exactly
`("DDK",)` — and carries the ruleset's full routed set, DDK included, in `ruleset_runs`. And
`TriageRunResult.nodes` now keeps every branch it recorded an outcome for: it was filtered to
`GRAPH_ORDER`, which has no DDK, so DDK's outcome was being computed and then dropped.

A test drives a real run with DDK forced into the execution set and asserts it finishes with a file
verdict.

### What a DDK branch would need

- `nodes/ddk.py` on the shared node shape, `(store, source, config, hint, *, run_dir)`, returning a
  `NodeResult` whose verdict carries `kind="ddk"`.
- A `ddk` entry in `GRAPH_ORDER` and in `run.py`'s dispatch table.
- A `ddk` kind line from TAXONOMY, or — better, and the direction this whole spec points — the
  ruleset becoming authoritative so that `routing` reads routed branches rather than kind states,
  and DDK needs no kind line at all.
- A subject. `specs/20260911-praat-ppg-detectors/design.md` measured 859 declared-DDK recordings
  never reaching the branch and gave `ddk.ppg_segment_rate_per_s` its operating point; what the
  branch would then *conclude* about a rate is not decided anywhere yet.

## What stage 2 must delete

Precisely these, once corpus agreement has been measured:

1. **`taxonomy.presence_floor`** in `data/config/default.yaml`, whole subtree, and the `floors`
   mapping in `nodes/taxonomy.py` that reads it.
2. **The kind-line machinery in `nodes/taxonomy.py`** that exists only to feed `kind_state`:
   `_line_state`, `_window_line`, `_span_line`, `_fold_speech_lines`, `_fold_authoritative_line`,
   `_acoustic_line`, `_window_evidence`, `_lexical_line`, `_span_label_evidence`,
   `_retired_voice_line`, `_unavailable_windows`, and the `lines` / `states` construction and the
   `kind` entities in `taxonomy()`. `_transcribed_span_ids` stays if AIRWAY still reads the same
   rule; check it before deleting.
3. **`routing.BRANCH_FOR_KIND`, `KIND_STATES`, `UNREADABLE`, `_classifications`, `_why`** and the
   whole kind-state arm of `routing()` — replaced by reading `ruleset_routing.routed`.
   `UNCLASSIFIED_BRANCHES` goes with them: with no kind lines there are no unclassified branches.
4. **The parallel columns added here.** `ruleset_will_run` and `ruleset_route_state` on
   `branch_decision`, and `ruleset_runs` / `ruleset_state` on `RoutingResult`, exist only to hold the
   two selections side by side. When one selection remains, `will_run` is it. Pre-alpha: rename and
   replace, no aliases.
5. **`authoritative` on the `ruleset_routing` measurement**, and the `try`/`except` in
   `_write_ruleset_routing` with `failed_route_attributes` beside it. Once the ruleset decides what
   runs, a failure to evaluate it is a failure of the node and must raise.
6. **`SCREENED_KINDS`**, and `kind_state` in `BranchDecision` / `FileVerdict.screened` / the
   `agreement` table in `vocabulary.py` — but only after deciding what the fold's agreement axis
   compares once TAXONOMY no longer classifies. That is a separate decision, not a deletion.
7. **The round trip.** See below; it is stage 2's other half.

## What must move in `features.py` so the two paths keep one definition

Not edited in this pass — another agent is in the file. The move is one refactor, mechanical:

**Split `extract_features` in two at its first line.**

```python
def reduce_records(
    records: Iterable[dict[str, Any]],
    *,
    run_dir: Path,
    stem: str,
    run_root: str,
    task_id: str,
    family: str,
    memberships: Mapping[str, LabelMembership],
    onomatopoeic: frozenset[str],
) -> RecordingFeatures:
    ...the current body, unchanged, with `store_path.parent` replaced by `run_dir`...


def extract_features(store_path: Path, stem, run_root, task_id, family, memberships, *, onomatopoeic):
    return reduce_records(read_store(store_path), run_dir=store_path.parent, stem=stem, ...)
```

Two things change inside the body, and nothing else:

- `_absorb_measurement(features, attributes, span_scores, store_path.parent)` takes `run_dir`
  directly. It is the only use of the path in the reduction; everything else reads records.
- `_ppg_summary`'s `run_dir` argument then comes from the caller rather than from the file's
  location, which is what removes the constraint that forced the serialisation under `run_dir`.

`live_evidence.read_live_features` then builds the record stream from the live store in memory —

```python
({"record": "entity", "id": e.id, "prov_type": e.prov_type, "extent": e.extent, "attributes": e.attributes}
 for e in store.entities())
```

plus the `wasInvalidatedBy` triples as relation records — and hands it to `reduce_records`. The temp
file, `EVIDENCE_PREFIX`, `EVIDENCE_SUFFIX` and the `finally` go. The equivalence tests are unchanged
and stop being tautological, which is when they start earning their place.

One caveat for whoever does it: `write_jsonl` uses `json.dumps(..., default=str)`, so today a store
attribute that is not JSON-native (a `Path`, a numpy scalar) round-trips as a string and the two
paths agree on the string. Reading the live store directly skips that coercion and the two paths
would then disagree on such an attribute. Nothing in the packaged gates reads one — but the
equivalence tests are where that would surface, and they should be kept pointed at the disk path for
exactly that reason.

## Not in scope, deliberately

- No new config key. `taxonomy.ruleset` is read as it stands.
- No threshold in code. Every number the reading uses comes from `taxonomy.ruleset`.
- Nothing deleted. The kind-line path is untouched and still decides.
- No corpus run. Agreement between the two selections is measured by the owner.
