# The branch foundation — the decisions `expected-patterns.md` could not express

[`expected-patterns.md`](expected-patterns.md) is the design: 58 `Expectation` rows over 13
`Pattern` kinds, nine entry points, a `Proposal`, a `proposer()` per branch and the shared helpers,
all of it Python that compiles and runs against a synthetic store. This document is what porting it
to `src/senselab` decided, in the four places the design leaves a choice to code, plus what did not
survive contact with the real objects.

The implementation is `src/senselab/audio/workflows/triage/nodes/branches.py`; its tests are
`src/tests/audio/workflows/triage/nodes/branches_test.py`. Nothing here implements a branch.

## D-F1. One new module beside `common.py`, not inside it

`nodes/common.py` is imported by ADMIT, PREPROCESS, TAXONOMY, REDACT, REPORT, VERDICT and the four
branches alike. None of the first six has a branch, an expectation or a proposal, and three of them
run before any branch does. Putting 58 rows of family data, 13 `Pattern` kinds, the two-mode
dispatch and the numpy array helpers there would:

* put branch vocabulary in every node's import path, including nodes that cannot reach it;
* add `numpy` and `routing_analysis.families` to `common.py`'s imports, which it currently has
  neither of — `common.py` imports nothing from `routing_analysis` at all;
* take a 481-line module past 1,300.

So `branches.py` sits beside it and imports from it (`find_measurement`, `live_entities`). The
direction is one-way and stays that way: `common.py` must not import `branches.py`.

The import of the family sets is `from ...routing_analysis.families import ...`, the submodule, not
`from ...routing_analysis import ...`, the package. `routing_analysis/__init__.py` re-exports
`gate_plot`, which imports matplotlib; `families.py` imports nothing from senselab and nothing
heavy. A branch must not pay a plotting import to ask what family it is on.

## D-F2. The declared task family is derived in one shared helper, called by `dispatch`

`align_*` takes a `task_family` and nothing passes one to a branch. Routing hands a branch one bit,
`will_run` (`run.py:298-300` passes `(store, "plain", config, hint)`), and no branch reads its own
`branch_decision`. Three places could derive it.

**Rejected: the runner.** `run.py` would have to grow the derivation and hand it to each branch,
which changes the branch signature the four implementations are written against and makes the runner
the place a *second* carrier has to be reconciled. A routing change is landing in parallel that makes
a declared task always route to its own branch; if the runner owned the derivation, that change and
this one would both be editing the same argument list.

**Rejected: each branch.** Four copies of BIDS-stem parsing inside four branch modules, which is
precisely what `AudioHints` exists to prevent, and four places to update when a carrier appears.

**Chosen: `declared_task_family(store, hint)` in `branches.py`, called by `dispatch`.** One
implementation, in-graph, reading carriers in order:

1. `hint.metadata["task_token"]` — the clean route. Written today only by this spec's own
   `runs/b2ai-v2/make_hints.py` and read by nothing in `src/senselab`; reading it here means it
   starts working the moment anything populates it, with no branch changing.
2. the `path` ADMIT writes onto the `recording` stream entity (`admit.py:101`), through
   `task_family(task_id_of(stem))`.

A third carrier — routing's declared task, when the parallel change lands — is one more entry in
`_declared_task_ids` and no change in any branch or in `run.py`. **That is the coordination
contract: the source is behind the helper, not in the caller.** `nodes/routing.py` is not touched.

`None` is returned, never raised, when no carrier names a family: an absent `recording` entity, a
`recording` with no `path`, a path that is not a BIDS stem, or a stem whose task id is `"unknown"`.
`None` takes the out-of-family mode, which is the safe arm — the branch annotates its speciality and
concludes nothing. There is no third arm.

The known loss is the design's own: `task_family` strips every trailing numeric segment
(`families.py:13`, `:143`), so the `fivebreaths` route and the `maximum-phonation-time-v2` effort
escalation are not recoverable from either carrier. A stem literally spelling `task-unknown` is also
read as no declaration; that is honest rather than correct, and it is cheaper than a sentinel that
could collide with a real family.

## D-F3. `dispatch` enforces the two rules the design states as prose

The design says a proposer's family is fixed by the branch, and that `detect_*` returns
`UNDETERMINED` always — "a rule, not a default". Both are enforced at the seam rather than trusted:

* every returned `Proposal` whose family is not the branch's own is a `ValueError`. `proposer()`
  already makes this unreachable through the minting helper, so what the check catches is a
  hand-built `Proposal`, which is exactly what a branch under time pressure writes.
* a `detect_*` result whose `done` is anything but `UNDETERMINED` is a `ValueError`. The design
  records an earlier draft giving SPEECH `done = (no lexical word was found)` over the AIRWAY
  families, which reads the absence of lexical content as a verdict on someone else's task. That
  specific defect is now a test failure rather than a review catch.

`dispatch` takes the two entry points as arguments rather than reading them out of a registry.
A registry in `branches.py` would have to import `airway.py`, `speech.py`, `voice.py` and the DDK
module, each of which imports `branches.py` — a four-way cycle for no gain.

## D-F4. `BranchParams` reads lazily, one key at a time

The design writes `Params` as a frozen dataclass with 40 eagerly-constructed fields. Every `p_*` in
it is unfitted, so all 38 numeric ones ship `null`, and reading a null raises. An eager record would
therefore raise at construction on every recording, failing the bodies that need none of the nulls
along with the bodies that need one.

`BranchParams` is a frozen dataclass holding the `TriageConfig` and one property per key, each
calling `config.require` on access. The property names are the design's own spelling (`p_score_min`,
`p_modulation_band_hz`), so porting a body is transcription. The failure a branch sees is
`branch.event_min_s has no value …`, naming the key it needed, on the recording it was asked about.

Properties rather than a `__getattr__` mapping: `__getattr__` returning `Any` would let
`params.p_peek_prominence_db` pass mypy and fail at runtime, and the whole point of this record is
to be the contract four separate implementations are checked against.

`p_normalise` gets no config key. It is a function, and a key naming one would be a plugin hook
nobody measured; it resolves to `consensus.vocabulary_key`, the normalisation the consensus
(`consensus.py:42`) and the stimulus alignment (`stimulus.py:35`) both declare. `p_label_sets` is
the one key that ships a value, and it takes no new decision: it is
`airway.labels_of_interest` split by which kind each label names. The rest is in
[`config-derivations.md`](config-derivations.md) § branch.

## D-F5. The propose path, and the two reserved attributes

`propose_span` is the only function that writes a proposal, and it writes a `span` entity carrying
`family`, `role`, the extent, the proposal's remaining attributes, `wasGeneratedBy`,
`wasAttributedTo` and one `wasDerivedFrom` per evidence id. It reuses the shape `speech.py:880-897`
already uses for a derived span; what it adds is that the derivation is required rather than
conditional.

Porting turned up a hole the design's version has. The design mints
`{"family": …, "role": …, **attributes}`, so an attribute named `family` is overridden — but the
attributes were spread *last* in the first port, and an attribute named `family` then silently
replaced the branch's own stamp. Two changes close it:

* `proposer()` refuses `family` or `role` in `**attributes` (`RESERVED_SPAN_ATTRIBUTES`), the same
  way `write_verdict` refuses the four reserved verdict keys;
* `role` and `extent` are **positional-only**, so `role=` reaches that check instead of colliding
  with the parameter, and `propose_span` spreads the attributes first and the two stamps last, so a
  hand-built `Proposal` cannot displace them either.

`write_findings` writes the other three kinds in the forms the design's own table names: `deviation`
and `contest` as assertions beside a span (`verb: "deviate"` / `"contest"`), `measure` as its own
measurement over its extent, and every `count` folded into **one** `counts` measurement whose
`entries` carry `found` beside `declared`. A count asserts no discrepancy. An unknown kind is
refused rather than dropped, because a finding nothing writes is a finding nobody sees — and
`refine` is exactly the kind a body might still reach for.

## What did not survive contact with the code

**The store facade is not real.** The design's bodies read `store.words`, `store.spans`,
`store.energy_envelope`, `store.level`, `store.stream_extent` — a named facade that does not exist
over `ProvStore`. The helpers are therefore written over the real objects: `Entity` with
`.attributes`, so `span.measure` becomes `span.attributes.get("measure")` behind
`spans_by_measure`, and `word.text` becomes `word_text(word)`. Four small frozen records
(`EnvelopeTrack`, `ContinuityTrack`, `PhonationTracks`, `SpectrogramBlock`) plus their `read_*`
loaders stand in for the four derivatives the instruments need, so `events_in_span` and
`train_rate_hz` are unit-testable on a hand-built array rather than only against a run directory.
Each loader returns `None` when the derivative, its path, its sidecar or its rate is absent — three
of the four branches would otherwise learn that separately, and at least one of them wrongly.

**A duplicate npz reader now exists.** `derivative_arrays` in `branches.py` does what `_npz` in
`figure.py:341` does, and what `voice.py:278` and `routing_analysis/features.py:676` do inline.
Collapsing the four is a separate change over files this work is not to touch; recorded here so it
is not rediscovered.

**`events_in_span` reports nothing on a flat-topped plateau.** The walk requires prominence over
`max(left_min, right_min)` — a trough on *both* sides — and the last sample of a digitally flat rise
has none on its left, so its prominence computes as 0. A clipped or limiter-flattened cough
therefore yields no event. This is the design's own arithmetic, ported unchanged and now pinned by a
test (`test_a_flat_topped_plateau_yields_no_event`) so an AIRWAY author meets it as a documented
property rather than as a silent zero. Whether the walk should smooth first or accept a one-sided
trough is a measurement, not a transcription choice.

**`spectral_balance_db` excludes the Nyquist bin.** Its high band is `[split_hz, sr/2)`, half-open,
so the bin at exactly `sr/2` is never counted, and a `split_hz` equal to Nyquist selects nothing and
returns NaN. One bin in 257 at a real transform length; pinned rather than changed.

**`Done` needed a `Literal`.** `UNDETERMINED = "UNDETERMINED"` infers as `str`, which makes every
`Result(UNDETERMINED, …)` a type error against `Done = bool | Literal["UNDETERMINED"]`. Declared
`UNDETERMINED: Literal["UNDETERMINED"]`.

**The 58 rows are 58.** `EXPECTATIONS` covers `AIRWAY_ELICITING` (11), `SPEECH_ELICITING` (31),
`VOICE_ELICITING` (6) and `SYLLABLE_REPETITION` (10) exactly, with the ten syllable families in both
SPEECH and DDK, verified against `families.py` by test rather than by count. The four
`VOICE_EXPECTATIONS_PENDING_DECLARATION` rows are kept as their own table and are **not** in
`EXPECTATIONS`: `cape-v-sentences`, `-v2`, `loudness` and `loudness-v2` are `LEXICAL_SPEECH`, so they
are out of family for VOICE under the reference family set, and `align_voice` must not reach them.
Changing that is a `families.py` decision, not a branch one.

## The signatures the four branch implementations are written against

```python
def align_<branch>(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result
def detect_<branch>(store: ProvStore, params: BranchParams) -> Result
```

and the node entry point, unchanged from what `run.py` already calls:

```python
def <branch>(store: ProvStore, source: str, config: TriageConfig,
             hint: AudioHints | None = None, *, run_dir: Path) -> NodeResult
```

whose body is

```python
result = dispatch("<BRANCH>", store, branch_params(config), hint,
                  align=align_<branch>, detect=detect_<branch>)
```

followed by `propose_spans` over `result.components`, `write_findings` over `result.deviations`, and
`write_verdict`. `run.py` changes not at all.
