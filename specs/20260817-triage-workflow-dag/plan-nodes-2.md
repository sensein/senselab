# Triage Node Implementation Plan — Part 2: SPEECH, VOICE, REDACT, VERDICT

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the last four nodes of the audio-triage workflow — SPEECH, VOICE, REDACT, VERDICT — over the merged foundation (`ProvStore`, the triage config, the vocabulary, and the six DSP/redaction tasks).

**Scope split:** This plan is Tasks 5–8. The first four nodes — ADMIT, PREPROCESS, TAXONOMY, AIRWAY — are Tasks 1–4 in `plan-nodes-1.md`, planned by a sibling. The two plans answer to the same design documents; where this plan names an element the earlier nodes write, it uses the design documents' own vocabulary (`prov_type` plus attributes), and **the plan reviewer reconciles the two plans' namings**. Nothing here plans, modifies, or assumes internals of the sibling's four nodes beyond what `preprocess.md`, `taxonomy.md`, `branch-airway.md` and `store.md` state.

**Architecture:** Each node is one module in `src/senselab/audio/workflows/triage/nodes/`, taking the store and the config, writing entities/activities/agents/relations to the store, and returning a `NodeProduct` — outcome + verdict + view, never a copy of the store's content. The branches are **not** concurrent: SPEECH reads AIRWAY's labelled spans (optionally — no labels means no withdrawals), VOICE's residual subtracts both AIRWAY's and SPEECH's claims and its read is not optional. Only `REDACT ∥ VOICE` survives as concurrency. No orchestrator is built here.

**Tech Stack:** Python 3.12, pydantic v2, numpy, scipy, pytest. uv for everything.

## Hard prerequisite: the separation backend is not on this branch

Verified by reading the tree at the start of planning: `design/triage-workflow-dag`'s
`src/senselab/audio/tasks/source_separation/api.py` has **only the unasdiff backend**, and its
containment check **refuses** any `HFModel` whose id does not start with `sensein/unasdiff` — so the
`separate_audios(audios, model=HFModel("alibabasglab/MossFormer2_SS_16K"), n_sources=2)` call that
`capability-map.md` §1.5 documents raises `ValueError` on this branch. `utils/clearvoice.py` does not
exist here either. Both exist on branch **`triage`** (verified:
`git show triage:src/senselab/audio/tasks/source_separation/api.py` dispatches through
`is_clearvoice_model_id` to `separate_audios_with_clearvoice(audios, model, device=None, timeout_s=None)`,
and refuses `n_sources != spec.expected_outputs`). `capability-map.md` describes the `triage` branch
accurately, not this one — the map is evidence, and on this point it is evidence about a different tree.

**Before Task 5's separation step, merge `triage` into this branch** and re-verify
`separate_audios`'s signature against the merged tree. Everything in Task 5 up to step 5, and all of
Tasks 6–8, has no dependency on the merge. (The sibling's plan carries the matching prerequisite for
HeAR; if it has already merged `triage`, re-verify and move on.)

## Global Constraints

Copied verbatim from the foundation plan; every one is binding here too.

- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates ~535 MB of frameworks. Run the directory you changed.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`.
- Google-style docstrings; line length 120; type hints required (mypy with the pydantic plugin). ruff applies to test code, so **every test class and function needs a docstring and every test function `-> None`**.
- **Rationale does not go in code.** Docstrings say what a thing is and how to call it. Measurements and rejected alternatives go in `specs/20260817-triage-workflow-dag/benchmarks/`.
- **No numeric constant appears in code.** Not as a signature default, not as a module-level constant. Every number lives in `data/config/default.yaml` with its derivation beside it, per `CLAUDE.md`: "Thresholds belong in `data/` with a written derivation, never as code literals." Functions take the values they need as arguments; the caller reads them from the config.
- **A value nobody has measured is `null` in the config, and reading it raises.** The loader names the parameter and points at `benchmarks/open.md`. This replaces keyword-only-without-default as the mechanism for making an unmeasured value impossible to use by accident.
- A model load passes a **resolved 40-hex commit SHA, never a ref**. There is an AST-sweep guard test (`src/tests/utils/revision_pinning_guard_test.py`).
- `uv sync` is subtractive — always pass `--all-extras`.
- Run `ruff format` before every commit.
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation shims.
- **A plan (or an implementation) supplying a value for anything in `benchmarks/open.md` is wrong.** Those keys stay `null`; tests exercise them through explicit YAML overrides, which is the intended production mechanism too.

**Reading this plan's constraints:** unit conversions (`padding_ms / 1000.0`) and mathematical
identities are not thresholds — the foundation's own `plan_redactions` contains exactly that
conversion; the numeric-constant rule bars unfitted *decisions*, not arithmetic. Test snippets in
this plan elide pytest fixture parameters (`tmp_path`, `monkeypatch`) and shared-fixture plumbing for
brevity; the implementer writes each test in full, with a docstring and `-> None`, taking every
fixture it uses.

## Assumed shared contract — the reviewer reconciles this against `plan-nodes-1.md` Task 1

The sibling's Task 1 defines the node contract. This plan **consumes** that contract and restates here
exactly what it depends on, mirroring the design documents' product sections. If the sibling's Task 1
names any of this differently, the reviewer reconciles; an implementer of Tasks 5–8 must read the merged
Task 1 as built, not this restatement.

**Module layout.** Nodes at `src/senselab/audio/workflows/triage/nodes/<name>.py` (`speech.py`,
`voice.py`, `redact.py`, `verdict.py`), shared helpers at
`src/senselab/audio/workflows/triage/elements.py`. Tests mirror at
`src/tests/audio/workflows/triage/nodes/<name>_test.py`.

**Node signature convention.** Every node takes `(store: ProvStore, config: TriageConfig, run_dir: Path)`
plus what its design doc names: `hint: AudioHints | None` for the branches, an `artifacts_dir` for
REDACT, nothing extra for VERDICT. Nodes load audio **through the store**: ADMIT/PREPROCESS write the
recording and its resampled/pre-emphasised derivatives as WAV sidecars under `run_dir`, each recorded as
a `stream` entity whose attributes carry the relative `path` and `sampling_rate`; a branch resolves the
entity, records `used`, and constructs `Audio(filepath=run_dir / attrs["path"])`. Heavy arrays (the
envelope, its floor) are `.npz` sidecars under `run_dir / "derivatives"` referenced by `measurement`
entities carrying `{"name", "path", "sampling_rate"}`.

**Helpers from `elements.py`** (signatures as the sibling's Task 1 produces them):

```python
@dataclass(frozen=True)
class NodeProduct:
    outcome: Outcome
    node_verdict: NodeVerdict
    verdict: dict[str, Any]            # the node-specific verdict mapping from its design doc
    view: list[str]                    # element ids this node authored or asserted over

def software_agent(store: ProvStore) -> str
def model_agent(store: ProvStore, *, model_id: str, commit_sha: str | None,
                unresolved_reason: str | None = None) -> str
def write_node_verdict(store: ProvStore, *, activity_id: str, node_verdict: NodeVerdict,
                       detail: dict[str, Any]) -> str          # a "verdict" entity; VERDICT reads these
def read_node_verdict_entities(store: ProvStore) -> list[Entity]
def node_verdict_from_entity(entity: Entity) -> NodeVerdict
def write_measurement(store: ProvStore, *, activity_id: str, name: str, attributes: dict[str, Any],
                      extent: tuple[float, float] | None = None, agent_id: str | None = None) -> str
def find_measurements(store: ProvStore, name: str) -> list[Entity]
def find_measurement(store: ProvStore, name: str) -> Entity | None
def write_assertion(store: ProvStore, *, verb: str, subject_id: str, activity_id: str,
                    attributes: dict[str, Any], agent_id: str | None = None,
                    answers: str | None = None) -> str
def assertions_about(store: ProvStore, subject_id: str, verb: str | None = None) -> list[Entity]
```

**What the store holds when Task 5 starts** (the design documents' vocabulary; authored by Tasks 1–4):

| element | `prov_type` | attributes this plan reads |
| --- | --- | --- |
| the recording | `stream` | `{"name": "recording", "path", "sampling_rate"}`, extent `(0.0, duration_s)` |
| resampled 16 kHz plain signal | `stream` | `{"name": "resampled_16k", "path", "sampling_rate"}` |
| ASR hypotheses (CrisperWhisper, Qwen) | `word` | `{"text", "asr_model", "score"?, "timestamp_source"?, "timestamp_model"?}`, extent `(start, end)` per word, `ScriptLine`-shaped timing |
| the envelope and its floor | `measurement` | `{"name": "energy_envelope" / "floor", "path", "sampling_rate"}` — npz sidecars |
| YAMNet Silence windows | `measurement` | `{"name": "silence", "windows": [{"start", "end", "score"}]}` |
| SQUIM (PREPROCESS's, per envelope span) | `measurement` | `{"name": "squim", ...}` — read for nothing here; SPEECH re-measures over its own spans |
| PREPROCESS's envelope spans | `span` | `{"peak_over_floor_db"}`, no label |
| TAXONOMY's predictions | `kind` | `{"kind": "airway"/"speech"/"voice_no_words", "state": "present"/"absent"/"undecided"/"not_screened", per-family evidence}` |
| AIRWAY's labels | `assertion` | `{"verb": "label", "label": ...}`, `wasDerivedFrom` the span it labels |
| every node's conclusion | `verdict` | written via `write_node_verdict`; `node_verdict_from_entity` reads it back |

**Where `NodeVerdict`s live (a decision `verdict.md`'s product section forces):** `verdict.md` says the
fold's view carries "the node verdict ids it folded" — verdicts have element ids, so **they live in the
store**, as `verdict` entities each node writes via `write_node_verdict` before returning. The
`NodeProduct` a node returns is a convenience for its caller; VERDICT reads only the store. `ran` is the
one thing the store cannot carry (only the runner knows a node **errored**), so VERDICT accepts it from
the caller, with a derived fallback (Task 8).

**Mocking boundary** (pattern from `src/tests/audio/workflows/pii_adapter_test.py`: patch where the
callee resolves the name). Every node module imports its model-calling functions **by name at module
top**; tests `monkeypatch.setattr(node_module, "<name>", fake)` on the node module. No test loads
YAMNet, SQUIM, pyannote, a second diarizer, MossFormer, ECAPA, an ASR model, Praat, or the PII
subprocess. Each fake's payload shape is verified in this plan against the real function's return
type, and each task's Interfaces block restates the shape its fakes must honour. Pure DSP runs real:
`fuse_word_streams`, envelope/floor reads, span grouping, `detect_disruptions`, `plan_redactions`,
`apply_redactions`, `extract_segments`, all store operations.

**Unmeasured `require()` keys make nodes refuse by design.** With the packaged config, SPEECH
(`speech.word_gap_ms`), VOICE (`phonation.f0_min_hz`, `f0_max_hz`, `hnr_floor_db`, `rms_floor`) and
REDACT (`redaction.padding_ms`) raise `ValueError` from `TriageConfig.require`, naming the key and
`benchmarks/open.md`. **Each node resolves every `require()` key it needs at entry, before its first
store write**, so an unmeasured key leaves the store untouched and the runner records the node as
`errored` — which VERDICT's fold turns into a flag for any kind the screen called present or undecided.
The rejected alternative — running the steps that need no unmeasured value and refusing mid-node — was
rejected because it leaves a partially-written store whose contents depend on which key was null, and
because a node whose write-set varies with the config's null pattern is harder to reason about in an
append-only store than one that either ran or did not. The cost, stated honestly: under the packaged
config a recording with no speech makes SPEECH **raise** rather than return its no-words `fail`,
because the gap threshold is resolved before the words are read. That is intended: the packaged config
is not runnable for these three nodes until overrides exist, and `benchmarks/open.md` says so. This is
intended behaviour, not a defect; each task has one test asserting it, and every other test supplies a
YAML override.

## Decisions this plan makes

Where the design admits more than one implementation (`capability-map.md` §5.2) or a parameter is
deliberately unmeasured, this plan decides and says so. An implementer must not silently re-decide one.
Numbered `N*` to stay distinct from the foundation review's `F-*` and the sibling's numbering; the
reviewer folds duplicates.

| # | point | decision |
| --- | --- | --- |
| N1 | `speech.word_gap_ms` is null and SPEECH cannot group words without it | `require()` at entry; the node refuses (raises) under the packaged config before touching the store. See "Unmeasured `require()` keys" above for the rejected alternative and its cost |
| N2 | VOICE's gate floors and F0 range are null and the gate cannot run ungated | same mechanism: `require()` at entry on all four `phonation.*` keys. The node's outcome under the packaged config is *unmeasurable-by-config* — surfaced as the raise, recorded by the runner as `errored`, flagged by VERDICT — never an invented floor and never a `fail`, because `fail` claims evidence of absence and an unrun gate has none |
| N3 | `redaction.padding_ms` is null | REDACT is importable and constructible; `require()` at entry makes it refuse to run without an override, per `redact.md` |
| N4 | SPEECH step 3's SQUIM half has no fitted cut (open.md) | YAMNet coverage alone decides corroboration; SQUIM STOI/SI-SDR are always recorded per span; the SQUIM vote — and therefore the instrument-disagreement flag — activates only when `speech.speech_test_stoi_floor` / `speech.speech_test_si_sdr_floor` (both new, both `null`) are supplied by override. While null, each span's corroboration records `squim_vote: "not_evaluated"` |
| N5 | YAMNet per-window score cut and per-span coverage cut | both are `yamnet.coverage_threshold` (0.5): the derivation measured the per-window label-score gap ("Speech 0.92 → 0.14"), and coverage is the fraction of overlapping windows clearing it; a span is YAMNet-confirmed when that fraction ≥ the same key. One measured value, two named uses, stated here. AIRWAY aggregates identically — reviewer reconciles |
| N6 | the second diarizer is unnamed in the design | config `speech.second_diarizer: null` (a model id, e.g. `"BUT-FIT/diarizen-wavlm-large-s80-md"` — note CC BY-NC 4.0 weights). While null, count ≠ 1 records `second_diarizer: "not_consulted"` and still flags; it never blocks |
| N7 | no target-match similarity threshold exists anywhere | config `speech.target_match_cosine: null`. It is `require()`d **only when the hint carries a target embedding** — a caller asking for a comparison the config cannot decide is refused rather than answered with an invented cut. Similarities are still computed and recorded per speaker (a measurement needs no threshold); only the match *decision* needs the key |
| N8 | "the recognizers disagree beyond threshold" flag has no measured threshold | config `speech.agreement_flag_floor: null`; while null the row is inert and the verdict records `agreement_flag: "not_evaluated"` |
| N9 | fabrication test names "no energy and no periodicity"; the periodicity gate is unmeasured | energy-only: a word over whose extent the envelope never exceeds the local floor is a candidate. The periodicity half waits on the phonation floors and is recorded as `periodicity: "not_evaluated"`. "Candidates survive" (the flag row) means the candidate set is non-empty after this test; each candidate word gets a `label` assertion `{"label": "fabrication_candidate"}` |
| N10 | SPEECH `refine`s a PREPROCESS span — overlap criterion unstated | any temporal intersection > 0 (`benchmarks/snr.md`: IoU falls to 0.17 while the verdict is still speech, so a fraction would drop true refinements) |
| N11 | PII scan granularity and locating a finding | scan per (speech span × recognizer): one `ScriptLine` per pair carrying the span's words as `chunks`, the span's extent, and the words' `timestamp_model`. A finding is located by normalized-token subsequence match of its text against the span's words; matched `word` elements get `label` assertions `{"label": "pii", "category": ...}` and the finding's extent is `[first matched word start, last matched word end]`; no match → the finding keeps the span extent and the verdict flags `pii_unlocated` |
| N12 | a finding whose speaker cannot be resolved (mixed span, unassigned words) | treated as target-overlapping — flagged. The rule can exempt only what it can attribute |
| N13 | `classify_audios` YAMNet defaults to `top_k=5`, so `Speech` can silently vanish (open.md) | config `yamnet.top_k: 521` — the size of YAMNet's label space, an identity, not a threshold; any smaller k makes a missing label unreadable-as-zero |
| N14 | REDACT "re-runs ASR" — one model or both | both recognizers PREPROCESS used, each loaded **at the commit the store's model agents recorded** (never a ref — the verification runs the recognizer that produced the words), and **any** finding in the verification scan is a `fail` — a new finding is exactly as unsafe as a survivor |
| N15 | REDACT on a store with no PII scan | refuses (raises): with no scan evidence it cannot distinguish "clean" from "unchecked", and `fail` would misreport ("a finding survived") while `pass` would launder an unexamined recording into `releasable`. The runner records `errored`; VERDICT reads no REDACT verdict → `not_assessed`, which is the designed answer for an unexamined recording |
| N16 | a PII detector failing **during REDACT's verification** | `fail(reason="verification could not run", survived=[])` → `withheld`. Examined-and-uncertifiable is `withheld`, not `not_assessed`; `survived` stays empty because nothing is known to have survived. `redact.md`'s "survived is non-empty only on fail" permits an empty-survived fail and the reason string carries the distinction |
| N17 | which audio REDACT redacts, and the artifacts it emits | the recording as admitted (the `stream` named `recording` — full fidelity, one clock shared with every extent in the store). Artifacts: redacted audio + redacted transcript; the figure is omitted (each artifact is independently optional per `redact.md`, and a released figure adds a rendering channel this plan does not open) |
| N18 | `branch-speech.md` names element kind `target_match`, absent from the foundation's `PROV_TYPE` | Task 5 adds `"target_match"` to `PROV_TYPE` in `src/senselab/utils/prov_store.py` (one token, guarded by a test) |
| N19 | attributing a store element to the node that wrote it (VOICE's residual needs "SPEECH's spans"; VERDICT orders verdicts) | Task 5 adds `ProvStore.get_activity(activity_id) -> Activity` (mirrors `get_entity`/`get_agent`); a node attributes an entity via `store.get_activity(store.generated_by(eid)).node`. An "airway-labelled span" is a `span` with a non-invalidated `label` assertion generated by an `AIRWAY` activity |
| N20 | VOICE's "intervals with energy" | contiguous runs where the envelope exceeds its local floor, read from PREPROCESS's npz sidecars. No new threshold: the floor is the measured one |
| N21 | VOICE's "F0 sits where the range serves two populations ambiguously" | a run is ambiguous when its median F0 times or divided by `phonation.period_doubling_factor` (config, `2.0` — the definition of period doubling, an identity) also lies inside `[f0_min_hz, f0_max_hz]`. A wide range flags everything, which is the design's point: the caller must state its population |
| N22 | VOICE's "near the interval's edge" flag — the measured intervals do not transfer to the implementation's units (config UNSET note) | config `phonation.hnr_floor_interval_db` / `phonation.rms_floor_interval` (both new, both `null`); when supplied as `[lo, hi]`, a run whose gate values at onset fall inside is flagged near-edge; while null the row is inert and the verdict records `gate_interval: "unmeasured"` |
| N23 | a gate-passing run in which Praat places no period marks | retained as a voiced-run span with `marks_n: 0` and `onset_kind: "criterion"` (there is no observed period to anchor it); a run with marks has `onset_kind: "period"`. Both always carry `offset_kind: "criterion"` — the design's two-kinds-of-edge rule made explicit in the attributes |
| N24 | the design's `f0_candidates` track has no producer in the phonation task | Task 6 adds `f0_track(audio, *, f0_min_hz, f0_max_hz, hop_s) -> (times, f0_hz, strength)` to `phonation/api.py` via Praat `to_pitch_cc` — the *selected* candidate per frame with its strength (the periodicity it travels with), not the full candidate list. Stated as a narrowing of the design's word; widening to full candidate lists is follow-on work if a consumer appears |
| N25 | "a hint asserts speech/phonation the branch did not find" — `may_contain` is an open vocabulary | config `speech.hint_tags` / `voice.hint_tags`: the tag lists that count as asserting each kind, seeded from the design documents' own member names (not fitted; extended by override). SPEECH additionally reads a non-empty `hint.expected_speech` as asserting speech |
| N26 | how VERDICT learns `ran`, and what a gated run looks like | `ran` is caller-supplied (only the runner can know `errored`); when omitted it is derived from the store — a node with a `verdict` entity is `completed`, otherwise `skipped`, and the derivation cannot see `errored` (stated limitation). The file-verdict entity carries `gated: true` when any kind predicted **absent** has no branch verdict — marking that the contradiction check did not happen, per `verdict.md` |
| N27 | TAXONOMY's `voice_no_words: "not_screened"` against the fold's `KindState` | VERDICT maps `not_screened` → `KindState.UNDECIDED`: the fold's undecided rows (pass → present, fail → absent, never-ran → flag) are exactly what an unscreened kind needs |
| N28 | measurements on separated streams vs. the recording (capability-map §4.5) | record both, never normalise, never compare: every `measurement` entity carries a `stream` attribute naming the stream entity it was taken on, and stream entities carry the `rms_scalar` ClearVoice reports. No code path compares a dBFS-referenced value across streams |

## Config additions (one edit to `data/config/default.yaml`, made in Task 5, extended in Task 6)

New keys only — the loader refuses overrides that introduce keys, so every key a test overrides must
exist in the packaged file. Each lands with a `#` comment naming its derivation or its open.md entry;
none of the `null`s may be given a value by this plan or its implementer.

```yaml
speech:
  word_gap_ms: null                  # exists already
  second_diarizer: null              # N6 — no measured ranking of second diarizers exists
  target_match_cosine: null          # N7 — open.md: no similarity threshold has been derived
  agreement_flag_floor: null         # N8
  speech_test_stoi_floor: null       # N4 — open.md: SQUIM thresholds over speech spans
  speech_test_si_sdr_floor: null     # N4
  hint_tags: [speech, read-speech]   # N25 — vocabulary from the design docs, not fitted

voice:
  hint_tags: [phonation, humming, sustained-vowel, voice]   # N25

yamnet:
  top_k: 521                         # N13 — the label space's size; an identity

phonation:
  period_doubling_factor: 2.0        # N21 — the definition of period doubling; an identity
  hnr_floor_interval_db: null        # N22 — benchmarks/voice.md measured an interval in other units
  rms_floor_interval: null           # N22
```

---

### Task 5: SPEECH

**Scope:** `nodes/speech.py` — all eight steps of `branch-speech.md` — plus the two one-token store
changes it forces (`PROV_TYPE` gains `"target_match"`, `ProvStore` gains `get_activity`) and the
config additions above. One node, one task, per this plan's charter; the eight steps are functions
inside one module, not eight modules.

**Design invariants restated for this task** (each is tested):

1. **Speech spans come from ASR word timings, not the envelope.** PREPROCESS's envelope spans are
   `refine`d (`wasDerivedFrom`) where they overlap a word-derived span and are never the source; a
   PREPROCESS span with no words is left alone. **This node runs no ASR** — it reads the two
   hypotheses PREPROCESS wrote and compares them.
2. **pyannote runs only over `[first word start, last word end]`.** `diarize_audios` takes whole audio
   only (verified: no interval argument exists), so the node crops the plain 16 kHz stream to the
   interval with `extract_segments` and **adds the interval's start back onto every returned
   segment's `start`/`end`** — the offset-shift is where an off-by-one lives, and it gets its own test.
3. **A diarizer segment overlapping an airway-labelled span is withdrawn (`wasInvalidatedBy`), never
   relabelled.** The entity stays in the store with the withdrawal recorded; the speaker count reads
   only un-withdrawn segments.
4. **Separation runs only when the count ≠ 1**, cannot serve ≥ 3 (the checkpoint separates exactly
   2 — the node reports that rather than separating into the wrong number), and its streams are
   `stream` entities; **every measurement records its stream** (N28).
5. **The PII scan reads both hypotheses; the decision is this branch's own and is speaker-scoped**:
   target-speaker findings flag; no-target means flag; a failed detector flags because "could not
   check" is not "clean" (`PiiScan.failures` honoured). Non-target-only findings with a known target
   do not flag — REDACT's scope differs deliberately.
6. **Quality (SQUIM + disruptions) is parallel and reported, never gating.** No quality reading
   changes the outcome; the SQUIM `fail` path is unreachable by design until `benchmarks/open.md`'s
   thresholds exist.
7. **A branch `fail` (no words from either recognizer) is normal** — the branch has no subject.
8. **No matched PII text in any entity, verdict, log or exception.** `pii` entities are built by
   projection (category + extent + provenance, never `.text`), per capability-map §3.4.

**Files:**
- Create: `src/senselab/audio/workflows/triage/nodes/__init__.py`, `src/senselab/audio/workflows/triage/nodes/speech.py`
- Modify: `src/senselab/utils/prov_store.py` (`PROV_TYPE` + `get_activity`), `src/senselab/audio/workflows/triage/data/config/default.yaml` (keys above)
- Test: `src/tests/audio/workflows/triage/nodes/speech_test.py`, plus two tests appended to `src/tests/utils/prov_store_test.py`

**Interfaces**

Consumes (all verified on this branch except where the merge prerequisite is named):
- `elements.py` helpers and `NodeProduct` (assumed shared contract above).
- `fuse_word_streams(word_streams: dict[str, list[dict]], *, weights=None, ...) -> list[dict]` from
  `senselab.audio.tasks.speech_to_text_ensemble.api` — runs **real** in tests (stdlib-only). Each
  returned word dict carries `text`, `start`, `end`, `confidence`, `existence_confidence`,
  `temporal_confidence`, `member_agreement`, `coverage`, `alternates`, `flags`. Note it imports
  `MIN_EVIDENCE_WEIGHT` from `workflows/audio_analysis/floors` (capability-map §4.9) — an existing
  coupling this task uses and must not extend.
- `classify_audios(audios, model="yamnet", top_k=cfg)` → `List[List[Dict]]`, per-window dicts with
  `start`, `end`, `labels`, `scores` (mocked).
- `extract_objective_quality_features_from_audios(audios, device=None) -> List[Dict[str, Any]]`,
  dicts with `stoi`, `pesq`, `si_sdr` (NaN on internal failure — kept as recorded values) (mocked).
- `diarize_audios(audios, model=None, ...) -> List[List[ScriptLine]]` and `PyannoteAudioModel`
  (mocked; the fake model object carries `path_or_uri` and `commit_sha` attributes because the real
  one resolves its commit at construction).
- `separate_audios(audios, model=HFModel("alibabasglab/MossFormer2_SS_16K"), n_sources=2, device=None,
  timeout_s=None) -> List[List[Audio]]` — **after the `triage` merge**; each output Audio carries
  `metadata["clearvoice"]` naming model, resolved commit, source index and the un-applied RMS scalar
  (mocked; the fake reproduces that metadata shape).
- `extract_speaker_embeddings_from_audios(audios, model=None, device=None) -> List[torch.Tensor]` and
  `SpeechBrainModel` (mocked).
- `scan_for_pii(inputs, detectors=None, ...) -> PiiScan | list[PiiScan]`; `PiiSpan` **is a
  `ScriptLine`** (foundation Task 1): findings from a scanned line inherit its `start`/`end`/`speaker`/
  `timestamp_model`; `PiiScan.failures` distinguishes could-not-check from clean (mocked).
- `extract_segments(data: List[Tuple[Audio, List[Tuple[float, float]]]]) -> List[List[Audio]]` (real).
- `detect_disruptions(audio, start_s, end_s, *, clip_headroom, min_clip_run, min_dropout_ms,
  discontinuity_threshold) -> Disruptions` (real).
- `AudioHints.target_speaker: TargetSpeakerEmbedding{vector, provenance: SpeakerEmbeddingProvenance}` —
  `provenance.model_id` + `provenance.model_commit_sha` (40-hex validated at construction) are the
  refusal gate for the comparison.

Produces:

```python
# src/senselab/audio/workflows/triage/nodes/speech.py
def speech(
    store: ProvStore,
    config: TriageConfig,
    run_dir: Path,
    hint: AudioHints | None = None,
    device: DeviceType | None = None,
) -> NodeProduct
```

`verdict` mapping (the design doc's product section, exactly):
`{"speaker_count", "target_speaker"?, "words_n", "speech_s", "pii": {"categories": [], "n",
"scanned_by": [], "failed": []}, "flags": []}`. `view` lists every element id authored or asserted
over; on a `flag`, the view **includes** the contested assertions (partial is a view, not a payload).

Store writes, by element kind (the design doc's table): consensus `word` entities (text, extent,
confidence from agreement, speaker, stream, `pii`/`fabrication_candidate` label assertions);
speech `span` entities (extent, corroboration attributes, `wasDerivedFrom` any overlapping PREPROCESS
span); one `interval` entity (the diarizer's window); `speaker` entities (diarizer segments, withdrawn
ones retained with `wasInvalidatedBy` and the overlapping airway span recorded on the withdrawal
activity's parameters); `stream` entities (one per separated source, `wasDerivedFrom` the recording's
stream); `pii` entities (category, extent, detectors ran/failed, which recognizer's hypothesis —
**never text**); `measurement` entities (SQUIM and disruptions per span, each carrying `stream`);
`target_match` entities (speaker, similarity, both embeddings' model + commit); one `verdict` entity.

**Mocking boundary for this task:** `classify_audios`, `extract_objective_quality_features_from_audios`,
`diarize_audios`, `PyannoteAudioModel`, `separate_audios`, `HFModel`,
`extract_speaker_embeddings_from_audios`, `SpeechBrainModel`, `scan_for_pii` — all patched **on
`nodes.speech`**. Everything else real.

- [ ] **Step 5.1: Store changes, with failing tests first**

Append to `src/tests/utils/prov_store_test.py`:

```python
def test_target_match_is_a_prov_type() -> None:
    """branch-speech.md's product table names target_match as an element kind."""
    store = ProvStore(run_id="t")
    eid = store.entity(prov_type="target_match", extent=None, attributes={"speaker": "SPEAKER_00"})
    assert store.get_entity(eid).prov_type == "target_match"


def test_get_activity_returns_what_activity_recorded() -> None:
    """An entity's author node is reachable: generated_by -> get_activity -> .node."""
    store = ProvStore(run_id="t")
    act = store.activity(node="SPEECH", step="diarize", parameters={})
    eid = store.entity(prov_type="speaker", extent=(1.0, 2.0), attributes={})
    store.was_generated_by(eid, act)
    assert store.get_activity(store.generated_by(eid)).node == "SPEECH"
```

Run: `uv run pytest src/tests/utils/prov_store_test.py -k "target_match or get_activity" -v` — FAIL
(unknown prov_type; no attribute `get_activity`).

Implement: add `"target_match"` to the `PROV_TYPE` literal; add

```python
def get_activity(self, activity_id: str) -> Activity:
    """Return one activity."""
    return self._activities[activity_id]
```

Re-run — PASS. Also run the whole store suite (`uv run pytest src/tests/utils/prov_store_test.py`) —
the JSONL round-trip and merge tests must stay green.

- [ ] **Step 5.2: Config keys**

Add the keys from the preamble's YAML block (speech/voice/yamnet/phonation additions) to
`data/config/default.yaml`, each with its `#` derivation comment, and extend the file's `UNSET, and
why` derivation note with the new null keys. Test (in `speech_test.py`):

```python
def test_new_speech_keys_exist_and_the_unmeasured_ones_raise() -> None:
    """Null keys are present (overridable) and refuse to be read as values."""
    cfg = load_triage_config()
    assert cfg.get("yamnet.top_k") == 521
    for key in (
        "speech.word_gap_ms", "speech.second_diarizer", "speech.target_match_cosine",
        "speech.agreement_flag_floor", "speech.speech_test_stoi_floor",
    ):
        with pytest.raises(ValueError, match="benchmarks/open.md|no value"):
            cfg.require(key)
```

- [ ] **Step 5.3: Test fixtures — a store as Tasks 1–4 leave it**

One conftest-level builder used by every test in this task (and reused by Tasks 6–7), with real code:

```python
"""SPEECH node tests. Every model call is faked at the node module; DSP and the store run real."""

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

SR = 16000
CW = "nyrahealth/CrisperWhisper"
QW = "Qwen/Qwen3-ASR-Flash"


def make_run(tmp_path, words_cw, words_qw, airway_label_extent=None, duration_s=6.0):
    """Build (store, config, run_dir) as ADMIT/PREPROCESS/TAXONOMY/AIRWAY leave them.

    words_*: list of (text, start, end). airway_label_extent: (start, end) to write a
    PREPROCESS span carrying an AIRWAY label assertion.
    """
    store = ProvStore(run_id="t")
    rng = np.random.default_rng(0)
    n = int(duration_s * SR)
    wave = np.zeros(n, dtype=np.float32)
    for _, s, e in [*words_cw, *words_qw]:
        wave[int(s * SR) : int(e * SR)] = 0.1 * rng.standard_normal(int(e * SR) - int(s * SR))
    if airway_label_extent:
        s, e = airway_label_extent
        wave[int(s * SR) : int(e * SR)] = 0.2 * rng.standard_normal(int(e * SR) - int(s * SR))
    run_dir = tmp_path / "run"
    (run_dir / "derivatives").mkdir(parents=True)
    Audio(waveform=wave[None, :], sampling_rate=SR).save_to_file(str(run_dir / "plain.wav"))

    pre = store.activity(node="PREPROCESS", step=None, parameters={})
    for name, path in (("recording", "plain.wav"), ("resampled_16k", "plain.wav")):
        sid = store.entity(prov_type="stream", extent=(0.0, duration_s),
                           attributes={"name": name, "path": path, "sampling_rate": SR})
        store.was_generated_by(sid, pre)

    # envelope + floor sidecars: envelope is above the floor exactly where the wave is non-zero
    env = np.full(n, -80.0)
    env[np.abs(wave) > 0] = -30.0
    floor = np.full(n, -60.0)
    np.savez(run_dir / "derivatives" / "envelope.npz", values=env, sampling_rate=SR)
    np.savez(run_dir / "derivatives" / "floor.npz", values=floor, sampling_rate=SR)
    for name, path in (("energy_envelope", "derivatives/envelope.npz"), ("floor", "derivatives/floor.npz")):
        mid = store.entity(prov_type="measurement", extent=None,
                           attributes={"name": name, "path": path, "sampling_rate": SR})
        store.was_generated_by(mid, pre)

    for model_id, words in ((CW, words_cw), (QW, words_qw)):
        for text, s, e in words:
            wid = store.entity(prov_type="word", extent=(s, e),
                               attributes={"text": text, "asr_model": model_id})
            store.was_generated_by(wid, pre)

    if airway_label_extent:
        s, e = airway_label_extent
        span = store.entity(prov_type="span", extent=(s, e), attributes={"peak_over_floor_db": 30.0})
        store.was_generated_by(span, pre)
        air = store.activity(node="AIRWAY", step="classify", parameters={})
        lab = store.entity(prov_type="assertion", extent=(s, e),
                           attributes={"verb": "label", "label": "Cough"})
        store.was_generated_by(lab, air)
        store.was_derived_from(lab, span)

    return store, load_triage_config(_override(tmp_path)), run_dir


def _override(tmp_path):
    """A YAML override supplying the unmeasured keys tests need — the production mechanism."""
    p = tmp_path / "override.yaml"
    p.write_text("speech:\n  word_gap_ms: 300\n")
    return p
```

The default fakes, installed by an autouse fixture so a test only overrides the call it is probing
(shapes verified against the real functions in the Interfaces block):

```python
@pytest.fixture(autouse=True)
def quiet_models(monkeypatch):
    """One speaker, speech-positive YAMNet, plausible SQUIM, no PII, no separation call."""
    def fake_yamnet(audios, model, top_k, **kw):
        dur = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        wins, t = [], 0.0
        while t < dur:
            wins.append({"start": t, "end": t + 0.96, "labels": ["Speech"], "scores": [0.9]})
            t += 0.48
        return [wins]

    def fake_diarize(audios, **kw):
        dur = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=dur)]]

    monkeypatch.setattr(speech_module, "classify_audios", fake_yamnet)
    monkeypatch.setattr(speech_module, "extract_objective_quality_features_from_audios",
                        lambda audios, device=None: [{"stoi": 0.9, "pesq": 3.0, "si_sdr": 18.0} for _ in audios])
    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    monkeypatch.setattr(speech_module, "PyannoteAudioModel",
                        lambda **kw: type("M", (), {"path_or_uri": kw["path_or_uri"],
                                                    "commit_sha": "a" * 40})())
    monkeypatch.setattr(speech_module, "separate_audios",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("separation must not run")))
    monkeypatch.setattr(speech_module, "scan_for_pii",
                        lambda inputs, **kw: [_clean_scan() for _ in inputs])
```

where `_clean_scan()` returns `PiiScan(spans=[], detectors_used=["presidio", "gliner", "rules"],
failures={})` — built from the real `PiiScan` type so a field rename breaks the tests.

- [ ] **Step 5.4: Write the failing tests**

The load-bearing ones in full; the enumerated rest follow the same pattern.

```python
def test_packaged_config_refuses_and_the_store_is_untouched() -> None:
    """word_gap_ms is null by design; the node raises at entry, before any store write."""
    store, _, run_dir = make_run(tmp_path, [("hi", 1.0, 1.3)], [("hi", 1.0, 1.3)])
    before = store.fingerprint()
    with pytest.raises(ValueError, match="speech.word_gap_ms"):
        speech_module.speech(store, load_triage_config(), run_dir)
    assert store.fingerprint() == before, "an unmeasured key must leave the store untouched"


def test_no_words_from_either_recognizer_is_a_normal_fail() -> None:
    """fail means this branch has no subject — a cough recording is not an error."""
    store, cfg, run_dir = make_run(tmp_path, [], [])
    product = speech_module.speech(store, cfg, run_dir)
    assert product.outcome is Outcome.FAIL
    assert store.entities("verdict"), "the verdict entity is written even on fail"


def test_spans_come_from_word_timings_and_refine_preprocess_spans() -> None:
    """Two word runs separated by more than word_gap_ms are two spans; an overlapping
    PREPROCESS span is refined (wasDerivedFrom), and one with no words is left alone."""
    words = [("one", 1.0, 1.2), ("two", 1.25, 1.5), ("three", 3.0, 3.4)]
    store, cfg, run_dir = make_run(tmp_path, words, words, airway_label_extent=(4.5, 5.0))
    pre_act = store.activity(node="PREPROCESS", step="spans", parameters={})
    pre_span = store.entity(prov_type="span", extent=(0.9, 1.6), attributes={"peak_over_floor_db": 20.0})
    store.was_generated_by(pre_span, pre_act)
    product = speech_module.speech(store, cfg, run_dir)
    speech_spans = [e for e in store.entities("span")
                    if store.get_activity(store.generated_by(e.id)).node == "SPEECH"]
    assert [tuple(round(x, 2) for x in s.extent) for s in sorted(speech_spans, key=lambda s: s.extent)] \
        == [(1.0, 1.5), (3.0, 3.4)]
    refined = [s for s in speech_spans if pre_span in store.derived_from(s.id)]
    assert len(refined) == 1, "any temporal intersection > 0 refines (N10); the airway span is untouched"


def test_pyannote_sees_only_the_word_interval_and_segments_are_offset_back() -> None:
    """The diarizer gets [first word start, last word end]; its clock is shifted back to the recording's."""
    seen = {}

    def fake_diarize(audios, **kw):
        seen["dur"] = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=seen["dur"])]]

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    words = [("one", 2.0, 2.3), ("two", 2.4, 2.8)]
    store, cfg, run_dir = make_run(tmp_path, words, words)
    speech_module.speech(store, cfg, run_dir)
    assert seen["dur"] == pytest.approx(0.8, abs=1 / SR), "cropped to the interval, not the file"
    seg = store.entities("speaker")[0]
    assert seg.extent == pytest.approx((2.0, 2.8)), "offset added back onto the returned segment"


def test_a_segment_overlapping_an_airway_label_is_withdrawn_not_relabelled() -> None:
    """wasInvalidatedBy, entity retained, excluded from the count."""
    words = [("one", 1.0, 1.4), ("two", 4.4, 4.8)]  # interval spans the airway label at 4.5-5.0
    store, cfg, run_dir = make_run(tmp_path, words, words, airway_label_extent=(4.5, 5.0))

    def fake_diarize(audios, **kw):
        return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=2.0),
                 ScriptLine(speaker="SPEAKER_01", start=3.4, end=3.8)]]  # 2nd overlaps label after offset

    monkeypatch.setattr(speech_module, "diarize_audios", fake_diarize)
    product = speech_module.speech(store, cfg, run_dir)
    speakers = store.entities("speaker")
    withdrawn = [s for s in speakers if store.is_invalidated(s.id)]
    assert len(speakers) == 2 and len(withdrawn) == 1
    assert withdrawn[0].attributes["speaker"] == "SPEAKER_01", "withdrawn, never relabelled"
    assert product.verdict["speaker_count"] == 1, "the count reads un-withdrawn segments only"


def test_count_two_separates_and_measurements_record_their_stream() -> None:
    """MossFormer runs at n_sources=2; streams become entities; SQUIM on a stream names it."""
    calls = {}

    def fake_separate(audios, model=None, n_sources=None, **kw):
        calls["n_sources"] = n_sources
        out = []
        for i in range(2):
            a = Audio(waveform=audios[0].waveform, sampling_rate=SR)
            a.metadata["clearvoice"] = {"model": "alibabasglab/MossFormer2_SS_16K",
                                        "commit": "b" * 40, "source_index": i, "rms_scalar": 0.31}
            out.append(a)
        return [out]

    monkeypatch.setattr(speech_module, "separate_audios", fake_separate)
    monkeypatch.setattr(speech_module, "diarize_audios", _two_speaker_fake)
    store, cfg, run_dir = make_run(tmp_path, WORDS, WORDS)
    product = speech_module.speech(store, cfg, run_dir)
    assert calls["n_sources"] == 2
    streams = [e for e in store.entities("stream") if "source_index" in e.attributes]
    assert {s.attributes["source_index"] for s in streams} == {0, 1}
    assert all("rms_scalar" in s.attributes for s in streams), "level died at -25 dBFS; the scalar is the record"
    squims = [m for m in store.entities("measurement") if m.attributes.get("name") == "squim"
              and store.get_activity(store.generated_by(m.id)).node == "SPEECH"]
    assert all("stream" in m.attributes for m in squims), "every measurement records its stream (N28)"
    assert Outcome.FLAG is product.outcome, "count != 1 flags"


def test_count_three_reports_rather_than_separating_wrong() -> None:
    """The checkpoint separates exactly two; >= 3 is reported, separate_audios is never called."""
    monkeypatch.setattr(speech_module, "diarize_audios", _three_speaker_fake)
    # autouse fake for separate_audios raises AssertionError if called
    store, cfg, run_dir = make_run(tmp_path, WORDS, WORDS)
    product = speech_module.speech(store, cfg, run_dir)
    assert product.outcome is Outcome.FLAG
    assert any("separation" in f for f in product.verdict["flags"])


def test_pii_decision_is_speaker_scoped() -> None:
    """Target-speaker finding flags; non-target-only does not; no target flags; failure flags."""
    # (a) finding on the target speaker's words -> flag
    # (b) same finding, attributed to the non-target speaker, target known -> no flag from PII
    # (c) same finding, no hint at all -> flag ("no speaker to exempt")
    # (d) clean spans but failures={"gliner": "load failed"} -> flag ("could not check")
    ...


def test_pii_entities_and_verdict_never_carry_matched_text() -> None:
    """Projection, not filtering: no store entity, verdict value or exception carries the match."""
    secret = "jane.doe@example.com"
    monkeypatch.setattr(speech_module, "scan_for_pii", _scan_finding(secret, category="EMAIL_ADDRESS"))
    store, cfg, run_dir = make_run(tmp_path, WORDS_WITH_EMAIL, WORDS_WITH_EMAIL)
    product = speech_module.speech(store, cfg, run_dir)
    dumped = json.dumps([(e.prov_type, e.attributes) for e in store.entities()
                         if e.prov_type != "word"], default=str)
    assert secret not in dumped, "pii/measurement/verdict entities are projections"
    assert secret not in json.dumps(product.verdict, default=str)
    pii = store.entities("pii")
    assert pii and pii[0].attributes["category"] == "EMAIL_ADDRESS" and pii[0].extent is not None


def test_target_without_commit_is_refused_and_flagged_without_an_embedding_call() -> None:
    """Embeddings from different models are not comparable; unprovenanced targets are refused."""
    ...


def test_target_with_provenance_requires_the_null_cosine_key() -> None:
    """A hint carrying a target under a null speech.target_match_cosine raises at entry (N7)."""
    ...


def test_quality_is_reported_never_gating() -> None:
    """Terrible SQUIM numbers and real disruptions leave a pass a pass."""
    monkeypatch.setattr(speech_module, "extract_objective_quality_features_from_audios",
                        lambda audios, device=None: [{"stoi": 0.1, "pesq": 1.0, "si_sdr": -10.0}
                                                     for _ in audios])
    store, cfg, run_dir = make_run(tmp_path, WORDS, WORDS)
    product = speech_module.speech(store, cfg, run_dir)
    assert product.outcome is Outcome.PASS
    dis = [m for m in store.entities("measurement") if m.attributes.get("name") == "disruptions"]
    assert dis and all("clipped_runs" in m.attributes for m in dis), "counts and extents, not a score"
```

Also written (same shapes, one line each here):
`test_fusion_runs_real_and_confidence_is_agreement_not_correctness` (two hypotheses disagreeing on one
word → that consensus word's confidence < the agreed words'; asserted on the real `fuse_word_streams`);
`test_a_word_over_no_energy_is_a_fabrication_candidate_and_flags` (word at 5.5–5.7 s where the fixture
envelope never exceeds the floor → `label` assertion `fabrication_candidate` on the word, flag; N9);
`test_yamnet_disconfirmation_flags` (fake YAMNet scores below `yamnet.coverage_threshold` over one span
→ flag, and the span's corroboration attributes carry the coverage that made it ambiguous);
`test_squim_vote_is_inert_while_thresholds_are_null` (`squim_vote: "not_evaluated"` recorded; no flag
fires on awful SQUIM; N4); `test_second_diarizer_null_records_not_consulted` (N6);
`test_straddling_word_is_marked_not_assigned` (word overlapping two segments → attribute
`speaker: None`, `speaker_note: "straddles"`); `test_hint_asserting_speech_not_found_flags` (hint with
`expected_speech` non-empty and zero words → the no-words `fail` becomes a `flag` per the outcome
table's hint row — the design's flag row outranks the fail row exactly when a hint contradicts it);
`test_flag_view_includes_contested_assertions` (partial-as-view);
`test_every_read_is_recorded_with_used` (the SPEECH activities' `used` targets include the word,
envelope, span and label entities the node read).

- [ ] **Step 5.5: Run them and watch them fail**

Run: `uv run pytest src/tests/audio/workflows/triage/nodes/speech_test.py -x -q` — FAIL (module
does not exist).

- [ ] **Step 5.6: Implement `nodes/speech.py`**

Module skeleton with the load-bearing logic in full; docstrings say what, `specs/` says why.

```python
"""The SPEECH branch: transcript agreement, spans from word timings, diarization, PII, quality."""

from __future__ import annotations

# module-top imports of every model-calling function, by name — the mocking boundary
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import extract_segments
from senselab.audio.tasks.source_separation.api import separate_audios
from senselab.audio.tasks.speaker_diarization.api import diarize_audios
from senselab.audio.tasks.speaker_embeddings.api import extract_speaker_embeddings_from_audios
from senselab.audio.tasks.speech_to_text_ensemble.api import fuse_word_streams
from senselab.text.tasks.pii_detection.api import scan_for_pii
from senselab.utils.data_structures import HFModel, PyannoteAudioModel, SpeechBrainModel

_NODE = "SPEECH"


def _required(config: TriageConfig, hint: AudioHints | None) -> dict[str, Any]:
    """Resolve every require() key at entry, so an unmeasured key precedes any store write."""
    values = {
        "word_gap_ms": config.require("speech.word_gap_ms"),
        "coverage_threshold": config.require("yamnet.coverage_threshold"),
        "yamnet_top_k": config.require("yamnet.top_k"),
        "clip_headroom": config.require("disruptions.clip_headroom"),
        "min_clip_run": config.require("disruptions.min_clip_run"),
        "min_dropout_ms": config.require("disruptions.min_dropout_ms"),
        "discontinuity_threshold": config.require("disruptions.discontinuity_threshold"),
    }
    if hint is not None and hint.target_speaker is not None:
        values["target_match_cosine"] = config.require("speech.target_match_cosine")
    return values


def _group_words_into_spans(words: list[dict], gap_ms: float) -> list[tuple[float, float, list[int]]]:
    """A span is the extent of a run of words; a gap over gap_ms starts a new run."""
    spans: list[tuple[float, float, list[int]]] = []
    for i, w in enumerate(sorted_words := sorted(words, key=lambda w: w["start"])):
        if spans and (w["start"] - spans[-1][1]) * 1000.0 <= gap_ms:
            start, _, members = spans[-1]
            spans[-1] = (start, max(spans[-1][1], w["end"]), [*members, i])
        else:
            spans.append((w["start"], w["end"], [i]))
    return spans


def _diarize_interval(store, activity, audio, interval, device):
    """pyannote over [first word start, last word end] only; segments shifted back."""
    t0, t1 = interval
    (cropped,) = extract_segments([(audio, [(t0, t1)])])[0]
    model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")
    agent = model_agent(store, model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
    store.was_associated_with(activity, agent)
    (segments,) = diarize_audios([cropped], model=model, device=device)
    shifted = [ScriptLine(speaker=s.speaker, start=(s.start or 0.0) + t0, end=(s.end or 0.0) + t0)
               for s in segments]
    return shifted, agent
```

Withdrawal, the PII decision and the projection, in full:

```python
def _airway_labelled_extents(store: ProvStore) -> list[tuple[float, float]]:
    """Spans carrying a non-invalidated label assertion authored by an AIRWAY activity (N19)."""
    out = []
    for span in store.entities("span"):
        for assertion in assertions_about(store, span.id, verb="label"):
            act = store.generated_by(assertion.id)
            if act and store.get_activity(act).node == "AIRWAY" and not store.is_invalidated(assertion.id):
                out.append(span.extent)
                break
    return out


def _decide_pii(findings, scans, target_word_extents, target_known) -> tuple[bool, list[str]]:
    """This branch's own rule over scan_for_pii's evidence — not decide_pii's.

    Flags when a finding overlaps the target speaker's words, when no target is known and
    anything was found, or when any detector failed: could-not-check is not clean.
    """
    reasons: list[str] = []
    for scan in scans:
        for detector, failure in scan.failures.items():
            reasons.append(f"pii detector {detector} did not run: {failure}")
    if findings and not target_known:
        reasons.append("pii found and no target speaker is known; there is no speaker to exempt")
    for finding in findings:
        if finding["speaker_resolved"] is False or _overlaps_any(finding["extent"], target_word_extents):
            reasons.append(f"pii ({finding['category']}) in or near target speech")
    return bool(reasons), reasons


def _pii_entity_attributes(finding_span, asr_model, detectors_used, failures) -> dict[str, Any]:
    """Projection: category and extent, never the matched text (capability-map 3.4)."""
    return {
        "category": finding_span.category,
        "source": finding_span.source,
        "asr_model": asr_model,
        "detectors_used": list(detectors_used),
        "detectors_failed": sorted(failures),
    }
```

The main `speech()` orchestrates the eight steps in design order, opening one activity per step
(`store.activity(node="SPEECH", step="transcript"...)` … `step="quality"`), recording `used` for every
entity read, and building the verdict mapping exactly as the product section names it. Step 8 runs
unconditionally after step 7 and touches nothing but `measurement` entities. Outcome assembly: `fail`
only from the no-words row (or its hint-contradicted `flag` variant); `flag` from the accumulated
reasons; otherwise `pass`. `write_node_verdict` is called on **every** path, including `fail`.

- [ ] **Step 5.7: Run and watch them pass**

`uv run pytest src/tests/audio/workflows/triage/nodes/speech_test.py src/tests/utils/prov_store_test.py -q`
Expected: all PASS. Then `uv run mypy src/senselab/audio/workflows/triage/`.

- [ ] **Step 5.8: `ruff format` + `ruff check`, commit** — `feat(triage): SPEECH — spans from words, interval-cropped diarization, speaker-scoped PII`

---

### Task 6: VOICE

**Scope:** `nodes/voice.py` — the residual, the gate, period marks, amplitudes — plus one addition to
the phonation task (`f0_track`, N24). **It measures; it does not classify**: no label space, no member
naming, no merged runs.

**Design invariants restated for this task** (each is tested):

1. **The residual is a store fold**: contiguous intervals where the envelope exceeds its local floor
   (N20), minus airway-**labelled** spans, minus SPEECH's speech spans. **A span AIRWAY proposed and
   declined to label is NOT excluded** — an unlabelled span is exactly where unclaimed vocalic
   activity sits. An empty residual is a normal `fail`.
2. **The gate is energy AND periodicity, and its floors are null by design.** The node refuses at
   entry under the packaged config (N2) — it never invents a floor and never reads unmeasurable as
   `fail`. Either condition alone admits something this branch should not claim (periodic room tone;
   broadband noise), so the AND is tested from both sides.
3. **Runs are elementary.** Two voiced runs separated by an unvoiced gap are two runs — no merge
   criterion exists because none has been measured.
4. **Period marks are a point process, not a contour**: per voiced run, ordered marks each carrying
   duration, amplitude, and placement; **absent** outside runs — not zero, not interpolated. Produced
   by the phonation task's `period_marks`.
5. **The onset is a period, the offset is a criterion** — two different kinds of quantity, recorded
   as `onset_kind` / `offset_kind` attributes so the product cannot present them as one (N23).

**Files:**
- Create: `src/senselab/audio/workflows/triage/nodes/voice.py`
- Modify: `src/senselab/audio/tasks/phonation/api.py` (add `f0_track`; N24)
- Test: `src/tests/audio/workflows/triage/nodes/voice_test.py`, plus `f0_track` tests appended to `src/tests/audio/tasks/phonation_test.py` (or wherever the foundation put the phonation tests — mirror it)

**Interfaces**

Consumes:
- store contents as Tasks 1–5 leave them: the `energy_envelope` / `floor` npz sidecar measurements,
  `span` entities (PREPROCESS's and SPEECH's, told apart via N19), AIRWAY `label` assertions,
  the `silence` measurement (read and recorded via `used`; the floor sidecar already embodies it),
  `stream` `resampled_16k` for the audio, `hint` for N25.
- `hnr_track(audio, *, f0_min_hz, hop_s, silence_threshold, periods_per_window) -> (times, hnr_db)` —
  real signature verified; silent frames carry Praat's floor value, so track and times stay aligned
  (mocked in node tests).
- `period_marks(audio, start_s, end_s, *, f0_min_hz, f0_max_hz) -> list[PeriodMark]` —
  `PeriodMark{time_s, period_s, amplitude}`; empty when Praat places no pulses — absent, not zero
  (mocked in node tests).
- `f0_track` — added by this task, mocked in node tests, tested real in the phonation task's tests.

Produces:

```python
# src/senselab/audio/tasks/phonation/api.py — addition
def f0_track(
    audio: Audio, *, f0_min_hz: float, f0_max_hz: float, hop_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F0 and its strength per frame via Praat's cc pitch: (times_s, f0_hz, strength).

    Unvoiced frames carry NaN in f0_hz with their strength retained, so F0 always travels
    with the periodicity that placed it and a reader cannot separate them.
    """

# src/senselab/audio/workflows/triage/nodes/voice.py
def voice(
    store: ProvStore,
    config: TriageConfig,
    run_dir: Path,
    hint: AudioHints | None = None,
) -> NodeProduct
```

`verdict` mapping (the design doc's product section, exactly):
`{"runs_n", "voiced_s", "f0_median_hz"?, "ambiguous_runs_n", "flags": []}`.

Store writes, by element kind: `span` (voiced run — extent, gate values at onset and offset,
`onset_kind`/`offset_kind`, the offset criterion named as which condition stopped holding:
`"hnr" | "rms" | "both" | "residual_end"`); `measurement` `period_marks` per run (ordered marks
inline — they are small — each `{time_s, period_s, amplitude}`); `measurement` `voice_tracks` (energy
RMS, HNR, F0-with-strength on the analysis hop, one npz sidecar, attributes carrying `path` and
`hop_s`); one `verdict` entity. Every entity read (envelope, floor, spans, labels, silence) gets a
`used` from the relevant activity.

Derived quantities, so no new constant appears: the RMS analysis window is
`periods_per_window / f0_min_hz` seconds — the same window Praat's harmonicity uses, an identity on
existing config keys; frames land on the `phonation.hop_s` grid so the three tracks share it.

**Mocking boundary for this task:** `hnr_track`, `period_marks`, `f0_track` patched on `nodes.voice`
(Praat is deterministic but slow and platform-sensitive; the phonation task's own tests own the real
calls). Everything else — the residual fold, the RMS track, the gate, run segmentation, the store —
runs real.

- [ ] **Step 6.1: `f0_track`, with failing tests first**

Real-parselmouth test on a synthesized tone (deterministic within tolerance), appended to the
foundation's phonation tests:

```python
def test_f0_track_places_a_steady_tone_and_keeps_strength_with_f0() -> None:
    """A synthetic 220 Hz tone reads near 220 Hz where voiced; unvoiced frames are NaN with strength."""
    sr = 16000
    t = np.arange(sr * 2) / sr
    tone = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    tone[: sr // 2] = 0.0  # half a second of silence first
    audio = Audio(waveform=tone[None, :], sampling_rate=sr)
    times, f0, strength = f0_track(audio, f0_min_hz=100.0, f0_max_hz=400.0, hop_s=0.01)
    assert times.shape == f0.shape == strength.shape
    voiced = ~np.isnan(f0)
    assert np.median(f0[voiced]) == pytest.approx(220.0, rel=0.02)
    assert np.isnan(f0[0]) and not np.isnan(strength[0]), "unvoiced is NaN f0, strength retained"
```

Run (FAIL: no `f0_track`), then implement in `phonation/api.py` via
`snd.to_pitch_cc(time_step=hop_s, pitch_floor=f0_min_hz, pitch_ceiling=f0_max_hz)`, reading
`pitch.selected_array["frequency"]` (0 → NaN) and `pitch.selected_array["strength"]`, `_require_parselmouth()`
first like its siblings. Run again — PASS. This is the one real-Praat step in this task; keep it in
the phonation test file so the node tests stay Praat-free.

- [ ] **Step 6.2: Write the failing node tests**

Fixture: reuse Task 5's `make_run` builder (move it to
`src/tests/audio/workflows/triage/nodes/conftest.py` in this step if Task 5 left it in the test
module), extended so the envelope sidecar can carry energetic intervals that are not word-covered,
plus a helper writing SPEECH's speech spans into the store the way Task 5 does (a `span` entity
generated by a `SPEECH` activity). Default fakes: `hnr_track` returns a constant track above any test
floor on the hop grid; `period_marks` returns marks every 1/220 s inside the queried extent;
`f0_track` matches. The override YAML supplies the four `phonation.*` nulls — the production
mechanism — with values chosen per-test (they are fixtures, not recommendations).

```python
def test_packaged_config_refuses_and_the_store_is_untouched() -> None:
    """The gate cannot run ungated: all four phonation.* keys are null by design (N2)."""
    store, _, run_dir = make_voice_run(tmp_path, energetic=[(1.0, 2.0)])
    before = store.fingerprint()
    with pytest.raises(ValueError, match="phonation\\."):
        voice_module.voice(store, load_triage_config(), run_dir)
    assert store.fingerprint() == before


def test_residual_subtracts_labelled_and_speech_but_not_unlabelled_spans() -> None:
    """energy minus airway-labelled minus speech; an unlabelled span is NOT excluded."""
    store, cfg, run_dir = make_voice_run(
        tmp_path,
        energetic=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        airway_labelled=[(1.0, 2.0)],
        speech_spans=[(3.0, 4.0)],
        unlabelled_spans=[(5.0, 6.0)],
    )
    product = voice_module.voice(store, cfg, run_dir)
    runs = [e for e in store.entities("span")
            if store.get_activity(store.generated_by(e.id)).node == "VOICE"]
    assert all(5.0 <= s <= 6.0 for r in runs for s in r.extent), (
        "only the unlabelled region survives the fold; unclaimed activity is exactly what VOICE is for"
    )


def test_empty_residual_is_a_normal_fail() -> None:
    """Every energetic interval belongs to another branch -> fail, with the verdict written."""
    store, cfg, run_dir = make_voice_run(tmp_path, energetic=[(1.0, 2.0)], airway_labelled=[(1.0, 2.0)])
    product = voice_module.voice(store, cfg, run_dir)
    assert product.outcome is Outcome.FAIL
    assert store.entities("verdict")


def test_the_gate_is_an_and_from_both_sides() -> None:
    """High HNR under the RMS floor is periodic room tone; high RMS under the HNR floor is noise.

    Neither passes alone. The fakes hold one condition and starve the other."""
    ...


def test_runs_are_elementary_never_merged() -> None:
    """A one-frame unvoiced gap yields two runs; nothing merges them."""
    # hnr_track fake dips below the floor for exactly one frame mid-interval
    ...
    assert product.verdict["runs_n"] == 2


def test_marks_are_absent_outside_runs_and_absent_is_not_zero() -> None:
    """period_marks is queried per run only; a markless gate-passing run records marks_n=0
    with onset_kind='criterion' (N23), distinct from a run nobody measured."""
    ...


def test_onset_is_a_period_and_offset_is_a_criterion() -> None:
    """A marked run's span starts at its first mark; both edge kinds are named in the attributes."""
    run = _single_marked_run(store)
    assert run.attributes["onset_kind"] == "period"
    assert run.attributes["offset_kind"] == "criterion"
    assert run.extent[0] == pytest.approx(first_mark_time)
    assert run.attributes["offset_criterion"] in {"hnr", "rms", "both", "residual_end"}


def test_period_doubling_alias_inside_the_range_flags() -> None:
    """median F0 * factor (or / factor) inside [f0_min, f0_max] -> ambiguous run, flagged (N21)."""
    # override f0 range [100, 500]; marks at 1/220 s -> 440 also in range -> ambiguous
    ...
    assert product.outcome is Outcome.FLAG
    assert product.verdict["ambiguous_runs_n"] == 1


def test_gate_interval_flag_is_inert_while_unmeasured() -> None:
    """phonation.*_interval keys are null: no near-edge flag fires; gate_interval: 'unmeasured' (N22)."""
    ...


def test_hint_asserting_phonation_not_found_flags() -> None:
    """hint.may_contain includes a voice.hint_tags tag and no run passes -> flag, not fail (N25)."""
    ...


def test_tracks_are_sidecars_on_the_hop_and_measurements_carry_used() -> None:
    """voice_tracks npz exists with hop_s recorded; the activities record what they read."""
    ...
```

- [ ] **Step 6.3: Run them and watch them fail**

`uv run pytest src/tests/audio/workflows/triage/nodes/voice_test.py -x -q` — FAIL (module absent).

- [ ] **Step 6.4: Implement `nodes/voice.py`**

The fold and the gate in full; the rest orchestration:

```python
def _required(config: TriageConfig) -> dict[str, Any]:
    """Resolve every require() key at entry (N2): the four unmeasured phonation keys plus the
    measured Praat parameters and the identity factor."""
    return {
        "f0_min_hz": config.require("phonation.f0_min_hz"),
        "f0_max_hz": config.require("phonation.f0_max_hz"),
        "hnr_floor_db": config.require("phonation.hnr_floor_db"),
        "rms_floor": config.require("phonation.rms_floor"),
        "hop_s": config.require("phonation.hop_s"),
        "silence_threshold": config.require("phonation.silence_threshold"),
        "periods_per_window": config.require("phonation.periods_per_window"),
        "doubling": config.require("phonation.period_doubling_factor"),
    }


def _residual(store: ProvStore, envelope: np.ndarray, floor: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Energetic intervals nobody else claimed: envelope > floor, minus airway-labelled spans,
    minus SPEECH's spans. Unlabelled spans are not subtracted."""
    energetic = _contiguous_true(envelope > floor, sr)
    claimed = _airway_labelled_extents(store) + [
        e.extent for e in store.entities("span")
        if (act := store.generated_by(e.id)) and store.get_activity(act).node == "SPEECH"
    ]
    return _subtract_intervals(energetic, claimed)


def _voiced_runs(times, hnr_db, rms, *, hnr_floor_db, rms_floor) -> list[tuple[int, int]]:
    """Maximal consecutive frames where BOTH conditions hold. Elementary: no merging."""
    ok = (hnr_db >= hnr_floor_db) & (rms >= rms_floor)
    return _runs_of_true(ok)
```

`_contiguous_true`, `_subtract_intervals`, `_runs_of_true` and the RMS track are small pure-numpy
helpers in the module, each with a docstring and no constants. Per residual interval the node runs
`hnr_track`/`f0_track` on the sliced audio, builds the RMS track on the same hop with the derived
window, gates, then per voiced run calls `period_marks` and assembles the span attributes: gate values
at both edges, `onset_kind`/`offset_kind`, the offset criterion (whichever of the two conditions
failed first at the offset frame, or `residual_end` when the run ran into the interval's edge),
`marks_n`. Ambiguity: `f0_median` from the run's marks; ambiguous when
`f0_median * doubling <= f0_max_hz or f0_median / doubling >= f0_min_hz` evaluates the alias inside
the range (both directions checked; the helper's test pins the arithmetic). `fail` when the residual
is empty or no run passes; the hint row (N25) upgrades either `fail` to `flag`. `write_node_verdict`
on every path.

- [ ] **Step 6.5: Run and watch them pass**

`uv run pytest src/tests/audio/workflows/triage/nodes/voice_test.py src/tests/audio/tasks/ -q -k "voice or phonation"`
Expected: all PASS. Then `uv run mypy src/senselab/audio/workflows/triage/ src/senselab/audio/tasks/phonation/`.

- [ ] **Step 6.6: `ruff format` + `ruff check`, commit** — `feat(triage): VOICE — the residual fold, the two-condition gate, marks not contours`

---

### Task 7: REDACT

**Scope:** `nodes/redact.py` — extents from every finding, padded and merged, applied, **verified on
its own output** — and the artifact/store separation that makes the output releasable.

**Design invariants restated for this task** (each is tested):

1. **Every finding is redacted, regardless of speaker.** SPEECH flags target-speaker PII because
   flagging is about which recordings need attention; redaction is about whether an artifact is safe
   to release, and a non-target speaker naming the participant is exactly as unsafe. The two scopes
   differ **deliberately** — a test pins that a non-target finding SPEECH did not flag is still
   redacted.
2. **Extents are padded outward by `redaction.padding_ms` and merged** (`plan_redactions` does both).
   The key is null — the margin must exceed the *worst* alignment edge error, which is unquantified —
   so the node is constructible but **refuses to run without an override** (N3).
3. **Verification is part of the node**: it re-runs ASR (both recognizers PREPROCESS used, N14) and
   the PII scan on its own output, plus the redacted transcript text, and **fails if any finding
   appears**. **The audio check is the weaker one**: ASR on redacted audio may fail to transcribe a
   region that still contains intelligible speech, so a clean re-scan bounds the failure and proves no
   negative — this sentence lives in this plan and the spec, not in the code (rationale rule); the
   verdict records `audio_check: "bounded"` so a reader of the product sees the caveat's flag without
   its prose.
4. **The store is never releasable**; the node writes new elements and artifacts and deletes nothing.
   **A released artifact shares no element ids with the store** — an id indexing both is a join key
   back to the PII. Enforced structurally (the artifact writer never receives a store id) and by test
   (no store id appears as a substring in any artifact byte).
5. **An error message is a disclosure path.** Extents handed to `plan_redactions` are built from `pii`
   entities, which carry category and extent and no text by construction; the node's membership check
   — every category non-empty and free of `+` (the reserved merge character) — is what secures
   `plan_redactions`'s raise as well as the artifact.

**Files:**
- Create: `src/senselab/audio/workflows/triage/nodes/redact.py`
- Test: `src/tests/audio/workflows/triage/nodes/redact_test.py`

**Interfaces**

Consumes:
- store contents as Task 5 leaves them: `pii` entities (category + extent + provenance, no text),
  `word` entities and their `pii` label assertions (for the transcript artifact), the `pii_scan`
  measurement SPEECH writes (`{"name": "pii_scan", "scanned_by": [...], "failed": [...]}`) — its
  **presence** is the evidence the recording was scanned (N15), the `recording` stream entity.
- `plan_redactions(extents, *, padding_ms: int) -> list[RedactionExtent]` — pads outward, merges,
  joins categories with `+`, raises `ValueError` naming bounds and category (never text) on an
  invalid extent (real).
- `apply_redactions(audio, extents) -> Audio` — silences, preserves duration, never mutates (real).
- `RedactionExtent{start, end, category}` (real).
- `transcribe_audios(audios, model, ...) -> List[ScriptLine]` and `HFModel` (mocked).
- `scan_for_pii` / `PiiScan` (mocked).
- `Audio.save_to_file(path)` — WAV default subtype FLOAT; out-of-range write refuses (real).

Produces:

```python
# src/senselab/audio/workflows/triage/nodes/redact.py
@dataclass(frozen=True)
class RedactProduct:
    product: NodeProduct
    artifacts: dict[str, Path]        # {"audio": ..., "transcript": ...}; empty on fail

def redact(
    store: ProvStore,
    config: TriageConfig,
    run_dir: Path,
    artifacts_dir: Path,
    device: DeviceType | None = None,
) -> RedactProduct
```

`verdict` mapping (the design doc's product section, exactly):
`{"redactions_n", "by_category": {}, "padding_ms", "verified": bool, "survived": [], "audio_check":
"bounded"}`. `survived` is non-empty only on the finding-survived `fail` and names **categories,
never matched text**; the could-not-verify `fail` carries `survived: []` with its reason (N16).
Outcome vocabulary: `fail(reason)` or `pass` — `redact.md` gives this node no `flag`.

Store writes: one `REDACT` activity per phase (`plan`, `apply`, `verify`); one entity per planned
extent (`prov_type="span"`, attributes `{"name": "redaction", "category"}`, `wasDerivedFrom` each
`pii` entity it covers); one `verdict` entity. `used` on every `pii` entity, the scan measurement,
the word entities read for the transcript, and the recording stream. Artifacts are written under
`artifacts_dir`, which must not contain or be contained by `run_dir` (checked at entry; the store and
the release directory must not be sweepable by one publish step — capability-map §3.4).

**Mocking boundary for this task:** `transcribe_audios`, `HFModel`, `scan_for_pii` patched on
`nodes.redact`. `plan_redactions` / `apply_redactions` / audio IO / the store run real.

- [ ] **Step 7.1: Write the failing tests**

Fixture: extend the shared conftest builder with `add_pii_finding(store, extent, category,
speaker=...)` writing a `pii` entity + the word-level `pii` label assertions + (once per store) the
`pii_scan` measurement, the way Task 5's node writes them. Default fakes: `transcribe_audios` returns
`[ScriptLine(text="", start=0.0, end=0.0)]` (nothing re-transcribed); `scan_for_pii` returns clean
scans with all three default detectors in `detectors_used`.

```python
def test_constructible_but_refuses_without_a_padding_override() -> None:
    """redaction.padding_ms is null by design: the module imports, the call refuses at entry (N3)."""
    store, _, run_dir = make_redact_run(tmp_path, findings=[((1.0, 1.4), "PERSON")])
    before = store.fingerprint()
    with pytest.raises(ValueError, match="redaction.padding_ms"):
        redact_module.redact(store, load_triage_config(), run_dir, tmp_path / "release")
    assert store.fingerprint() == before


def test_every_finding_is_redacted_regardless_of_speaker() -> None:
    """A non-target finding SPEECH did not flag is exactly as unsafe to release."""
    store, cfg, run_dir = make_redact_run(
        tmp_path,
        findings=[((1.0, 1.4), "PERSON", "SPEAKER_00"), ((3.0, 3.5), "LOCATION", "SPEAKER_01")],
    )
    result = redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    assert result.product.verdict["redactions_n"] == 2
    audio = Audio(filepath=result.artifacts["audio"])
    x = np.asarray(audio.waveform)[0]
    pad = cfg.require("redaction.padding_ms") / 1000.0
    for s, e in ((1.0, 1.4), (3.0, 3.5)):
        assert not x[int((s - pad + EDGE) * SR) : int((e + pad - EDGE) * SR)].any(), "silenced, padded outward"


def test_padded_overlapping_extents_merge_and_categories_join() -> None:
    """Two findings whose padded extents touch become one redaction; an audible sliver between
    two separate redactions is the failure merging exists to prevent."""
    store, cfg, run_dir = make_redact_run(
        tmp_path, findings=[((1.0, 1.2), "PERSON"), ((1.25, 1.5), "LOCATION")]
    )  # override padding makes them overlap
    result = redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    assert result.product.verdict["redactions_n"] == 1
    assert result.product.verdict["by_category"] == {"PERSON+LOCATION": 1}


def test_a_category_containing_plus_is_refused_by_the_node_not_discovered_later() -> None:
    """+ is reserved for merged categories; a label carrying it would silently decompose (invariant 5)."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=[((1.0, 1.4), "A+B")])
    with pytest.raises(ValueError, match="reserved") as err:
        redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    assert "A+B" in str(err.value), "the message names the category and bounds only"
    # pii entities carry no text field at all, so the exception cannot quote a match


def test_verification_reruns_both_recognizers_and_a_surviving_finding_fails() -> None:
    """Any finding in the re-scan — survivor or new — withholds the artifact and names categories only."""
    seen_models: list[str] = []

    def fake_transcribe(audios, model, **kw):
        seen_models.append(str(model.path_or_uri))
        return [ScriptLine(text="jane doe", start=1.0, end=1.4)]

    monkeypatch.setattr(redact_module, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(redact_module, "scan_for_pii", _scan_finding("jane doe", "PERSON"))
    store, cfg, run_dir = make_redact_run(tmp_path, findings=[((1.0, 1.4), "PERSON")])
    result = redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    assert result.product.outcome is Outcome.FAIL
    assert result.product.verdict["survived"] == ["PERSON"], "categories, never matched text"
    assert "jane doe" not in json.dumps(result.product.verdict)
    assert sorted(seen_models) == sorted({CW, QW}), "both recognizers PREPROCESS used (N14)"
    assert result.artifacts == {}, "a failed verification releases nothing"


def test_verification_scans_the_redacted_transcript_alongside_the_audio() -> None:
    """A finding surviving only in the transcript artifact is caught by the same gate."""
    ...


def test_a_failed_detector_during_verification_withholds() -> None:
    """could-not-verify is fail(survived=[]) -> withheld, not a pass and not not_assessed (N16)."""
    monkeypatch.setattr(redact_module, "scan_for_pii", _failing_scan({"gliner": "load failed"}))
    ...
    assert result.product.outcome is Outcome.FAIL
    assert result.product.verdict["survived"] == []


def test_released_artifacts_share_no_element_ids_with_the_store() -> None:
    """An id indexing both the store and a released artifact is a join key back to the PII."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=[((1.0, 1.4), "PERSON")])
    result = redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    ids = [e.id for e in store.entities()]
    for path in result.artifacts.values():
        blob = path.read_bytes()
        for eid in ids:
            assert eid.encode() not in blob


def test_the_source_is_not_destroyed_and_the_store_only_grows() -> None:
    """Redaction writes; deletion is an operator decision with its own authorisation."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=[((1.0, 1.4), "PERSON")])
    entities_before = {e.id for e in store.entities()}
    redact_module.redact(store, cfg, run_dir, tmp_path / "release")
    assert (run_dir / "plain.wav").exists()
    assert entities_before <= {e.id for e in store.entities()}, "append-only: nothing removed"


def test_an_unscanned_store_is_refused_not_certified() -> None:
    """No pii_scan measurement means 'unchecked', which must not launder into releasable (N15)."""
    store, cfg, run_dir = make_redact_run(tmp_path, findings=[], scanned=False)
    with pytest.raises(ValueError, match="no PII scan"):
        redact_module.redact(store, cfg, run_dir, tmp_path / "release")


def test_zero_findings_still_verifies_before_passing() -> None:
    """A clean scan's artifact is verified like any other; verification is part of the node."""
    ...
    assert result.product.verdict["verified"] is True and result.product.verdict["redactions_n"] == 0


def test_artifacts_dir_nested_in_run_dir_is_refused() -> None:
    """The store's directory and the release directory must not be one publish step apart."""
    with pytest.raises(ValueError, match="artifacts_dir"):
        redact_module.redact(store, cfg, run_dir, run_dir / "release")


def test_transcript_artifact_replaces_findings_with_category_placeholders() -> None:
    """Words inside planned extents render as [CATEGORY]; padded-in neighbours go with them,
    matching what the audio lost; no timestamps, no ids, no matched text."""
    ...
```

- [ ] **Step 7.2: Run them and watch them fail**

`uv run pytest src/tests/audio/workflows/triage/nodes/redact_test.py -x -q` — FAIL (module absent).

- [ ] **Step 7.3: Implement `nodes/redact.py`**

The extent construction and verification in full:

```python
_RESERVED_CATEGORY_CHAR = "+"  # plan_redactions' merge separator; a string, not a threshold


def _extents_from_findings(store: ProvStore) -> list[RedactionExtent]:
    """Every pii entity, regardless of speaker; the membership check that secures the error path."""
    extents = []
    for finding in store.entities("pii"):
        category = finding.attributes.get("category", "")
        if not category or _RESERVED_CATEGORY_CHAR in category:
            raise ValueError(
                f"category {category!r} is empty or contains the reserved merge character; "
                "it cannot be planned without silently decomposing on re-planning"
            )
        if finding.extent is None:
            raise ValueError(f"pii finding {finding.id} has no extent; nothing locatable can be redacted")
        extents.append(RedactionExtent(start=finding.extent[0], end=finding.extent[1], category=category))
    return extents


def _verify(redacted: Audio, transcript_text: str, asr_models: list[tuple[str, str]], device) -> tuple[bool, list[str], bool]:
    """Re-run both recognizers and the scan on the node's own output.

    Returns (verified, survived_categories, scan_ran). Any finding anywhere fails; a detector
    failure means the check did not run, which is not a clean result.
    """
    hypotheses = []
    for model_id, commit_sha in asr_models:  # (id, sha) pairs read from the store's model agents
        model = HFModel(path_or_uri=model_id, revision=commit_sha)
        (line,) = transcribe_audios([redacted], model=model, device=device)
        hypotheses.append(flatten_script_line(line))
    scans = scan_for_pii([*hypotheses, transcript_text])
    scans = scans if isinstance(scans, list) else [scans]
    if any(s.failures for s in scans):
        return False, [], False
    survived = sorted({span.category for s in scans for span in s.spans})
    return not survived, survived, True
```

`redact()` order: entry checks (`require("redaction.padding_ms")`, artifacts_dir outside the store tree, the
`pii_scan` measurement's presence) → `_extents_from_findings` → `plan_redactions` → load the
`recording` stream → `apply_redactions` → build the transcript from SPEECH's consensus words with
planned-extent words replaced by `[CATEGORY]` → **verify** → only on verified success write both
artifacts to `artifacts_dir` and return them; on any verification failure write nothing releasable,
return `artifacts={}` with the `fail` verdict. Store writes as the Interfaces block lists; the verdict
entity's `detail` carries the verdict mapping. The artifact writer is a private function that takes
waveform, transcript text and paths — **it has no store parameter**, so it cannot embed an element id.

- [ ] **Step 7.4: Run and watch them pass**

`uv run pytest src/tests/audio/workflows/triage/nodes/redact_test.py -q`, then
`uv run mypy src/senselab/audio/workflows/triage/`.

- [ ] **Step 7.5: `ruff format` + `ruff check`, commit** — `feat(triage): REDACT — every finding, padded and merged, verified on its own output`

---

### Task 8: VERDICT

**Scope:** `nodes/verdict.py`. **The fold already exists** — `fold_file_verdict` in
`vocabulary.py`, tested by the foundation — and this node **wires store contents into it; it does not
reimplement it**. Everything in this task is reading, mapping and recording: node verdicts out of
`verdict` entities, TAXONOMY's `kind` entities into `KindState`, REDACT's outcome into `Release`, the
fold's result into one new `verdict` entity with `used` edges to everything it folded.

**The fold's actual signature, verified on this branch** (the node conforms to it, changes nothing
in it):

```python
def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    kind_predictions: Mapping[str, KindState],
    ran: Mapping[str, RunState],
    release: Release = Release.NOT_ASSESSED,
) -> FileVerdict
```

with `Release.RELEASABLE / WITHHELD / NOT_ASSESSED`, `RunState.COMPLETED / SKIPPED / ERRORED`, and
`FileVerdict{triage, release, kinds, reasons, ran}`. The fold already implements: ADMIT-fail first
(rule 1), any flag or contradiction (rule 2), every-kind-absent (rule 3, a **different** `fail` from
ADMIT's — could-not-measure vs measured-and-empty), otherwise pass; the contradiction table including
the never-ran rows; undecided resolution. None of that is re-tested here beyond one integration case
— the node's tests target the **wiring**.

**Design invariants restated for this task** (each is tested at the wiring level):

1. **A branch `fail` is not a file `fail`** — a cough recording where SPEECH failed against an
   absent-speech prediction folds to `pass`.
2. **ADMIT-fail and every-kind-absent are distinct `fail`s** with different reasons; a consumer that
   cannot tell them apart treats an empty recording as a broken one.
3. **`not_assessed` is not `releasable`** — no REDACT verdict in the store must never map to
   `RELEASABLE`, and the node never defaults `release`.
4. **A branch that never ran is not a branch that failed** — `ran` is caller-supplied because only the
   runner can know `errored`; the store-derived fallback is stated as unable to see `errored` (N26).
5. **Decides nothing a node has not decided** — where nodes contradict each other it records both and
   flags; `reasons` carries every contribution, not only the deciding one.

**Files:**
- Create: `src/senselab/audio/workflows/triage/nodes/verdict.py`
- Test: `src/tests/audio/workflows/triage/nodes/verdict_test.py`

**Interfaces**

Consumes: `fold_file_verdict`, `FileVerdict`, `NodeVerdict`, `Outcome`, `KindState`, `RunState`,
`Release` from `vocabulary.py`; `read_node_verdict_entities` / `node_verdict_from_entity` /
`write_node_verdict` / `software_agent` from `elements.py`; `kind` entities as `taxonomy.md`'s product
writes them (`{"kind", "state"}` with `state` ∈ present/absent/undecided/**not_screened**);
`ProvStore.get_activity` (N19). **No models — no mocks in this task.**

Produces:

```python
# src/senselab/audio/workflows/triage/nodes/verdict.py
@dataclass(frozen=True)
class VerdictProduct:
    file_verdict: FileVerdict
    view: list[str]                   # the file-verdict entity id, then every id it folded

def verdict(
    store: ProvStore,
    config: TriageConfig,
    ran: Mapping[str, RunState] | None = None,
) -> VerdictProduct
```

(`config` is taken for signature uniformity and the activity's `config_hash` parameter; VERDICT has no
thresholds — "any threshold that would turn a flag into a pass" is out of scope by design.)

Store writes: one `VERDICT` activity; one `verdict` entity whose attributes carry
`{"scope": "file", "triage", "release", "kinds", "ran", "gated", "reasons": [...]}` (reasons as plain
dicts of `NodeVerdict` fields); `used` from the activity to **every** node-verdict entity and `kind`
entity folded; `wasGeneratedBy` + `wasAttributedTo` the software agent. Node-verdict entities are
distinguished from the file verdict by the `scope` attribute (node verdicts carry `scope: "node"` via
`write_node_verdict`; if the sibling's Task 1 chose a different discriminator, use that — reviewer
reconciles).

The mappings, stated once and tested:

| store fact | fold input |
| --- | --- |
| `verdict` entity per node, in graph order `ADMIT, PREPROCESS, TAXONOMY, AIRWAY, SPEECH, VOICE, REDACT` | `node_verdicts` (a vocabulary list, not a number) |
| `kind` entity `state: "not_screened"` | `KindState.UNDECIDED` (N27) |
| `kind` entity states otherwise | `KindState(state)` |
| no `kind` entities at all (TAXONOMY absent) | `kind_predictions = {}`; the file verdict records `screened: false` |
| REDACT verdict `outcome: pass` | `Release.RELEASABLE` — **for its artifacts only**, never the store |
| REDACT verdict `outcome: fail` | `Release.WITHHELD` |
| no REDACT verdict | `Release.NOT_ASSESSED` |
| `ran` omitted by the caller | derived: node has a verdict entity → `COMPLETED`, else `SKIPPED`; cannot see `ERRORED` (N26) |
| any kind predicted `absent` with no branch verdict | `gated: true` on the file-verdict entity — the contradiction check did not happen for it |

- [ ] **Step 8.1: Write the failing tests**

Fixture: a builder writing node-verdict and kind entities directly through `write_node_verdict` and
`store.entity(prov_type="kind", ...)` — no branch nodes run in these tests.

```python
def test_a_branch_fail_against_an_absent_kind_is_a_file_pass() -> None:
    """A cough recording: airway present+pass, speech absent+fail, voice absent-by-resolution."""
    store = make_verdict_store(
        node_verdicts=[
            ("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.PASS, None),
            ("AIRWAY", Outcome.PASS, "airway"), ("SPEECH", Outcome.FAIL, "speech"),
            ("VOICE", Outcome.FAIL, "voice_no_words"),
        ],
        kinds={"airway": "present", "speech": "absent", "voice_no_words": "not_screened"},
    )
    result = verdict_module.verdict(store, cfg)
    assert result.file_verdict.triage is Outcome.PASS, "a branch fail is not a file fail"
    assert result.file_verdict.kinds["voice_no_words"] is KindState.ABSENT, (
        "not_screened maps to UNDECIDED (N27), which VOICE's fail resolves to absent"
    )


def test_admit_fail_and_every_kind_absent_are_distinct_fails() -> None:
    """could-not-measure and measured-and-empty carry different reasons, in different shapes."""
    broken = verdict_module.verdict(make_verdict_store(
        node_verdicts=[("ADMIT", Outcome.FAIL, None)], kinds={}), cfg)
    empty = verdict_module.verdict(make_verdict_store(
        node_verdicts=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.FAIL, None)],
        kinds={"airway": "absent", "speech": "absent", "voice_no_words": "absent"}), cfg)
    assert broken.file_verdict.triage is empty.file_verdict.triage is Outcome.FAIL
    assert broken.file_verdict.reasons[0].node == "ADMIT"
    assert any("every kind is absent" in r.why for r in empty.file_verdict.reasons)
    assert not any("every kind is absent" in r.why for r in broken.file_verdict.reasons)


def test_release_mapping_and_not_assessed_is_not_releasable() -> None:
    """No REDACT verdict -> NOT_ASSESSED; fail -> WITHHELD; pass -> RELEASABLE. Never a default."""
    none_ran = verdict_module.verdict(make_verdict_store(node_verdicts=BASE, kinds=KINDS), cfg)
    assert none_ran.file_verdict.release is Release.NOT_ASSESSED
    assert none_ran.file_verdict.release is not Release.RELEASABLE, (
        "a recording with no speech has no scan; unexamined must not read as cleared"
    )
    withheld = verdict_module.verdict(make_verdict_store(
        node_verdicts=[*BASE, ("REDACT", Outcome.FAIL, None)], kinds=KINDS), cfg)
    assert withheld.file_verdict.release is Release.WITHHELD
    released = verdict_module.verdict(make_verdict_store(
        node_verdicts=[*BASE, ("REDACT", Outcome.PASS, None)], kinds=KINDS), cfg)
    assert released.file_verdict.release is Release.RELEASABLE


def test_contradiction_wiring_resolves_the_kind_and_flags() -> None:
    """absent-predicted kind whose branch passed -> flag, kind resolved present, both visible."""
    result = verdict_module.verdict(make_verdict_store(
        node_verdicts=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
        kinds={"speech": "absent"}), cfg)
    assert result.file_verdict.triage is Outcome.FLAG
    assert result.file_verdict.kinds["speech"] is KindState.PRESENT
    kind_entities = store.entities("kind")
    assert kind_entities[0].attributes["state"] == "absent", (
        "TAXONOMY's assertion stays in the store; the resolution is this node's, and both remain"
    )


def test_a_present_kind_whose_branch_never_ran_flags() -> None:
    """The absence of evidence, on a kind the graph was asked about, is a gap a human sees."""
    result = verdict_module.verdict(
        make_verdict_store(node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"airway": "present"}),
        cfg, ran={"ADMIT": RunState.COMPLETED, "AIRWAY": RunState.SKIPPED},
    )
    assert result.file_verdict.triage is Outcome.FLAG


def test_ran_is_derived_when_omitted_and_cannot_see_errored() -> None:
    """verdict entity -> completed, none -> skipped; the docstring states the errored blindness (N26)."""
    result = verdict_module.verdict(make_verdict_store(node_verdicts=BASE, kinds=KINDS), cfg)
    assert result.file_verdict.ran["ADMIT"] is RunState.COMPLETED
    assert result.file_verdict.ran["REDACT"] is RunState.SKIPPED


def test_gated_run_is_marked() -> None:
    """An absent kind with no branch verdict marks gated: the contradiction check did not happen."""
    verdict_module.verdict(make_verdict_store(
        node_verdicts=[("ADMIT", Outcome.PASS, None)], kinds={"speech": "absent"}), cfg)
    file_entity = _file_verdict_entity(store)
    assert file_entity.attributes["gated"] is True


def test_the_fold_is_wired_not_reimplemented() -> None:
    """The node's module calls fold_file_verdict; no second fold lives here."""
    import inspect
    src = inspect.getsource(verdict_module)
    assert "fold_file_verdict(" in src
    assert "_BRANCH_FOR_KIND" not in src, "the kind->branch table stays in vocabulary.py"


def test_every_folded_id_is_used_and_the_view_leads_with_the_file_verdict() -> None:
    """used edges to every node-verdict and kind entity; view = [file id, *folded ids]."""
    ...


def test_reasons_carry_every_contribution() -> None:
    """A flag naming one cause hides the others; every node's verdict appears in reasons."""
    ...
```

- [ ] **Step 8.2: Run them and watch them fail**

`uv run pytest src/tests/audio/workflows/triage/nodes/verdict_test.py -x -q` — FAIL (module absent).

- [ ] **Step 8.3: Implement `nodes/verdict.py`**

The whole node is small enough to state nearly in full:

```python
_NODE = "VERDICT"
_GRAPH_ORDER = ("ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH", "VOICE", "REDACT")
_NOT_SCREENED = "not_screened"


def _node_verdicts_in_graph_order(store: ProvStore) -> list[tuple[Entity, NodeVerdict]]:
    """Node-scope verdict entities, ordered by the graph, unknown nodes last in store order."""
    pairs = [(e, node_verdict_from_entity(e)) for e in read_node_verdict_entities(store)
             if e.attributes.get("scope") == "node"]
    return sorted(pairs, key=lambda p: _GRAPH_ORDER.index(p[1].node)
                  if p[1].node in _GRAPH_ORDER else len(_GRAPH_ORDER))


def _kind_predictions(store: ProvStore) -> tuple[dict[str, KindState], list[str]]:
    """TAXONOMY's kind entities as KindStates; not_screened is UNDECIDED (N27)."""
    predictions, ids = {}, []
    for e in store.entities("kind"):
        state = e.attributes["state"]
        predictions[e.attributes["kind"]] = (
            KindState.UNDECIDED if state == _NOT_SCREENED else KindState(state)
        )
        ids.append(e.id)
    return predictions, ids


def _release_from(verdicts: Sequence[NodeVerdict]) -> Release:
    """REDACT's outcome, for its artifacts only; absent means unexamined, never releasable."""
    redact = next((v for v in verdicts if v.node == "REDACT"), None)
    if redact is None:
        return Release.NOT_ASSESSED
    return Release.WITHHELD if redact.outcome is Outcome.FAIL else Release.RELEASABLE
```

`verdict()` assembles: derived-or-supplied `ran`; `fold_file_verdict(node_verdicts, predictions, ran,
release=_release_from(...))`; `gated = any(state is KindState.ABSENT and kind has no branch verdict)`;
writes the file-verdict entity (attributes as the Interfaces block lists, `scope: "file"`), records
`used` for every folded id, and returns `VerdictProduct(file_verdict, [file_id, *folded_ids])`. Its
docstring states what the derived `ran` cannot see. No numbers anywhere; the only tables are
vocabulary.

- [ ] **Step 8.4: Run and watch them pass**

`uv run pytest src/tests/audio/workflows/triage/nodes/ -q` — the whole nodes suite, serially.
Then `uv run mypy src/senselab/audio/workflows/triage/` and, once, the full triage tree:
`uv run pytest src/tests/audio/workflows/triage/ src/tests/utils/prov_store_test.py -q`.

- [ ] **Step 8.5: `ruff format` + `ruff check`, commit** — `feat(triage): VERDICT — wire the store into the existing fold; two axes, never collapsed`

---

## Self-review against the four node documents

Checked line-by-line against `branch-speech.md`, `branch-voice.md`, `redact.md`, `verdict.md` after
drafting. Covered and pinned by tests: spans-from-words (not envelope); no ASR in SPEECH;
interval-restricted diarization with the offset shift; withdrawal never relabelling; separation only
on count ≠ 1 with ≥ 3 reported; stream-recorded measurements; both-hypothesis PII scan with the
speaker-scoped decision and the failures row; quality parallel and non-gating; no matched text
anywhere; the residual fold with unlabelled spans kept; the two-condition gate refusing to run
ungated; elementary runs; marks absent-not-zero; onset-period/offset-criterion; redact-everything
scope; outward padding with merge; verification on both recognizers plus the transcript; no shared
ids; store never releasable; source not destroyed; branch-fail ≠ file-fail; the two distinct fails;
`not_assessed` ≠ `releasable`; never-ran ≠ failed; reasons complete.

**Gaps this plan leaves open, deliberately or honestly:**

1. **SPEECH's figure is not planned.** `branch-speech.md` names "one aligned figure per recording" as
   an artifact beside the store. `plot_aligned_panels` exists and capability-map §1.4 maps the
   panels, but the figure must honour PII word-markings (§3.4: "the figure renders words and would
   otherwise leak what the scan just found"), which makes it a rendering task with its own redaction
   rule. Deferred to a follow-on task rather than half-planned here; until it exists the product's
   `figure` slot is absent, which the design tolerates for REDACT's artifacts but not, strictly, for
   SPEECH's product. **This is a real deviation from `branch-speech.md`'s product table.**
2. **Fabrication candidates still have no consumer** (open.md says so; N9 keeps them a flag and a
   label). The flag row "fabrication candidates survive" is implemented as "the candidate set is
   non-empty", which is the weakest honest reading of "survive".
3. **`f0_candidates` is narrowed to the selected candidate with strength** (N24). The design's word
   is "candidates"; full per-frame candidate lists need deeper parselmouth surgery and have no
   consumer yet.
4. **VOICE's "loud phonation" and "maximum phonation time"** are members the design itself says the
   branch cannot label; nothing here computes the run-vs-run contrast or the named offset criterion
   beyond recording which gate condition released. Consumers compute them from the marks — stated,
   not planned.
5. **The `pii_scan` measurement's exact name/shape is a Task 5 invention** (`{"name": "pii_scan",
   "scanned_by", "failed"}`) that Task 7's N15 gate depends on; if the reviewer renames it in
   reconciliation, both tasks move together.
6. **REDACT's verification cannot re-run subprocess-venv recognizers cheaply in CI** — the node tests
   mock them, so the end-to-end verification path is exercised only with fakes. A future integration
   test on a GPU host (ORCD recipe exists) would close it.
7. **`speech.hint_tags` / `voice.hint_tags` (N25) are seeded vocabularies, not measurements.** They
   are config values with a derivation note saying exactly that; a reviewer preferring a different
   hint-matching rule changes one key, not code.

**Capability-map corrections found while planning** (the map is evidence, not gospel):

- §1.5's `separate_audios(model=HFModel("alibabasglab/MossFormer2_SS_16K"), ...)` row and §4.5's
  `utils.clearvoice` reference describe branch `triage`, **not** `design/triage-workflow-dag`: on this
  branch `source_separation/api.py` is unasdiff-only and *refuses* that model id, and
  `utils/clearvoice.py` does not exist. Hence the merge prerequisite at the top of this plan.
- §1.5's PII rows ("`PiiSpan` has no offsets and no times", "MISSING — locating a finding") are
  **stale on this branch**: foundation Task 1 landed `PiiSpan(ScriptLine)`, and findings from a
  scanned line inherit its extent and speaker. The projection rule ("never matched text") remains the
  node's job and is planned in Tasks 5 and 7.
- §1.7's redaction rows ("nothing in senselab redacts anything") are likewise stale: the foundation's
  `redaction` task (`RedactionExtent`, `plan_redactions`, `apply_redactions`) exists on this branch
  with exactly the pad/merge/`+`-join semantics `redact.md` requires.
