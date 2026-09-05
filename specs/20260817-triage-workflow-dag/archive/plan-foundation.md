# Triage Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the element store and the five missing DSP/redaction tasks that all nine triage nodes depend on.

**Architecture:** An append-only element store in `utils/` carries elements and assertions with provenance; a `workflows/triage/` package holds the triage vocabulary; five new tasks under `audio/tasks/` supply the DSP the design needs and senselab lacks. No node logic here — nodes are a follow-on plan. Every model call the nodes need already exists.

**Tech Stack:** Python 3.12, pydantic v2, numpy, scipy, soundfile, pytest. uv for everything.

## Global Constraints

- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`.** Each xdist worker duplicates ~535 MB of frameworks. Run the directory you changed.
- Tests live in `src/tests/` mirroring the package, named `*_test.py`.
- Google-style docstrings; line length 120; type hints required (mypy with the pydantic plugin).
- **Rationale does not go in code.** Docstrings say what a thing is and how to call it. Measurements and rejected alternatives go in `specs/20260817-triage-workflow-dag/benchmarks/`.
- **No numeric constant appears in code.** Not as a signature default, not as a module-level constant.
  Every number lives in `data/config/default.yaml` with its derivation beside it, per `CLAUDE.md`:
  "Thresholds belong in `data/` with a written derivation, never as code literals." Functions take the
  values they need as arguments; the caller reads them from the config.
- **A value nobody has measured is `null` in the config, and reading it raises.** The loader names the
  parameter and points at `benchmarks/open.md`. This replaces keyword-only-without-default as the
  mechanism for making an unmeasured value impossible to use by accident.
- A model load passes a **resolved 40-hex commit SHA, never a ref**. There is an AST-sweep guard test.
- `uv sync` is subtractive — always pass `--all-extras`.
- Run `ruff format` before every commit.

**Design source of truth:** `specs/20260817-triage-workflow-dag/` — `store.md` first, then the node documents. `capability-map.md` maps every requirement to existing code. `benchmarks/open.md` lists what is deliberately unmeasured.

---

### Task 1: `PiiSpan` becomes a `ScriptLine`

**Why first:** `capability-map.md` names this the highest-leverage change in the codebase. A finding can say what it is but not where, so SPEECH's speaker-scoped PII rule and all of REDACT are unimplementable.

**Why subclass rather than compose:** a PII finding *is* a timed piece of text, which is what `ScriptLine` already is. Subclassing gives `text`, `start`, `end`, `speaker`, `score`, `chunks` and `timestamp_model` natively — no `.line` indirection and no `start_s` helper duplicating `start`. `timestamp_model` is the load-bearing one: it exists because two recognizers timed by the same aligner agree for reasons unrelated to the audio, so a finding must record which producer timed it.

Verified before planning this: `PiiSpan` is a plain `@dataclass` (not frozen), every construction in the tree is keyword-only, its existing `text` and `score` fields already exist on `ScriptLine` with the same meaning, and `ScriptLine` requires `text` or `speaker` — which a finding always has. One incompatibility exists and is handled in Step 4.

**Files:**
- Modify: `src/senselab/text/tasks/pii_detection/api.py` (`PiiSpan`, `_materialize_spans`, `scan_for_pii`)
- Modify: `src/senselab/audio/workflows/audio_analysis/pii.py:275` (one line — `asdict` no longer applies)
- Test: `src/tests/text/tasks/pii_detection_test.py`

**Interfaces:**
- Consumes: `ScriptLine` from `senselab.utils.data_structures`.
- Produces: `class PiiSpan(ScriptLine)` adding `category: str`, `source: str`, `asr_model: str`. Location is `span.start` / `span.end`; attribution is `span.speaker`. Task 9 (redaction) and the node plan's SPEECH step 7 consume those directly.
- **Every existing `PiiSpan(...)` construction keeps working** — all are keyword-only and pass only fields the subclass still has.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_finding_is_a_script_line() -> None:
    """An unlocated finding is a ScriptLine with start and end still None."""
    from senselab.text.tasks.pii_detection.api import PiiSpan
    from senselab.utils.data_structures import ScriptLine

    span = PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9)
    assert isinstance(span, ScriptLine), "a finding is a timed piece of text, not a parallel type"
    assert span.start is None and span.end is None, "unlocated until something locates it"


def test_a_located_finding_reports_its_extent_and_speaker_natively() -> None:
    """Location is the inherited start/end/speaker, with no parallel start_s name."""
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane Doe", category="PERSON", source="presidio", asr_model="w",
        start=11.9, end=12.4, speaker="SPEAKER_00",
    )
    assert (span.start, span.end) == (11.9, 12.4)
    assert span.speaker == "SPEAKER_00"
    assert not hasattr(span, "start_s"), "no parallel name for a field that already exists"


def test_the_timing_provenance_lives_on_the_finding() -> None:
    """A finding records which producer timed it, via the inherited timestamp_model."""
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane", category="PERSON", source="presidio", asr_model="w",
        start=1.0, end=1.3, timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    assert span.timestamp_model == "Qwen/Qwen3-ForcedAligner-0.6B", (
        "two recognizers timed by one aligner are not independent witnesses"
    )


def test_scanning_a_script_line_carries_its_timing_onto_every_finding() -> None:
    """Every finding from a scanned line inherits that line's extent and speaker."""
    from senselab.text.tasks.pii_detection.api import scan_for_pii
    from senselab.utils.data_structures import ScriptLine

    line = ScriptLine(text="call alice@example.com", speaker="SPEAKER_01", start=3.0, end=4.5)
    scan = scan_for_pii(line, detectors=["rules"])
    assert scan.spans, "rules detector found nothing"
    assert all(sp.start == 3.0 and sp.end == 4.5 for sp in scan.spans)
    assert all(sp.speaker == "SPEAKER_01" for sp in scan.spans)


def test_scanning_a_bare_string_leaves_the_finding_unlocated() -> None:
    """A bare string carries no timing, so its findings claim none."""
    from senselab.text.tasks.pii_detection.api import scan_for_pii

    scan = scan_for_pii("call alice@example.com", detectors=["rules"])
    assert scan.spans
    assert all(sp.start is None for sp in scan.spans), "a bare string has no timing to claim"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/tests/text/tasks/pii_detection_test.py -k "script_line or located or provenance or unlocated" -v`
Expected: FAIL — `assert isinstance(span, ScriptLine)` fails; `PiiSpan` is a dataclass.

- [ ] **Step 3: Make `PiiSpan` a `ScriptLine`**

In `api.py`, add the import and replace the dataclass. Remove the `@dataclass` decorator:

```python
from senselab.utils.data_structures import ScriptLine
```

```python
class PiiSpan(ScriptLine):
    """One PII detection, as the timed line it sits in.

    A ``ScriptLine``, so ``text``, ``start``, ``end``, ``speaker``, ``score`` and ``timestamp_model``
    are inherited and mean what they mean everywhere else. ``start`` and ``end`` are None until
    something locates the finding — scanning a bare string cannot.

    Attributes:
        category: The entity type, e.g. ``"PERSON"``, ``"EMAIL_ADDRESS"``.
        source: The detector that found it, e.g. ``"presidio"`` or ``"gliner/<label>"``.
        asr_model: Identifier of the scanned input.
    """

    category: str
    source: str
    asr_model: str
```

- [ ] **Step 4: Fix the one incompatibility**

`src/senselab/audio/workflows/audio_analysis/pii.py:275` calls `dataclasses.asdict(s)`, which only works on a dataclass. Replace it:

```python
        "spans": [{**s.model_dump(exclude_none=True), "perturbation": report.perturbation} for s in report.spans],
```

`exclude_none=True` keeps the serialised span close to what `asdict` produced — a `ScriptLine` has more fields than the old dataclass, and an unlocated finding leaves most of them None. Remove `asdict` from that file's imports if nothing else uses it.

- [ ] **Step 5: Carry the line's timing through the scanner**

The real `_materialize_spans` (`api.py:326`) already has two behaviours this step must not lose:
it dedupes on `(category, text.strip().lower(), source)` — so "Jane Doe" from Presidio and
"jane doe " from the rules cascade are one finding per detector, which
`_compute_detection_confidence`'s agreement counting depends on — and it skips a raw span whose
`text` is empty, whitespace-only or missing, reading every key with a `.get(...) or <default>`.
That guard matters *more* after the subclassing: `ScriptLine` accepts `text=""`, so an empty
finding would materialise and inflate `n_spans`, and a missing `text` alongside no `speaker`
would raise `ValidationError` instead of being skipped. Keep the body and add only the timing
copy:

```python
def _materialize_spans(
    raw_spans: list[dict[str, Any]], source_id: str, line: ScriptLine | None = None
) -> list[PiiSpan]:
    """Build deduped ``PiiSpan`` objects from one input's raw subprocess spans.

    Dedupe key is ``(category, normalized_text, source)`` so a single entity detected by
    both Presidio and GLiNER counts once per detector rather than once per phrasing —
    mirrors the per-ASR-model dedup the ``audio_analysis`` workflow's per-pass PII adapter
    does on top of this.

    Args:
        raw_spans: Raw span dicts for one input, as returned under one
            ``spans_by_asr`` key by ``detect_pii_via_subprocess``.
        source_id: Identifier for the scanned input, stored on ``PiiSpan.asr_model``.
            Standalone text has no ASR backend; this reuses that field as the
            per-input identifier ``_compute_detection_confidence`` groups by, rather
            than adding a parallel field for a single-input-per-report caller.
        line: The line scanned, when the input was one. Its extent, speaker and timing
            provenance are copied onto every finding; a bare-string input leaves them None.

    Returns:
        Deduped ``PiiSpan`` list for the input.
    """
    seen: set[tuple[str, str, str]] = set()
    spans: list[PiiSpan] = []
    timing: dict[str, Any] = (
        {
            "start": line.start,
            "end": line.end,
            "speaker": line.speaker,
            "timestamp_source": line.timestamp_source,
            "timestamp_model": line.timestamp_model,
        }
        if line is not None
        else {}
    )
    for raw in raw_spans:
        text: str = raw.get("text") or ""
        category: str = raw.get("category") or ""
        source: str = raw.get("source") or "unknown"
        normalized = text.strip().lower()
        dedup_key: tuple[str, str, str] = (category, normalized, source)
        if not normalized or dedup_key in seen:
            continue
        seen.add(dedup_key)
        score = raw.get("score")
        spans.append(
            PiiSpan(
                text=text,
                category=category,
                source=source,
                asr_model=source_id,
                score=float(score) if score is not None else None,
                **timing,
            )
        )
    return spans
```

The call site is the final `return _finish(...)` of `scan_for_pii` (`api.py:503-511`), a list
comprehension over `range(len(texts))` with no per-item `ScriptLine` in scope. Replace the whole
comprehension with a loop over `items` — in scope from the top of the function — so each input's
line can travel to its spans:

```python
    scans: list[PiiScan] = []
    for i, item in enumerate(items):
        line = item if isinstance(item, ScriptLine) else None
        scans.append(
            PiiScan(
                spans=_materialize_spans(spans_by_index_raw.get(str(i), []), str(i), line=line),
                detectors_used=list(detectors_used),
                failures=dict(failures),
            )
        )
    return _finish(scans)
```

- [ ] **Step 6: Run the new tests and everything that touches PII**

Run: `uv run pytest src/tests/text/tasks/ src/tests/audio/tasks/pii_detection_test.py src/tests/audio/workflows/pii_adapter_test.py -v`
Expected: the five new tests PASS and **every existing test passes unchanged**. No test in the tree
asserts the exact key set of a serialised span — `pii_adapter_test.py:102-104` checks only that
`payload["spans"]` is truthy and that every span carries `perturbation` and `asr_model` — so a
failure anywhere means the change was not additive and the production code is wrong. The one real
shape change is in Step 4's serialisation: `model_dump(exclude_none=True)` **drops** `score` when it
is None, where `asdict` always emitted `"score": None`. `global_summary.py` reads `s.score` as an
attribute and is unaffected, but any external consumer of `pii.json` that indexes `span["score"]`
must switch to `.get("score")`.

- [ ] **Step 7: Commit**

```bash
uv run ruff format src/senselab/text/tasks/pii_detection/api.py src/senselab/audio/workflows/audio_analysis/pii.py src/tests/text/tasks/pii_detection_test.py
git add src/senselab/text/tasks/pii_detection/api.py src/senselab/audio/workflows/audio_analysis/pii.py src/tests/text/tasks/pii_detection_test.py
git commit -m "feat(pii): a finding is a ScriptLine

A detection could say what it was but not where, so neither a speaker-scoped rule nor
redaction could act on one. PiiSpan now subclasses ScriptLine, which is what a timed piece
of text already is here, so text, start, end, speaker, score and timestamp_model are
inherited rather than reinvented. timestamp_model is the load-bearing inheritance: two
recognizers timed by the same aligner agree for reasons unrelated to the audio, and a
finding has to record which producer timed it.

Subclassed rather than composed: a .line field with start_s and end_s helpers would still
be parallel names for start and end. One incompatibility, handled -- audio_analysis called
dataclasses.asdict on a span, which now uses model_dump."
```

---

### Task 2: The provenance store

**Files:**
- Create: `src/senselab/utils/prov_store.py`
- Test: `src/tests/utils/prov_store_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Entity`, `Activity`, `Agent`, `ProvStore` with `.entity()`, `.activity()`, `.agent()`, the six relation methods, `.write_jsonl()`, `.read_jsonl()`, `.merge()`, `.fingerprint()`. Tasks 3–10 and every node consume these.

**W3C PROV, modelled directly — no library.** No `prov` or `rdflib` dependency is added. The three node
types and six relations are the PROV data model's own; the JSONL is PROV-JSON-shaped so it can be exported
later without re-modelling.

**Do not add a fourth 40-hex validator.** `audio_hints.py:47` `SpeakerEmbeddingProvenance` and
`signal.py:56` `SignalProvenance` already validate resolved commits. `Agent` follows their shape:
`commit_sha` for the resolved value, `unresolved_reason` for the case where resolution failed. A field
named `revision` would mean the *ref* in this codebase, which is the opposite of what is wanted.

Top-level `utils/`, not `utils/data_structures/`, to avoid that package's `__init__` fan-in.

- [ ] **Step 1: Write the failing tests**

```python
"""The provenance store: PROV entities, activities, agents; append-only and order-independent."""

from __future__ import annotations

import pytest

from senselab.utils.prov_store import Activity, Agent, Entity, ProvStore


def _store() -> ProvStore:
    return ProvStore(run_id="run-1")


class TestPurpose:
    """A store records what was produced, by what, using what."""

    def test_an_entity_records_the_activity_that_generated_it(self) -> None:
        """wasGeneratedBy replaces an author field."""
        s = _store()
        act = s.activity(node="PREPROCESS", step="spans", parameters={"k_db": 18.0})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={"peak_over_floor_db": 31.4})
        s.was_generated_by(ent, act)
        assert s.generated_by(ent) == act
        assert s.get_entity(ent).extent == (1.0, 2.0)

    def test_used_records_what_a_node_read(self) -> None:
        """The relation that makes dependency order inspectable rather than inferred."""
        s = _store()
        upstream = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        s.used(act, upstream)
        assert s.uses_of(act) == [upstream]

    def test_an_assertion_is_an_entity_derived_from_what_it_is_about(self) -> None:
        """label/confirm/contest are entities, so a confirm can name the assertion it answers."""
        s = _store()
        span = s.entity(prov_type="span", extent=(7.9, 8.5), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        label = s.entity(prov_type="assertion", extent=None, attributes={"verb": "label", "value": "Cough"})
        s.was_generated_by(label, act)
        s.was_derived_from(label, span)
        confirm = s.entity(prov_type="assertion", extent=None, attributes={"verb": "confirm"})
        s.was_derived_from(confirm, label)
        assert s.derived_from(confirm) == [label]
        assert s.derived_from(label) == [span]


class TestAgents:
    """An agent may be a model, and its commit may be unknown."""

    def test_a_resolved_commit_is_accepted(self) -> None:
        """A resolved commit is accepted."""
        s = _store()
        sha = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
        a = s.agent(agent_type="model", model_id="google/hear", commit_sha=sha)
        assert s.get_agent(a).commit_sha == sha

    def test_a_ref_masquerading_as_a_commit_is_refused(self) -> None:
        """A ref masquerading as a commit is refused."""
        s = _store()
        with pytest.raises(ValueError, match="40-hex"):
            s.agent(agent_type="model", model_id="google/hear", commit_sha="main")

    def test_an_unresolved_commit_is_representable_rather_than_fatal(self) -> None:
        """A Hub outage must degrade, not block every write."""
        s = _store()
        a = s.agent(agent_type="model", model_id="google/hear", unresolved_reason="hub 503")
        got = s.get_agent(a)
        assert got.commit_sha is None and got.unresolved_reason == "hub 503"

    def test_a_model_agent_needs_one_of_the_two(self) -> None:
        """A model agent needs one of the two."""
        s = _store()
        with pytest.raises(ValueError, match="commit_sha or unresolved_reason"):
            s.agent(agent_type="model", model_id="google/hear")

    def test_an_activity_records_which_agent_ran_it(self) -> None:
        """An activity records which agent ran it."""
        s = _store()
        a = s.agent(agent_type="software", version="0.1.0")
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        s.was_associated_with(act, a)
        assert s.associated_with(act) == [a]


class TestInvalidation:
    """Withdrawal keeps the entity."""

    def test_an_invalidated_entity_is_still_readable(self) -> None:
        """An invalidated entity is still readable."""
        s = _store()
        seg = s.entity(prov_type="speaker", extent=(7.9, 9.0), attributes={"speaker": "SPEAKER_00"})
        act = s.activity(node="SPEECH", step="withdraw", parameters={"reason": "airway span"})
        s.was_invalidated_by(seg, act)
        assert s.is_invalidated(seg)
        assert s.get_entity(seg).attributes["speaker"] == "SPEAKER_00"

    def test_nothing_can_be_deleted(self) -> None:
        """Nothing can be deleted."""
        s = _store()
        assert not hasattr(s, "delete_entity")
        assert not hasattr(s, "remove_relation")


class TestOrderIndependence:
    """Append-only makes a merge a set union."""

    def test_merging_in_either_order_gives_the_same_fingerprint(self) -> None:
        """Merging in either order gives the same fingerprint."""
        a, b = _store(), _store()
        a.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        b.entity(prov_type="word", extent=(1.1, 1.4), attributes={"word": "hello"})
        assert ProvStore.merge([a, b]).fingerprint() == ProvStore.merge([b, a]).fingerprint()


class TestRoundTrip:
    """PROV-JSON-shaped JSONL survives a round trip."""

    def test_entities_activities_agents_and_relations_all_return(self, tmp_path: Path) -> None:
        """Entities activities agents and relations all return."""
        s = _store()
        ag = s.agent(agent_type="model", model_id="google/hear", commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f")
        act = s.activity(node="AIRWAY", step="classify", parameters={"labels": ["Cough"]})
        ent = s.entity(prov_type="span", extent=(7.9, 8.5), attributes={})
        s.was_generated_by(ent, act)
        s.was_associated_with(act, ag)
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back.fingerprint() == s.fingerprint()
        assert back.associated_with(act) == [ag]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/tests/utils/prov_store_test.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'senselab.utils.prov_store'`

- [ ] **Step 3: Implement**

```python
"""An append-only provenance store, modelled on W3C PROV.

Entities are what the graph believes exists, activities are node executions, agents are what acted.
Relations are PROV's own. Nothing is modified after it is added, so merging two stores is a set union
and is order-independent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

PROV_TYPE = Literal[
    "span", "word", "speaker", "interval", "measurement", "kind", "stream", "pii", "verdict", "assertion"
]
AGENT_TYPE = Literal["model", "software"]
RELATION = Literal[
    "wasGeneratedBy", "used", "wasAssociatedWith", "wasAttributedTo", "wasDerivedFrom", "wasInvalidatedBy"
]

_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Entity:
    """Something the graph believes exists."""

    id: str
    prov_type: PROV_TYPE
    extent: tuple[float, float] | None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Activity:
    """One node execution, or one step of one."""

    id: str
    node: str
    step: str | None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Agent:
    """What acted: a model at a resolved commit, or the software itself.

    Attributes:
        id: The agent's id.
        agent_type: ``"model"`` or ``"software"``.
        model_id: The model's identifier, for a model agent.
        commit_sha: A resolved 40-hex commit, when resolution succeeded.
        unresolved_reason: Why the commit is unknown, when it is. A provenance model that cannot say
            "I could not resolve this" forces either a lie or a crash.
        version: Software version, for a software agent.
    """

    id: str
    agent_type: AGENT_TYPE
    model_id: str | None = None
    commit_sha: str | None = None
    unresolved_reason: str | None = None
    version: str | None = None


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ProvStore:
    """An append-only PROV document.

    Args:
        run_id: Mixed into every id so two runs never collide.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty store."""
        self.run_id = run_id
        self._entities: dict[str, Entity] = {}
        self._activities: dict[str, Activity] = {}
        self._agents: dict[str, Agent] = {}
        self._relations: list[tuple[RELATION, str, str]] = []

    def entity(self, *, prov_type: PROV_TYPE, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """Add an entity.

        Args:
            prov_type: What sort of thing it is.
            extent: ``(start, end)`` in seconds, or None for something without one.
            attributes: Whatever describes it.

        Returns:
            Its id.
        """
        eid = f"{prov_type}-{_digest([self.run_id, prov_type, extent, attributes])}"
        self._entities[eid] = Entity(id=eid, prov_type=prov_type, extent=extent, attributes=dict(attributes))
        return eid

    def activity(self, *, node: str, step: str | None, parameters: dict[str, Any]) -> str:
        """Add an activity.

        Args:
            node: The node executing.
            step: Which step of it, when a node has several.
            parameters: The values it ran with.

        Returns:
            Its id.
        """
        aid = f"act-{_digest([self.run_id, node, step, parameters])}"
        self._activities[aid] = Activity(id=aid, node=node, step=step, parameters=dict(parameters))
        return aid

    def agent(
        self,
        *,
        agent_type: AGENT_TYPE,
        model_id: str | None = None,
        commit_sha: str | None = None,
        unresolved_reason: str | None = None,
        version: str | None = None,
    ) -> str:
        """Add an agent.

        Args:
            agent_type: ``"model"`` or ``"software"``.
            model_id: Required for a model agent.
            commit_sha: A resolved 40-hex commit.
            unresolved_reason: Why the commit is unknown, if it is.
            version: Software version, for a software agent.

        Returns:
            Its id.

        Raises:
            ValueError: If ``commit_sha`` is not 40 hex characters, or a model agent supplies neither a
                commit nor a reason it is missing.
        """
        if commit_sha is not None and not _SHA.match(commit_sha):
            raise ValueError(
                f"commit_sha must be a resolved 40-hex commit, got {commit_sha!r}. A ref recorded as a "
                "commit makes the provenance confidently wrong."
            )
        if agent_type == "model" and commit_sha is None and unresolved_reason is None:
            raise ValueError("a model agent needs commit_sha or unresolved_reason; silence is not a third option")
        gid = f"agent-{_digest([self.run_id, agent_type, model_id, commit_sha, unresolved_reason, version])}"
        self._agents[gid] = Agent(
            id=gid,
            agent_type=agent_type,
            model_id=model_id,
            commit_sha=commit_sha,
            unresolved_reason=unresolved_reason,
            version=version,
        )
        return gid

    def _relate(self, relation: RELATION, source: str, target: str) -> None:
        self._relations.append((relation, source, target))

    def was_generated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an activity produced an entity."""
        self._relate("wasGeneratedBy", entity_id, activity_id)

    def used(self, activity_id: str, entity_id: str) -> None:
        """Record that an activity read an entity."""
        self._relate("used", activity_id, entity_id)

    def was_associated_with(self, activity_id: str, agent_id: str) -> None:
        """Record which agent ran an activity."""
        self._relate("wasAssociatedWith", activity_id, agent_id)

    def was_attributed_to(self, entity_id: str, agent_id: str) -> None:
        """Record which agent is answerable for an entity."""
        self._relate("wasAttributedTo", entity_id, agent_id)

    def was_derived_from(self, entity_id: str, source_entity_id: str) -> None:
        """Record that an entity refines or answers another, keeping both."""
        self._relate("wasDerivedFrom", entity_id, source_entity_id)

    def was_invalidated_by(self, entity_id: str, activity_id: str) -> None:
        """Record that an entity should no longer be read as what it was. It is not removed."""
        self._relate("wasInvalidatedBy", entity_id, activity_id)

    def get_entity(self, entity_id: str) -> Entity:
        """Return one entity."""
        return self._entities[entity_id]

    def get_agent(self, agent_id: str) -> Agent:
        """Return one agent."""
        return self._agents[agent_id]

    def entities(self, prov_type: PROV_TYPE | None = None) -> list[Entity]:
        """Return entities, optionally of one type."""
        return [e for e in self._entities.values() if prov_type is None or e.prov_type == prov_type]

    def _targets(self, relation: RELATION, source: str) -> list[str]:
        return [t for r, s, t in self._relations if r == relation and s == source]

    def generated_by(self, entity_id: str) -> str | None:
        """Return the activity that generated an entity, or None."""
        found = self._targets("wasGeneratedBy", entity_id)
        return found[0] if found else None

    def uses_of(self, activity_id: str) -> list[str]:
        """Return the entities an activity read."""
        return self._targets("used", activity_id)

    def associated_with(self, activity_id: str) -> list[str]:
        """Return the agents associated with an activity."""
        return self._targets("wasAssociatedWith", activity_id)

    def derived_from(self, entity_id: str) -> list[str]:
        """Return the entities an entity was derived from."""
        return self._targets("wasDerivedFrom", entity_id)

    def is_invalidated(self, entity_id: str) -> bool:
        """Whether an entity has been invalidated."""
        return bool(self._targets("wasInvalidatedBy", entity_id))

    def write_jsonl(self, path: str | Path) -> None:
        """Write the store as one PROV-JSON-shaped record per line."""
        lines = [json.dumps({"record": "entity", **asdict(e)}, sort_keys=True, default=str) for e in self._entities.values()]
        lines += [json.dumps({"record": "activity", **asdict(a)}, sort_keys=True, default=str) for a in self._activities.values()]
        lines += [json.dumps({"record": "agent", **asdict(g)}, sort_keys=True, default=str) for g in self._agents.values()]
        lines += [json.dumps({"record": "relation", "relation": r, "source": s, "target": t}, sort_keys=True) for r, s, t in self._relations]
        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, run_id: str = "read") -> "ProvStore":
        """Read a store back."""
        store = cls(run_id=run_id)
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.pop("record")
            if kind == "entity":
                extent = rec.pop("extent")
                store._entities[rec["id"]] = Entity(extent=tuple(extent) if extent else None, **rec)
            elif kind == "activity":
                store._activities[rec["id"]] = Activity(**rec)
            elif kind == "agent":
                store._agents[rec["id"]] = Agent(**rec)
            else:
                store._relations.append((rec["relation"], rec["source"], rec["target"]))
        return store

    @classmethod
    def merge(cls, stores: Sequence["ProvStore"]) -> "ProvStore":
        """Union several stores. Append-only makes this order-independent."""
        out = cls(run_id="merged")
        for s in stores:
            out._entities.update(s._entities)
            out._activities.update(s._activities)
            out._agents.update(s._agents)
        seen: set[tuple[RELATION, str, str]] = set()
        for s in stores:
            for rel in s._relations:
                if rel not in seen:
                    seen.add(rel)
                    out._relations.append(rel)
        return out

    def fingerprint(self) -> str:
        """A content hash that ignores insertion order."""
        return _digest(
            {
                "e": sorted(self._entities),
                "act": sorted(self._activities),
                "ag": sorted(self._agents),
                "r": sorted(f"{r}:{s}:{t}" for r, s, t in self._relations),
            }
        )
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/utils/prov_store_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/utils/prov_store.py src/tests/utils/prov_store_test.py
git add src/senselab/utils/prov_store.py src/tests/utils/prov_store_test.py
git commit -m "feat(prov): an append-only provenance store on the W3C PROV model

Entities, activities and agents with PROV's own relations, modelled directly rather than
via a library -- no prov or rdflib dependency, and the JSONL is PROV-JSON-shaped so it can
be exported later without re-modelling.

PROV supplies what an ad-hoc vocabulary was missing. used() records what a node read, so
dependency order is queryable rather than inferred. wasDerivedFrom keeps the source, which
is what refine needed. wasInvalidatedBy marks an entity unusable without deleting it, which
is what withdraw needed.

An agent carries commit_sha or unresolved_reason, following SpeakerEmbeddingProvenance
rather than adding a fourth 40-hex validator, and deliberately not naming the field
revision -- which means the ref in this codebase. A model agent must supply one of the two:
a provenance model that cannot say 'I could not resolve this' forces a lie or a crash."
```

---

### Task 3: The triage configuration

**Files:**
- Create: `src/senselab/audio/workflows/triage/data/config/default.yaml`
- Create: `src/senselab/audio/workflows/triage/config.py`
- Test: `src/tests/audio/workflows/triage/config_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `TriageConfig` with `.require(path) -> float | int | str` and `.get(path, default=None)`, plus `load_triage_config(override=None) -> TriageConfig` carrying `.name`, `.version`, `.config_hash`. **Every later task reads its numbers from this.**

Follows the pattern of `audio_analysis/data/run_config/default.yaml` and `run_config.py`: one versioned
file, `derivation` as a config *value* so editing it changes the hash, whole-file overrides deep-merged,
and `{name, version, config_hash}` stamped into artifacts.

- [ ] **Step 1: Write the failing tests**

```python
"""The triage configuration: every number, its derivation, and what happens when one is unset."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config


class TestMeasuredValues:
    """A value with a derivation is readable."""

    def test_the_measured_values_are_present(self) -> None:
        """The measured values are present."""
        cfg = load_triage_config()
        assert cfg.require("envelope.lowpass_hz") == 40.0
        assert cfg.require("spans.onset_drop_db") == 15.0
        assert cfg.require("spans.offset_fraction") == 0.7
        assert cfg.require("spans.k_db.airway") == 18.0
        assert cfg.require("preemphasis.coefficient") == 0.97
        assert cfg.require("floor.eval_grid_s") == 0.1
        assert cfg.require("phonation.periods_per_window") == 4.5

    def test_identity_travels_with_the_config(self) -> None:
        """Identity travels with the config."""
        cfg = load_triage_config()
        assert cfg.name == "senselab-triage/default"
        assert isinstance(cfg.version, int)
        assert len(cfg.config_hash) == 16


class TestUnsetValues:
    """A number nobody measured must be impossible to use by accident."""

    def test_reading_an_unset_value_raises_and_names_it(self) -> None:
        """Reading an unset value raises and names it."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="phonation.hnr_floor_db"):
            cfg.require("phonation.hnr_floor_db")

    def test_the_error_points_at_what_would_settle_it(self) -> None:
        """The error points at what would settle it."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="benchmarks/open.md"):
            cfg.require("redaction.padding_ms")

    def test_every_unset_value_is_null_rather_than_absent(self) -> None:
        """Absent is a typo; null is a decision not yet taken."""
        cfg = load_triage_config()
        for path in ("phonation.hnr_floor_db", "phonation.rms_floor", "redaction.padding_ms",
                     "speech.word_gap_ms", "quality.stoi_floor", "taxonomy.min_families.airway"):
            assert cfg.get(path, "MISSING") is None, f"{path} must be present and null"

    def test_get_returns_a_default_instead_of_raising(self) -> None:
        """Get returns a default instead of raising."""
        cfg = load_triage_config()
        assert cfg.get("phonation.hnr_floor_db", 8.0) == 8.0


class TestOverrides:
    """Whole-file overrides, and the hash follows the merged mapping."""

    def test_an_override_supplies_an_unset_value(self, tmp_path: Path) -> None:
        """An override supplies an unset value."""
        override = tmp_path / "o.yaml"
        override.write_text("redaction:\n  padding_ms: 250\n")
        cfg = load_triage_config(override)
        assert cfg.require("redaction.padding_ms") == 250

    def test_an_override_changes_the_hash(self, tmp_path: Path) -> None:
        """An override changes the hash."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  onset_drop_db: 12.0\n")
        assert load_triage_config(override).config_hash != load_triage_config().config_hash

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        """An unknown key is refused rather than ignored."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  onset_drpo_db: 12.0\n")
        with pytest.raises(ValueError, match="onset_drpo_db"):
            load_triage_config(override)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/tests/audio/workflows/triage/config_test.py -v`
Expected: FAIL at collection — no module `senselab.audio.workflows.triage.config`.

- [ ] **Step 3: Write `data/config/default.yaml`**

```yaml
# The triage workflow configuration. One file, versioned, with its derivation written down.
#
# No number appears in the code. A value here is either measured -- with the measurement named -- or
# `null`, meaning nobody has measured it and reading it raises. There is no third state: a default
# chosen to make a signature tidy is an unmeasured decision with a public interface.
#
# `derivation` below is a config *value*, not a comment: the loader hashes the merged mapping, so
# editing a word of it changes `config_hash` and two behaviourally identical runs report different
# identities. Corrections to its prose belong in `#` comments like this one.
version: 1
name: senselab-triage/default

derivation: |
  Envelope lowpass 40 Hz, zero-phase -- benchmarks/preprocess-params.md. Sweeping the cutoff against
  six labelled events, a wider band makes onsets worse (median 144 ms at 320 Hz against 63 ms at
  40 Hz) because it tracks pre-event fluctuation a fixed threshold then fires on. 40 Hz is the
  modulation bandwidth the envelope is for, not an onset-precision choice. Zero-phase beats causal,
  63.5 ms against 90.1 ms median, which makes the envelope offline-only.

  Floor: rolling 3 s window, 10th percentile, in dBFS -- benchmarks/spans.md. Local and absolute
  because a global anchor fails two ways: a floor-anchored gate loses the quieter event as noise
  raises the floor, and a peak-anchored one moves 49.1 dB within one recording and is destroyed by a
  single 30 ms click. Normalising the envelope by its own maximum is the fault both share.

  Span onset drop 15 dB, peak-anchored -- benchmarks/spans.md. 5 of 6 labelled onsets inside their
  declared windows, against 2 of 6 for a floor-referenced threshold on the same envelope.

  Span offset 0.7 of each event's own range, 120 ms hangover -- benchmarks/spans.md. Median offset
  error 84.3 ms against 573.9 ms for a fixed peak-10 dB. A fixed drop cannot serve both a 20 dB
  mouth sound and a 57 dB cough. The hangover must be shorter than the shortest event to be bounded:
  at 250 ms it overshoots a 202 ms click by 418 ms.

  Span propose K, per reader -- benchmarks/spans.md and snr.md. 18 dB for airway, whose events stand
  53-57 dB above the floor. 12 dB detects a speech span to +10 dB SNR and still survives an injected
  click; 8 dB collapses the clean-file span set from six to two by merging, so lower is not freely
  better. SPEECH no longer reads these spans -- it derives its own from word timings -- so only the
  airway value is in use.

  Pre-emphasis 0.97 -- conventional in speech analysis, not fitted here. It raises event-to-floor
  contrast on every labelled event and most on the two hardest, cough 1 by +10.95 dB and the mouth
  sound by +7.36 dB, both of which carry 14-16% of their energy in 4-8 kHz.

  Spectrograms 5 ms and 20 ms window, 5 ms hop -- benchmarks/preprocess-params.md. At F0 88.1 Hz the
  glottal period is 11.4 ms, so 10 ms resolves neither harmonics (150 Hz against 88 Hz spacing) nor
  pulses (0.88 of a period). Two windows rather than a compromise between them.

  Gammatone 40 ERB channels, 80-7800 Hz, 5 ms hop -- conventional auditory-filterbank settings.

  Praat harmonicity settings hop_s 0.01, silence_threshold 0.1, periods_per_window 4.5 --
  Praat's own documented defaults for the cc method, which extract_harmonicity_descriptors
  (praat_parselmouth.py:569) already uses. Conventional, not fitted here.

  Floor evaluation grid 0.1 s -- a granularity rather than a threshold: the percentile is evaluated
  every 100 ms and interpolated between, which is finer than the 3 s window it summarises.

  Disruption parameters -- conventional. A single sample at full scale is not clipping, which is what
  min_clip_run is for. The counts these produce are exact; what has no measured value is the
  tolerance, which is quality.disruption_* below.

  UNSET, and why -- benchmarks/open.md carries each of these:
    phonation.hnr_floor_db, phonation.rms_floor: the gate's interval was measured as normalised
      autocorrelation, (0.44, 0.933) and (0.0007, 0.0161) on one recording, and the implementation now
      uses Praat harmonicity in dB. The units differ, so the interval does not transfer and neither
      floor has a value. CLAUDE.md records a related trap: a 2-10 dB HNR ramp under which ordinary
      voiced speech, median 8.12 dB, read as only partly voiced.
    redaction.padding_ms: must exceed the *worst* alignment edge error, which is unquantified. The
      median will not do -- of the two boundary failures, an audible fragment of a name and a clipped
      neighbour, only one is recoverable.
    speech.word_gap_ms: any value is a claim about what makes one utterance.
    taxonomy.min_families.*: the asymmetry is known -- airway has three eligible families, speech two
      -- but neither count has a derived value.
    quality.stoi_floor, quality.pesq_floor, quality.disruption_*: no labelled quality verdicts exist,
      so SPEECH's quality fail is unreachable by design until they do.

resample:
  target_hz: 16000

preemphasis:
  enabled: true
  coefficient: 0.97

envelope:
  lowpass_hz: 40.0
  filter_order: 4
  zero_phase: true

floor:
  window_s: 3.0
  percentile: 10.0
  eval_grid_s: 0.1

spans:
  onset_drop_db: 15.0
  offset_fraction: 0.7
  hangover_ms: 120
  min_duration_ms: 50
  min_separation_ms: 150
  k_db:
    airway: 18.0

spectrogram:
  wideband_window_ms: 5.0
  narrowband_window_ms: 20.0
  hop_ms: 5.0

gammatone:
  n_channels: 40
  low_hz: 80.0
  high_hz: 7800.0
  hop_ms: 5.0

hear:
  window_s: 2.0
  label_floor: 0.5

yamnet:
  silence_threshold: 0.5
  coverage_threshold: 0.5

phonation:
  f0_min_hz: null
  f0_max_hz: null
  hnr_floor_db: null
  rms_floor: null
  hop_s: 0.01
  silence_threshold: 0.1
  periods_per_window: 4.5

speech:
  word_gap_ms: null

taxonomy:
  min_families:
    airway: null
    speech: null

quality:
  stoi_floor: null
  pesq_floor: null
  disruption_clipped_s_max: null
  disruption_dropout_s_max: null

disruptions:
  clip_headroom: 0.999
  min_clip_run: 3
  min_dropout_ms: 10.0
  discontinuity_threshold: 0.5

redaction:
  padding_ms: null
```

- [ ] **Step 4: Write `config.py`**

```python
"""Loading the triage configuration.

Every number the triage workflow uses lives in ``data/config/default.yaml`` beside the measurement that
produced it. A value nobody has measured is ``null`` there, and reading it raises rather than returning a
number nobody chose.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DEFAULT = Path(__file__).parent / "data" / "config" / "default.yaml"
_OPEN_QUESTIONS = "specs/20260817-triage-workflow-dag/benchmarks/open.md"


@dataclass(frozen=True)
class TriageConfig:
    """One resolved configuration.

    Attributes:
        name: The configuration's name.
        version: Schema version of the file.
        config_hash: Hash of the merged mapping, so a run's configuration can be named.
        values: The merged mapping.
    """

    name: str
    version: int
    config_hash: str
    values: dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        """Read a value, returning ``default`` when it is absent or null.

        Args:
            path: Dotted path, e.g. ``"spans.onset_drop_db"``.
            default: Returned when the value is missing or null.

        Returns:
            The value, or ``default``.
        """
        node: Any = self.values
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def require(self, path: str) -> Any:
        """Read a value that must have been measured.

        Args:
            path: Dotted path.

        Returns:
            The value.

        Raises:
            ValueError: If the value is absent, or is null because nobody has measured it.
        """
        found = self.get(path, None)
        if found is None:
            raise ValueError(
                f"{path} has no value in {self.name}. It is null because nobody has measured it — see "
                f"{_OPEN_QUESTIONS} for what would settle it. Supply it with a config override rather "
                "than defaulting it here."
            )
        return found


def _merge(base: dict[str, Any], over: dict[str, Any], trail: str = "") -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in over.items():
        where = f"{trail}.{key}" if trail else key
        if key not in out:
            raise ValueError(f"unknown configuration key {where!r}; overrides may not introduce keys")
        if isinstance(value, dict) and isinstance(out[key], dict):
            out[key] = _merge(out[key], value, where)
        else:
            out[key] = value
    return out


def load_triage_config(override: str | Path | None = None) -> TriageConfig:
    """Load the packaged configuration, deep-merging one override over it.

    Args:
        override: Path to a partial YAML. Its keys must already exist in the packaged file — a typo
            is refused rather than silently ignored.

    Returns:
        The resolved configuration, carrying the hash of the merged mapping.

    Raises:
        ValueError: If the override introduces a key the packaged file does not have.
    """
    values = yaml.safe_load(_DEFAULT.read_text())
    if override is not None:
        values = _merge(values, yaml.safe_load(Path(override).read_text()) or {})
    digest = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return TriageConfig(name=values["name"], version=int(values["version"]), config_hash=digest, values=values)
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest src/tests/audio/workflows/triage/config_test.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/config_test.py
git add src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/config_test.py
git commit -m "feat(triage): one versioned config, with every number's derivation beside it

No number appears in the triage code -- not as a signature default, not as a module
constant. CLAUDE.md already required this and the audio_analysis run_config already
demonstrates the shape: derivation as a config value so editing it changes the hash, and
whole-file overrides that refuse an unknown key rather than ignoring it.

A value nobody has measured is null, and require() raises naming the parameter and pointing
at benchmarks/open.md. That replaces keyword-only-without-default as the mechanism: an
unmeasured number is now impossible to use by accident rather than merely conspicuous.
Eleven are null today, including both phonation floors -- whose measured interval was in
autocorrelation units the Praat implementation does not use."
```

---

### Task 4: The envelope task

**Files:**
- Create: `src/senselab/audio/tasks/envelope/__init__.py`, `src/senselab/audio/tasks/envelope/api.py`
- Test: `src/tests/audio/tasks/envelope_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `hilbert_envelope_dbfs(audio, *, lowpass_hz, filter_order) -> np.ndarray`, `rolling_floor_dbfs(envelope_db, sampling_rate, *, window_s, percentile) -> np.ndarray`. Task 5 consumes both.

Every parameter is required: the values live in `envelope.*` and `floor.*` of Task 3's config with
their derivation, and no number appears in this module. Derivation: `benchmarks/preprocess-params.md`
(zero-phase, 40 Hz) and `benchmarks/spans.md` (why local and absolute).

- [ ] **Step 1: Write the failing tests**

```python
"""The Hilbert envelope in dBFS and its rolling local floor."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.envelope import hilbert_envelope_dbfs, rolling_floor_dbfs

SR = 16000


def _tone(seconds: float, amp: float, freq: float = 200.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    return Audio(waveform=(amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :], sampling_rate=SR)


def _env(audio: Audio) -> np.ndarray:
    """Run the envelope with the config's measured values; the fixture supplies the literals."""
    return hilbert_envelope_dbfs(audio, lowpass_hz=40.0, filter_order=4)


class TestEnvelopeIsAbsolute:
    """The envelope is dBFS, never normalised by the input's own maximum."""

    def test_a_half_scale_tone_sits_near_minus_six_dbfs(self) -> None:
        """A 0.5-amplitude tone reads close to -6 dBFS, pinning the absolute reference."""
        env = _env(_tone(1.0, 0.5))
        mid = env[SR // 4 : -SR // 4]
        assert -7.5 < float(np.median(mid)) < -4.5

    def test_scaling_the_input_shifts_the_envelope_by_the_same_amount(self) -> None:
        """A 20 dB input change moves the envelope 20 dB — absolute, not max-normalised."""
        loud = float(np.median(_env(_tone(1.0, 0.5))[SR // 4 : -SR // 4]))
        quiet = float(np.median(_env(_tone(1.0, 0.05))[SR // 4 : -SR // 4]))
        assert loud - quiet == pytest.approx(20.0, abs=1.0), "dBFS is absolute, not max-normalised"

    def test_a_loud_click_elsewhere_does_not_move_the_rest(self) -> None:
        """One 30 ms click cannot rescale the envelope far from it."""
        quiet = _tone(2.0, 0.05)
        clicked = quiet.waveform.numpy().copy() if hasattr(quiet.waveform, "numpy") else np.array(quiet.waveform)
        clicked[0, SR : SR + 480] += 0.95
        a = _env(quiet)
        b = _env(Audio(waveform=clicked.astype(np.float32), sampling_rate=SR))
        early = slice(SR // 8, SR // 2)
        assert float(np.median(b[early])) == pytest.approx(float(np.median(a[early])), abs=0.5)


class TestRollingFloor:
    """The floor tracks the recording rather than summarising it."""

    def test_the_floor_tracks_a_level_change_rather_than_averaging_it(self) -> None:
        """A -60 to -30 dB step moves the floor to each level, not to their mean."""
        env = np.concatenate([np.full(5 * SR, -60.0), np.full(5 * SR, -30.0)])
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0)
        assert fl[SR] == pytest.approx(-60.0, abs=1.0)
        assert fl[9 * SR] == pytest.approx(-30.0, abs=1.0)

    def test_the_floor_is_one_value_per_sample(self) -> None:
        """The floor track has the envelope's own shape."""
        env = np.full(3 * SR, -50.0)
        assert rolling_floor_dbfs(env, SR, window_s=3.0, percentile=10.0).shape == env.shape
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/envelope_test.py -v`
Expected: FAIL at collection — no module `senselab.audio.tasks.envelope`.

- [ ] **Step 3: Implement `api.py`**

```python
"""The broadband amplitude envelope, in dBFS, and a floor that tracks the recording."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from senselab.audio.data_structures import Audio


def hilbert_envelope_dbfs(audio: Audio, *, lowpass_hz: float, filter_order: int) -> np.ndarray:
    """The analytic-signal magnitude, lowpassed, in dBFS.

    The filter is zero-phase, so the envelope is offline-only.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        lowpass_hz: Cutoff of the zero-phase Butterworth lowpass. Read it from
            ``envelope.lowpass_hz`` in the triage config.
        filter_order: Order of the Butterworth design. Read it from ``envelope.filter_order``.

    Returns:
        One dBFS value per input sample. Absolute, never normalised by the input's maximum.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    b, a = butter(filter_order, lowpass_hz / (audio.sampling_rate / 2), "low")
    env = np.maximum(filtfilt(b, a, np.abs(hilbert(x))), 1e-12)
    return 20.0 * np.log10(env)


def rolling_floor_dbfs(
    envelope_db: np.ndarray,
    sampling_rate: int,
    *,
    window_s: float,
    percentile: float,
    eval_grid_s: float,
) -> np.ndarray:
    """A low percentile of the envelope over a sliding window.

    Args:
        envelope_db: Output of :func:`hilbert_envelope_dbfs`.
        sampling_rate: Samples per second of ``envelope_db``.
        window_s: Width of the sliding window. Read it from ``floor.window_s``.
        percentile: Which percentile within the window is the floor. Config `floor.percentile`.
        eval_grid_s: How often the percentile is evaluated before interpolation. Config
            `floor.eval_grid_s` — a granularity, not a threshold, but it is still a number and so
            still belongs in the config. Read it from
            ``floor.percentile``.

    Returns:
        One floor value per sample of ``envelope_db``.
    """
    n = len(envelope_db)
    half = int(window_s * sampling_rate) // 2
    step = max(1, int(eval_grid_s * sampling_rate))
    centres = np.arange(0, n, step)
    vals = [float(np.percentile(envelope_db[max(0, c - half) : min(n, c + half)], percentile)) for c in centres]
    return np.interp(np.arange(n), centres, vals)
```

And `__init__.py`:

```python
"""Amplitude envelope and local floor."""

from senselab.audio.tasks.envelope.api import hilbert_envelope_dbfs, rolling_floor_dbfs

__all__ = [
    "hilbert_envelope_dbfs",
    "rolling_floor_dbfs",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/envelope_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/envelope/ src/tests/audio/tasks/envelope_test.py
git add src/senselab/audio/tasks/envelope/ src/tests/audio/tasks/envelope_test.py
git commit -m "feat(envelope): Hilbert envelope in dBFS with a rolling local floor

Absolute rather than max-normalised, so one loud sample cannot rescale the analysis,
and a floor that tracks the recording rather than summarising it. Both properties are
tested directly: an injected click leaves the rest of the envelope within 0.5 dB."
```

---

### Task 5: The spans task

**Files:**
- Create: `src/senselab/audio/tasks/spans/__init__.py`, `src/senselab/audio/tasks/spans/api.py`
- Test: `src/tests/audio/tasks/spans_test.py`

**Interfaces:**
- Consumes: `hilbert_envelope_dbfs`, `rolling_floor_dbfs` from Task 4.
- Produces: `Span` (frozen dataclass: `start`, `end`, `peak_over_floor_db`) and
  `propose_spans(envelope_db, floor_db, sampling_rate, *, k_db, onset_drop_db, offset_fraction, hangover_ms, min_duration_ms, min_separation_ms) -> list[Span] | NoContrast`.
  `NoContrast` is a distinct return, not an empty list.

Every rule parameter is required, `min_separation_ms` included: the values live in `spans.*` of
Task 3's config with their derivation, and no number appears in this module. Derivation:
`benchmarks/spans.md`.

- [ ] **Step 1: Write the failing tests**

```python
"""Span proposal: the gate, the peak-anchored onset, the range-relative offset."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.tasks.spans import NoContrast, Span, propose_spans

SR = 16000


def _envelope(events: list[tuple[float, float, float]], seconds: float = 14.0, floor: float = -55.0) -> np.ndarray:
    env = np.full(int(seconds * SR), floor)
    for start, end, peak in events:
        env[int(start * SR) : int(end * SR)] = peak
    return env


def _propose(env: np.ndarray, floor: np.ndarray) -> list[Span] | NoContrast:
    """Run propose_spans with the config's measured values; the fixture supplies the literals."""
    return propose_spans(
        env, floor, SR, k_db=18.0, onset_drop_db=15.0, offset_fraction=0.7,
        hangover_ms=120, min_duration_ms=50, min_separation_ms=150,
    )


class TestGate:
    """Only a peak that clears k_db over the local floor proposes a span."""

    def test_a_peak_below_k_is_not_proposed(self) -> None:
        """Of one 10 dB peak and one 35 dB peak, only the one clearing 18 dB becomes a span."""
        env = _envelope([(2.0, 2.5, -45.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert not isinstance(out, NoContrast)
        assert len(out) == 1, "the 10 dB event must be gated out, not merely narrowed"
        (span,) = out
        assert span.start == pytest.approx(8.0, abs=0.05)
        assert span.end == pytest.approx(8.5, abs=0.05)

    def test_a_peak_above_k_is_proposed(self) -> None:
        """A 35 dB event over the floor clears the 18 dB gate."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert len(out) == 1 and isinstance(out[0], Span)

    def test_two_events_far_apart_stay_two_spans(self) -> None:
        """Distant events do not merge."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert len(out) == 2


class TestNoContrast:
    """An unmeasurable recording is a distinct value, not a quiet one."""

    def test_no_peak_anywhere_is_no_contrast_not_an_empty_list(self) -> None:
        """A flat envelope 1 dB over its floor yields NoContrast naming the gate."""
        env = np.full(int(3.0 * SR), -30.0)
        out = _propose(env, np.full_like(env, -29.0))
        assert isinstance(out, NoContrast)
        assert "18" in out.reason


class TestKIsRequired:
    """No rule parameter has a default."""

    def test_k_has_no_default(self) -> None:
        """Calling without the rule parameters is a TypeError, not a silent default."""
        env = _envelope([(2.0, 2.5, -20.0)])
        with pytest.raises(TypeError):
            propose_spans(env, np.full_like(env, -55.0), SR)  # type: ignore[call-arg]


class TestSpanCarriesItsContrast:
    """A span reports how far its peak stood above the floor."""

    def test_peak_over_floor_travels_with_the_span(self) -> None:
        """A -20 dB event over a -55 dB floor reports 35 dB of contrast."""
        env = _envelope([(2.0, 2.5, -20.0)])
        (span,) = _propose(env, np.full_like(env, -55.0))
        assert span.peak_over_floor_db == pytest.approx(35.0, abs=0.5)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/spans_test.py -v`
Expected: FAIL at collection — no module `senselab.audio.tasks.spans`.

- [ ] **Step 3: Implement `api.py`**

```python
"""Proposing spans from an envelope and its local floor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Span:
    """A proposed span, carrying no label.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        peak_over_floor_db: How far the span's peak stood above the local floor.
    """

    start: float
    end: float
    peak_over_floor_db: float


@dataclass(frozen=True)
class NoContrast:
    """No peak anywhere rose the required amount above the local floor.

    Distinct from an empty span list: an unmeasurable recording must not read as a quiet one.

    Attributes:
        reason: What was required and what was found.
    """

    reason: str


def propose_spans(
    envelope_db: np.ndarray,
    floor_db: np.ndarray,
    sampling_rate: int,
    *,
    k_db: float,
    onset_drop_db: float,
    offset_fraction: float,
    hangover_ms: int,
    min_duration_ms: int,
    min_separation_ms: int,
) -> list[Span] | NoContrast:
    """Propose spans from an envelope, anchoring the onset to each event's own peak.

    Args:
        envelope_db: Envelope in dBFS.
        floor_db: Local floor, same length as ``envelope_db``.
        sampling_rate: Samples per second.
        k_db: How far above the local floor a peak must rise to be proposed. Per reader: read it
            from ``spans.k_db.<reader>`` in the triage config.
        onset_drop_db: Walk back from the peak to ``peak - onset_drop_db``. Read it from
            ``spans.onset_drop_db``.
        offset_fraction: Walk forward to ``peak - offset_fraction * (peak - floor)``. Read it from
            ``spans.offset_fraction``.
        hangover_ms: The offset closes only after this long continuously below threshold. Must be shorter
            than the shortest event to be bounded. Read it from ``spans.hangover_ms``.
        min_duration_ms: Discard spans shorter than this. Read it from ``spans.min_duration_ms``.
        min_separation_ms: Minimum distance between two proposed peaks. Read it from
            ``spans.min_separation_ms``.

    Returns:
        Merged spans in time order, or :class:`NoContrast` when no peak clears ``k_db``.
    """
    above = envelope_db - floor_db
    peaks, _ = find_peaks(above, height=k_db, distance=int(min_separation_ms * sampling_rate / 1000))
    if len(peaks) == 0:
        return NoContrast(
            reason=f"no peak rose {k_db} dB above the local floor; the largest rose {float(above.max()):.1f} dB"
        )
    hang = int(hangover_ms * sampling_rate / 1000)
    found: list[Span] = []
    for p in peaks:
        peak = float(envelope_db[p])
        i = int(p)
        while i > 0 and envelope_db[i] > peak - onset_drop_db:
            i -= 1
        threshold = peak - offset_fraction * (peak - float(floor_db[p]))
        j = int(p)
        while j < len(envelope_db) - 1:
            window = envelope_db[j : j + hang]
            if len(window) and window.max() <= threshold:
                break
            j += 1
        if (j - i) >= min_duration_ms * sampling_rate / 1000:
            found.append(Span(start=i / sampling_rate, end=j / sampling_rate, peak_over_floor_db=peak - float(floor_db[p])))
    found.sort(key=lambda s: s.start)
    merged: list[Span] = []
    for span in found:
        if merged and span.start <= merged[-1].end:
            last = merged[-1]
            merged[-1] = Span(
                start=last.start,
                end=max(last.end, span.end),
                peak_over_floor_db=max(last.peak_over_floor_db, span.peak_over_floor_db),
            )
        else:
            merged.append(span)
    return merged
```

And `__init__.py`:

```python
"""Span proposal from an envelope."""

from senselab.audio.tasks.spans.api import NoContrast, Span, propose_spans

__all__ = [
    "NoContrast",
    "Span",
    "propose_spans",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/spans_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Reproduce the benchmark on the reference recording**

Run: `uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/localgate.py` — the span
producer, applying the same rules (`K` = 18 dB, onset `peak − 15 dB`, offset fraction 0.7, 120 ms
hangover, 50 ms minimum, 150 ms separation). `floor.py` beside it is a floor comparison, not the
span producer. Caveat before running: `s2b.py` reads its stage-1 envelope from job scratch on the
machine that produced the benchmark (`/Users/satra/.claude/jobs/295c3f8a/tmp/stage1.npz`); on any
other host it cannot run and this step is a recorded expectation, not a check.
Expected: **six** spans at `K = 18 dB` on the clean file on the clean reference recording — the `K` = 18 dB row of
`benchmarks/spans.md`'s "Propose threshold `K`" table. That file records the count, not the
extents, so the comparison is on the count. Note also that `s2b.py` gates against a *scalar*
floor where `propose_spans` takes the rolling floor track, so boundaries may differ by samples;
a different span *count* means the implementation diverges from the measured rules — fix the
implementation, not the benchmark.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/spans/ src/tests/audio/tasks/spans_test.py
git add src/senselab/audio/tasks/spans/ src/tests/audio/tasks/spans_test.py
git commit -m "feat(spans): peak-anchored onset, range-relative offset, no_contrast as a value

Every rule parameter is a required keyword argument with no default -- the values live in
the triage config beside their derivation, and k_db is additionally per-reader. NoContrast
is a distinct return rather than an empty list, so an unmeasurable recording cannot read
as a quiet one."
```

---

### Task 6: The gammatone task

**Files:**
- Create: `src/senselab/audio/tasks/gammatone/__init__.py`, `src/senselab/audio/tasks/gammatone/api.py`
- Test: `src/tests/audio/tasks/gammatone_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gammatone_filterbank(audio, *, n_channels, low_hz, high_hz, hop_s) -> tuple[np.ndarray, np.ndarray]` returning `(centre_frequencies, energy_db)` of shape `(n_channels,)` and `(n_channels, n_frames)`.

Every parameter is required: the values live in `gammatone.*` of Task 3's config, and no number
appears in this module.

- [ ] **Step 1: Write the failing tests**

```python
"""The gammatone filterbank."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.gammatone import erb_space, gammatone_filterbank

SR = 16000


def _tone(freq: float, seconds: float = 1.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    return Audio(waveform=(0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None, :], sampling_rate=SR)


class TestErbSpacing:
    """Centre frequencies sit on the ERB-rate scale."""

    def test_centres_span_the_requested_range_and_increase(self) -> None:
        """Forty centres run 80 to 7800 Hz, ascending."""
        cf = erb_space(80.0, 7800.0, 40)
        assert len(cf) == 40
        assert cf[0] == pytest.approx(80.0, abs=1.0)
        assert cf[-1] == pytest.approx(7800.0, abs=10.0)
        assert np.all(np.diff(cf) > 0)

    def test_spacing_is_wider_at_high_frequency(self) -> None:
        """ERB spacing widens with frequency, unlike a linear grid."""
        cf = erb_space(80.0, 7800.0, 40)
        assert (cf[-1] - cf[-2]) > (cf[1] - cf[0]) * 5


class TestFilterbank:
    """The bank resolves frequency into the right channel at the right shape."""

    def test_a_tone_excites_the_channel_nearest_its_frequency(self) -> None:
        """A 1 kHz tone lands in the channel centred nearest 1 kHz."""
        cf, energy = gammatone_filterbank(_tone(1000.0), n_channels=40, low_hz=80.0, high_hz=7800.0, hop_s=0.005)
        loudest = int(np.argmax(energy.mean(axis=1)))
        assert abs(cf[loudest] - 1000.0) < 250.0

    def test_shape_is_channels_by_frames(self) -> None:
        """The output is (n_channels, n_frames) with the centres alongside."""
        cf, energy = gammatone_filterbank(
            _tone(1000.0, seconds=2.0), n_channels=24, low_hz=80.0, high_hz=7800.0, hop_s=0.01
        )
        assert energy.shape[0] == 24 == len(cf)
        assert energy.shape[1] == pytest.approx(2.0 / 0.01, abs=2)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/gammatone_test.py -v`
Expected: FAIL at collection.

- [ ] **Step 3: Implement `api.py`**

```python
"""An ERB-spaced gammatone filterbank."""

from __future__ import annotations

import numpy as np
from scipy.signal import gammatone, hilbert, lfilter

from senselab.audio.data_structures import Audio


def erb_space(low_hz: float, high_hz: float, n_channels: int) -> np.ndarray:
    """Centre frequencies equally spaced on the ERB-rate scale.

    Args:
        low_hz: Lowest centre frequency.
        high_hz: Highest centre frequency.
        n_channels: How many channels.

    Returns:
        Centre frequencies in Hz, ascending.
    """
    to_erb = lambda f: 21.4 * np.log10(4.37e-3 * f + 1.0)  # noqa: E731
    from_erb = lambda e: (10.0 ** (e / 21.4) - 1.0) / 4.37e-3  # noqa: E731
    return from_erb(np.linspace(to_erb(low_hz), to_erb(high_hz), n_channels))


def gammatone_filterbank(
    audio: Audio,
    *,
    n_channels: int,
    low_hz: float,
    high_hz: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Energy per auditory channel over time.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        n_channels: Number of ERB-spaced channels. Read it from ``gammatone.n_channels``.
        low_hz: Lowest centre frequency. Read it from ``gammatone.low_hz``.
        high_hz: Highest centre frequency. Read it from ``gammatone.high_hz``.
        hop_s: Frame hop for the energy summary. Read it from ``gammatone.hop_ms``.

    Returns:
        ``(centre_frequencies, energy_db)`` with shapes ``(n_channels,)`` and ``(n_channels, n_frames)``.
        ``energy_db`` is **absolute dBFS**, never normalised by the bank's own maximum: one loud
        sample must not rescale the whole view, which is the fault `benchmarks/spans.md` records for
        the envelope and which applies identically here.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    sr = audio.sampling_rate
    cf = erb_space(low_hz, high_hz, n_channels)
    hop = max(1, int(hop_s * sr))
    n_frames = len(x) // hop
    out = np.zeros((n_channels, n_frames))
    for k, centre in enumerate(cf):
        b, a = gammatone(centre, "iir", fs=sr)
        magnitude = np.abs(hilbert(lfilter(b, a, x)))
        out[k] = magnitude[: n_frames * hop].reshape(n_frames, hop).mean(axis=1)
    return cf, 20.0 * np.log10(out + 1e-10)
```

And `__init__.py`:

```python
"""Gammatone filterbank."""

from senselab.audio.tasks.gammatone.api import erb_space, gammatone_filterbank

__all__ = ["erb_space", "gammatone_filterbank"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/gammatone_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/gammatone/ src/tests/audio/tasks/gammatone_test.py
git add src/senselab/audio/tasks/gammatone/ src/tests/audio/tasks/gammatone_test.py
git commit -m "feat(gammatone): ERB-spaced auditory filterbank

Channel count, band edges and hop are required arguments read from the triage config,
which carries 40 channels over 80-7800 Hz on a 5 ms hop; scipy.signal.gammatone does the
filtering. Tested by exciting a single channel with a pure tone."
```

---

### Task 7: The HeAR whole-span buffer

**Why:** `capability-map.md` found that HeAR's module actively refuses the padded input AIRWAY specifies. The path works only because a buffer of exactly 32000 samples passes its length check. That coincidence needs a named function rather than each caller rediscovering it.

**Files:**
- Modify: `src/senselab/audio/tasks/health_acoustics/hear.py` (add the function; do not change the existing guard)
- Test: `src/tests/audio/tasks/health_acoustics/hear_test.py` — the tree has a **package** here, not a `health_acoustics_test.py` module. Add to the existing file.

**Interfaces:**
- Consumes: nothing. It takes `start_s`/`end_s` as floats, so a caller may pass a `Span`'s fields or any other pair — no dependency on Task 5.
- Produces: `span_to_hear_buffer(audio, start_s, end_s, *, placement="centre") -> Audio` returning exactly 2 s at the input rate.

- [ ] **Step 1: Write the failing tests**

```python
def test_span_to_hear_buffer_is_exactly_two_seconds() -> None:
    """The buffer is exactly the detector's 2 s window at the input rate."""
    import numpy as np

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

    sr = 16000
    audio = Audio(waveform=np.random.default_rng(0).standard_normal((1, 5 * sr)).astype("float32") * 0.1,
                  sampling_rate=sr)
    buf = span_to_hear_buffer(audio, 1.0, 1.35)
    assert buf.waveform.shape[-1] == 2 * sr
    assert buf.sampling_rate == sr


def test_span_to_hear_buffer_centres_the_span_and_zeroes_the_rest() -> None:
    """Everything outside the span is silence, never neighbouring audio."""
    import numpy as np

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

    sr = 16000
    x = np.ones((1, 3 * sr), dtype="float32")
    buf = span_to_hear_buffer(Audio(waveform=x, sampling_rate=sr), 1.0, 1.5)
    w = np.asarray(buf.waveform).squeeze()
    span_len = int(0.5 * sr)
    offset = (2 * sr - span_len) // 2
    assert np.all(w[:offset] == 0.0), "outside the span must be silence, not neighbouring audio"
    assert np.all(w[offset : offset + span_len] == 1.0)


def test_a_span_longer_than_two_seconds_is_refused() -> None:
    """A span the window cannot hold raises rather than being truncated."""
    import numpy as np
    import pytest

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

    sr = 16000
    audio = Audio(waveform=np.zeros((1, 5 * sr), dtype="float32"), sampling_rate=sr)
    with pytest.raises(ValueError, match="longer than the 2 s"):
        span_to_hear_buffer(audio, 1.0, 4.0)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/health_acoustics/hear_test.py -k span_to_hear -v`
Expected: FAIL — `ImportError: cannot import name 'span_to_hear_buffer'`.

- [ ] **Step 3: Implement in `hear.py`**

The window length is not re-derived: `hear.py` already names it — `HEAR_WINDOW_SAMPLES = 32000`
(`hear.py:126`) and `HEAR_WINDOW_SECONDS` (`hear.py:130`) — and the function is added in the same
module, so it reads those.

```python
def span_to_hear_buffer(audio: Audio, start_s: float, end_s: float, *, placement: str = "centre") -> Audio:
    """Place one span inside a 2 s buffer containing nothing else.

    The detector's graph accepts exactly 2 s. A span shorter than that is placed in a silent buffer so the
    model sees the span and silence, never a neighbouring event.

    Args:
        audio: The recording, at 16 kHz.
        start_s: Span onset.
        end_s: Span offset.
        placement: ``"centre"``, ``"start"`` or ``"end"`` — where in the buffer the span sits.

    Returns:
        Audio of exactly 2 s at the input's sampling rate.

    Raises:
        ValueError: If the span is longer than the window, or ``placement`` is not one of the three.
    """
    sr = audio.sampling_rate
    want = int(round(HEAR_WINDOW_SECONDS * sr))
    x = np.asarray(audio.waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    segment = x[int(start_s * sr) : int(end_s * sr)]
    if len(segment) > want:
        raise ValueError(
            f"span {start_s:.3f}-{end_s:.3f}s is {len(segment) / sr:.3f}s, longer than the "
            f"{HEAR_WINDOW_SECONDS:g} s the detector accepts. Split it or classify a sub-span."
        )
    offsets = {"centre": (want - len(segment)) // 2, "start": 0, "end": want - len(segment)}
    if placement not in offsets:
        raise ValueError(f"placement must be one of {sorted(offsets)}, got {placement!r}")
    buffer = np.zeros(want, dtype=np.float32)
    off = offsets[placement]
    buffer[off : off + len(segment)] = segment
    return Audio(waveform=buffer[None, :], sampling_rate=sr)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/health_acoustics/ -v`
Expected: the three new tests PASS and the existing ones still pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/health_acoustics/hear.py src/tests/audio/tasks/health_acoustics/hear_test.py
git add src/senselab/audio/tasks/health_acoustics/hear.py src/tests/audio/tasks/health_acoustics/hear_test.py
git commit -m "feat(hear): name the whole-span buffer instead of rediscovering it

The detector accepts exactly 2 s, so classifying a shorter span means placing it in a
silent buffer. That worked only because a buffer of exactly 32000 samples passes the
module's length check; it is now a function with the placement stated and a span longer
than 2 s refused rather than truncated."
```

---

### Task 8: The phonation task

**Files:**
- Create: `src/senselab/audio/tasks/phonation/__init__.py`, `src/senselab/audio/tasks/phonation/api.py`
- Test: `src/tests/audio/tasks/phonation_test.py`

**Interfaces:**
- Consumes: `PARSELMOUTH_AVAILABLE`, `get_sound` and `parselmouth` from
  `senselab.audio.tasks.features_extraction.praat_parselmouth` — the same Praat route
  `extract_harmonicity_descriptors` (`praat_parselmouth.py:569`, via `to_harmonicity_cc()`) and
  `extract_jitter` (`praat_parselmouth.py:1101`, via `"To PointProcess (periodic, cc)"` at `:1115`)
  already use. Read those before writing anything.
- Produces: `hnr_track(audio, *, f0_min_hz, hop_s, silence_threshold, periods_per_window) -> tuple[np.ndarray, np.ndarray]` giving `(times_s, hnr_db)`; and `period_marks(audio, start_s, end_s, *, f0_min_hz, f0_max_hz) -> list[PeriodMark]`. The VOICE node consumes both.

**On Praat, not hand-rolled.** An earlier draft measured periodicity as a biased `np.correlate`
normalised by `ac[0]`, which carries a `(1 − lag/N)` taper: with a frame of three longest periods,
its ceiling for a 100 Hz buzz is exactly 0.80, and for the reference voice at 87 Hz with
`f0_min_hz = 60` it is 0.77 — below the 0.933 `benchmarks/voice.md` recorded, so the gate interval
and the estimator were on two different scales. Praat's cc method is the normalised
cross-correlation that quantity actually is, and senselab already wraps it.

**The units change with it.** The gate's measurement is now **HNR in dB**, not normalised
autocorrelation. `benchmarks/voice.md`'s interval — periodicity `(0.44, 0.933)`, RMS
`(0.0007, 0.0161)` — was measured in the old units and **does not transfer**, so
`phonation.hnr_floor_db` and `phonation.rms_floor` are `null` in `data/config/default.yaml` and
`require()` raises on both (Task 3 already tests exactly that). This task ships the measurement
primitives; the VOICE gate stays unreachable until someone measures voiced/unvoiced HNR and RMS on
labelled data — `benchmarks/open.md` records what would settle it.

**Period marks are Praat pulse times.** Jitter is defined between consecutive periods; marks placed
at integer multiples of a frame-wise argmax lag can differ only by lag quantisation (62.5 µs at
16 kHz), which forecloses the measurement the point process exists for. The pulse times are read
out of the PointProcess object (`Get number of points` / `Get time from index`).

`f0_min_hz` and `f0_max_hz` are keyword-only with no default: `benchmarks/voice.md` records that no
single range serves both low adult male and infant voices. Every other parameter is required too —
no number appears in this module.

- [ ] **Step 1: Write the failing tests**

```python
"""The HNR track and the glottal period marks, both through Praat."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.phonation import PeriodMark, hnr_track, period_marks

SR = 16000


def _buzz(f0: float, seconds: float = 1.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(0.3 / (h + 1) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(6))
    return Audio(waveform=wave.astype(np.float32)[None, :], sampling_rate=SR)


def _noise(seconds: float = 1.0) -> Audio:
    rng = np.random.default_rng(0)
    return Audio(waveform=(0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)[None, :], sampling_rate=SR)


def _hnr(audio: Audio) -> np.ndarray:
    """Run hnr_track with Praat's documented cc settings; the fixture supplies the literals."""
    _, hnr_db = hnr_track(audio, f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5)
    return hnr_db


class TestHnr:
    """The gate's measurement is harmonicity in dB, from Praat's cc method."""

    def test_a_buzz_is_harmonic_and_noise_is_not(self) -> None:
        """A 100 Hz harmonic buzz reads far above 20 dB HNR; white noise reads below 0 dB."""
        assert float(np.median(_hnr(_buzz(100.0)))) > 20.0
        assert float(np.median(_hnr(_noise()))) < 0.0

    def test_the_track_carries_its_own_times(self) -> None:
        """Times and values come back paired, equally long and increasing."""
        times, hnr_db = hnr_track(
            _buzz(100.0), f0_min_hz=60.0, hop_s=0.01, silence_threshold=0.1, periods_per_window=4.5
        )
        assert times.shape == hnr_db.shape
        assert np.all(np.diff(times) > 0)

    def test_every_parameter_is_required(self) -> None:
        """No default stands in for a value the caller did not choose."""
        with pytest.raises(TypeError):
            hnr_track(_buzz(100.0))  # type: ignore[call-arg]


class TestPeriodMarks:
    """Pulse times come from Praat's point process, not from integer-lag multiples."""

    def test_marks_are_spaced_by_one_period(self) -> None:
        """A 100 Hz buzz yields pulses one 10 ms period apart across the span."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert len(marks) > 40
        gaps = np.diff([m.time_s for m in marks])
        assert float(np.median(gaps)) == pytest.approx(0.01, abs=0.001)

    def test_each_mark_carries_its_period_and_amplitude(self) -> None:
        """Jitter and shimmer read consecutive periods, so each mark carries both quantities."""
        marks = period_marks(_buzz(100.0), 0.2, 0.4, f0_min_hz=60.0, f0_max_hz=400.0)
        m = marks[len(marks) // 2]
        assert isinstance(m, PeriodMark)
        assert m.period_s == pytest.approx(0.01, abs=0.002)
        assert m.amplitude > 0.0

    def test_mark_times_stay_in_the_recordings_clock(self) -> None:
        """Praat places pulses on the waveform inside the span, in absolute time."""
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert 0.2 <= marks[0].time_s <= 0.25
        assert marks[-1].time_s <= 0.8

    def test_noise_yields_no_marks(self) -> None:
        """White noise gives Praat no periodic pulses, so the answer is absent, not zero."""
        assert period_marks(_noise(), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0) == []
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py -v`
Expected: FAIL at collection.

- [ ] **Step 3: Implement `api.py`**

```python
"""Harmonicity and glottal period marks, through Praat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    PARSELMOUTH_AVAILABLE,
    get_sound,
    parselmouth,
)


@dataclass(frozen=True)
class PeriodMark:
    """One glottal period, delimited by two consecutive Praat pulses.

    Attributes:
        time_s: The opening pulse, in the recording's clock.
        period_s: Time to the next pulse.
        amplitude: Peak absolute amplitude of the waveform within the period.
    """

    time_s: float
    period_s: float
    amplitude: float


def _require_parselmouth() -> None:
    """Raise when parselmouth is not installed."""
    if not PARSELMOUTH_AVAILABLE:
        raise ModuleNotFoundError(
            "`parselmouth` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )


def hnr_track(
    audio: Audio,
    *,
    f0_min_hz: float,
    hop_s: float,
    silence_threshold: float,
    periods_per_window: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Harmonics-to-noise ratio over time, in dB, via Praat's cc method.

    Args:
        audio: The recording. ``get_sound`` handles channel merging and resampling.
        f0_min_hz: Lowest F0 the analysis considers. Read it from ``phonation.f0_min_hz``.
        hop_s: Praat's ``time_step``.
        silence_threshold: Praat's silence threshold.
        periods_per_window: Praat's analysis window length, in periods of ``f0_min_hz``.

    Returns:
        ``(times_s, hnr_db)``, one value per frame. Frames Praat judges silent carry its floor
        value rather than being dropped, so the track and its times stay aligned.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    harmonicity = snd.to_harmonicity_cc(
        time_step=hop_s,
        minimum_pitch=f0_min_hz,
        silence_threshold=silence_threshold,
        periods_per_window=periods_per_window,
    )
    times = np.asarray(harmonicity.xs(), dtype=np.float64)
    values = np.asarray(harmonicity.values, dtype=np.float64).squeeze(0)
    return times, values


def period_marks(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
) -> list[PeriodMark]:
    """Glottal period marks inside one span, from Praat's point process.

    Args:
        audio: The recording.
        start_s: Span onset.
        end_s: Span offset.
        f0_min_hz: Lowest F0 to search. Read it from ``phonation.f0_min_hz``.
        f0_max_hz: Highest F0 to search. Read it from ``phonation.f0_max_hz``.

    Returns:
        One mark per pair of consecutive pulses whose gap is a plausible period — within
        ``[1/f0_max_hz, 1/f0_min_hz]``; a longer gap is a voicing break, not a period. Empty when
        Praat places no pulses — absent, not zero.

    Raises:
        ModuleNotFoundError: If parselmouth is not installed.
    """
    _require_parselmouth()
    snd = get_sound(audio)
    part = snd.extract_part(from_time=start_s, to_time=end_s, preserve_times=True)
    point_process = parselmouth.praat.call(part, "To PointProcess (periodic, cc)", f0_min_hz, f0_max_hz)
    n_points = int(parselmouth.praat.call(point_process, "Get number of points"))
    pulses = [float(parselmouth.praat.call(point_process, "Get time from index", i)) for i in range(1, n_points + 1)]
    x = np.asarray(part.values, dtype=np.float64).squeeze(0)
    t0 = float(part.xmin)
    sr = 1.0 / float(part.dx)
    marks: list[PeriodMark] = []
    for opening, closing in zip(pulses, pulses[1:]):
        period = closing - opening
        if not (1.0 / f0_max_hz <= period <= 1.0 / f0_min_hz):
            continue
        i0 = max(0, int((opening - t0) * sr))
        i1 = min(len(x), int((closing - t0) * sr))
        amplitude = float(np.abs(x[i0:i1]).max()) if i1 > i0 else 0.0
        marks.append(PeriodMark(time_s=opening, period_s=period, amplitude=amplitude))
    return marks
```

And `__init__.py`:

```python
"""Phonation measurements through Praat."""

from senselab.audio.tasks.phonation.api import PeriodMark, hnr_track, period_marks

__all__ = ["PeriodMark", "hnr_track", "period_marks"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
git add src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
git commit -m "feat(phonation): HNR track and period marks through Praat

A hand-rolled biased autocorrelation normalised at lag zero carries a (1 - lag/N) taper,
capping a 100 Hz buzz at 0.80 and putting benchmarks/voice.md's 0.933 out of reach, so the
gate interval and the estimator were on two different scales. Praat's cc harmonicity and
point process are already wrapped in features_extraction and measure the real quantities.
The gate's unit is now HNR in dB, so the autocorrelation interval does not transfer and
both phonation floors stay null in the config until measured on labelled data. Period
marks are Praat pulse times rather than integer-lag multiples -- jitter is defined between
consecutive periods and lag quantisation destroys it. The F0 search range stays
keyword-only with no default, because no single range serves both low adult and infant
voices."
```

---

### Task 9: The redaction task

**Files:**
- Create: `src/senselab/audio/tasks/redaction/__init__.py`, `src/senselab/audio/tasks/redaction/api.py`
- Test: `src/tests/audio/tasks/redaction_test.py`

**Interfaces:**
- Consumes: `PiiSpan` (Task 1) for its `start`/`end` — the inherited `ScriptLine` fields; Task 1's tests pin that no `start_s` alias exists.
- Produces: `RedactionExtent`, `plan_redactions(extents, *, padding_ms) -> list[RedactionExtent]`, `apply_redactions(audio, extents) -> Audio`.

`padding_ms` is keyword-only with no default: `benchmarks/open.md` records that the margin must exceed the *worst* alignment edge error, which is unquantified.

- [ ] **Step 1: Write the failing tests**

```python
"""Redaction: pad outward, merge overlaps, silence the audio."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.redaction import RedactionExtent, apply_redactions, plan_redactions

SR = 16000


class TestPlanning:
    """Padding and merging happen before any audio is touched."""

    def test_padding_is_required(self) -> None:
        """padding_ms has no default; the margin is unmeasured and must be supplied."""
        with pytest.raises(TypeError):
            plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")])  # type: ignore[call-arg]

    def test_extents_are_padded_outward_on_both_sides(self) -> None:
        """100 ms of padding widens (1.0, 1.2) to (0.9, 1.3)."""
        (out,) = plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")], padding_ms=100)
        assert out.start == pytest.approx(0.9)
        assert out.end == pytest.approx(1.3)

    def test_padding_never_produces_a_negative_start(self) -> None:
        """Padding clamps at the start of the recording."""
        (out,) = plan_redactions([RedactionExtent(0.02, 0.1, "PERSON")], padding_ms=100)
        assert out.start == 0.0

    def test_extents_that_overlap_after_padding_are_merged(self) -> None:
        """Two paddings that touch become one redaction carrying both categories."""
        out = plan_redactions(
            [RedactionExtent(1.0, 1.1, "PERSON"), RedactionExtent(1.25, 1.35, "DATE")], padding_ms=100
        )
        assert len(out) == 1, "an audible sliver between two redactions is a leak"
        assert out[0].category == "PERSON+DATE"


class TestApplying:
    """Applying silences exactly the planned extents and nothing else."""

    def test_the_redacted_region_is_silent_and_the_rest_is_untouched(self) -> None:
        """Samples inside the extent go to zero; samples outside keep their values."""
        x = np.ones((1, 3 * SR), dtype="float32")
        audio = Audio(waveform=x, sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.5, "PERSON")])
        w = np.asarray(out.waveform).squeeze()
        assert np.all(w[int(1.0 * SR) : int(1.5 * SR)] == 0.0)
        assert np.all(w[: int(1.0 * SR)] == 1.0)
        assert np.all(w[int(1.5 * SR) :] == 1.0)

    def test_duration_is_preserved(self) -> None:
        """Silencing replaces samples; it never cuts them out."""
        audio = Audio(waveform=np.ones((1, 3 * SR), dtype="float32"), sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.5, "PERSON")])
        assert np.asarray(out.waveform).shape[-1] == 3 * SR
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/redaction_test.py -v`
Expected: FAIL at collection.

- [ ] **Step 3: Implement `api.py`**

```python
"""Planning and applying redactions over audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from senselab.audio.data_structures import Audio


@dataclass(frozen=True)
class RedactionExtent:
    """A region to remove.

    Attributes:
        start: Onset in seconds.
        end: Offset in seconds.
        category: What was found here. Never the matched text.
    """

    start: float
    end: float
    category: str


def plan_redactions(extents: Sequence[RedactionExtent], *, padding_ms: int) -> list[RedactionExtent]:
    """Pad every extent outward and merge those that then overlap.

    Args:
        extents: Regions to redact.
        padding_ms: Margin added to each side. No default: it must exceed the worst edge error of whatever
            produced the extents, and that error is unquantified.

    Returns:
        Padded, merged extents in time order. Categories of merged extents are joined with ``+``.
    """
    pad = padding_ms / 1000.0
    widened = sorted(
        (RedactionExtent(max(0.0, e.start - pad), e.end + pad, e.category) for e in extents),
        key=lambda e: e.start,
    )
    merged: list[RedactionExtent] = []
    for extent in widened:
        if merged and extent.start <= merged[-1].end:
            last = merged[-1]
            categories = last.category.split("+")
            if extent.category not in categories:
                categories.append(extent.category)
            merged[-1] = RedactionExtent(last.start, max(last.end, extent.end), "+".join(categories))
        else:
            merged.append(extent)
    return merged


def apply_redactions(audio: Audio, extents: Sequence[RedactionExtent]) -> Audio:
    """Silence every extent, preserving duration.

    Args:
        audio: The recording.
        extents: Regions to silence. Pass the output of :func:`plan_redactions`, not raw findings.

    Returns:
        A new ``Audio``. The input is not modified.
    """
    x = np.array(np.asarray(audio.waveform, dtype=np.float32), copy=True)
    if x.ndim == 1:
        x = x[None, :]
    sr = audio.sampling_rate
    for extent in extents:
        x[:, max(0, int(extent.start * sr)) : min(x.shape[-1], int(extent.end * sr))] = 0.0
    return Audio(waveform=x, sampling_rate=sr)
```

And `__init__.py`:

```python
"""Audio redaction."""

from senselab.audio.tasks.redaction.api import RedactionExtent, apply_redactions, plan_redactions

__all__ = ["RedactionExtent", "apply_redactions", "plan_redactions"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/redaction_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/redaction/ src/tests/audio/tasks/redaction_test.py
git add src/senselab/audio/tasks/redaction/ src/tests/audio/tasks/redaction_test.py
git commit -m "feat(redaction): pad outward, merge overlaps, silence in place

padding_ms is keyword-only with no default because it must exceed the worst edge error
of whatever produced the extents, and that error is unquantified. Overlapping padded
extents merge, since redacting them separately leaves an audible sliver. Duration is
preserved and the category never carries the matched text."
```

---

### Task 10: The triage vocabulary

**Files:**
- Create: `src/senselab/audio/workflows/triage/__init__.py`, `src/senselab/audio/workflows/triage/vocabulary.py`
- Test: `src/tests/audio/workflows/triage/vocabulary_test.py`, `src/tests/audio/workflows/triage/__init__.py`

**Interfaces:**
- Consumes: nothing. The fold takes plain verdicts and mappings, so it is testable without a store; the nodes are what read and write the `ProvStore` from Task 2.
- Produces: `Outcome` (`PASS`/`FLAG`/`FAIL`), `KindState`, `RunState`, `Release`, `NodeVerdict`, `FileVerdict`, and `fold_file_verdict(node_verdicts, kind_predictions, ran, release=Release.NOT_ASSESSED) -> FileVerdict`. The follow-on node plan consumes all of these.

Derivation: `verdict.md`.

- [ ] **Step 1: Write the failing tests**

```python
"""The file-level fold: a branch fail is not a file fail."""

from __future__ import annotations

from senselab.audio.workflows.triage.vocabulary import (
    KindState,
    NodeVerdict,
    Outcome,
    RunState,
    fold_file_verdict,
)


def _v(node: str, outcome: Outcome, kind: str | None = None) -> NodeVerdict:
    return NodeVerdict(node=node, outcome=outcome, kind=kind, why="test")


class TestBranchFailIsNotFileFail:
    """A branch with no subject fails without failing the file."""

    def test_a_branch_failing_on_an_absent_kind_is_expected(self) -> None:
        """SPEECH failing where TAXONOMY predicted no speech leaves the file a pass."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway"),
                           _v("SPEECH", Outcome.FAIL, "speech")],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.PASS


class TestContradictions:
    """A branch outcome disagreeing with TAXONOMY's prediction is a flag."""

    def test_present_kind_with_a_failing_branch_flags(self) -> None:
        """A kind predicted present whose branch found no subject flags the file."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("SPEECH", Outcome.FAIL, "speech")],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert any("contradiction" in r.why for r in out.reasons)

    def test_absent_kind_with_a_passing_branch_flags_and_resolves_the_kind(self) -> None:
        """A kind predicted absent whose branch passed flags, and the kind resolves present."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert out.kinds["airway"] is KindState.PRESENT


class TestNeverRan:
    """A branch that never ran is read against what its kind predicted."""

    def test_a_skipped_branch_on_a_present_kind_flags(self) -> None:
        """Skipping the branch of a kind predicted present flags the file."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.FLAG

    def test_a_skipped_branch_on_an_absent_kind_is_expected(self) -> None:
        """Skipping the branch of a kind predicted absent is the graph working as designed."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.PASS


class TestOrdering:
    """The distinct fail cases stay distinct, and no reason is dropped."""

    def test_admit_failing_wins_over_everything(self) -> None:
        """ADMIT failing is the file verdict regardless of what the branches said."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.FAIL), _v("AIRWAY", Outcome.FLAG, "airway")],
            kind_predictions={},
            ran={"ADMIT": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[0].node == "ADMIT"

    def test_every_kind_absent_is_a_different_fail_from_admit(self) -> None:
        """No branch having a subject fails the file with a reason that is not ADMIT's."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"airway": KindState.ABSENT, "speech": KindState.ABSENT},
            ran={},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[-1].node != "ADMIT"

    def test_reasons_carry_every_contribution_not_only_the_deciding_one(self) -> None:
        """Two flagging branches both appear in the reasons, not just the first."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.FLAG, "airway"),
                           _v("VOICE", Outcome.FLAG, "voice_no_words")],
            kind_predictions={"airway": KindState.PRESENT, "voice_no_words": KindState.PRESENT},
            ran={"AIRWAY": RunState.COMPLETED, "VOICE": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert len([r for r in out.reasons if r.outcome is Outcome.FLAG]) == 2
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/workflows/triage/vocabulary_test.py -v`
Expected: FAIL at collection.

- [ ] **Step 3: Implement `vocabulary.py`**

```python
"""The triage graph's shared vocabulary and the file-level fold."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class Outcome(Enum):
    """What a node concluded."""

    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class KindState(Enum):
    """Whether a kind is in the recording."""

    PRESENT = "present"
    ABSENT = "absent"
    UNDECIDED = "undecided"


class RunState(Enum):
    """Whether a node ran at all."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERRORED = "errored"


class Release(Enum):
    """Whether a redacted artifact may be handed on."""

    RELEASABLE = "releasable"
    WITHHELD = "withheld"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class NodeVerdict:
    """One node's conclusion."""

    node: str
    outcome: Outcome
    kind: str | None
    why: str


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording."""

    triage: Outcome
    release: Release
    kinds: dict[str, KindState]
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)


_BRANCH_FOR_KIND = {"airway": "AIRWAY", "speech": "SPEECH", "voice_no_words": "VOICE"}


def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    kind_predictions: Mapping[str, KindState],
    ran: Mapping[str, RunState],
    release: Release = Release.NOT_ASSESSED,
) -> FileVerdict:
    """Combine every node's verdict into one for the recording.

    A branch ``fail`` means that branch had no subject, which is normal. It is read against what TAXONOMY
    predicted for its kind, and a disagreement between the two is a flag.

    Args:
        node_verdicts: Every node's conclusion, in graph order.
        kind_predictions: TAXONOMY's prediction per kind.
        ran: Whether each node ran.
        release: REDACT's release state, if it ran.

    Returns:
        The file verdict, carrying every contributing reason rather than only the deciding one.
    """
    reasons: list[NodeVerdict] = list(node_verdicts)
    kinds = dict(kind_predictions)
    by_kind = {v.kind: v for v in node_verdicts if v.kind}

    contradictions: list[NodeVerdict] = []
    for kind, predicted in kind_predictions.items():
        node = _BRANCH_FOR_KIND.get(kind, kind.upper())
        verdict = by_kind.get(kind)
        state = ran.get(node)
        if verdict is None:
            if state in (RunState.SKIPPED, RunState.ERRORED) and predicted in (
                KindState.PRESENT,
                KindState.UNDECIDED,
            ):
                contradictions.append(
                    NodeVerdict(node, Outcome.FLAG, kind, f"contradiction: {kind} was {predicted.value} and {node} never ran")
                )
            continue
        if predicted is KindState.PRESENT and verdict.outcome is Outcome.FAIL:
            contradictions.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"contradiction: {kind} predicted present, {node} found no subject")
            )
        elif predicted is KindState.ABSENT and verdict.outcome is Outcome.PASS:
            kinds[kind] = KindState.PRESENT
            contradictions.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"contradiction: {kind} predicted absent, {node} passed")
            )
        elif predicted is KindState.UNDECIDED:
            kinds[kind] = KindState.PRESENT if verdict.outcome is Outcome.PASS else KindState.ABSENT
    reasons.extend(contradictions)

    admit = next((v for v in node_verdicts if v.node == "ADMIT"), None)
    if admit and admit.outcome is Outcome.FAIL:
        return FileVerdict(Outcome.FAIL, release, kinds, [admit], dict(ran))
    if any(v.outcome is Outcome.FLAG for v in reasons):
        return FileVerdict(Outcome.FLAG, release, kinds, reasons, dict(ran))
    if kinds and all(s is KindState.ABSENT for s in kinds.values()):
        reasons.append(NodeVerdict("VERDICT", Outcome.FAIL, None, "every kind is absent; no branch had a subject"))
        return FileVerdict(Outcome.FAIL, release, kinds, reasons, dict(ran))
    return FileVerdict(Outcome.PASS, release, kinds, reasons, dict(ran))
```

And `__init__.py` for both the package and the test package:

```python
"""The triage workflow."""

from senselab.audio.workflows.triage.vocabulary import (
    FileVerdict,
    KindState,
    NodeVerdict,
    Outcome,
    Release,
    RunState,
    fold_file_verdict,
)

__all__ = [
    "FileVerdict",
    "KindState",
    "NodeVerdict",
    "Outcome",
    "Release",
    "RunState",
    "fold_file_verdict",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/workflows/triage/ -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole changed surface and lint**

Run:
```bash
uv run pytest src/tests/utils/prov_store_test.py src/tests/audio/tasks/ src/tests/audio/workflows/triage/ src/tests/text/tasks/ -q
uv run ruff check src/senselab src/tests
uv run mypy src/senselab/utils/prov_store.py src/senselab/audio/workflows/triage/ src/senselab/audio/tasks/envelope src/senselab/audio/tasks/spans src/senselab/audio/tasks/gammatone src/senselab/audio/tasks/phonation src/senselab/audio/tasks/redaction src/senselab/audio/tasks/disruptions
```
Expected: tests pass, ruff clean, mypy clean.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/
git add src/senselab/audio/workflows/triage/ src/tests/audio/workflows/triage/
git commit -m "feat(triage): the shared vocabulary and the file-level fold

A branch fail is not a file fail -- it means that branch had no subject, which is
normal. Outcomes are read against TAXONOMY's prediction, and the two disagreeing is a
flag, as is a branch that never ran on a kind predicted present. ADMIT failing and
every-kind-absent are distinct fails with distinct reasons, and the verdict carries
every contributing reason rather than only the deciding one."
```

---

### Task 11: The disruptions task

**Files:**
- Create: `src/senselab/audio/tasks/disruptions/__init__.py`, `src/senselab/audio/tasks/disruptions/api.py`
- Test: `src/tests/audio/tasks/disruptions_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Disruptions` (frozen dataclass) and `detect_disruptions(audio, start_s, end_s, *, clip_headroom, min_clip_run, min_dropout_ms, discontinuity_threshold) -> Disruptions`.

Derivation: `branch-speech.md` step 8. Counts and extents, never a score. The four parameters are
required arguments; their values live in `disruptions.*` of Task 3's config, whose derivation records
that they are conventional rather than fitted. The *tolerance* — how much is too much — has no value
(`quality.disruption_*` is null) and is not this function's business.

- [ ] **Step 1: Write the failing tests**

```python
"""Recording disruptions: clipping, dropouts, discontinuities, DC offset."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.disruptions import Disruptions, detect_disruptions

SR = 16000


def _audio(x: np.ndarray) -> Audio:
    return Audio(waveform=x.astype("float32")[None, :], sampling_rate=SR)


def _tone(seconds: float = 1.0, amp: float = 0.5, freq: float = 200.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _detect(audio: Audio, start_s: float, end_s: float) -> Disruptions:
    """Run detect_disruptions with the config's conventional values; the fixture supplies the literals."""
    return detect_disruptions(
        audio, start_s, end_s, clip_headroom=0.999, min_clip_run=3, min_dropout_ms=10.0,
        discontinuity_threshold=0.5,
    )


class TestClipping:
    """Clipping is a run of samples at the headroom, never a single sample."""

    def test_a_clean_tone_has_no_clipping(self) -> None:
        """A half-scale tone reports zero clipped runs and zero clipped time."""
        d = _detect(_audio(_tone()), 0.0, 1.0)
        assert d.clipped_runs == 0
        assert d.clipped_s == 0.0

    def test_a_saturated_tone_is_clipped(self) -> None:
        """A tone clipped at full scale reports a run on every half cycle."""
        d = _detect(_audio(np.clip(_tone(amp=2.0), -1.0, 1.0)), 0.0, 1.0)
        assert d.clipped_runs >= 100, "200 Hz for 1 s saturates on every half cycle"
        assert d.clipped_s > 0.1

    def test_a_single_full_scale_sample_is_not_clipping(self) -> None:
        """One sample at full scale is below min_clip_run and does not count."""
        x = _tone()
        x[5000] = 1.0
        assert _detect(_audio(x), 0.0, 1.0).clipped_runs == 0


class TestDropouts:
    """A dropout is a run of exact zeros at least min_dropout_ms long."""

    def test_a_zero_run_is_a_dropout(self) -> None:
        """A 250 ms zero run is one dropout of that duration."""
        x = _tone()
        x[4000:8000] = 0.0
        d = _detect(_audio(x), 0.0, 1.0)
        assert d.dropout_runs == 1
        assert d.dropout_s == pytest.approx(4000 / SR, abs=0.002)

    def test_a_run_shorter_than_the_minimum_is_not_a_dropout(self) -> None:
        """A 20-sample zero run is shorter than 10 ms and does not count."""
        x = _tone()
        x[4000:4020] = 0.0
        assert _detect(_audio(x), 0.0, 1.0).dropout_runs == 0


class TestDiscontinuities:
    """A discontinuity is a sample-to-sample jump beyond the threshold."""

    def test_a_step_is_a_discontinuity(self) -> None:
        """A 0.9 step in an otherwise smooth tone is counted."""
        x = _tone(amp=0.1)
        x[8000:] += 0.9
        assert _detect(_audio(x), 0.0, 1.0).discontinuities >= 1

    def test_a_smooth_tone_has_none(self) -> None:
        """A continuous tone produces zero discontinuities."""
        assert _detect(_audio(_tone()), 0.0, 1.0).discontinuities == 0


class TestDcOffset:
    """DC offset is the mean sample value over the span."""

    def test_a_bias_is_reported(self) -> None:
        """A +0.2 bias reads back as 0.2."""
        d = _detect(_audio(_tone() + 0.2), 0.0, 1.0)
        assert d.dc_offset == pytest.approx(0.2, abs=0.01)

    def test_a_centred_signal_reports_near_zero(self) -> None:
        """A zero-mean tone reads near zero."""
        assert abs(_detect(_audio(_tone()), 0.0, 1.0).dc_offset) < 0.01


class TestScoping:
    """Measurement is scoped to the requested span."""

    def test_only_the_requested_span_is_measured(self) -> None:
        """Clipping in the first second is invisible to a span that excludes it."""
        x = _tone(seconds=3.0)
        x[: SR] = np.clip(_tone(amp=2.0)[: SR], -1.0, 1.0)
        assert _detect(_audio(x), 1.5, 2.5).clipped_runs == 0
        assert _detect(_audio(x), 0.0, 1.0).clipped_runs > 0

    def test_a_clean_span_reports_zero_rather_than_nothing(self) -> None:
        """A clean span is zeros — a different statement from a span nobody measured."""
        d = _detect(_audio(_tone()), 0.0, 1.0)
        assert isinstance(d, Disruptions)
        assert (d.clipped_runs, d.dropout_runs, d.discontinuities) == (0, 0, 0)
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/disruptions_test.py -v`
Expected: FAIL at collection — no module `senselab.audio.tasks.disruptions`.

- [ ] **Step 3: Implement `api.py`**

```python
"""Detecting recording disruptions within a span.

Counts and extents, never a score. How much disruption makes a span unusable is a tolerance nobody has
derived, and it is the caller's decision rather than this module's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio


@dataclass(frozen=True)
class Disruptions:
    """What was found in one span.

    Attributes:
        start: Span onset in seconds.
        end: Span offset in seconds.
        clipped_runs: Number of runs of consecutive samples at or beyond the headroom.
        clipped_s: Total duration of those runs.
        dropout_runs: Number of runs of exact zeros at least ``min_dropout_ms`` long.
        dropout_s: Total duration of those runs.
        discontinuities: Number of sample-to-sample jumps exceeding the threshold.
        dc_offset: Mean sample value over the span.
    """

    start: float
    end: float
    clipped_runs: int
    clipped_s: float
    dropout_runs: int
    dropout_s: float
    discontinuities: int
    dc_offset: float


def _runs(mask: np.ndarray, minimum: int) -> tuple[int, int]:
    """Count runs of True at least ``minimum`` long, and their total length.

    Args:
        mask: Boolean array.
        minimum: Shortest run that counts.

    Returns:
        ``(run_count, total_samples)``.
    """
    if not mask.any():
        return 0, 0
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))
    lengths = [e - s for s, e in zip(starts, ends) if e - s >= minimum]
    return len(lengths), int(sum(lengths))


def detect_disruptions(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    clip_headroom: float,
    min_clip_run: int,
    min_dropout_ms: float,
    discontinuity_threshold: float,
) -> Disruptions:
    """Measure disruptions inside one span.

    Args:
        audio: The recording. A multi-channel input is averaged.
        start_s: Span onset.
        end_s: Span offset.
        clip_headroom: A sample at or beyond this absolute value counts as clipped. Read it from
            ``disruptions.clip_headroom``.
        min_clip_run: Shortest run of clipped samples that counts as a clipping event. Read it from
            ``disruptions.min_clip_run``.
        min_dropout_ms: Shortest run of exact zeros that counts as a dropout. Read it from
            ``disruptions.min_dropout_ms``.
        discontinuity_threshold: Absolute sample-to-sample jump that counts as a discontinuity. Read
            it from ``disruptions.discontinuity_threshold``.

    Returns:
        The span's disruptions. Every count is exact; a clean span reports zeros.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    sr = audio.sampling_rate
    segment = x[max(0, int(start_s * sr)) : min(len(x), int(end_s * sr))]
    if segment.size == 0:
        return Disruptions(start_s, end_s, 0, 0.0, 0, 0.0, 0, 0.0)
    clip_runs, clip_n = _runs(np.abs(segment) >= clip_headroom, min_clip_run)
    drop_runs, drop_n = _runs(segment == 0.0, max(1, int(min_dropout_ms * sr / 1000)))
    jumps = int(np.count_nonzero(np.abs(np.diff(segment)) > discontinuity_threshold))
    return Disruptions(
        start=start_s,
        end=end_s,
        clipped_runs=clip_runs,
        clipped_s=clip_n / sr,
        dropout_runs=drop_runs,
        dropout_s=drop_n / sr,
        discontinuities=jumps,
        dc_offset=float(segment.mean()),
    )
```

And `__init__.py`:

```python
"""Recording disruptions."""

from senselab.audio.tasks.disruptions.api import Disruptions, detect_disruptions

__all__ = [
    "Disruptions",
    "detect_disruptions",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/disruptions_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/disruptions/ src/tests/audio/tasks/disruptions_test.py
git add src/senselab/audio/tasks/disruptions/ src/tests/audio/tasks/disruptions_test.py
git commit -m "feat(disruptions): clipping, dropouts, discontinuities and DC offset per span

SQUIM is a speech-quality estimator trained on particular degradations, so hard clipping
can read as acceptable or as generic noise rather than as the defect it is. These are
counts and extents rather than a score, scoped to a span, and a clean span reports zeros
-- which is a different statement from a span nobody measured. The four detection
parameters are required arguments read from the triage config, whose derivation records
that they are conventional rather than fitted. How much is too much is a tolerance nobody
has derived and is the caller's decision."
```

---

## What this plan does not build

- **The nine node implementations.** They consume everything above and are a second plan.
- **The Nextflow orchestration**, which already exists at `specs/20260817-triage-workflow-dag/nextflow/` and lints clean with a passing stub run.
- **Anything requiring an undecided parameter.** SQUIM thresholds, the phonation gate's floors and the redaction margin have no justified value: each is `null` in Task 3's config, `require()` raises on it naming what would settle it, and a caller must supply an override. Where such a value is also a function parameter — `padding_ms` — it is keyword-only with no default. `benchmarks/open.md` records them all.

## Known deviations from the design, to resolve before the node plan

- `resample_audios` designs its anti-alias filter at the target rate and applies it at the source rate. Unverified as a defect; a sweep test would settle it.
- `classify_audios` defaults to `top_k=5` for YAMNet, so `Silence` can be absent rather than low, and applies softmax to AST, which is wrong for multi-label AudioSet. The node plan must pass explicit arguments.
- `MossFormer2_SS_16K` RMS-normalises its input to −25 dBFS, so no dBFS-referenced measurement on a separated stream is comparable with one on the recording.
