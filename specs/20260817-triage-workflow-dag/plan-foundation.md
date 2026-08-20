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
- **Parameters the design leaves undecided must be keyword-only with no default**, so their absence is visible in the signature. These are: the phonation gate floors, SQUIM thresholds, the redaction padding margin, the word-gap threshold, `min_families` per kind.
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
- Produces: `class PiiSpan(ScriptLine)` adding `category: str`, `source: str`, `asr_model: str`. Location is `span.start` / `span.end`; attribution is `span.speaker`. Task 8 (redaction) and the node plan's SPEECH step 7 consume those directly.
- **Every existing `PiiSpan(...)` construction keeps working** — all are keyword-only and pass only fields the subclass still has.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_finding_is_a_script_line():
    from senselab.text.tasks.pii_detection.api import PiiSpan
    from senselab.utils.data_structures import ScriptLine

    span = PiiSpan(text="Jane Doe", category="PERSON", source="presidio", asr_model="w", score=0.9)
    assert isinstance(span, ScriptLine), "a finding is a timed piece of text, not a parallel type"
    assert span.start is None and span.end is None, "unlocated until something locates it"


def test_a_located_finding_reports_its_extent_and_speaker_natively():
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane Doe", category="PERSON", source="presidio", asr_model="w",
        start=11.9, end=12.4, speaker="SPEAKER_00",
    )
    assert (span.start, span.end) == (11.9, 12.4)
    assert span.speaker == "SPEAKER_00"
    assert not hasattr(span, "start_s"), "no parallel name for a field that already exists"


def test_the_timing_provenance_lives_on_the_finding():
    from senselab.text.tasks.pii_detection.api import PiiSpan

    span = PiiSpan(
        text="Jane", category="PERSON", source="presidio", asr_model="w",
        start=1.0, end=1.3, timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
    )
    assert span.timestamp_model == "Qwen/Qwen3-ForcedAligner-0.6B", (
        "two recognizers timed by one aligner are not independent witnesses"
    )


def test_scanning_a_script_line_carries_its_timing_onto_every_finding():
    from senselab.text.tasks.pii_detection.api import scan_for_pii
    from senselab.utils.data_structures import ScriptLine

    line = ScriptLine(text="call alice@example.com", speaker="SPEAKER_01", start=3.0, end=4.5)
    scan = scan_for_pii(line, detectors=["rules"])
    assert scan.spans, "rules detector found nothing"
    assert all(sp.start == 3.0 and sp.end == 4.5 for sp in scan.spans)
    assert all(sp.speaker == "SPEAKER_01" for sp in scan.spans)


def test_scanning_a_bare_string_leaves_the_finding_unlocated():
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

In `_materialize_spans`, accept the scanned line and copy its timing onto every span:

```python
def _materialize_spans(
    raw_spans: list[dict[str, Any]], source_id: str, line: ScriptLine | None = None
) -> list[PiiSpan]:
    """Turn raw detector output into ``PiiSpan``s.

    Args:
        raw_spans: Detector dicts carrying ``text``, ``category`` and ``source``.
        source_id: Identifier of the scanned input.
        line: The line scanned, when the input was one. Its extent, speaker and timing provenance are
            copied onto every finding.

    Returns:
        One ``PiiSpan`` per distinct (category, text, source).
    """
    seen: set[tuple[str, str, str]] = set()
    out: list[PiiSpan] = []
    timing = (
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
        key = (raw["category"], raw["text"], raw["source"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            PiiSpan(
                text=raw["text"],
                category=raw["category"],
                source=raw["source"],
                asr_model=source_id,
                score=raw.get("score"),
                **timing,
            )
        )
    return out
```

At the call site in `scan_for_pii`, pass the input through when it is a `ScriptLine`:

```python
        line = item if isinstance(item, ScriptLine) else None
        spans = _materialize_spans(raw, source_id=source_id, line=line)
```

- [ ] **Step 6: Run the new tests and everything that touches PII**

Run: `uv run pytest src/tests/text/tasks/ src/tests/audio/tasks/pii_detection_test.py src/tests/audio/workflows/pii_adapter_test.py -v`
Expected: the five new tests PASS and **every existing test passes unchanged**, except any that asserts the exact key set of a serialised span — `pii_adapter_test.py` is the one that might. If it does, the artifact genuinely gained keys and the assertion should be updated to match; if any *other* test fails, the change was not additive and the production code is wrong.

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

### Task 2: The element store mechanism

**Files:**
- Create: `src/senselab/utils/element_store.py`
- Test: `src/tests/utils/element_store_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Element`, `Assertion`, `ModelProvenance`, `ElementStore` with `.add_element()`, `.assert_over()`, `.elements()`, `.assertions_for()`, `.write_jsonl()`, `.read_jsonl()`, `.merge()`. Tasks 3–9 and every node consume these.

Top-level `utils/`, not `utils/data_structures/`, to avoid that package's `__init__` fan-in.

- [ ] **Step 1: Write the failing tests**

```python
"""The element store: append-only, provenance-carrying, order-independent."""

from __future__ import annotations

import pytest

from senselab.utils.element_store import Assertion, Element, ElementStore, ModelProvenance


def _store() -> ElementStore:
    return ElementStore(run_id="run-1")


class TestAppendOnly:
    def test_an_element_keeps_its_author_and_evidence(self):
        s = _store()
        eid = s.add_element(kind="span", extent=(1.0, 2.0), author="PREPROCESS",
                            evidence={"peak_over_floor_db": 31.4})
        (el,) = s.elements()
        assert el.id == eid and el.kind == "span" and el.extent == (1.0, 2.0)
        assert el.evidence["peak_over_floor_db"] == 31.4

    def test_an_assertion_carries_its_own_id_so_confirm_can_name_it(self):
        s = _store()
        eid = s.add_element(kind="span", extent=(1.0, 2.0), author="PREPROCESS", evidence={})
        aid = s.assert_over(eid, verb="label", value="Cough", author="AIRWAY", evidence={"coverage": 1.0})
        cid = s.assert_over(eid, verb="confirm", value=aid, author="AIRWAY", evidence={"yamnet": 1.0})
        assert cid != aid
        verbs = {a.verb: a for a in s.assertions_for(eid)}
        assert verbs["confirm"].value == aid

    def test_a_contest_does_not_remove_the_label_it_contests(self):
        s = _store()
        eid = s.add_element(kind="span", extent=(11.75, 13.16), author="PREPROCESS", evidence={})
        aid = s.assert_over(eid, verb="label", value="Cough", author="AIRWAY", evidence={})
        s.assert_over(eid, verb="contest", value=aid, author="AIRWAY", evidence={"yamnet": "Speech"})
        verbs = [a.verb for a in s.assertions_for(eid)]
        assert verbs == ["label", "contest"], "both survive, in order"

    def test_nothing_can_be_deleted(self):
        s = _store()
        assert not hasattr(s, "delete_element")
        assert not hasattr(s, "remove_assertion")


class TestProvenance:
    def test_a_model_authored_element_requires_a_resolved_sha(self):
        s = _store()
        with pytest.raises(ValueError, match="40-hex"):
            s.add_element(kind="span", extent=None, author="AIRWAY", evidence={},
                          model=ModelProvenance(model_id="google/hear", revision="main"))

    def test_a_forty_hex_revision_is_accepted(self):
        s = _store()
        sha = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
        eid = s.add_element(kind="measurement", extent=None, author="AIRWAY", evidence={},
                            model=ModelProvenance(model_id="google/hear", revision=sha))
        assert s.element(eid).model.revision == sha


class TestOrderIndependence:
    def test_merging_in_any_order_gives_the_same_store(self):
        a, b = _store(), _store()
        a.add_element(kind="span", extent=(1.0, 2.0), author="PREPROCESS", evidence={})
        b.add_element(kind="word", extent=(1.1, 1.4), author="SPEECH", evidence={"word": "hello"})
        one = ElementStore.merge([a, b]).fingerprint()
        two = ElementStore.merge([b, a]).fingerprint()
        assert one == two


class TestRoundTrip:
    def test_jsonl_survives_a_round_trip(self, tmp_path):
        s = _store()
        eid = s.add_element(kind="span", extent=(1.0, 2.0), author="PREPROCESS", evidence={"k": 18})
        s.assert_over(eid, verb="label", value="Cough", author="AIRWAY", evidence={})
        p = tmp_path / "store.jsonl"
        s.write_jsonl(p)
        back = ElementStore.read_jsonl(p)
        assert back.fingerprint() == s.fingerprint()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest src/tests/utils/element_store_test.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'senselab.utils.element_store'`

- [ ] **Step 3: Implement**

```python
"""An append-only store of elements and the assertions nodes make over them.

Elements are things the graph believes might exist; assertions are later claims about them. Nothing is
deleted and nothing is overwritten, so merging two stores is a set union and is order-independent.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

KIND = Literal["span", "word", "speaker", "interval", "measurement", "kind", "stream", "pii", "run", "verdict"]
VERB = Literal["label", "confirm", "contest", "refine", "withdraw", "measure"]

_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelProvenance:
    """The model behind an element or assertion.

    Attributes:
        model_id: The model's identifier, e.g. ``"google/hear"``.
        revision: A resolved 40-hex commit SHA. A ref is refused.
    """

    model_id: str
    revision: str

    def __post_init__(self) -> None:
        """Reject a revision that is not a resolved commit.

        Raises:
            ValueError: If ``revision`` is not 40 hex characters.
        """
        if not _SHA.match(self.revision):
            raise ValueError(
                f"revision must be a resolved 40-hex commit SHA, got {self.revision!r}. "
                "Recording a ref makes the provenance confidently wrong."
            )


@dataclass(frozen=True)
class Element:
    """Something the graph believes might exist."""

    id: str
    kind: KIND
    extent: tuple[float, float] | None
    author: str
    evidence: dict[str, Any] = field(default_factory=dict)
    model: ModelProvenance | None = None


@dataclass(frozen=True)
class Assertion:
    """A later claim about an element."""

    id: str
    element_id: str
    verb: VERB
    value: Any
    author: str
    evidence: dict[str, Any] = field(default_factory=dict)
    model: ModelProvenance | None = None


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ElementStore:
    """An append-only collection of elements and assertions.

    Args:
        run_id: Identifier for the run, mixed into every id so two runs never collide.
    """

    def __init__(self, run_id: str) -> None:
        """Create an empty store."""
        self.run_id = run_id
        self._elements: dict[str, Element] = {}
        self._assertions: list[Assertion] = []

    def add_element(
        self,
        *,
        kind: KIND,
        extent: tuple[float, float] | None,
        author: str,
        evidence: dict[str, Any],
        model: ModelProvenance | None = None,
    ) -> str:
        """Append an element.

        Args:
            kind: What sort of thing this is.
            extent: ``(start, end)`` in seconds, or None for a file-level element.
            author: The node proposing it.
            evidence: Whatever the author measured, verbatim.
            model: Provenance, required when a model produced the element.

        Returns:
            The element's id.
        """
        eid = f"{kind}-{_digest({'r': self.run_id, 'k': kind, 'x': extent, 'a': author, 'e': evidence})}"
        self._elements[eid] = Element(
            id=eid, kind=kind, extent=extent, author=author, evidence=dict(evidence), model=model
        )
        return eid

    def assert_over(
        self,
        element_id: str,
        *,
        verb: VERB,
        value: Any,
        author: str,
        evidence: dict[str, Any],
        model: ModelProvenance | None = None,
    ) -> str:
        """Append an assertion about an element.

        Args:
            element_id: The element being asserted over.
            verb: What kind of claim this is.
            value: The claim's payload — a label, or the id of the assertion being confirmed or contested.
            author: The node making the claim.
            evidence: Whatever the author measured.
            model: Provenance, required when a model produced the claim.

        Returns:
            The assertion's own id, which ``confirm`` and ``contest`` name.

        Raises:
            KeyError: If ``element_id`` is not in the store.
        """
        if element_id not in self._elements:
            raise KeyError(f"no element {element_id!r} to assert over")
        aid = f"a-{_digest({'r': self.run_id, 'e': element_id, 'v': verb, 'x': value, 'a': author, 'd': evidence})}"
        self._assertions.append(
            Assertion(
                id=aid, element_id=element_id, verb=verb, value=value,
                author=author, evidence=dict(evidence), model=model,
            )
        )
        return aid

    def element(self, element_id: str) -> Element:
        """Return one element.

        Args:
            element_id: Its id.

        Returns:
            The element.
        """
        return self._elements[element_id]

    def elements(self, kind: KIND | None = None) -> list[Element]:
        """Return elements, optionally of one kind.

        Args:
            kind: Restrict to this kind, or None for all.

        Returns:
            Elements in insertion order.
        """
        return [e for e in self._elements.values() if kind is None or e.kind == kind]

    def assertions_for(self, element_id: str) -> list[Assertion]:
        """Return every assertion about one element, in the order they were made.

        Args:
            element_id: The element's id.

        Returns:
            Its assertions.
        """
        return [a for a in self._assertions if a.element_id == element_id]

    def write_jsonl(self, path: str | Path) -> None:
        """Write the store as one JSON object per line.

        Args:
            path: Destination file.
        """
        lines = [json.dumps({"type": "element", **asdict(e)}, sort_keys=True, default=str) for e in self._elements.values()]
        lines += [json.dumps({"type": "assertion", **asdict(a)}, sort_keys=True, default=str) for a in self._assertions]
        Path(path).write_text("\n".join(lines) + "\n")

    @classmethod
    def read_jsonl(cls, path: str | Path, run_id: str = "read") -> "ElementStore":
        """Read a store back.

        Args:
            path: The file to read.
            run_id: Run id for the reconstructed store.

        Returns:
            The store.
        """
        store = cls(run_id=run_id)
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            kind = rec.pop("type")
            model = rec.pop("model", None)
            prov = ModelProvenance(**model) if model else None
            if kind == "element":
                extent = rec.pop("extent")
                store._elements[rec["id"]] = Element(
                    extent=tuple(extent) if extent else None, model=prov, **rec
                )
            else:
                store._assertions.append(Assertion(model=prov, **rec))
        return store

    @classmethod
    def merge(cls, stores: Sequence["ElementStore"]) -> "ElementStore":
        """Union several stores.

        Append-only means no element is ever mutated, so the union is order-independent.

        Args:
            stores: The stores to merge.

        Returns:
            One store holding every element and assertion.
        """
        out = cls(run_id="merged")
        for s in stores:
            out._elements.update(s._elements)
        seen: set[str] = set()
        for s in stores:
            for a in s._assertions:
                if a.id not in seen:
                    seen.add(a.id)
                    out._assertions.append(a)
        return out

    def fingerprint(self) -> str:
        """A content hash that ignores insertion order.

        Returns:
            A hex digest identical for stores holding the same elements and assertions.
        """
        eids = sorted(self._elements)
        aids = sorted(a.id for a in self._assertions)
        return _digest({"e": eids, "a": aids})
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/utils/element_store_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/utils/element_store.py src/tests/utils/element_store_test.py
git add src/senselab/utils/element_store.py src/tests/utils/element_store_test.py
git commit -m "feat(store): an append-only element store with provenance

Elements and assertions, nothing deleted and nothing overwritten, so merging two
stores is a set union and order-independent -- verified by fingerprint equality under
reordering. A model-authored record must carry a resolved 40-hex commit; a ref is
refused at construction."
```

---

### Task 3: The envelope task

**Files:**
- Create: `src/senselab/audio/tasks/envelope/__init__.py`, `src/senselab/audio/tasks/envelope/api.py`
- Test: `src/tests/audio/tasks/envelope_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `hilbert_envelope_dbfs(audio) -> np.ndarray`, `rolling_floor_dbfs(envelope_db, sr, window_s=3.0, percentile=10.0) -> np.ndarray`. Task 4 consumes both.

Derivation: `benchmarks/preprocess-params.md` (zero-phase, 40 Hz) and `benchmarks/spans.md` (why local and absolute).

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


class TestEnvelopeIsAbsolute:
    def test_a_half_scale_tone_sits_near_minus_six_dbfs(self):
        env = hilbert_envelope_dbfs(_tone(1.0, 0.5))
        mid = env[SR // 4 : -SR // 4]
        assert -7.5 < float(np.median(mid)) < -4.5

    def test_scaling_the_input_shifts_the_envelope_by_the_same_amount(self):
        loud = float(np.median(hilbert_envelope_dbfs(_tone(1.0, 0.5))[SR // 4 : -SR // 4]))
        quiet = float(np.median(hilbert_envelope_dbfs(_tone(1.0, 0.05))[SR // 4 : -SR // 4]))
        assert loud - quiet == pytest.approx(20.0, abs=1.0), "dBFS is absolute, not max-normalised"

    def test_a_loud_click_elsewhere_does_not_move_the_rest(self):
        quiet = _tone(2.0, 0.05)
        clicked = quiet.waveform.numpy().copy() if hasattr(quiet.waveform, "numpy") else np.array(quiet.waveform)
        clicked[0, SR : SR + 480] += 0.95
        a = hilbert_envelope_dbfs(quiet)
        b = hilbert_envelope_dbfs(Audio(waveform=clicked.astype(np.float32), sampling_rate=SR))
        early = slice(SR // 8, SR // 2)
        assert float(np.median(b[early])) == pytest.approx(float(np.median(a[early])), abs=0.5)


class TestRollingFloor:
    def test_the_floor_tracks_a_level_change_rather_than_averaging_it(self):
        env = np.concatenate([np.full(5 * SR, -60.0), np.full(5 * SR, -30.0)])
        fl = rolling_floor_dbfs(env, SR, window_s=1.0, percentile=10.0)
        assert fl[SR] == pytest.approx(-60.0, abs=1.0)
        assert fl[9 * SR] == pytest.approx(-30.0, abs=1.0)

    def test_the_floor_is_one_value_per_sample(self):
        env = np.full(3 * SR, -50.0)
        assert rolling_floor_dbfs(env, SR).shape == env.shape
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

ENVELOPE_LOWPASS_HZ = 40.0
FLOOR_WINDOW_S = 3.0
FLOOR_PERCENTILE = 10.0


def hilbert_envelope_dbfs(audio: Audio, lowpass_hz: float = ENVELOPE_LOWPASS_HZ) -> np.ndarray:
    """The analytic-signal magnitude, lowpassed, in dBFS.

    The filter is zero-phase, so the envelope is offline-only.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        lowpass_hz: Cutoff of the zero-phase Butterworth lowpass.

    Returns:
        One dBFS value per input sample. Absolute, never normalised by the input's maximum.
    """
    x = np.asarray(audio.waveform, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=0)
    b, a = butter(4, lowpass_hz / (audio.sampling_rate / 2), "low")
    env = np.maximum(filtfilt(b, a, np.abs(hilbert(x))), 1e-12)
    return 20.0 * np.log10(env)


def rolling_floor_dbfs(
    envelope_db: np.ndarray,
    sampling_rate: int,
    window_s: float = FLOOR_WINDOW_S,
    percentile: float = FLOOR_PERCENTILE,
) -> np.ndarray:
    """A low percentile of the envelope over a sliding window.

    Args:
        envelope_db: Output of :func:`hilbert_envelope_dbfs`.
        sampling_rate: Samples per second of ``envelope_db``.
        window_s: Width of the sliding window.
        percentile: Which percentile within the window is the floor.

    Returns:
        One floor value per sample of ``envelope_db``.
    """
    n = len(envelope_db)
    half = int(window_s * sampling_rate) // 2
    step = max(1, int(0.1 * sampling_rate))
    centres = np.arange(0, n, step)
    vals = [float(np.percentile(envelope_db[max(0, c - half) : min(n, c + half)], percentile)) for c in centres]
    return np.interp(np.arange(n), centres, vals)
```

And `__init__.py`:

```python
"""Amplitude envelope and local floor."""

from senselab.audio.tasks.envelope.api import (
    ENVELOPE_LOWPASS_HZ,
    FLOOR_PERCENTILE,
    FLOOR_WINDOW_S,
    hilbert_envelope_dbfs,
    rolling_floor_dbfs,
)

__all__ = [
    "ENVELOPE_LOWPASS_HZ",
    "FLOOR_PERCENTILE",
    "FLOOR_WINDOW_S",
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

### Task 4: The spans task

**Files:**
- Create: `src/senselab/audio/tasks/spans/__init__.py`, `src/senselab/audio/tasks/spans/api.py`
- Test: `src/tests/audio/tasks/spans_test.py`

**Interfaces:**
- Consumes: `hilbert_envelope_dbfs`, `rolling_floor_dbfs` from Task 3.
- Produces: `Span` (frozen dataclass: `start`, `end`, `peak_over_floor_db`) and
  `propose_spans(envelope_db, floor_db, sampling_rate, *, k_db, onset_drop_db=15.0, offset_fraction=0.7, hangover_ms=120, min_duration_ms=50) -> list[Span] | NoContrast`.
  `NoContrast` is a distinct return, not an empty list.

Derivation: `benchmarks/spans.md`.

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


class TestGate:
    def test_a_peak_below_k_is_not_proposed(self):
        env = _envelope([(2.0, 2.5, -45.0)])
        out = propose_spans(env, np.full_like(env, -55.0), SR, k_db=18.0)
        assert out == []

    def test_a_peak_above_k_is_proposed(self):
        env = _envelope([(2.0, 2.5, -20.0)])
        out = propose_spans(env, np.full_like(env, -55.0), SR, k_db=18.0)
        assert len(out) == 1 and isinstance(out[0], Span)

    def test_two_events_far_apart_stay_two_spans(self):
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = propose_spans(env, np.full_like(env, -55.0), SR, k_db=18.0)
        assert len(out) == 2


class TestNoContrast:
    def test_no_peak_anywhere_is_no_contrast_not_an_empty_list(self):
        env = np.full(int(3.0 * SR), -30.0)
        out = propose_spans(env, np.full_like(env, -29.0), SR, k_db=18.0)
        assert isinstance(out, NoContrast)
        assert "18" in out.reason


class TestKIsRequired:
    def test_k_has_no_default(self):
        env = _envelope([(2.0, 2.5, -20.0)])
        with pytest.raises(TypeError):
            propose_spans(env, np.full_like(env, -55.0), SR)  # type: ignore[call-arg]


class TestSpanCarriesItsContrast:
    def test_peak_over_floor_travels_with_the_span(self):
        env = _envelope([(2.0, 2.5, -20.0)])
        (span,) = propose_spans(env, np.full_like(env, -55.0), SR, k_db=18.0)
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

ONSET_DROP_DB = 15.0
OFFSET_FRACTION = 0.7
HANGOVER_MS = 120
MIN_DURATION_MS = 50
MIN_SEPARATION_MS = 150


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
    onset_drop_db: float = ONSET_DROP_DB,
    offset_fraction: float = OFFSET_FRACTION,
    hangover_ms: int = HANGOVER_MS,
    min_duration_ms: int = MIN_DURATION_MS,
) -> list[Span] | NoContrast:
    """Propose spans from an envelope, anchoring the onset to each event's own peak.

    Args:
        envelope_db: Envelope in dBFS.
        floor_db: Local floor, same length as ``envelope_db``.
        sampling_rate: Samples per second.
        k_db: How far above the local floor a peak must rise to be proposed. No default: the value is
            per-reader and unmeasured across readers.
        onset_drop_db: Walk back from the peak to ``peak - onset_drop_db``.
        offset_fraction: Walk forward to ``peak - offset_fraction * (peak - floor)``.
        hangover_ms: The offset closes only after this long continuously below threshold. Must be shorter
            than the shortest event to be bounded.
        min_duration_ms: Discard spans shorter than this.

    Returns:
        Merged spans in time order, or :class:`NoContrast` when no peak clears ``k_db``.
    """
    above = envelope_db - floor_db
    peaks, _ = find_peaks(above, height=k_db, distance=int(MIN_SEPARATION_MS * sampling_rate / 1000))
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

from senselab.audio.tasks.spans.api import (
    HANGOVER_MS,
    MIN_DURATION_MS,
    OFFSET_FRACTION,
    ONSET_DROP_DB,
    NoContrast,
    Span,
    propose_spans,
)

__all__ = [
    "HANGOVER_MS",
    "MIN_DURATION_MS",
    "OFFSET_FRACTION",
    "ONSET_DROP_DB",
    "NoContrast",
    "Span",
    "propose_spans",
]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/spans_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Reproduce the benchmark on the reference recording**

Run: `uv run python specs/20260817-triage-workflow-dag/benchmarks/scripts/floor.py`
Expected: five spans at 2.32–3.29, 5.32–6.22, 7.92–8.51, 9.61–9.96, 11.75–13.16 s, matching `benchmarks/spans.md`. If they differ, the implementation diverges from the measured rules — fix the implementation, not the benchmark.

- [ ] **Step 6: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/spans/ src/tests/audio/tasks/spans_test.py
git add src/senselab/audio/tasks/spans/ src/tests/audio/tasks/spans_test.py
git commit -m "feat(spans): peak-anchored onset, range-relative offset, no_contrast as a value

k_db is keyword-only with no default because it is per-reader and no cross-reader value
was measured. NoContrast is a distinct return rather than an empty list, so an
unmeasurable recording cannot read as a quiet one."
```

---

### Task 5: The gammatone task

**Files:**
- Create: `src/senselab/audio/tasks/gammatone/__init__.py`, `src/senselab/audio/tasks/gammatone/api.py`
- Test: `src/tests/audio/tasks/gammatone_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `gammatone_filterbank(audio, *, n_channels=40, low_hz=80.0, high_hz=7800.0, hop_s=0.005) -> tuple[np.ndarray, np.ndarray]` returning `(centre_frequencies, energy_db)` of shape `(n_channels,)` and `(n_channels, n_frames)`.

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
    def test_centres_span_the_requested_range_and_increase(self):
        cf = erb_space(80.0, 7800.0, 40)
        assert len(cf) == 40
        assert cf[0] == pytest.approx(80.0, abs=1.0)
        assert cf[-1] == pytest.approx(7800.0, abs=10.0)
        assert np.all(np.diff(cf) > 0)

    def test_spacing_is_wider_at_high_frequency(self):
        cf = erb_space(80.0, 7800.0, 40)
        assert (cf[-1] - cf[-2]) > (cf[1] - cf[0]) * 5


class TestFilterbank:
    def test_a_tone_excites_the_channel_nearest_its_frequency(self):
        cf, energy = gammatone_filterbank(_tone(1000.0))
        loudest = int(np.argmax(energy.mean(axis=1)))
        assert abs(cf[loudest] - 1000.0) < 250.0

    def test_shape_is_channels_by_frames(self):
        cf, energy = gammatone_filterbank(_tone(1000.0, seconds=2.0), n_channels=24, hop_s=0.01)
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

N_CHANNELS = 40
LOW_HZ = 80.0
HIGH_HZ = 7800.0
HOP_S = 0.005


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
    n_channels: int = N_CHANNELS,
    low_hz: float = LOW_HZ,
    high_hz: float = HIGH_HZ,
    hop_s: float = HOP_S,
) -> tuple[np.ndarray, np.ndarray]:
    """Energy per auditory channel over time.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        n_channels: Number of ERB-spaced channels.
        low_hz: Lowest centre frequency.
        high_hz: Highest centre frequency.
        hop_s: Frame hop for the energy summary.

    Returns:
        ``(centre_frequencies, energy_db)`` with shapes ``(n_channels,)`` and ``(n_channels, n_frames)``.
        ``energy_db`` is relative to the bank's own maximum.
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
    db = 20.0 * np.log10(out + 1e-10)
    return cf, db - db.max()
```

And `__init__.py`:

```python
"""Gammatone filterbank."""

from senselab.audio.tasks.gammatone.api import HIGH_HZ, HOP_S, LOW_HZ, N_CHANNELS, erb_space, gammatone_filterbank

__all__ = ["HIGH_HZ", "HOP_S", "LOW_HZ", "N_CHANNELS", "erb_space", "gammatone_filterbank"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/gammatone_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/gammatone/ src/tests/audio/tasks/gammatone_test.py
git add src/senselab/audio/tasks/gammatone/ src/tests/audio/tasks/gammatone_test.py
git commit -m "feat(gammatone): ERB-spaced auditory filterbank

Forty channels from 80 to 7800 Hz via scipy.signal.gammatone, summarised on a 5 ms hop.
Tested by exciting a single channel with a pure tone."
```

---

### Task 6: The HeAR whole-span buffer

**Why:** `capability-map.md` found that HeAR's module actively refuses the padded input AIRWAY specifies. The path works only because a buffer of exactly 32000 samples passes its length check. That coincidence needs a named function rather than each caller rediscovering it.

**Files:**
- Modify: `src/senselab/audio/tasks/health_acoustics/hear.py` (add the function; do not change the existing guard)
- Test: `src/tests/audio/tasks/health_acoustics_test.py`

**Interfaces:**
- Consumes: nothing. It takes `start_s`/`end_s` as floats, so a caller may pass a `Span`'s fields or any other pair — no dependency on Task 4.
- Produces: `span_to_hear_buffer(audio, start_s, end_s, *, placement="centre") -> Audio` returning exactly 2 s at the input rate.

- [ ] **Step 1: Write the failing tests**

```python
def test_span_to_hear_buffer_is_exactly_two_seconds():
    import numpy as np

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

    sr = 16000
    audio = Audio(waveform=np.random.default_rng(0).standard_normal((1, 5 * sr)).astype("float32") * 0.1,
                  sampling_rate=sr)
    buf = span_to_hear_buffer(audio, 1.0, 1.35)
    assert buf.waveform.shape[-1] == 2 * sr
    assert buf.sampling_rate == sr


def test_span_to_hear_buffer_centres_the_span_and_zeroes_the_rest():
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


def test_a_span_longer_than_two_seconds_is_refused():
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

Run: `uv run pytest src/tests/audio/tasks/health_acoustics_test.py -k span_to_hear -v`
Expected: FAIL — `ImportError: cannot import name 'span_to_hear_buffer'`.

- [ ] **Step 3: Implement in `hear.py`**

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
        ValueError: If the span is longer than 2 s, or ``placement`` is not one of the three.
    """
    sr = audio.sampling_rate
    want = 2 * sr
    x = np.asarray(audio.waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    segment = x[int(start_s * sr) : int(end_s * sr)]
    if len(segment) > want:
        raise ValueError(
            f"span {start_s:.3f}-{end_s:.3f}s is {len(segment) / sr:.3f}s, longer than the 2 s the detector "
            "accepts. Split it or classify a sub-span."
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

Run: `uv run pytest src/tests/audio/tasks/health_acoustics_test.py -v`
Expected: the three new tests PASS and the existing ones still pass.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/health_acoustics/hear.py src/tests/audio/tasks/health_acoustics_test.py
git add src/senselab/audio/tasks/health_acoustics/hear.py src/tests/audio/tasks/health_acoustics_test.py
git commit -m "feat(hear): name the whole-span buffer instead of rediscovering it

The detector accepts exactly 2 s, so classifying a shorter span means placing it in a
silent buffer. That worked only because a buffer of exactly 32000 samples passes the
module's length check; it is now a function with the placement stated and a span longer
than 2 s refused rather than truncated."
```

---

### Task 7: The phonation task

**Files:**
- Create: `src/senselab/audio/tasks/phonation/__init__.py`, `src/senselab/audio/tasks/phonation/api.py`
- Test: `src/tests/audio/tasks/phonation_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `periodicity_track(audio, *, hop_s=0.01, f0_min_hz, f0_max_hz) -> tuple[np.ndarray, np.ndarray]` giving `(periodicity, f0_hz)`; and `period_marks(audio, start_s, end_s, *, f0_min_hz, f0_max_hz) -> list[PeriodMark]`.

`f0_min_hz` and `f0_max_hz` are keyword-only with no default: `benchmarks/voice.md` records that no single range serves both low adult male and infant voices.

- [ ] **Step 1: Write the failing tests**

```python
"""Periodicity and the period-mark point process."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.phonation import PeriodMark, period_marks, periodicity_track

SR = 16000


def _buzz(f0: float, seconds: float = 1.0) -> Audio:
    t = np.arange(int(seconds * SR)) / SR
    wave = sum(0.3 / (h + 1) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(6))
    return Audio(waveform=wave.astype(np.float32)[None, :], sampling_rate=SR)


def _noise(seconds: float = 1.0) -> Audio:
    rng = np.random.default_rng(0)
    return Audio(waveform=(0.1 * rng.standard_normal(int(seconds * SR))).astype(np.float32)[None, :], sampling_rate=SR)


class TestPeriodicity:
    def test_a_buzz_is_periodic_and_noise_is_not(self):
        p_voiced, _ = periodicity_track(_buzz(100.0), f0_min_hz=60.0, f0_max_hz=400.0)
        p_noise, _ = periodicity_track(_noise(), f0_min_hz=60.0, f0_max_hz=400.0)
        assert float(np.median(p_voiced)) > 0.9
        assert float(np.median(p_noise)) < 0.5

    def test_f0_is_recovered(self):
        _, f0 = periodicity_track(_buzz(120.0), f0_min_hz=60.0, f0_max_hz=400.0)
        assert float(np.median(f0)) == pytest.approx(120.0, abs=4.0)

    def test_the_search_range_is_required(self):
        with pytest.raises(TypeError):
            periodicity_track(_buzz(120.0))  # type: ignore[call-arg]


class TestPeriodMarks:
    def test_marks_are_spaced_by_one_period(self):
        marks = period_marks(_buzz(100.0), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0)
        assert len(marks) > 40
        gaps = np.diff([m.time_s for m in marks])
        assert float(np.median(gaps)) == pytest.approx(0.01, abs=0.001)

    def test_each_mark_carries_its_period_and_amplitude(self):
        marks = period_marks(_buzz(100.0), 0.2, 0.4, f0_min_hz=60.0, f0_max_hz=400.0)
        m = marks[len(marks) // 2]
        assert isinstance(m, PeriodMark)
        assert m.period_s == pytest.approx(0.01, abs=0.002)
        assert m.amplitude > 0.0

    def test_noise_yields_no_marks(self):
        assert period_marks(_noise(), 0.2, 0.8, f0_min_hz=60.0, f0_max_hz=400.0) == []
```

- [ ] **Step 2: Run and watch them fail**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py -v`
Expected: FAIL at collection.

- [ ] **Step 3: Implement `api.py`**

```python
"""Periodicity, F0 candidates, and glottal period marks as a point process."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from senselab.audio.data_structures import Audio

HOP_S = 0.01
PERIODICITY_FOR_MARKS = 0.5


@dataclass(frozen=True)
class PeriodMark:
    """One glottal period boundary.

    Attributes:
        time_s: Where the boundary sits.
        period_s: Duration of the period beginning here.
        amplitude: Peak absolute amplitude within the period.
        peak: The normalised autocorrelation value that placed it.
    """

    time_s: float
    period_s: float
    amplitude: float
    peak: float


def _mono(audio: Audio) -> np.ndarray:
    x = np.asarray(audio.waveform, dtype=np.float64)
    return x.mean(axis=0) if x.ndim > 1 else x


def _autocorr_peak(frame: np.ndarray, sr: int, f0_min_hz: float, f0_max_hz: float) -> tuple[float, float]:
    frame = frame - frame.mean()
    if not np.any(frame):
        return 0.0, 0.0
    ac = np.correlate(frame, frame, "full")[len(frame) - 1 :]
    if ac[0] <= 0:
        return 0.0, 0.0
    ac = ac / ac[0]
    lo, hi = int(sr / f0_max_hz), min(int(sr / f0_min_hz), len(ac) - 1)
    if hi <= lo:
        return 0.0, 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    return float(ac[lag]), sr / lag


def periodicity_track(
    audio: Audio, *, f0_min_hz: float, f0_max_hz: float, hop_s: float = HOP_S
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised autocorrelation peak and its F0, per frame.

    Args:
        audio: Mono audio. A multi-channel input is averaged.
        f0_min_hz: Lowest F0 to search. No default: no single range serves both low adult and infant voices.
        f0_max_hz: Highest F0 to search.
        hop_s: Frame hop. The frame is three times the longest searched period.

    Returns:
        ``(periodicity, f0_hz)``, one value per frame. ``f0_hz`` is meaningless where periodicity is low
        and is returned alongside it so the two cannot be separated.
    """
    x = _mono(audio)
    sr = audio.sampling_rate
    frame_len = int(3 * sr / f0_min_hz)
    hop = max(1, int(hop_s * sr))
    starts = range(0, max(1, len(x) - frame_len), hop)
    pairs = [_autocorr_peak(x[s : s + frame_len], sr, f0_min_hz, f0_max_hz) for s in starts]
    if not pairs:
        return np.zeros(0), np.zeros(0)
    per, f0 = zip(*pairs)
    return np.asarray(per), np.asarray(f0)


def period_marks(
    audio: Audio,
    start_s: float,
    end_s: float,
    *,
    f0_min_hz: float,
    f0_max_hz: float,
    periodicity_floor: float = PERIODICITY_FOR_MARKS,
) -> list[PeriodMark]:
    """Place glottal period boundaries inside one span.

    Args:
        audio: The recording.
        start_s: Span onset.
        end_s: Span offset.
        f0_min_hz: Lowest F0 to search.
        f0_max_hz: Highest F0 to search.
        periodicity_floor: Below this autocorrelation peak, no mark is placed.

    Returns:
        Marks in time order. Empty when the span never clears ``periodicity_floor`` — absent, not zero.
    """
    x = _mono(audio)
    sr = audio.sampling_rate
    i0, i1 = int(start_s * sr), int(end_s * sr)
    frame_len = int(3 * sr / f0_min_hz)
    marks: list[PeriodMark] = []
    cursor = i0
    while cursor + frame_len < i1:
        peak, f0 = _autocorr_peak(x[cursor : cursor + frame_len], sr, f0_min_hz, f0_max_hz)
        if peak < periodicity_floor or f0 <= 0:
            cursor += max(1, int(0.005 * sr))
            continue
        period = int(sr / f0)
        marks.append(
            PeriodMark(
                time_s=cursor / sr,
                period_s=period / sr,
                amplitude=float(np.abs(x[cursor : cursor + period]).max()),
                peak=peak,
            )
        )
        cursor += period
    return marks
```

And `__init__.py`:

```python
"""Periodicity and period marks."""

from senselab.audio.tasks.phonation.api import HOP_S, PeriodMark, period_marks, periodicity_track

__all__ = ["HOP_S", "PeriodMark", "period_marks", "periodicity_track"]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest src/tests/audio/tasks/phonation_test.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
uv run ruff format src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
git add src/senselab/audio/tasks/phonation/ src/tests/audio/tasks/phonation_test.py
git commit -m "feat(phonation): periodicity track and period marks as a point process

Period marks rather than an F0 contour: at 87 Hz one period is 11.4 ms, so a fixed-hop
contour is coarser than what it samples and jitter is unrecoverable from it. The F0
search range is keyword-only with no default, because no single range serves both low
adult and infant voices."
```

---

### Task 8: The redaction task

**Files:**
- Create: `src/senselab/audio/tasks/redaction/__init__.py`, `src/senselab/audio/tasks/redaction/api.py`
- Test: `src/tests/audio/tasks/redaction_test.py`

**Interfaces:**
- Consumes: `PiiSpan` (Task 1) for its `start_s`/`end_s`.
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
    def test_padding_is_required(self):
        with pytest.raises(TypeError):
            plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")])  # type: ignore[call-arg]

    def test_extents_are_padded_outward_on_both_sides(self):
        (out,) = plan_redactions([RedactionExtent(1.0, 1.2, "PERSON")], padding_ms=100)
        assert out.start == pytest.approx(0.9)
        assert out.end == pytest.approx(1.3)

    def test_padding_never_produces_a_negative_start(self):
        (out,) = plan_redactions([RedactionExtent(0.02, 0.1, "PERSON")], padding_ms=100)
        assert out.start == 0.0

    def test_extents_that_overlap_after_padding_are_merged(self):
        out = plan_redactions(
            [RedactionExtent(1.0, 1.1, "PERSON"), RedactionExtent(1.25, 1.35, "DATE")], padding_ms=100
        )
        assert len(out) == 1, "an audible sliver between two redactions is a leak"
        assert out[0].category == "PERSON+DATE"


class TestApplying:
    def test_the_redacted_region_is_silent_and_the_rest_is_untouched(self):
        x = np.ones((1, 3 * SR), dtype="float32")
        audio = Audio(waveform=x, sampling_rate=SR)
        out = apply_redactions(audio, [RedactionExtent(1.0, 1.5, "PERSON")])
        w = np.asarray(out.waveform).squeeze()
        assert np.all(w[int(1.0 * SR) : int(1.5 * SR)] == 0.0)
        assert np.all(w[: int(1.0 * SR)] == 1.0)
        assert np.all(w[int(1.5 * SR) :] == 1.0)

    def test_duration_is_preserved(self):
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

### Task 9: The triage vocabulary

**Files:**
- Create: `src/senselab/audio/workflows/triage/__init__.py`, `src/senselab/audio/workflows/triage/vocabulary.py`
- Test: `src/tests/audio/workflows/triage/vocabulary_test.py`, `src/tests/audio/workflows/triage/__init__.py`

**Interfaces:**
- Consumes: nothing. The fold takes plain verdicts and mappings, so it is testable without a store; the nodes are what read and write `ElementStore` from Task 2.
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
    def test_a_branch_failing_on_an_absent_kind_is_expected(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway"),
                           _v("SPEECH", Outcome.FAIL, "speech")],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.PASS


class TestContradictions:
    def test_present_kind_with_a_failing_branch_flags(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("SPEECH", Outcome.FAIL, "speech")],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert any("contradiction" in r.why for r in out.reasons)

    def test_absent_kind_with_a_passing_branch_flags_and_resolves_the_kind(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert out.kinds["airway"] is KindState.PRESENT


class TestNeverRan:
    def test_a_skipped_branch_on_a_present_kind_flags(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.FLAG

    def test_a_skipped_branch_on_an_absent_kind_is_expected(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.PASS


class TestOrdering:
    def test_admit_failing_wins_over_everything(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.FAIL), _v("AIRWAY", Outcome.FLAG, "airway")],
            kind_predictions={},
            ran={"ADMIT": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[0].node == "ADMIT"

    def test_every_kind_absent_is_a_different_fail_from_admit(self):
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"airway": KindState.ABSENT, "speech": KindState.ABSENT},
            ran={},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[-1].node != "ADMIT"

    def test_reasons_carry_every_contribution_not_only_the_deciding_one(self):
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
uv run pytest src/tests/utils/element_store_test.py src/tests/audio/tasks/ src/tests/audio/workflows/triage/ src/tests/text/tasks/ -q
uv run ruff check src/senselab src/tests
uv run mypy src/senselab/utils/element_store.py src/senselab/audio/workflows/triage/ src/senselab/audio/tasks/envelope src/senselab/audio/tasks/spans src/senselab/audio/tasks/gammatone src/senselab/audio/tasks/phonation src/senselab/audio/tasks/redaction
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

### Task 10: The disruptions task

**Files:**
- Create: `src/senselab/audio/tasks/disruptions/__init__.py`, `src/senselab/audio/tasks/disruptions/api.py`
- Test: `src/tests/audio/tasks/disruptions_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Disruptions` (frozen dataclass) and `detect_disruptions(audio, start_s, end_s, *, clip_headroom=0.999, min_clip_run=3, min_dropout_ms=10.0, discontinuity_threshold=0.5) -> Disruptions`.

Derivation: `branch-speech.md` step 8. Counts and extents, never a score. The four parameters are
conventional rather than fitted, and the docstring says so; the *tolerance* — how much is too much — has
no value and is not this function's business.

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


class TestClipping:
    def test_a_clean_tone_has_no_clipping(self):
        d = detect_disruptions(_audio(_tone()), 0.0, 1.0)
        assert d.clipped_runs == 0
        assert d.clipped_s == 0.0

    def test_a_saturated_tone_is_clipped(self):
        d = detect_disruptions(_audio(np.clip(_tone(amp=2.0), -1.0, 1.0)), 0.0, 1.0)
        assert d.clipped_runs >= 100, "200 Hz for 1 s saturates on every half cycle"
        assert d.clipped_s > 0.1

    def test_a_single_full_scale_sample_is_not_clipping(self):
        x = _tone()
        x[5000] = 1.0
        assert detect_disruptions(_audio(x), 0.0, 1.0).clipped_runs == 0


class TestDropouts:
    def test_a_zero_run_is_a_dropout(self):
        x = _tone()
        x[4000:8000] = 0.0
        d = detect_disruptions(_audio(x), 0.0, 1.0)
        assert d.dropout_runs == 1
        assert d.dropout_s == pytest.approx(4000 / SR, abs=0.002)

    def test_a_run_shorter_than_the_minimum_is_not_a_dropout(self):
        x = _tone()
        x[4000:4020] = 0.0
        assert detect_disruptions(_audio(x), 0.0, 1.0).dropout_runs == 0


class TestDiscontinuities:
    def test_a_step_is_a_discontinuity(self):
        x = _tone(amp=0.1)
        x[8000:] += 0.9
        assert detect_disruptions(_audio(x), 0.0, 1.0).discontinuities >= 1

    def test_a_smooth_tone_has_none(self):
        assert detect_disruptions(_audio(_tone()), 0.0, 1.0).discontinuities == 0


class TestDcOffset:
    def test_a_bias_is_reported(self):
        d = detect_disruptions(_audio(_tone() + 0.2), 0.0, 1.0)
        assert d.dc_offset == pytest.approx(0.2, abs=0.01)

    def test_a_centred_signal_reports_near_zero(self):
        assert abs(detect_disruptions(_audio(_tone()), 0.0, 1.0).dc_offset) < 0.01


class TestScoping:
    def test_only_the_requested_span_is_measured(self):
        x = _tone(seconds=3.0)
        x[: SR] = np.clip(_tone(amp=2.0)[: SR], -1.0, 1.0)
        assert detect_disruptions(_audio(x), 1.5, 2.5).clipped_runs == 0
        assert detect_disruptions(_audio(x), 0.0, 1.0).clipped_runs > 0

    def test_a_clean_span_reports_zero_rather_than_nothing(self):
        d = detect_disruptions(_audio(_tone()), 0.0, 1.0)
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

CLIP_HEADROOM = 0.999
MIN_CLIP_RUN = 3
MIN_DROPOUT_MS = 10.0
DISCONTINUITY_THRESHOLD = 0.5


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
    clip_headroom: float = CLIP_HEADROOM,
    min_clip_run: int = MIN_CLIP_RUN,
    min_dropout_ms: float = MIN_DROPOUT_MS,
    discontinuity_threshold: float = DISCONTINUITY_THRESHOLD,
) -> Disruptions:
    """Measure disruptions inside one span.

    The four parameters are conventional rather than fitted: a single sample at full scale is not
    clipping, which is why ``min_clip_run`` exists, and the values are the usual ones rather than values
    derived from labelled verdicts.

    Args:
        audio: The recording. A multi-channel input is averaged.
        start_s: Span onset.
        end_s: Span offset.
        clip_headroom: A sample at or beyond this absolute value counts as clipped.
        min_clip_run: Shortest run of clipped samples that counts as a clipping event.
        min_dropout_ms: Shortest run of exact zeros that counts as a dropout.
        discontinuity_threshold: Absolute sample-to-sample jump that counts as a discontinuity.

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

from senselab.audio.tasks.disruptions.api import (
    CLIP_HEADROOM,
    DISCONTINUITY_THRESHOLD,
    MIN_CLIP_RUN,
    MIN_DROPOUT_MS,
    Disruptions,
    detect_disruptions,
)

__all__ = [
    "CLIP_HEADROOM",
    "DISCONTINUITY_THRESHOLD",
    "MIN_CLIP_RUN",
    "MIN_DROPOUT_MS",
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
-- which is a different statement from a span nobody measured. How much is too much is a
tolerance nobody has derived and is the caller's decision."
```

---

## What this plan does not build

- **The nine node implementations.** They consume everything above and are a second plan.
- **The Nextflow orchestration**, which already exists at `specs/20260817-triage-workflow-dag/nextflow/` and lints clean with a passing stub run.
- **Anything requiring an undecided parameter.** SQUIM thresholds, the phonation gate's floors and the redaction margin have no justified value, so the functions above take them as keyword-only arguments with no default. A caller must supply one, and `benchmarks/open.md` records what would settle each.

## Known deviations from the design, to resolve before the node plan

- `resample_audios` designs its anti-alias filter at the target rate and applies it at the source rate. Unverified as a defect; a sweep test would settle it.
- `classify_audios` defaults to `top_k=5` for YAMNet, so `Silence` can be absent rather than low, and applies softmax to AST, which is wrong for multi-label AudioSet. The node plan must pass explicit arguments.
- `MossFormer2_SS_16K` RMS-normalises its input to −25 dBFS, so no dBFS-referenced measurement on a separated stream is comparable with one on the recording.
