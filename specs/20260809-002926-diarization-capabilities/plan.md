# Diarization Capabilities Implementation Plan

> **Verification status (2026-08-13, commit `ad4fffa2`):** every task below was verified complete against the code on branch `feat/diarization-backends`. Boxes are ticked at *task-deliverable* granularity — the deliverable was confirmed present in the tree, not each TDD step observed independently.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every diarization backend a declared, machine-readable record of which `ScriptLine` fields it populates, what its `speaker` labels denote, and what its speaker ceiling is — so a consumer can branch before paying for a 16 GB download.

**Architecture:** A frozen `DiarizationCapabilities` dataclass in its own light module (no backend imports). Each backend module declares a `CAPABILITIES` constant beside itself. `api.py`, which already imports all six backends, maps model-id prefix → record and exposes `capabilities_for()`. `ScriptLine` does not change.

**Tech Stack:** Python 3.12, `uv`, pytest (serial), pydantic-free plain `dataclasses`.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **`ScriptLine` must not change.** It already provides a uniform key set and is shared by ASR, forced alignment and the workflow's harvesters.
- **No `audio_analysis` wiring.** The existing `ROLE_LABEL_ONLY_PREFIXES` list keeps working; migrating those guards to read `speaker_label_kind` is a separate change.
- **No runtime conformance check.** Do not validate a backend's output against its declaration on every call.
- **No rewriting of speaker label values.** Only the key structure and its declared meaning are harmonised.
- **`max_speakers=None` means unmeasured, not unlimited.** Only child-adult (2) and Sortformer (4) have known ceilings. Do not guess the others from model cards.
- **`labels_stable_across_files` defaults to `False`** — the safe direction. Only DiariZen is measured.
- **Every Python command runs through `uv run`.** Never bare `python` or `pip`.
- **Never run `pytest -n auto`** or any `-n` flag. Serial only, scoped to what changed.
- **Never `git add -A` unqualified.** Use a pathspec: `git add -A -- src/ docs/ scripts/`.
- **No test may construct an `HFModel`** without `monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)` — an unmocked construction triggers a real `snapshot_download`.
- **Test isolation via `monkeypatch`**, never by clearing a module-level cache.
- **Test docstrings in Google style**: one-line summary, blank line, body, closing quotes on their own line. Ruff enforces `D205`/`D209` here.
- **Check machine load before running tests** (`uptime` against 10 cores). This box is often busy; a run showing minutes elapsed against seconds of CPU is contention, not a hang.

## Measured values (from the H100 run — use these exact values)

| Backend | model-id prefix | `populates_text` | `speaker_label_kind` | `labels_stable_across_files` | `max_speakers` | `honors_speaker_hints` |
|---|---|---|---|---|---|---|
| Pyannote | *(default branch, no prefix)* | `False` | `"identity"` | `False` | `None` | `True` |
| NeMo Sortformer | `nvidia/diar_sortformer` | `False` | `"identity"` | `False` | `4` | `False` |
| VibeVoice-ASR-HF | `microsoft/VibeVoice-ASR` | `True` | `"identity"` | `False` | `None` | `False` |
| USC-SAIL child-adult | `AlexXu811/whisper-child-adult` | `False` | `"role"` | `False` | `2` | `False` |
| MOSS-Transcribe-Diarize | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | `True` | `"identity"` | `False` | `None` | `False` |
| DiariZen | `BUT-FIT/diarizen` | `False` | `"identity"` | `False` | `None` | `False` |

Pyannote is the only backend honouring `num_speakers`/`min_speakers`/`max_speakers`: the other five all call `_warn_if_speaker_hints_ignored` (`api.py:167,174,183,190,199`).

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/senselab/audio/tasks/speaker_diarization/capabilities.py` | The frozen dataclass and its validation. No backend imports — must stay light enough to import anywhere. | Create |
| `src/senselab/audio/tasks/speaker_diarization/{pyannote,nvidia,vibevoice,child_adult,moss,diarizen}.py` | Each declares its own `CAPABILITIES` constant | Modify |
| `src/senselab/audio/tasks/speaker_diarization/api.py` | Prefix → record mapping; `capabilities_for()` | Modify |
| `src/senselab/model_registry.yaml` | `capabilities:` block per diarization entry | Modify |
| `scripts/generate_model_registry.py` | Render the new columns | Modify |
| `src/senselab/model_registry.md` | Regenerated, never hand-edited | Modify |
| `src/tests/audio/tasks/speaker_diarization_capabilities_test.py` | All capability tests | Create |

---

### Task 1: The `DiarizationCapabilities` record

**Files:**
- Create: `src/senselab/audio/tasks/speaker_diarization/capabilities.py`
- Test: `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `DiarizationCapabilities(populates_text: bool, speaker_label_kind: Literal["identity", "role"], labels_stable_across_files: bool, max_speakers: int | None, honors_speaker_hints: bool)` — a frozen dataclass. Task 2 instantiates it six times.

- [x] **Step 1: Write the failing test**

Create `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`:

```python
"""Declared capabilities for the diarization backends."""

import dataclasses

import pytest

from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities


def test_record_is_frozen() -> None:
    """A capability record is a declaration, not mutable state.

    If a caller could mutate one, two callers would disagree about what a backend
    can do, and the disagreement would depend on import order.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.max_speakers = 4  # type: ignore[misc]


def test_max_speakers_none_is_allowed_and_means_unmeasured() -> None:
    """None is 'nobody has measured this', not 'unlimited'.

    Four of six backends have no published or measured ceiling. Encoding that as
    None keeps it distinguishable from a real limit, so the NeMo probe can fill it
    in later without anyone having guessed in the meantime.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    assert caps.max_speakers is None


def test_max_speakers_must_be_at_least_one_when_given() -> None:
    """A ceiling of zero or less describes nothing that can diarize.

    Catches a typo or an off-by-one in a declaration at construction time rather
    than as a confusing empty result much later.
    """
    with pytest.raises(ValueError, match="max_speakers"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="identity",
            labels_stable_across_files=False,
            max_speakers=0,
            honors_speaker_hints=False,
        )


def test_speaker_label_kind_rejects_an_unknown_value() -> None:
    """Only 'identity' and 'role' are meaningful.

    The distinction decides whether labels may reach embedding clustering, so a
    third value silently defaulting to one branch would be a correctness bug.
    """
    with pytest.raises(ValueError, match="speaker_label_kind"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="cluster",  # type: ignore[arg-type]
            labels_stable_across_files=False,
            max_speakers=None,
            honors_speaker_hints=False,
        )
```

- [x] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'senselab.audio.tasks.speaker_diarization.capabilities'`.

- [x] **Step 3: Implement**

Create `src/senselab/audio/tasks/speaker_diarization/capabilities.py`:

```python
"""What each diarization backend actually provides.

Six backends reach :func:`diarize_audios` and share a return type while disagreeing
about almost everything else. Measured across three recordings on an H100: ``text`` is
populated by exactly two of six; ``speaker`` denotes an identity for five and a *role*
for the USC-SAIL child-adult classifier; DiariZen's VBx clustering numbers speakers per
audio, so the same run produced ``['1','2']`` for one file and ``['0','0','1','0']`` for
another. None of that was discoverable without running the model.

This module declares it instead. The record is static rather than returned per call
because the question a caller needs answered — "can this give me more than two
speakers?" — has to be answerable *before* paying for a 16 GB download and a GPU minute.

``ScriptLine`` deliberately does not change. It already provides a uniform key set, and
it is shared by ASR, forced alignment and the workflow's harvesters, so reshaping it for
a diarization-specific gap would be the wrong blast radius.

This module imports no backend, so it stays cheap to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

SpeakerLabelKind = Literal["identity", "role"]

_VALID_LABEL_KINDS = ("identity", "role")


@dataclass(frozen=True)
class DiarizationCapabilities:
    """What one diarization backend populates, and what its labels mean.

    Attributes:
        populates_text: Whether the backend fills ``ScriptLine.text``. Without this a
            consumer cannot tell "this backend does not transcribe" from "this segment
            had no words" — both look like ``text=None``.
        speaker_label_kind: ``"identity"`` when ``speaker`` names *who* is talking,
            ``"role"`` when it names *what kind* of talker (child-adult emits
            CHILD/ADULT/OVERLAP). Role labels must not reach embedding clustering: a
            per-role centroid blends distinct speakers under one label.
        labels_stable_across_files: Whether label ``"1"`` in one file denotes the same
            speaker as ``"1"`` in another. False for any backend that numbers per audio.
        max_speakers: The backend's ceiling, or ``None`` when nobody has measured it.
            ``None`` does **not** mean unlimited.
        honors_speaker_hints: Whether ``num_speakers``/``min_speakers``/``max_speakers``
            passed to :func:`diarize_audios` do anything. Five of six ignore them.
    """

    populates_text: bool
    speaker_label_kind: SpeakerLabelKind
    labels_stable_across_files: bool
    max_speakers: Optional[int]
    honors_speaker_hints: bool

    def __post_init__(self) -> None:
        """Reject declarations that cannot describe a real backend."""
        if self.speaker_label_kind not in _VALID_LABEL_KINDS:
            raise ValueError(
                f"speaker_label_kind must be one of {_VALID_LABEL_KINDS}, got {self.speaker_label_kind!r}. "
                "The distinction decides whether these labels may reach embedding clustering."
            )
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError(
                f"max_speakers must be >= 1 or None (unmeasured), got {self.max_speakers!r}. "
                "None means nobody has measured the ceiling; it does not mean unlimited."
            )
```

- [x] **Step 4: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
```

Expected: PASS, 4 tests.

- [x] **Step 5: Lint, type-check, commit**

```bash
uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/senselab/
git add -A -- src/
git commit -m "feat(speaker_diarization): DiarizationCapabilities record

Declares what a backend populates and what its speaker labels denote. Frozen,
because two callers disagreeing about a backend's abilities depending on import
order is worse than no declaration at all.

max_speakers=None means unmeasured, not unlimited, and the validator says so —
four of six backends have no measured ceiling yet."
```

---

### Task 2: Declare all six backends and expose the lookup

**Files:**
- Modify: `src/senselab/audio/tasks/speaker_diarization/pyannote.py`, `nvidia.py`, `vibevoice.py`, `child_adult.py`, `moss.py`, `diarizen.py`
- Modify: `src/senselab/audio/tasks/speaker_diarization/api.py`
- Test: `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`

**Interfaces:**
- Consumes: `DiarizationCapabilities` from Task 1.
- Produces: `CAPABILITIES: DiarizationCapabilities` in each of the six backend modules, and in `api.py`:
  `capabilities_for(model_id: str) -> DiarizationCapabilities` — returns the record for whichever backend `diarize_audios` would dispatch that id to, falling back to Pyannote's record for any unmatched id, exactly as the dispatch itself falls back.

- [x] **Step 1: Write the failing tests**

Append to `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`:

```python
from senselab.audio.tasks.speaker_diarization.api import ROLE_LABEL_ONLY_PREFIXES, capabilities_for

_ALL_BACKEND_IDS = (
    "pyannote/speaker-diarization-community-1",
    "nvidia/diar_sortformer_4spk-v1",
    "microsoft/VibeVoice-ASR-HF",
    "AlexXu811/whisper-child-adult",
    "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "BUT-FIT/diarizen-wavlm-large-s80-md",
)


@pytest.mark.parametrize("model_id", _ALL_BACKEND_IDS)
def test_every_dispatchable_backend_declares_capabilities(model_id: str) -> None:
    """A backend reachable from diarize_audios must say what it provides.

    This is the test that stops a seventh backend being added without declaring
    itself, which is how the current situation arose: six backends, no declarations,
    and the only way to learn the differences was to run each one.
    """
    caps = capabilities_for(model_id)
    assert isinstance(caps, DiarizationCapabilities)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("microsoft/VibeVoice-ASR-HF", True),
        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", True),
        ("BUT-FIT/diarizen-wavlm-large-s80-md", False),
        ("AlexXu811/whisper-child-adult", False),
        ("pyannote/speaker-diarization-community-1", False),
        ("nvidia/diar_sortformer_4spk-v1", False),
    ],
)
def test_populates_text_matches_what_was_measured(model_id: str, expected: bool) -> None:
    """Exactly the two joint ASR+diarization backends fill `text`.

    Measured on an H100: VibeVoice returned 7 segments all carrying text, MOSS 6,
    while DiariZen (10) and child-adult (19) returned none. A consumer reading
    text=None otherwise cannot tell a backend limitation from an empty segment.
    """
    assert capabilities_for(model_id).populates_text is expected


def test_child_adult_is_a_two_speaker_role_classifier() -> None:
    """Count and kind are separate facts about the same backend.

    child-adult can only ever emit CHILD/ADULT, making it a 2-speaker diarizer by
    count. But its labels denote roles, which is what decides they must not reach
    embedding clustering. A 2-speaker identity diarizer would share the ceiling and
    need different handling, so one field cannot carry both.
    """
    caps = capabilities_for("AlexXu811/whisper-child-adult")
    assert caps.max_speakers == 2
    assert caps.speaker_label_kind == "role"


def test_sortformer_declares_the_ceiling_in_its_own_name() -> None:
    """`diar_sortformer_4spk` tops out at four."""
    assert capabilities_for("nvidia/diar_sortformer_4spk-v1").max_speakers == 4


@pytest.mark.parametrize(
    "model_id",
    [
        "pyannote/speaker-diarization-community-1",
        "microsoft/VibeVoice-ASR-HF",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "BUT-FIT/diarizen-wavlm-large-s80-md",
    ],
)
def test_unmeasured_ceilings_are_none_not_guessed(model_id: str) -> None:
    """Four backends have no measured ceiling, so they declare None.

    None means unmeasured. The NeMo synthetic-speaker probe fills these in with a
    number that carries its measurement; a value copied from a model card would be
    exactly the unfitted literal this repo's conventions warn against.
    """
    assert capabilities_for(model_id).max_speakers is None


def test_only_pyannote_honors_speaker_hints() -> None:
    """Five of six ignore num_speakers, and api.py already warns when they do.

    Declaring it lets a caller avoid passing a hint that will be dropped, rather
    than discovering it in a log line after the run.
    """
    assert capabilities_for("pyannote/speaker-diarization-community-1").honors_speaker_hints is True
    for model_id in _ALL_BACKEND_IDS:
        if model_id.startswith("pyannote/"):
            continue
        assert capabilities_for(model_id).honors_speaker_hints is False


def test_diarizen_labels_are_not_stable_across_files() -> None:
    """VBx clusters per audio, so a label means nothing outside its own file.

    Measured: the same run produced ['1','2'] for one recording and
    ['0','0','1','0'] for another. A consumer joining on label across files would
    silently merge unrelated speakers.
    """
    assert capabilities_for("BUT-FIT/diarizen-wavlm-large-s80-md").labels_stable_across_files is False


def test_role_kind_agrees_with_the_existing_prefix_list() -> None:
    """The new declaration and the old ROLE_LABEL_ONLY_PREFIXES must not diverge.

    Both encode "these labels are roles, keep them out of the identity axis". While
    both exist, a backend appearing in one and not the other is a latent bug — the
    audio_analysis guards read the prefix list, and future code will read the record.
    """
    for model_id in _ALL_BACKEND_IDS:
        in_prefix_list = any(model_id.startswith(p) for p in ROLE_LABEL_ONLY_PREFIXES)
        is_role = capabilities_for(model_id).speaker_label_kind == "role"
        assert in_prefix_list == is_role, f"{model_id}: prefix list says {in_prefix_list}, record says {is_role}"


def test_an_unknown_model_id_falls_back_like_the_dispatch_does() -> None:
    """An unmatched id resolves to Pyannote, mirroring diarize_audios' own fallback.

    Returning None instead would make every caller write the same None-check for a
    case the dispatch itself treats as ordinary.
    """
    assert capabilities_for("some/unknown-diarizer").honors_speaker_hints is True
```

- [x] **Step 2: Run them and watch them fail**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
```

Expected: FAIL — `ImportError: cannot import name 'capabilities_for'`.

- [x] **Step 3: Declare `CAPABILITIES` in each backend module**

Add to each module, near its other module-level constants. Use the exact values from the "Measured values" table above. For example, in `vibevoice.py`:

```python
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities

CAPABILITIES = DiarizationCapabilities(
    populates_text=True,  # joint ASR+diarization: measured 7/7 segments carried text
    speaker_label_kind="identity",
    labels_stable_across_files=False,  # per-audio numbering; not measured otherwise
    max_speakers=None,  # unmeasured — pending the NeMo synthetic-speaker probe
    honors_speaker_hints=False,  # api.py warns that num_speakers is dropped here
)
```

The other five, in full — every non-obvious value carries a comment saying what it was measured from, which is this repo's convention:

```python
# child_adult.py
CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="role",  # CHILD/ADULT/OVERLAP name a role, not a speaker
    labels_stable_across_files=False,
    max_speakers=2,  # it can only ever emit CHILD and ADULT
    honors_speaker_hints=False,
)

# moss.py
CAPABILITIES = DiarizationCapabilities(
    populates_text=True,  # joint ASR+diarization: measured 6/6 segments carried text
    speaker_label_kind="identity",  # emits S01/S02 tags parsed from its transcript
    labels_stable_across_files=False,
    max_speakers=None,  # unmeasured — pending the NeMo synthetic-speaker probe
    honors_speaker_hints=False,
)

# diarizen.py
CAPABILITIES = DiarizationCapabilities(
    populates_text=False,  # measured: 10/10 segments had no text
    speaker_label_kind="identity",
    # Measured: VBx clusters per audio, so the same run gave ['1','2'] for one file
    # and ['0','0','1','0'] for another. A label means nothing outside its own file.
    labels_stable_across_files=False,
    max_speakers=None,  # unmeasured — pending the NeMo synthetic-speaker probe
    honors_speaker_hints=False,
)

# nvidia.py  (NeMo Sortformer)
CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="identity",
    labels_stable_across_files=False,
    max_speakers=4,  # declared by the checkpoint's own name: diar_sortformer_4spk
    honors_speaker_hints=False,
)

# pyannote.py
CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="identity",  # SPEAKER_00, SPEAKER_01, ...
    labels_stable_across_files=False,
    max_speakers=None,  # unmeasured — pending the NeMo synthetic-speaker probe
    honors_speaker_hints=True,  # the only backend that acts on num_speakers
)
```

- [x] **Step 4: Add the lookup to `api.py`**

`api.py` imports the backends' **functions**, not their modules (verified: `from ...child_adult import diarize_audios_with_child_adult`, and so on for all six). So import the constants directly, with aliases, in the same style:

```python
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.audio.tasks.speaker_diarization.child_adult import CAPABILITIES as _CHILD_ADULT_CAPS
from senselab.audio.tasks.speaker_diarization.diarizen import CAPABILITIES as _DIARIZEN_CAPS
from senselab.audio.tasks.speaker_diarization.moss import CAPABILITIES as _MOSS_CAPS
from senselab.audio.tasks.speaker_diarization.nvidia import CAPABILITIES as _SORTFORMER_CAPS
from senselab.audio.tasks.speaker_diarization.pyannote import CAPABILITIES as _PYANNOTE_CAPS
from senselab.audio.tasks.speaker_diarization.vibevoice import CAPABILITIES as _VIBEVOICE_CAPS
```

There is no `_SORTFORMER_PREFIXES` constant yet — the Sortformer branch checks its prefix inline. Add one beside the existing four so the dispatch and the lookup share it:

```python
_SORTFORMER_PREFIXES = ("nvidia/diar_sortformer",)
```

and use it in the dispatch too, replacing the inline literal, so the two cannot drift.

Then:

```python
_CAPABILITIES_BY_PREFIX: tuple[tuple[tuple[str, ...], DiarizationCapabilities], ...] = (
    (_SORTFORMER_PREFIXES, _SORTFORMER_CAPS),
    (_VIBEVOICE_PREFIXES, _VIBEVOICE_CAPS),
    (_CHILD_ADULT_PREFIXES, _CHILD_ADULT_CAPS),
    (_MOSS_PREFIXES, _MOSS_CAPS),
    (_DIARIZEN_PREFIXES, _DIARIZEN_CAPS),
)


def capabilities_for(model_id: str) -> DiarizationCapabilities:
    """Return what the backend handling ``model_id`` provides.

    Mirrors :func:`diarize_audios`' own dispatch, including its fallback: an id
    matching no prefix resolves to Pyannote, because that is the backend that would
    actually run it. Returning ``None`` instead would make every caller write the
    same check for a case the dispatch treats as ordinary.
    """
    for prefixes, caps in _CAPABILITIES_BY_PREFIX:
        if any(model_id.startswith(p) for p in prefixes):
            return caps
    return _PYANNOTE_CAPS
```

- [x] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
```

Expected: PASS, 20 tests.

- [x] **Step 6: Confirm nothing else regressed**

```bash
uptime   # check load first; this box is often busy
uv run pytest src/tests/audio/tasks/speaker_diarization_test.py -q
```

Expected: unchanged from before this task (26 passed, 6 skipped).

- [x] **Step 7: Lint, type-check, commit**

```bash
uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/senselab/
git add -A -- src/
git commit -m "feat(speaker_diarization): declare capabilities for all six backends

Each backend declares what it populates beside its own implementation; api.py maps
prefix to record and falls back to Pyannote exactly as the dispatch does.

A test asserts the new speaker_label_kind agrees with the existing
ROLE_LABEL_ONLY_PREFIXES list, so the two encodings of 'these are roles, keep them
out of the identity axis' cannot diverge while both exist."
```

---

### Task 3: Surface capabilities in the model registry

**Files:**
- Modify: `src/senselab/model_registry.yaml`
- Modify: `scripts/generate_model_registry.py`
- Modify: `src/senselab/model_registry.md` (regenerated, never hand-edited)
- Test: `src/tests/audio/tasks/speaker_diarization_capabilities_test.py`

**Interfaces:**
- Consumes: `capabilities_for` from Task 2.
- Produces: no importable interface. Deliverable is the registry entries plus a test binding them to the code.

- [x] **Step 1: Write the failing test**

Append to the capabilities test file:

```python
def test_registry_capabilities_match_the_code() -> None:
    """The YAML and the backend declarations must agree.

    The registry is what a human reads when choosing a model; the code is what runs.
    Two sources of truth are acceptable here only because this test makes drift a
    test failure rather than a surprise.
    """
    import yaml

    from senselab.audio.tasks.speaker_diarization.api import capabilities_for

    registry = yaml.safe_load(
        (Path(__file__).parents[3] / "senselab" / "model_registry.yaml").read_text()
    )

    def _walk(node: object) -> "Iterator[dict]":
        if isinstance(node, dict):
            if "model_id" in node and "capabilities" in node:
                yield node
            for value in node.values():
                yield from _walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from _walk(value)

    checked = 0
    for entry in _walk(registry):
        caps = capabilities_for(entry["model_id"])
        declared = entry["capabilities"]
        assert declared["populates_text"] == caps.populates_text, entry["model_id"]
        assert declared["speaker_label_kind"] == caps.speaker_label_kind, entry["model_id"]
        assert declared["max_speakers"] == caps.max_speakers, entry["model_id"]
        checked += 1
    assert checked == 6, f"expected 6 diarization entries with capabilities, found {checked}"
```

Add `from pathlib import Path` and `from typing import Iterator` to the test module's imports.

- [x] **Step 2: Run it and watch it fail**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -k registry -q
```

Expected: FAIL — `assert 0 == 6`, because no entry has a `capabilities` block yet.

- [x] **Step 3: Add the `capabilities` block to each diarization entry**

For each of the six diarization entries in `src/senselab/model_registry.yaml`, add a nested block using the exact values from the "Measured values" table. For example:

```yaml
  capabilities:
    populates_text: false
    speaker_label_kind: identity
    labels_stable_across_files: false
    max_speakers: null
    honors_speaker_hints: false
```

`max_speakers: null` is deliberate and means unmeasured. Do not substitute a number.

- [x] **Step 4: Teach the generator to render them**

`scripts/generate_model_registry.py` already renders an optional `License` column only for task sections where some entry declares one — `has_license = any("license" in m for m in task_models)` at line ~32, with the header and each row made conditional on it. Mirror that exactly:

```python
        # `capabilities` is an optional key, present only on diarization entries.
        # Gate the columns on the section, like `license` above, so unrelated task
        # tables are not widened with columns that would be empty in every row.
        has_caps = any("capabilities" in m for m in task_models)
        ...
        if has_caps:
            header += " Speakers | Text |"
            separator += "---|---|"
```

and in the row loop:

```python
            if has_caps:
                caps = m.get("capabilities", {})
                max_spk = caps.get("max_speakers")
                # An em dash, never "unlimited": null means nobody has measured the
                # ceiling. Rendering it as unlimited would invent a capability.
                speakers = "—" if max_spk is None else str(max_spk)
                text_col = "yes" if caps.get("populates_text") else "no"
```

Append `speakers` and `text_col` to that section's row `print(...)`, matching how `license_` is appended in the existing branch.

- [x] **Step 5: Regenerate and verify**

```bash
uv run python scripts/generate_model_registry.py
git diff --stat src/senselab/model_registry.md
uv run python scripts/generate_model_registry.py && git diff --exit-code src/senselab/model_registry.md && echo "generator is idempotent"
```

Expected: the `.md` changes once, then a second run produces no diff.

- [x] **Step 6: Run the tests**

```bash
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py -q
```

Expected: PASS, 21 tests.

- [x] **Step 7: Full check and commit**

```bash
uptime
uv run ruff format --check src/ scripts/ && uv run ruff check src/ && uv run mypy src/senselab/
uv run --with codespell==2.4.2 codespell src/senselab/model_registry.yaml src/senselab/model_registry.md
uv run pytest src/tests/audio/tasks/speaker_diarization_capabilities_test.py src/tests/audio/tasks/speaker_diarization_test.py -q
git add -A -- src/ scripts/
git commit -m "feat(model_registry): surface diarization capabilities

The registry is what a human reads when choosing a model; the backend constants are
what runs. A test asserts they agree, so drift is a test failure rather than a
surprise for whoever trusts the table.

max_speakers renders as an em dash where unmeasured, never as 'unlimited' — four of
six backends have no measured ceiling until the NeMo probe runs."
```

---

## After this plan

`max_speakers` is `None` for Pyannote, VibeVoice, MOSS and DiariZen. The NeMo synthetic-speaker probe (separate spec) measures each backend's accuracy from 1 to 8 speakers against known ground truth and replaces those `None`s with numbers that carry their derivation. Updating them is a one-line change per backend plus a registry regeneration, and the tests above will fail until the YAML and the code are updated together — which is the intended behaviour.
