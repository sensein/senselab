"""The reference and status pages: generated from the registry, the catalogue and the specs, and readable by docs."""

import importlib
import re
from pathlib import Path

import pytest
from senselab._api.describe import describe
from senselab.reference import _render, _specs
from senselab_core.kernel.declare import REGISTRY

import senselab

PAGES = ("tasks", "models", "datasets", "status")


def _tasks() -> list[str]:
    return [s.name for s in REGISTRY.tasks()]


LEDGER = """
| Area (1.x paths) | Where | Mark | Reconciled changes |
|---|---|---|---|
| `src/senselab/audio/tasks/` | on alpha | `abc123` | - |
| `packages/x` | moved | `abc123` | - |

## Moved in Phase 2

| 1.x module | Moved to | Task |
|---|---|---|
| `src/senselab/audio/tasks/speech_to_text/` `huggingface.py` | `a/b.py` with `engine: ct2 \\| transformers` | 13 |
"""

REGISTER = """
| Id | Src | Finding | Resolution | Status |
|---|---|---|---|---|
| row-1 | A | one | fix | resolved in design |
| row-2 | A | two | fix | open, owner: Phase 3 |
| row-3 | A | three | fix | scheduled |
"""

DESIGN = """
**Phase 2: tasks and backends.**
- Build:
  - things
- Exit:
  - each call switches by profile alone;
  - parity holds within tolerance, on CPU
    and on one CUDA host

**Phase 3: plans.**
- Exit:
  - other
"""


@pytest.mark.logic
def test_tables_split_cells_on_unescaped_bars_and_drop_the_separator() -> None:
    """Tables split cells on unescaped bars and drop the separator."""
    (rows,) = [t for t in _specs.tables(LEDGER) if t[0][0] == "1.x module"]
    assert rows[1][1] == "`a/b.py` with `engine: ct2 | transformers`"
    assert len(rows) == 2


@pytest.mark.logic
def test_ledger_marks_and_moved_rows_are_read_by_heading() -> None:
    """Ledger marks and moved rows are read by heading."""
    assert [r["Where"] for r in _specs.ledger_marks(LEDGER)] == ["on alpha", "moved"]
    assert [r["Task"] for r in _specs.moved_in_phase(LEDGER, 2)] == ["13"]
    assert _specs.moved_in_phase(LEDGER, 3) == []


@pytest.mark.logic
def test_only_open_and_scheduled_register_rows_are_listed() -> None:
    """Only open and scheduled register rows are listed."""
    assert [r["Id"] for r in _specs.open_register_rows(REGISTER)] == ["row-2", "row-3"]


@pytest.mark.logic
def test_phase_exits_stop_at_the_next_phase_and_join_wrapped_lines() -> None:
    """Phase exits stop at the next phase and join wrapped lines."""
    assert _specs.phase_exits(DESIGN, 2) == [
        "each call switches by profile alone",
        "parity holds within tolerance, on CPU and on one CUDA host",
    ]
    assert _specs.phase_exits(DESIGN, 9) == []


@pytest.mark.logic
def test_absent_specs_read_as_empty() -> None:
    """Absent specs read as empty."""
    assert _specs.read_spec("migration-ledger.md", root=Path("/nonexistent")) == ""
    assert _specs.open_register_rows("") == []
    assert _specs.ledger_marks("") == []


@pytest.mark.logic
def test_task_state_follows_the_ledger_and_the_registry() -> None:
    """Task state follows the ledger and the registry."""
    states = {s["task"]: s for s in _render.task_states(REGISTRY, LEDGER, root=None)}
    assert set(states) == set(_tasks())
    assert states["speech_to_text"]["ported"]
    assert not states["diarize"]["ported"]
    assert all(s["declared"] and s["implemented"] for s in states.values())
    assert not any(s["benchmarked"] for s in states.values())


@pytest.mark.contract
def test_describe_names_a_task_with_a_text_port() -> None:
    """Describe names a task with a text port."""
    inputs = describe("pii_spans")["inputs"]["properties"]
    assert "description" in inputs["text"]


@pytest.mark.contract
@pytest.mark.parametrize("page", PAGES)
def test_each_page_module_documents_itself_with_its_badge(page: str) -> None:
    """Each page module documents itself with its badge."""
    module = importlib.import_module(f"senselab.reference.{page}")
    assert module.__doc__ is not None
    assert module.__doc__.startswith(_render.BADGES["2x"])


@pytest.mark.contract
def test_tasks_page_has_a_section_per_declared_task_and_every_default_model() -> None:
    """Tasks page has a section per declared task and every default model."""
    page = _render.tasks_page()
    for name in _tasks():
        assert f"### `{name}`" in page
    for name in _tasks():
        default = describe(name)["default_model"]
        assert default is None or default in page


@pytest.mark.contract
def test_models_page_lists_the_whole_catalogue_by_revision() -> None:
    """Models page lists the whole catalogue by revision."""
    page = _render.models_page()
    for model in senselab.models():
        assert f"`{model['id']}`" in page
        assert model["revision"][:12] in page


@pytest.mark.contract
def test_datasets_page_names_each_manifest_and_how_to_fetch_it() -> None:
    """Datasets page names each manifest and how to fetch it."""
    root = _specs.checkout_root()
    assert root is not None
    page = _render.datasets_page(root)
    for manifest in (root / "benchmarks" / "manifests").glob("*.yaml"):
        assert f"senselab data fetch benchmarks/manifests/{manifest.name}" in page


@pytest.mark.contract
def test_status_page_reads_the_registry_and_the_checkout_specs() -> None:
    """Status page reads the registry and the checkout specs."""
    page = _render.status_page()
    for name in _tasks():
        assert f"| `{name}` |" in page
    assert "## Consolidation inventory" in page
    assert _specs.phase_exits(_specs.read_spec("design.md"), 2)[0] in page


@pytest.mark.contract
@pytest.mark.parametrize("page", PAGES)
def test_pages_use_no_verdict_or_decision_vocabulary_of_their_own(page: str) -> None:
    """Pages use no verdict or decision vocabulary of their own."""
    text = importlib.import_module(f"senselab.reference.{page}").__doc__ or ""
    own = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    assert not re.search(r"verdict|decision", own, re.IGNORECASE)
