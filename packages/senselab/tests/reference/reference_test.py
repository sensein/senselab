"""The reference and status pages: generated from the registry, the catalogue and the specs, and readable by docs."""

import importlib
import re
import wave
from pathlib import Path

import pytest
from senselab._api.describe import describe
from senselab.reference import _api, _render, _specs
from senselab_core.kernel.declare import REGISTRY
from senselab_core.runtime.profiles import ExecutionProfile

import senselab

PAGES = ("tasks", "models", "datasets", "status", "plans", "runs", "executors", "profiles", "librispeech_example")


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
def test_phase_names_lose_their_ids() -> None:
    """A phase name that cites an ADR is published without the id."""
    state = "| Phase | State |\n|---|---|\n| 2: tasks, parity, ADR-010 | done 2026-11-03 |\n"
    state += "| 3: plans (P3-1) | in progress |\n"
    assert _specs.phase_states(state) == {"2: tasks, parity": "done", "3: plans": "in progress"}


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
def test_pages_use_no_retired_vocabulary_anywhere(page: str) -> None:
    """Pages use no retired triage vocabulary, tables included."""
    text = importlib.import_module(f"senselab.reference.{page}").__doc__ or ""
    assert not re.search(r"verdict|decision|\breject\b", text, re.IGNORECASE)


@pytest.mark.contract
@pytest.mark.parametrize("page", PAGES)
def test_pages_publish_no_spec_ids_or_paths(page: str) -> None:
    """Pages publish no register, principle, hardening or ADR id and no spec path."""
    text = importlib.import_module(f"senselab.reference.{page}").__doc__ or ""
    assert not re.search(r"\b(?:ADR|P2|[PRHD])-\d|\bH\d+[a-z]?\b|specs/", text)


@pytest.mark.contract
@pytest.mark.parametrize("page", PAGES)
def test_every_page_is_built_in_a_checkout(page: str) -> None:
    """Every page has content in a checkout and none is the unavailable notice."""
    text = importlib.import_module(f"senselab.reference.{page}").__doc__ or ""
    assert len(text) > 300
    assert _render.UNAVAILABLE not in text


@pytest.mark.logic
def test_a_page_without_sources_is_a_notice_not_a_crash() -> None:
    """A page whose sources are absent becomes the unavailable notice."""

    def missing() -> str:
        raise FileNotFoundError("no checkout")

    assert _render.or_unavailable(missing).startswith(_render.UNAVAILABLE)


@pytest.mark.logic
def test_scrub_removes_ids_and_their_parentheses() -> None:
    """Scrub removes ids and the parentheses that held them."""
    assert _specs.scrub("switches by profile alone (P-8)") == "switches by profile alone"
    assert _specs.scrub("one store (ADR-002), a lock") == "one store, a lock"


@pytest.mark.logic
def test_open_items_count_by_owner_phase() -> None:
    """Open items are counted by owner phase and name no row."""
    assert _specs.open_by_phase(REGISTER) == {"Phase 3": 1, "unassigned": 1}


@pytest.mark.logic
def test_signature_is_one_parameter_a_line_without_self_or_object_reprs() -> None:
    """A signature drops self, shows long annotations as dots and no memory address."""

    class Clock:
        pass

    class Thing:
        def go(
            self,
            plan: str,
            *,
            clock: object = Clock(),  # noqa: B008
            values: "Mapping[str, StringConstraints(x)]" = None,  # type: ignore[name-defined]  # noqa: F821
        ) -> int:
            return 0

    text = _api.signature(Thing.go)
    assert text == "(\n    plan: str,\n    *,\n    clock: object = ...,\n    values: ... = None,\n) -> int"
    assert _api.signature(Thing) == "()"


@pytest.mark.logic
def test_prose_turns_raises_and_attributes_blocks_into_bullets() -> None:
    """Google-style blocks become bullet lists."""
    text = "Do it.\n\nRaises:\n    ValueError: it is bad,\n        very.\n    KeyError: absent.\n\nAfter."
    assert (
        _api.prose(text)
        == "Do it.\n\n\n**Raises**\n\n- `ValueError`: it is bad,\n  very.\n- `KeyError`: absent.\n\nAfter."
    )


@pytest.mark.contract
@pytest.mark.parametrize("page", PAGES)
def test_pages_hold_no_memory_address(page: str) -> None:
    """A generated page names no object by its address, so two builds give one page."""
    text = importlib.import_module(f"senselab.reference.{page}").__doc__ or ""
    assert not re.search(r"0x[0-9a-f]{6,}", text)


@pytest.mark.contract
def test_profiles_page_names_every_key_of_the_packaged_profiles() -> None:
    """Every key of the packaged profiles, as the format reads it, has a row on the profiles page."""
    from senselab.reference import _code_pages  # noqa: PLC0415 - beside its test

    page = _code_pages.profiles_page()
    for name in ("laptop", "slurm"):
        for key in _code_pages.profile_rows(name):
            assert f"`{key}`" in page
    loaded = ExecutionProfile.load("laptop").model_dump(mode="json", exclude={"format"})
    for key in loaded:
        assert f"`{key}" in page


@pytest.mark.contract
def test_profiles_page_states_the_site_profile_rules() -> None:
    """The site-profile section says placement reads no scheduler, where QOS limits go and how to calibrate."""
    page = _render_page("profiles")
    for phrase in (
        "generic profiles only",
        "invisible",
        "placement hook",
        "--measures-root",
        "calibration.measures_roots",
    ):
        assert phrase in page


def _render_page(name: str) -> str:
    return importlib.import_module(f"senselab.reference.{name}").__doc__ or ""


@pytest.mark.contract
def test_librispeech_example_code_plans_and_explains(tmp_path: Path) -> None:
    """The code of the example page plans a task over a BIDS tree and `explain` names the model the page names."""
    root = tmp_path / "tree"
    audio = root / "sub-1" / "ses-01" / "audio"
    audio.mkdir(parents=True)
    (root / "dataset_description.json").write_text('{"Name": "t", "BIDSVersion": "1.10.0"}\n', encoding="utf-8")
    for run in (1, 2):
        with wave.open(str(audio / f"sub-1_ses-01_task-read_run-{run}_audio.wav"), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16_000)
            out.writeframes(b"\x00\x01" * 1600 * run)
    page = _render_page("librispeech_example")
    assert 'senselab.use(execution="laptop")' in page
    with senselab.use(execution="laptop"):
        placed = senselab.plan(senselab.SpeechToText(), senselab.Dataset.bids(root))
        text = placed.explain()
    default = senselab.describe("speech_to_text")["default_model"]
    assert placed.hash in text
    assert default in text
    assert default.split("@")[0] in page
    assert "node call (speech_to_text)" in text


@pytest.mark.contract
def test_status_page_lists_the_executors_and_exit_scenarios_of_the_checkout() -> None:
    """The status page rows of executors and scenarios are read from the exit checks."""
    root = _specs.checkout_root()
    assert root is not None
    assert [r[0] for r in _render.scenario_rows(root)] == ["`resample`", "`stt`"]
    rows = {(r[0], r[1]): r for r in _render.executor_rows(root)}
    assert rows[("`InProcess`", "this process")][2:] == ("`laptop`", "yes")
    assert rows[("`Pool`", "`Slurm`")][2:] == ("`slurm`", "yes")
    assert rows[("`Pool`", "`Local`")][2] == "-"
    assert _render.executor_rows(None)[0][3] == "-"


@pytest.mark.contract
def test_librispeech_example_constants_still_describe_the_run_the_page_shows() -> None:
    """The recorded run equals the catalogue default and the exit scenario, or the example is stale."""
    from senselab.reference import _code_pages  # noqa: PLC0415 - beside its test
    from senselab_core.runtime.config import load_yaml  # noqa: PLC0415 - beside its test

    root = _specs.checkout_root()
    assert root is not None
    step = load_yaml(root / "benchmarks" / "exit" / "exit.yaml", "exit checks")["steps"][_code_pages.EXAMPLE_STEP]
    run = _code_pages.EXAMPLE_RUN
    default = senselab.describe("speech_to_text")["default_model"]
    assert default == run["model"]
    assert step["items"] == run["items"]
    page = _render_page("librispeech_example")
    assert _code_pages.EXAMPLE_HASH in page
    assert f"{run['interrupted_at']} of {run['items']}" in page
    assert "bids_tree.from_librispeech" in page


@pytest.mark.contract
def test_profiles_page_says_coordinators_count_against_the_submit_limit() -> None:
    """The site-profile section names the per-user submit limit that coordinators count against."""
    assert "coordinators are\n  jobs too" in _render_page("profiles")
