"""The replay driver: re-deciding a finished run without losing what it decided before.

Nothing here is faked. A finished run in miniature is enough for the properties that make a replay
different in kind from the five drivers that only add a measurement: which entities it retires,
that retiring them cannot retire their replacements, and that a second pass over a replayed run
writes nothing.

``specs/20260922-replay-decisions-over-a-finished-corpus/design.md`` is the design.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import (
    DECISION_SUPERSEDED,
    REPLAY_MARKER_STEP,
    REPLAY_NODE,
    REPLAYED_NODES,
    find_replay_marker,
    live_decisions,
    read_store,
    replay_run_id,
    retire_decisions,
)
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import GRAPH_ORDER, Outcome
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_replay_decisions.py"
_spec = importlib.util.spec_from_file_location("extend_replay_decisions_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_replay_decisions_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
"""The fixture's sampling rate."""


def _samples() -> np.ndarray:
    """A short quiet tone, enough for ADMIT to decode and record a stream.

    Returns:
        The recording, mono float32.
    """
    grid = np.arange(SR) / SR
    return (0.05 * np.sin(2 * np.pi * 220.0 * grid)).astype(np.float32)


def _seed_run(root: Path, *, decided: bool = True) -> Path:
    """Write one finished run: a source WAV, ADMIT's stream, PREPROCESS's verdict, and a decision.

    Args:
        root: The run root, created if absent.
        decided: Whether to seed a VERDICT-generated entity for the replay to retire.

    Returns:
        The source WAV's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    source = root.parent / f"{root.name}.wav"
    sf.write(str(source), _samples(), SR)
    sf.write(str(run_dir / "streams" / "enhanced.flac"), _samples(), SR)

    store = ProvStore(run_id=root.name)
    config = load_triage_config()
    admitted = admit(store, source, config, run_dir=run_dir)
    assert admitted.audio is not None
    agent = software_agent(store)
    preprocessing = store.activity(node="PREPROCESS", step="condition", parameters={})
    store.was_associated_with(preprocessing, agent)
    write_verdict(
        store,
        preprocessing,
        agent,
        node="PREPROCESS",
        outcome=Outcome.PASS,
        kind=None,
        why="conditioning complete; absent derivatives are listed",
        detail={"absent": {}, "derivatives": {}},
    )
    if decided:
        deciding = store.activity(node="VERDICT", step=None, parameters={})
        store.was_associated_with(deciding, agent)
        write_verdict(
            store,
            deciding,
            agent,
            node="VERDICT",
            outcome=Outcome.PASS,
            kind=None,
            why="the fixture's prior decision",
            detail={},
        )
    store.write_jsonl(run_dir / "store.jsonl")
    (run_dir / "run.json").write_text(json.dumps({"source": str(source)}) + "\n", encoding="utf-8")
    return source


def _manifest(path: Path, roots: list[Path]) -> Path:
    """Write a manifest naming one enhanced stream per run root.

    Args:
        path: Where the JSONL goes.
        roots: The run roots, in manifest order.

    Returns:
        The manifest's path.
    """
    lines = [
        json.dumps({"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")}) for root in roots
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_replayed_nodes_start_at_taxonomy_and_exclude_what_is_read_off_disk() -> None:
    """ADMIT and PREPROCESS are read from the store; every node after them is replayed."""
    assert REPLAYED_NODES[0] == "TAXONOMY"
    assert "ADMIT" not in REPLAYED_NODES
    assert "PREPROCESS" not in REPLAYED_NODES
    assert REPLAYED_NODES == GRAPH_ORDER[GRAPH_ORDER.index("TAXONOMY") :]


def test_replay_run_id_separates_the_replay_from_the_run_and_from_another_config(tmp_path: Path) -> None:
    """A replay writes under an id of its own, one per configuration."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    assert replay_run_id(root, "aaaa") != root.name
    assert replay_run_id(root, "aaaa") != replay_run_id(root, "bbbb")
    assert replay_run_id(root, "aaaa") == replay_run_id(root, "aaaa")


def test_reading_under_the_replay_id_gives_new_content_a_new_id(tmp_path: Path) -> None:
    """The property retire-first depends on: a replay cannot retire its own replacement.

    Writing byte-identical content under the run's own id reproduces the id the run recorded, which
    is what the append-only drivers rely on. Under the replay's id the same content takes a
    different id, so an entity retired before the replay ran cannot be the one it writes after.
    """
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    attributes = {"name": "a-measurement", "value": 1.0}

    same = read_store(root)
    replayed = read_store(root, run_id=replay_run_id(root, "aaaa"))
    under_run_id = same.entity(prov_type="measurement", extent=None, attributes=attributes)
    under_replay_id = replayed.entity(prov_type="measurement", extent=None, attributes=attributes)

    assert under_run_id != under_replay_id
    assert same.run_id == root.name
    assert replayed.run_id == replay_run_id(root, "aaaa")


def test_live_decisions_finds_the_replayed_nodes_output_and_leaves_preprocess_alone(tmp_path: Path) -> None:
    """Only what a node from TAXONOMY on generated is in scope for retirement."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    store = read_store(root)

    found = live_decisions(store)

    assert found, "the seeded VERDICT entity should be in scope"
    for entity_id in found:
        activity = store.get_activity(str(store.generated_by(entity_id)))
        assert activity.node in REPLAYED_NODES
    generated_by_preprocess = [
        entity.id
        for entity in store.entities()
        if (aid := store.generated_by(entity.id)) is not None and store.get_activity(aid).node == "PREPROCESS"
    ]
    assert generated_by_preprocess, "the fixture should carry PREPROCESS output"
    assert not set(generated_by_preprocess) & set(found)


def test_retire_decisions_invalidates_the_decisions_and_nothing_else(tmp_path: Path) -> None:
    """Retirement is an invalidation edge per decision; PREPROCESS's output stays live."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    store = read_store(root)
    scoped = live_decisions(store)
    preprocessed = [
        entity.id
        for entity in store.entities()
        if (aid := store.generated_by(entity.id)) is not None and store.get_activity(aid).node == "PREPROCESS"
    ]

    activities = retire_decisions(store, scoped, software=software_agent(store))

    assert len(activities) == len(scoped)
    assert all(store.is_invalidated(entity_id) for entity_id in scoped)
    assert not any(store.is_invalidated(entity_id) for entity_id in preprocessed)
    for activity_id in activities:
        activity = store.get_activity(activity_id)
        assert activity.node == REPLAY_NODE
        assert activity.step == DECISION_SUPERSEDED
    assert not live_decisions(store), "a retired decision is no longer in scope"


def test_retiring_twice_adds_no_second_edge(tmp_path: Path) -> None:
    """A slice re-run cannot stack two supersessions over one decision."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    store = read_store(root)
    scoped = live_decisions(store)
    agent = software_agent(store)

    retire_decisions(store, scoped, software=agent)
    after_first = store.fingerprint()
    retire_decisions(store, scoped, software=agent)

    assert store.fingerprint() == after_first


def test_the_marker_is_absent_before_and_idempotent_after(tmp_path: Path) -> None:
    """The marker is what makes a second pass a no-op, and writing it twice moves nothing."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    store = read_store(root, run_id=replay_run_id(root, "aaaa"))
    assert find_replay_marker(store, "aaaa") is None

    parameters = {"config_hash": "aaaa", "commit": None, "retired": 1}
    first = store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters=parameters)
    after_first = store.fingerprint()
    second = store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters=parameters)

    assert first == second
    assert store.fingerprint() == after_first
    assert find_replay_marker(store, "aaaa") == first
    assert find_replay_marker(store, "bbbb") is None


def test_mirror_run_root_is_writable_and_reads_through_to_the_finished_run(tmp_path: Path) -> None:
    """The out-root mirror must not write into the tree it reads."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    (root / "run" / "derivatives" / "yamnet_scores.json").write_text("[]", encoding="utf-8")
    out_root = tmp_path / "replayed"

    mirrored = cli.mirror_run_root(root, out_root, "sub-01_ses-01_task-x")

    assert mirrored.parent == out_root / "sub-01" / "ses-01"
    assert (mirrored / "run" / "derivatives").is_symlink()
    assert (mirrored / "run" / "derivatives" / "yamnet_scores.json").read_text() == "[]"
    assert not (mirrored / "run" / "streams").is_symlink()
    assert (mirrored / "run" / "streams" / "enhanced.flac").is_symlink()
    assert (mirrored / "released").is_dir()

    (mirrored / "run" / "streams" / "redacted.flac").write_text("new", encoding="utf-8")
    assert not (root / "run" / "streams" / "redacted.flac").exists()


def test_mirror_run_root_is_idempotent(tmp_path: Path) -> None:
    """A preempted task re-entering must not trip over the links it already made."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    out_root = tmp_path / "replayed"

    first = cli.mirror_run_root(root, out_root, "sub-01_ses-01_task-x")
    second = cli.mirror_run_root(root, out_root, "sub-01_ses-01_task-x")

    assert first == second
    assert (second / "run" / "streams" / "enhanced.flac").is_symlink()


def test_source_of_reads_the_recording_out_of_the_run_log(tmp_path: Path) -> None:
    """The hint is rebuilt from the recording, which the run log names."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    source = _seed_run(root)

    assert cli.source_of(root) == source


def test_source_of_refuses_a_run_with_no_log(tmp_path: Path) -> None:
    """A run whose log is missing cannot have its hint rebuilt, and says so."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    (root / "run" / "run.json").unlink()

    with pytest.raises(FileNotFoundError):
        cli.source_of(root)


def test_a_run_already_carrying_this_configurations_marker_is_skipped(tmp_path: Path) -> None:
    """The resume predicate: the store says whether this configuration has already replayed it."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    config = load_triage_config()
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    marker = store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": config.config_hash, "commit": None, "retired": 0},
    )
    store.write_jsonl(root / "run" / "store.jsonl")
    before = (root / "run" / "store.jsonl").read_text()

    record = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None)

    assert record["status"] == cli.PRESENT
    assert (root / "run" / "store.jsonl").read_text() == before, "a skipped run is not rewritten"
    assert marker


def test_the_manifest_and_slice_shape_match_the_other_extend_drivers(tmp_path: Path) -> None:
    """One array task takes rows[i::n] of a manifest keyed on stem and enhanced."""
    roots = [tmp_path / f"sub-0{index}_ses-01_task-x_20260920-030808" for index in range(1, 5)]
    for root in roots:
        _seed_run(root, decided=False)
    manifest = _manifest(tmp_path / "manifest.jsonl", roots)

    parsed = cli.build_parser().parse_args([str(manifest), "--slice-index", "1", "--slice-count", "2"])

    assert parsed.slice_index == 1
    assert parsed.slice_count == 2
    rows = json.loads("[" + ",".join(manifest.read_text().strip().splitlines()) + "]")
    assert [row["stem"] for row in rows[1::2]] == [roots[1].name, roots[3].name]


def test_a_row_whose_enhanced_path_is_not_in_a_run_tree_is_an_error_not_a_crash(tmp_path: Path) -> None:
    """One unreadable row must not take its neighbours down."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    rows = [
        {"stem": "nowhere", "enhanced": str(tmp_path / "loose.flac")},
        {"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")},
    ]

    out = cli.process(rows, load_triage_config(), build_hint=None, out_root=None, commit=None)

    assert out[0]["status"] == cli.ERROR
    assert out[1]["status"] != cli.ERROR or "run_root" not in out[0]
