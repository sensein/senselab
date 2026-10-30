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
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio, AudioHints, ExpectedSpeech
from senselab.audio.data_structures.audio_hints import SpeakerEmbeddingProvenance, TargetSpeakerEmbedding
from senselab.audio.workflows.triage import extend as extend_module
from senselab.audio.workflows.triage.background_model import write_view_arrays
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import (
    AUDIO_UNVERIFIED,
    AUDIO_VERIFIED,
    DECISION_SUPERSEDED,
    HINT_GIVEN,
    HINT_RECORDED,
    REPLAY_MARKER_STEP,
    REPLAY_NODE,
    REPLAYED_NODES,
    ReplayInputs,
    ReplayRefused,
    check_replay_inputs,
    find_replay_marker,
    live_decisions,
    read_store,
    realign_stimulus,
    record_replayed_hint,
    replay_decisions,
    replay_run_id,
    retire_decisions,
    write_store,
)
from senselab.audio.workflows.triage.hint_record import (
    HINT,
    HINT_DIGEST,
    HINT_FIELD_DIGESTS,
    NO_HINT,
    changed_fields,
    hint_digest,
)
from senselab.audio.workflows.triage.hint_record import expected_speech_digest as prompts_digest
from senselab.audio.workflows.triage.identity import declared_task_id, identity_of, recording_entity
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_measurement, software_agent, write_stream, write_verdict
from senselab.audio.workflows.triage.nodes.preprocess import STIMULUS_MEASUREMENT, write_stimulus_alignment
from senselab.audio.workflows.triage.stimulus import StimulusAlignment
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


def _seed_run(root: Path, *, decided: bool = True, hint: AudioHints | None = None) -> Path:
    """Write one finished run: a source WAV, ADMIT's stream, PREPROCESS's verdict, and a decision.

    Args:
        root: The run root, created if absent.
        decided: Whether to seed a VERDICT-generated entity for the replay to retire.
        hint: The hint ADMIT is given, and so records.

    Returns:
        The source WAV's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    source = root.parent / f"{root.name}.wav"
    sf.write(str(source), _samples(), SR)
    for name in ("enhanced", "plain", "redacted", "separated_0"):
        sf.write(str(run_dir / "streams" / f"{name}.flac"), _samples(), SR)

    store = ProvStore(run_id=root.name)
    config = load_triage_config()
    admitted = admit(store, source, config, hint, run_dir=run_dir)
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


def test_replayed_nodes_start_at_background_and_exclude_what_is_read_off_disk() -> None:
    """ADMIT, PREPROCESS and SESSION are read from the store; every node from BACKGROUND on but COHORT is replayed."""
    assert REPLAYED_NODES[0] == "BACKGROUND"
    assert {"ADMIT", "PREPROCESS", "SESSION", "COHORT"}.isdisjoint(REPLAYED_NODES)
    assert REPLAYED_NODES == tuple(n for n in GRAPH_ORDER[GRAPH_ORDER.index("BACKGROUND") :] if n != "COHORT")


def test_replay_run_id_separates_the_replay_from_the_run_and_from_another_config(tmp_path: Path) -> None:
    """A replay writes under an id of its own, one per configuration."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    assert replay_run_id(root, "aaaa") != root.name
    assert replay_run_id(root, "aaaa") != replay_run_id(root, "bbbb")
    assert replay_run_id(root, "aaaa") == replay_run_id(root, "aaaa")
    assert replay_run_id(root, "aaaa", "c" * 40) != replay_run_id(root, "aaaa", "d" * 40)


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


def test_the_marker_is_absent_for_another_revision_under_the_same_configuration(tmp_path: Path) -> None:
    """A code change under an unchanged configuration replays again rather than reading as done."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    store = read_store(root, run_id=replay_run_id(root, "aaaa", "c" * 40))
    marker = store.activity(
        node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={"config_hash": "aaaa", "commit": "c" * 40, "retired": 1}
    )
    assert find_replay_marker(store, "aaaa", "c" * 40) == marker
    assert find_replay_marker(store, "aaaa", "d" * 40) is None
    assert find_replay_marker(store, "aaaa") == marker


def test_mirror_run_root_is_writable_and_reads_through_to_the_finished_run(tmp_path: Path) -> None:
    """The out-root mirror must not write into the tree it reads."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    (root / "run" / "derivatives" / "yamnet_scores.json").write_text("[]", encoding="utf-8")
    out_root = tmp_path / "replayed"

    mirrored = cli.mirror_run_root(root, out_root, identity_of("sub-01_ses-01_task-x"))

    assert mirrored.parent == out_root / "sub-01" / "ses-01"
    assert not (mirrored / "run" / "derivatives").is_symlink()
    assert (mirrored / "run" / "derivatives" / "yamnet_scores.json").is_symlink()
    assert (mirrored / "run" / "derivatives" / "yamnet_scores.json").read_text() == "[]"
    assert not (mirrored / "run" / "streams").is_symlink()
    assert (mirrored / "run" / "streams" / "enhanced.flac").is_symlink()
    assert (mirrored / "released").is_dir()

    assert not (mirrored / "run" / "streams" / "redacted.flac").exists()


def test_a_stream_a_replayed_node_writes_is_never_a_link_into_the_finished_run(tmp_path: Path) -> None:
    """REDACT and SPEECH write into ``streams/``; a link there would carry the write to the source."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of("sub-01_ses-01_task-x"))

    links = {path.name for path in (mirrored / "run" / "streams").iterdir()}
    assert links == {"enhanced.flac", "plain.flac"}, "only the streams a replay reads may be linked"


def test_a_replayed_redact_write_leaves_the_finished_tree_byte_identical(tmp_path: Path) -> None:
    """The defect: 17,924 ``redacted.flac`` under the 2026-09-19 corpus were overwritten this way."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    before = {path.name: path.read_bytes() for path in sorted((root / "run" / "streams").iterdir()) if path.is_file()}

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of("sub-01_ses-01_task-x"))
    for stem in ("redacted", "separated_0"):
        write_stream(Audio(waveform=_samples()[None, :] * 0.5, sampling_rate=SR), mirrored / "run", stem)

    after = {path.name: path.read_bytes() for path in sorted((root / "run" / "streams").iterdir()) if path.is_file()}
    assert after == before, "a replayed run must not write into the tree it reads"
    assert (mirrored / "run" / "streams" / "redacted.flac").read_bytes() != before["redacted.flac"]


def test_the_derivatives_a_replay_rewrites_leave_the_finished_tree_byte_identical(tmp_path: Path) -> None:
    """The defect: replayed BACKGROUND rewrote ``background_view.npz`` through a linked ``derivatives/``."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    finished = root / "run" / "derivatives"
    np.savez(finished / "background_view.npz", marker=np.array([1]))
    np.savez(finished / "stimulus_alignment.npz", marker=np.array([2]))
    before = {path.name: path.read_bytes() for path in sorted(finished.iterdir())}

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of("sub-01_ses-01_task-x"))
    store = ProvStore(run_id="replayed")
    frames = SimpleNamespace(times_s=np.arange(3.0), band_db=np.zeros((3, 2)), level_db=np.zeros(3))
    floor = SimpleNamespace(
        band_db=np.zeros(2), own_db=np.zeros(2), residual_db=None, source="quiet_frames", quiet_s=1.0
    )
    reading = SimpleNamespace(frames=frames, floor=floor, impulses=(), regions=())
    write_view_arrays(mirrored / "run" / "derivatives" / "background_view.npz", reading)  # type: ignore[arg-type]
    alignment = StimulusAlignment(expected=(), units=(), unexpected=(), provenance={})
    write_stimulus_alignment(store, run_dir=mirrored / "run", consensus_id="c", alignment=alignment, terminators=".")

    after = {path.name: path.read_bytes() for path in sorted(finished.iterdir())}
    assert after == before, "a replayed run must not write into the tree it reads"
    for name in ("background_view.npz", "stimulus_alignment.npz"):
        assert not (mirrored / "run" / "derivatives" / name).is_symlink()


def test_a_mirror_whose_derivatives_link_the_whole_directory_is_rebuilt(tmp_path: Path) -> None:
    """An out root an earlier driver populated must not let a rewrite delete the finished run's file."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    finished = root / "run" / "derivatives"
    np.savez(finished / "background_view.npz", marker=np.array([1]))
    before = (finished / "background_view.npz").read_bytes()
    old = tmp_path / "replayed" / "sub-01" / "ses-01" / root.name / "run"
    old.mkdir(parents=True)
    (old / "derivatives").symlink_to(finished)

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of("sub-01_ses-01_task-x"))
    reading = SimpleNamespace(
        frames=SimpleNamespace(times_s=np.arange(3.0), band_db=np.zeros((3, 2)), level_db=np.zeros(3)),
        floor=SimpleNamespace(band_db=np.zeros(2), own_db=np.zeros(2), residual_db=None, source="q", quiet_s=1.0),
        impulses=(),
        regions=(),
    )
    write_view_arrays(mirrored / "run" / "derivatives" / "background_view.npz", reading)  # type: ignore[arg-type]

    assert not (mirrored / "run" / "derivatives").is_symlink()
    assert (finished / "background_view.npz").read_bytes() == before


def test_a_stream_write_replaces_the_entry_rather_than_following_it(tmp_path: Path) -> None:
    """The chokepoint's own property, independent of how any mirror was built."""
    source = tmp_path / "source.flac"
    sf.write(str(source), _samples(), SR)
    run_dir = tmp_path / "run"
    (run_dir / "streams").mkdir(parents=True)
    (run_dir / "streams" / "redacted.flac").symlink_to(source)
    before = source.read_bytes()

    write_stream(Audio(waveform=_samples()[None, :] * 0.5, sampling_rate=SR), run_dir, "redacted")

    assert source.read_bytes() == before
    assert not (run_dir / "streams" / "redacted.flac").is_symlink()


def test_mirror_run_root_is_idempotent(tmp_path: Path) -> None:
    """A preempted task re-entering must not trip over the links it already made."""
    root = tmp_path / "finished" / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    out_root = tmp_path / "replayed"

    first = cli.mirror_run_root(root, out_root, identity_of("sub-01_ses-01_task-x"))
    second = cli.mirror_run_root(root, out_root, identity_of("sub-01_ses-01_task-x"))

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
        parameters={"config_hash": config.config_hash, "commit": None, "retired": 0, "hint_digest": NO_HINT},
    )
    store.write_jsonl(root / "run" / "store.jsonl")
    before = (root / "run" / "store.jsonl").read_text()

    record = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None)

    assert record["status"] == cli.PRESENT
    assert (root / "run" / "store.jsonl").read_text() == before, "a skipped run is not rewritten"
    assert marker


def test_a_marker_under_another_hint_does_not_skip_the_replay(tmp_path: Path) -> None:
    """A corrected sidecar is a new hint: the run replays rather than reading as already done."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    config = load_triage_config()
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": config.config_hash, "commit": None, "retired": 0, "hint_digest": "another"},
    )
    assert find_replay_marker(store, config.config_hash, None, NO_HINT) is None
    assert find_replay_marker(store, config.config_hash, None, "another") is not None
    assert find_replay_marker(store, config.config_hash) is not None


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


def test_each_replayed_row_lands_in_the_row_log_as_it_completes(tmp_path: Path) -> None:
    """The slice log holds a record per finished row, in order, before the slice ends."""
    from senselab.audio.workflows.triage.extend import SliceLog

    rows = [{"stem": f"nowhere-{n}", "enhanced": str(tmp_path / f"loose-{n}.flac")} for n in range(3)]
    path = tmp_path / "slices" / "replay-decisions-slice-0-of-1.jsonl"
    log = SliceLog(path, slice_index=0, slice_count=1, total=len(rows))
    seen: list[int] = []
    original = log.add

    def _watch(record: dict, seconds: float) -> None:
        original(record, seconds)
        seen.append(len(path.read_text(encoding="utf-8").splitlines()))

    log.add = _watch  # type: ignore[method-assign]
    try:
        out = cli.process(rows, load_triage_config(), build_hint=None, out_root=None, commit=None, log=log)
    finally:
        log.close()
    assert seen == [1, 2, 3], "a row was not on disk when the next one started"
    assert [json.loads(line)["stem"] for line in path.read_text(encoding="utf-8").splitlines()] == [
        row["stem"] for row in out
    ]


def test_a_reading_that_must_be_read_again_is_named_and_fails_the_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row whose REVIEW could not be carried forward is counted, listed by stem, and exits 3."""
    roots = [_seed_run(tmp_path / f"run{n}") for n in range(2)]
    manifest = _manifest(tmp_path / "manifest.jsonl", roots)
    states = iter(["carried", "needs_reread"])

    def _fake(run_root: Path, config: object, **kwargs: object) -> dict:
        return {"status": "ok", "retired": 0, "review": next(states), "review_why": None}

    monkeypatch.setattr(cli, "replay_one", _fake)
    summary = cli.run_slice(
        manifest,
        slice_index=0,
        slice_count=1,
        config=load_triage_config(),
        log_dir=tmp_path / "logs",
        hints=None,
        out_root=None,
        commit=None,
    )
    stems = [json.loads(line)["stem"] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert summary["reviews"] == {"carried": 1, "needs_reread": 1}
    assert summary["needs_reread"] == [stems[1]]


def test_the_driver_exits_3_when_any_reading_must_be_read_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exit code says a GPU node's result was lost, so a pipeline cannot run on past it."""
    roots = [_seed_run(tmp_path / "run0")]
    manifest = _manifest(tmp_path / "manifest.jsonl", roots)
    monkeypatch.setattr(
        cli,
        "replay_one",
        lambda run_root, config, **kwargs: {"status": "ok", "retired": 0, "review": "needs_reread", "review_why": "x"},
    )
    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)]) == 3


def _stub_replay(errors: dict[str, str]):  # type: ignore[no-untyped-def]  # noqa: ANN202
    """A replay_decisions stand-in that writes nothing and reports these node errors."""
    from senselab.audio.workflows.triage.extend import ReplayOutcome

    def _replay(store: object, config: object, hint: object, **kwargs: object) -> ReplayOutcome:
        return ReplayOutcome(retired=0, states={}, errors=dict(errors), released={}, summary={}, marker="m")

    return _replay


def test_a_replay_whose_node_raised_is_errored_and_names_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that landed with a node error is not ``ok``: the row says which node failed."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({"SPEECH": "RuntimeError: no libtorchcodec"}))

    record = cli.replay_one(root, load_triage_config(), build_hint=None, out_root=None, stem=root.name, commit=None)

    assert record["status"] == cli.ERRORED
    assert record["errored_nodes"] == ["SPEECH"]


def test_a_replay_with_no_node_error_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The errored status is only for a raised node; a clean replay stays ``ok``."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))

    record = cli.replay_one(root, load_triage_config(), build_hint=None, out_root=None, stem=root.name, commit=None)

    assert record["status"] == cli.OK
    assert record["errored_nodes"] == []


def test_the_driver_exits_1_when_any_row_is_errored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A node error fails the array task, so a chain cannot run on past it."""
    manifest = _manifest(tmp_path / "manifest.jsonl", [_seed_run(tmp_path / "run0")])
    monkeypatch.setattr(
        cli,
        "replay_one",
        lambda run_root, config, **kwargs: {"status": cli.ERRORED, "retired": 0, "errored_nodes": ["SPEECH"]},
    )
    assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)]) == 1


def test_force_replays_a_run_that_already_carries_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--force`` bypasses the resume predicate; without it the same run is ``present``."""
    root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
    _seed_run(root)
    config = load_triage_config()
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": config.config_hash, "commit": None, "retired": 0, "hint_digest": NO_HINT},
    )
    store.write_jsonl(root / "run" / "store.jsonl")
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))

    skipped = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None)
    forced = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None, force=True)

    assert skipped["status"] == cli.PRESENT
    assert forced["status"] == cli.OK


HINT_COUGH = AudioHints(metadata={"task_token": "voluntary-cough"})
HINT_RAINBOW = AudioHints(metadata={"task_token": "rainbow-passage"})


def _checked(root: Path, **kwargs: Any) -> ReplayInputs:  # noqa: ANN401
    store = read_store(root)
    return check_replay_inputs(store, root, load_triage_config(), **{"hint": None, "hint_given": False, **kwargs})


class TestTheReplayChecksTheRunsRecord:
    """What a replay or re-fold compares before it changes anything (plan step 3, check list)."""

    def test_the_admitted_recording_is_verified(self, tmp_path: Path) -> None:
        """Re-hashed, it matches what ADMIT recorded."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root)
        assert _checked(root).audio == AUDIO_VERIFIED

    def test_a_different_recording_is_refused(self, tmp_path: Path) -> None:
        """Bytes that differ from the record are a different recording."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        source = _seed_run(root)
        sf.write(str(source), _samples() * 0.5, SR)
        with pytest.raises(ReplayRefused, match="not the recording"):
            _checked(root)

    def test_a_recording_that_is_gone_is_unverified_not_refused(self, tmp_path: Path) -> None:
        """Nothing to compare is not evidence of a mistake."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root).unlink()
        assert _checked(root).audio == AUDIO_UNVERIFIED

    def test_another_profile_name_is_refused_unless_allowed(self, tmp_path: Path) -> None:
        """A run made with one dataset's profile is not replayed under another's by accident."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        source = _seed_run(root)
        recorded = {"name": "another-dataset", "version": "1", "digest": "d"}
        (root / "run" / "run.json").write_text(json.dumps({"source": str(source), "config": {"profile": recorded}}))
        with pytest.raises(ReplayRefused, match="another-dataset"):
            _checked(root)
        assert _checked(root, allow_profile_change=True).profile["recorded"] == recorded

    def test_a_run_with_no_recorded_profile_is_accepted(self, tmp_path: Path) -> None:
        """A run from before profiles: nothing to compare."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root)
        assert _checked(root).profile is None

    def test_the_recorded_hint_is_reused_without_hints(self, tmp_path: Path) -> None:
        """No --hints reads the hint the run recorded rather than replaying with none."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        inputs = _checked(root)
        assert (inputs.hint, inputs.hint_from, inputs.changed) == (HINT_COUGH, HINT_RECORDED, [])

    def test_a_store_that_recorded_no_hint_is_refused_without_hints_or_no_hint(self, tmp_path: Path) -> None:
        """A store from before ADMIT recorded hints cannot say whether one was used."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root)
        store = read_store(root)
        [stream] = store.entities("stream")
        for key in (HINT, HINT_DIGEST, HINT_FIELD_DIGESTS):
            del stream.attributes[key]
        with pytest.raises(ReplayRefused, match="records no hint"):
            check_replay_inputs(store, root, load_triage_config(), hint=None, hint_given=False)
        accepted = check_replay_inputs(store, root, load_triage_config(), hint=None, hint_given=False, no_hint=True)
        assert (accepted.hint, accepted.changed) == (None, None)

    def test_a_given_hint_that_differs_is_accepted_and_named(self, tmp_path: Path) -> None:
        """The caller's correction wins; the fields it moved are recorded."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        inputs = _checked(root, hint=HINT_RAINBOW, hint_given=True)
        assert (inputs.hint_from, inputs.changed) == (HINT_GIVEN, ["metadata.task_token"])

    def test_a_refold_refuses_a_changed_hint(self, tmp_path: Path) -> None:
        """Re-folding over branch reports made under another hint would mix the two."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        with pytest.raises(ReplayRefused, match="replay the run instead"):
            _checked(root, hint=HINT_RAINBOW, hint_given=True, allow_hint_change=False)
        assert _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False).changed == []

    def test_a_recorded_target_speaker_is_not_reused(self, tmp_path: Path) -> None:
        """Only its digest is stored, so the hint cannot be rebuilt from the record."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        speaker = TargetSpeakerEmbedding(vector=[0.1, 0.2], provenance=SpeakerEmbeddingProvenance(model_id="m"))
        _seed_run(root, hint=AudioHints(target_speaker=speaker))
        with pytest.raises(ReplayRefused, match="target speaker"):
            _checked(root)


class TestAChangedHintBecomesTheRunsRecord:
    """A replay with a changed hint records it, so every stage and the next pass read it."""

    def test_the_identity_is_superseded_and_the_marker_names_the_change(self, tmp_path: Path) -> None:
        """The recording stream is replaced; the marker says what moved and who was affected."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        config = load_triage_config()
        store = read_store(root, run_id=replay_run_id(root, config.config_hash))
        inputs = check_replay_inputs(store, root, config, hint=HINT_RAINBOW, hint_given=True)
        outcome = replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=root / "run",
            artifacts_dir=root / "released",
            summary_dir=root / "summary",
            inputs=inputs,
        )
        assert declared_task_id(store) == "rainbow-passage"
        assert changed_fields(store, HINT_RAINBOW) == []
        marker = store.get_activity(outcome.marker).parameters
        assert marker["hint_changed"] == ["metadata.task_token"]
        assert marker["identity_change"]["to"]["declared_task_id"] == "rainbow-passage"
        assert find_replay_marker(store, config.config_hash, None, hint_digest(HINT_RAINBOW)) == outcome.marker

    def test_an_unchanged_hint_rewrites_no_record(self, tmp_path: Path) -> None:
        """The same hint again leaves the recording stream as ADMIT wrote it."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        store = read_store(root)
        inputs = check_replay_inputs(store, root, load_triage_config(), hint=HINT_COUGH, hint_given=True)
        assert not inputs.hint_moved


class TestTheStimulusAlignmentFollowsTheExpectedSpeech:
    """Replay rebuilds the alignment where the expected speech it was aligned against moved."""

    def _store_with_alignment(self, digest: str | None) -> ProvStore:
        store = ProvStore(run_id="s")
        activity = store.activity(
            node="PREPROCESS", step=STIMULUS_MEASUREMENT, parameters={"expected_speech_digest": digest}
        )
        entity = store.entity(prov_type="measurement", extent=None, attributes={"name": STIMULUS_MEASUREMENT})
        store.was_generated_by(entity, activity)
        return store

    def test_matching_expected_speech_is_left_alone(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing moved, nothing is rebuilt."""
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        store = self._store_with_alignment(prompts_digest(hint.expected_speech))
        monkeypatch.setattr(extend_module, "stimulus_alignment", lambda *a, **k: pytest.fail("rebuilt"))
        assert realign_stimulus(store, load_triage_config(), hint, run_dir=tmp_path, software="s") is None

    def test_moved_expected_speech_is_realigned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The old alignment is retired and a new one is written."""
        store = self._store_with_alignment("another")
        monkeypatch.setattr(extend_module, "stimulus_alignment", lambda *a, **k: "new-alignment")
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        assert realign_stimulus(store, load_triage_config(), hint, run_dir=tmp_path, software="s") == "new-alignment"
        assert find_measurement(store, STIMULUS_MEASUREMENT) is None

    def test_a_rebuild_that_fails_leaves_the_old_alignment_standing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The old one is retired only once a replacement exists."""
        store = self._store_with_alignment("another")

        def fail(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
            raise LookupError("no live consensus_transcript")

        monkeypatch.setattr(extend_module, "stimulus_alignment", fail)
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        assert realign_stimulus(store, load_triage_config(), hint, run_dir=tmp_path, software="s") is None
        assert find_measurement(store, STIMULUS_MEASUREMENT) is not None

    def test_a_hint_with_no_expected_speech_retires_the_alignment(self, tmp_path: Path) -> None:
        """Nothing to align against: the alignment goes, with nothing in its place."""
        store = self._store_with_alignment("another")
        assert realign_stimulus(store, load_triage_config(), AudioHints(), run_dir=tmp_path, software="s") is None
        assert find_measurement(store, STIMULUS_MEASUREMENT) is None

    def test_an_alignment_that_recorded_no_digest_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written before alignments recorded one: unknown, so not rebuilt."""
        store = self._store_with_alignment(None)
        monkeypatch.setattr(extend_module, "stimulus_alignment", lambda *a, **k: pytest.fail("rebuilt"))
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        assert realign_stimulus(store, load_triage_config(), hint, run_dir=tmp_path, software="s") is None


class TestTheRecordAReplayWrites:
    """The record a changed hint writes is always live, and reads the file ADMIT read."""

    def _store(self, audio_file: str, path: str) -> ProvStore:
        store = ProvStore(run_id="r")
        store.activity(node="ADMIT", step=None, parameters={"audio_file": audio_file})
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": path})
        return store

    def test_switching_back_to_an_earlier_hint_leaves_a_live_record(self) -> None:
        """Content-addressed ids must not hand back a record that was already retired."""
        store = self._store("/d/sub-01_ses-01_task-x.wav", "/d/sub-01_ses-01_task-x.wav")
        for hint in (HINT_RAINBOW, HINT_COUGH, HINT_RAINBOW):
            record_replayed_hint(store, hint, software="s")
            assert recording_entity(store) is not None
        assert declared_task_id(store) == "rainbow-passage"

    def test_the_identity_is_read_off_the_file_admit_was_given(self) -> None:
        """ADMIT records the resolved path; the stem it read was the one it was given."""
        store = self._store("/bids/sub-01_ses-01_task-loudness-1.wav", "/exports/platform-0042.wav")
        record_replayed_hint(store, None, software="s")
        assert declared_task_id(store) == "loudness-1"


class TestTheRecordingCheckReadsTheStatFirst:
    """A recording whose size and modification time match the record is not re-hashed."""

    def test_an_unchanged_recording_is_not_hashed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading every source end to end on every pass is the cost this avoids."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        _seed_run(root)
        monkeypatch.setattr(extend_module, "file_digest", lambda *a, **k: pytest.fail("hashed"))
        assert _checked(root).audio == AUDIO_VERIFIED

    def test_a_touched_recording_is_hashed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A changed modification time sends the check to the bytes."""
        root = tmp_path / "sub-01_ses-01_task-x_20260920-030808"
        source = _seed_run(root)
        os.utime(source, ns=(1, 1))
        hashed: list[Path] = []
        real = extend_module.file_digest

        def counting(path: Path) -> Any:  # noqa: ANN401
            hashed.append(path)
            return real(path)

        monkeypatch.setattr(extend_module, "file_digest", counting)
        assert _checked(root).audio == AUDIO_VERIFIED
        assert hashed == [source]


def test_an_out_root_mirror_is_placed_where_the_replay_leaves_the_identity() -> None:
    """A given hint's identity places it; else the recorded one; else, for a run recording none, the stem's."""
    stem = "sub-01_ses-01_task-x"
    hint = AudioHints(metadata={"participant_id": "sub-02", "session_id": "ses-05"})
    moved = cli.replay_identity(None, stem, hint, hint_given=True)
    assert (moved.participant, moved.session) == ("sub-02", "ses-05")
    assert cli.replay_identity(None, stem, None, hint_given=False) == identity_of(stem)
    recorded = identity_of(stem, hint)
    assert cli.replay_identity(recorded, stem, None, hint_given=False) == recorded
