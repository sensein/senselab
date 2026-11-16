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
import yaml  # type: ignore[import-untyped]

from senselab.audio.data_structures import Audio, AudioHints, ExpectedSpeech
from senselab.audio.data_structures.audio_hints import SpeakerEmbeddingProvenance, TargetSpeakerEmbedding
from senselab.audio.workflows.triage import extend as extend_module
from senselab.audio.workflows.triage.background_model import write_view_arrays
from senselab.audio.workflows.triage.config import ACTIVE, load_triage_config
from senselab.audio.workflows.triage.dataset_profile import deactivate_profile
from senselab.audio.workflows.triage.extend import (
    AUDIO_STAT_MATCHED,
    AUDIO_UNVERIFIED,
    AUDIO_VERIFIED,
    CORPUS_INPUTS_KEY,
    DECISION_SUPERSEDED,
    HINT_GIVEN,
    HINT_RECORDED,
    OFF_STAGES_KEY,
    PROFILE_KEY,
    REFOLD_MARKER_STEP,
    REFOLD_NODE,
    REPLAY_MARKER_STEP,
    REPLAY_NODE,
    REPLAYED_NODES,
    REVIEW_KEY,
    REVIEW_STATES,
    IdentityChangeRefused,
    ReplayInputs,
    ReplayRefused,
    check_replay_inputs,
    check_run_inputs,
    corpus_inputs_digest,
    decided_under,
    find_replay_marker,
    live_decisions,
    read_store,
    realign_stimulus,
    record_moved,
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
from senselab.audio.workflows.triage.identity import (
    DECLARED_FROM,
    DECLARED_PARTICIPANT,
    DECLARED_SESSION,
    DECLARED_TASK_ID,
    declared_task_id,
    identity_of,
    recorded_identity,
    recording_entity,
)
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_measurement, software_agent, write_stream, write_verdict
from senselab.audio.workflows.triage.nodes.preprocess import STIMULUS_MEASUREMENT, write_stimulus_alignment
from senselab.audio.workflows.triage.stimulus import StimulusAlignment
from senselab.audio.workflows.triage.task_catalogue import TASK_EXPECTS_DIGEST, TASK_FROM
from senselab.audio.workflows.triage.vocabulary import GRAPH_ORDER, Outcome
from senselab.audio.workflows.triage.workflow import OFF_IN_WORKFLOW
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.conftest import hint_for
from tests.conftest import FIXTURE_PROFILE

pytestmark = pytest.mark.usefixtures("triage_profile")
"""The drivers' internals these tests call read the fixture profile's data; a driver run is given ``--profile``."""

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
    config = load_triage_config(profile=ACTIVE)
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
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
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    store = read_store(root, run_id=replay_run_id(root, "aaaa", "c" * 40))
    marker = store.activity(
        node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={"config_hash": "aaaa", "commit": "c" * 40, "retired": 1}
    )
    assert find_replay_marker(store, "aaaa", "c" * 40) == marker
    assert find_replay_marker(store, "aaaa", "d" * 40) is None
    assert find_replay_marker(store, "aaaa") == marker


def test_a_marker_written_under_the_legacy_config_hash_still_reads_as_done(tmp_path: Path) -> None:
    """A replay recorded before 2026-11-14 hashed the profile's bytes; unchanged files make it this configuration's."""
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    config = load_triage_config(profile=ACTIVE)
    assert config.legacy_config_hash is not None and config.legacy_config_hash != config.config_hash
    store = read_store(root, run_id=replay_run_id(root, config.legacy_config_hash))
    parameters = {"config_hash": config.legacy_config_hash, "commit": None, "retired": 1}
    marker = store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters=parameters)
    assert find_replay_marker(store, config.config_hashes) == marker
    assert find_replay_marker(store, config.config_hash) is None


def test_mirror_run_root_is_writable_and_reads_through_to_the_finished_run(tmp_path: Path) -> None:
    """The out-root mirror must not write into the tree it reads."""
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)
    (root / "run" / "derivatives" / "yamnet_scores.json").write_text("[]", encoding="utf-8")
    out_root = tmp_path / "replayed"

    mirrored = cli.mirror_run_root(root, out_root, identity_of(hint_for("p01", "s1", "x")))

    assert mirrored.parent == out_root / "p01" / "s1"
    assert not (mirrored / "run" / "derivatives").is_symlink()
    assert (mirrored / "run" / "derivatives" / "yamnet_scores.json").is_symlink()
    assert (mirrored / "run" / "derivatives" / "yamnet_scores.json").read_text() == "[]"
    assert not (mirrored / "run" / "streams").is_symlink()
    assert (mirrored / "run" / "streams" / "enhanced.flac").is_symlink()
    assert (mirrored / "released").is_dir()

    assert not (mirrored / "run" / "streams" / "redacted.flac").exists()


def test_the_mirror_links_the_runs_log_so_later_passes_check_its_record(tmp_path: Path) -> None:
    """Review K3: a pass over the mirror reads the finished run's run.json, so the recording is verified."""
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root, hint=HINT_COUGH)
    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of(hint_for("p01", "s1", "x")))
    assert (mirrored / "run" / "run.json").is_symlink()
    assert (mirrored / "run" / "run.json").read_text() == (root / "run" / "run.json").read_text()
    store = read_store(root)
    checked = check_replay_inputs(store, mirrored, load_triage_config(profile=ACTIVE), hint=None, hint_given=False)
    assert checked.audio == AUDIO_VERIFIED


def test_a_stream_a_replayed_node_writes_is_never_a_link_into_the_finished_run(tmp_path: Path) -> None:
    """REDACT and SPEECH write into ``streams/``; a link there would carry the write to the source."""
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of(hint_for("p01", "s1", "x")))

    links = {path.name for path in (mirrored / "run" / "streams").iterdir()}
    assert links == {"enhanced.flac", "plain.flac"}, "only the streams a replay reads may be linked"


def test_a_replayed_redact_write_leaves_the_finished_tree_byte_identical(tmp_path: Path) -> None:
    """The defect: 17,924 ``redacted.flac`` under the 2026-09-19 corpus were overwritten this way."""
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)
    before = {path.name: path.read_bytes() for path in sorted((root / "run" / "streams").iterdir()) if path.is_file()}

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of(hint_for("p01", "s1", "x")))
    for stem in ("redacted", "separated_0"):
        write_stream(Audio(waveform=_samples()[None, :] * 0.5, sampling_rate=SR), mirrored / "run", stem)

    after = {path.name: path.read_bytes() for path in sorted((root / "run" / "streams").iterdir()) if path.is_file()}
    assert after == before, "a replayed run must not write into the tree it reads"
    assert (mirrored / "run" / "streams" / "redacted.flac").read_bytes() != before["redacted.flac"]


def test_the_derivatives_a_replay_rewrites_leave_the_finished_tree_byte_identical(tmp_path: Path) -> None:
    """The defect: replayed BACKGROUND rewrote ``background_view.npz`` through a linked ``derivatives/``."""
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)
    finished = root / "run" / "derivatives"
    np.savez(finished / "background_view.npz", marker=np.array([1]))
    np.savez(finished / "stimulus_alignment.npz", marker=np.array([2]))
    before = {path.name: path.read_bytes() for path in sorted(finished.iterdir())}

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of(hint_for("p01", "s1", "x")))
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
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)
    finished = root / "run" / "derivatives"
    np.savez(finished / "background_view.npz", marker=np.array([1]))
    before = (finished / "background_view.npz").read_bytes()
    old = tmp_path / "replayed" / "p01" / "s1" / root.name / "run"
    old.mkdir(parents=True)
    (old / "derivatives").symlink_to(finished)

    mirrored = cli.mirror_run_root(root, tmp_path / "replayed", identity_of(hint_for("p01", "s1", "x")))
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
    root = tmp_path / "finished" / "rec-0001_20260920-030808"
    _seed_run(root)
    out_root = tmp_path / "replayed"

    first = cli.mirror_run_root(root, out_root, identity_of(hint_for("p01", "s1", "x")))
    second = cli.mirror_run_root(root, out_root, identity_of(hint_for("p01", "s1", "x")))

    assert first == second
    assert (second / "run" / "streams" / "enhanced.flac").is_symlink()


def test_source_of_reads_the_recording_out_of_the_run_log(tmp_path: Path) -> None:
    """The hint is rebuilt from the recording, which the run log names."""
    root = tmp_path / "rec-0001_20260920-030808"
    source = _seed_run(root)

    assert cli.source_of(root) == source


def test_source_of_refuses_a_run_with_no_log(tmp_path: Path) -> None:
    """A run whose log is missing cannot have its hint rebuilt, and says so."""
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    (root / "run" / "run.json").unlink()

    with pytest.raises(FileNotFoundError):
        cli.source_of(root)


def test_a_run_already_carrying_this_configurations_marker_is_skipped(tmp_path: Path) -> None:
    """The resume predicate: the store says whether this configuration has already replayed it."""
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    config = load_triage_config(profile=ACTIVE)
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    marker = store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={
            "config_hash": config.config_hash,
            "commit": None,
            "retired": 0,
            "hint_digest": NO_HINT,
            CORPUS_INPUTS_KEY: corpus_inputs_digest(store),
        },
    )
    store.write_jsonl(root / "run" / "store.jsonl")
    before = (root / "run" / "store.jsonl").read_text()

    record = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None)

    assert record["status"] == cli.PRESENT
    assert (root / "run" / "store.jsonl").read_text() == before, "a skipped run is not rewritten"
    assert marker


def test_a_marker_under_another_hint_does_not_skip_the_replay(tmp_path: Path) -> None:
    """A corrected sidecar is a new hint: the run replays rather than reading as already done."""
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    config = load_triage_config(profile=ACTIVE)
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
    roots = [tmp_path / f"rec-000{index}_20260920-030808" for index in range(1, 5)]
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
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    rows = [
        {"stem": "nowhere", "enhanced": str(tmp_path / "loose.flac")},
        {"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")},
    ]

    out = cli.process(rows, load_triage_config(profile=ACTIVE), build_hint=None, out_root=None, commit=None)

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
        out = cli.process(
            rows, load_triage_config(profile=ACTIVE), build_hint=None, out_root=None, commit=None, log=log
        )
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
        config=load_triage_config(profile=ACTIVE),
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
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({"SPEECH": "RuntimeError: no libtorchcodec"}))

    record = cli.replay_one(
        root, load_triage_config(profile=ACTIVE), build_hint=None, out_root=None, stem=root.name, commit=None
    )

    assert record["status"] == cli.ERRORED
    assert record["errored_nodes"] == ["SPEECH"]


def test_a_replay_with_no_node_error_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The errored status is only for a raised node; a clean replay stays ``ok``."""
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))

    record = cli.replay_one(
        root, load_triage_config(profile=ACTIVE), build_hint=None, out_root=None, stem=root.name, commit=None
    )

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
    root = tmp_path / "rec-0001_20260920-030808"
    _seed_run(root)
    config = load_triage_config(profile=ACTIVE)
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={
            "config_hash": config.config_hash,
            "commit": None,
            "retired": 0,
            "hint_digest": NO_HINT,
            CORPUS_INPUTS_KEY: corpus_inputs_digest(store),
        },
    )
    store.write_jsonl(root / "run" / "store.jsonl")
    monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))

    skipped = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None)
    forced = cli.replay_one(root, config, build_hint=None, out_root=None, stem=root.name, commit=None, force=True)

    assert skipped["status"] == cli.PRESENT
    assert forced["status"] == cli.OK


def _mark_replayed(root: Path, config: Any, **recorded: Any) -> None:  # noqa: ANN401
    """Write this configuration's replay marker into a run's store, with any further parameters."""
    store = read_store(root, run_id=replay_run_id(root, config.config_hash))
    store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={
            "config_hash": config.config_hash,
            "commit": None,
            "retired": 0,
            "hint_digest": NO_HINT,
            CORPUS_INPUTS_KEY: corpus_inputs_digest(store),
            **recorded,
        },
    )
    store.write_jsonl(root / "run" / "store.jsonl")


def test_a_present_or_failed_replay_leaves_no_out_root_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Satra review 2026-11-14, verifier round 2: the mirror is made only for a replay that runs.

    It is kept only if the replay lands; a failed replay removes what it created and nothing an earlier call made.
    """
    config = load_triage_config(profile=ACTIVE)
    present = tmp_path / "rec-0001_20260920-030808"
    _seed_run(present)
    _mark_replayed(present, config)
    out_root = tmp_path / "replayed"
    record = cli.replay_one(present, config, build_hint=None, out_root=out_root, stem=present.name, commit=None)
    assert record["status"] == cli.PRESENT
    assert not out_root.exists()

    failing = tmp_path / "rec-0002_20260920-030808"
    _seed_run(failing)

    def _raise(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cli, "replay_decisions", _raise)
    for existing in (False, True):
        if existing:
            (out_root / "earlier").mkdir(parents=True)
        record = cli.replay_one(failing, config, build_hint=None, out_root=out_root, stem=failing.name, commit=None)
        assert record["status"] == cli.ERROR and "disk full" in record[cli.DERIVATION]
        if existing:
            assert sorted(path.name for path in out_root.iterdir()) == ["earlier"], "an earlier call's tree is kept"
        else:
            assert not out_root.exists()


def test_a_resubmitted_slice_s_present_rows_keep_the_review_state_their_replay_recorded(tmp_path: Path) -> None:
    """Satra review 2026-11-14, verifier round 2: rows already replayed report the review state of that replay.

    A resubmitted slice starts a fresh row log; its already-replayed rows come back ``present`` with the state the
    replay's marker recorded, so a reading that must be read again is still named (and exits 3). A marker written
    before the state was recorded reports none.
    """
    config = load_triage_config(profile=ACTIVE)
    roots = [tmp_path / f"rec-000{n}_20260920-030808" for n in range(1, 4)]
    for root in roots:
        _seed_run(root)
    _mark_replayed(roots[0], config, **{REVIEW_KEY: "needs_reread", "review_why": "the transcript moved"})
    _mark_replayed(roots[1], config, **{REVIEW_KEY: "carried", "review_why": None})
    _mark_replayed(roots[2], config)
    manifest = _manifest(tmp_path / "manifest.jsonl", roots)
    summary = cli.run_slice(
        manifest,
        slice_index=0,
        slice_count=1,
        config=config,
        log_dir=tmp_path / "logs",
        hints=None,
        out_root=None,
        commit=None,
    )
    assert summary["counts"] == {cli.PRESENT: 3}
    assert summary["reviews"] == {"needs_reread": 1, "carried": 1}
    assert summary["needs_reread"] == [roots[0].name]
    rows = [json.loads(line) for line in Path(summary["log"]).read_text(encoding="utf-8").splitlines()]
    by_stem = {row["stem"]: row for row in rows if "stem" in row}
    assert by_stem[roots[0].name]["review_why"] == "the transcript moved"
    assert "review" not in by_stem[roots[2].name]


HINT_COUGH = AudioHints(metadata={"task_token": "voluntary-cough"})
HINT_RAINBOW = AudioHints(metadata={"task_token": "rainbow-passage"})


def _checked(root: Path, **kwargs: Any) -> ReplayInputs:  # noqa: ANN401
    store = read_store(root)
    return check_replay_inputs(
        store, root, load_triage_config(profile=ACTIVE), **{"hint": None, "hint_given": False, **kwargs}
    )


class TestTheReplayChecksTheRunsRecord:
    """What a replay or re-fold compares before it changes anything (plan step 3, check list)."""

    def test_the_admitted_recording_is_verified(self, tmp_path: Path) -> None:
        """Re-hashed, it matches what ADMIT recorded."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        assert _checked(root).audio == AUDIO_VERIFIED

    def test_a_different_recording_is_refused(self, tmp_path: Path) -> None:
        """Bytes that differ from the record are a different recording."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root)
        sf.write(str(source), _samples() * 0.5, SR)
        with pytest.raises(ReplayRefused, match="not the recording"):
            _checked(root)

    def test_a_recording_that_is_gone_is_unverified_not_refused(self, tmp_path: Path) -> None:
        """Nothing to compare is not evidence of a mistake."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root).unlink()
        assert _checked(root).audio == AUDIO_UNVERIFIED

    def test_another_profile_name_is_refused_unless_allowed(self, tmp_path: Path) -> None:
        """A run made with one dataset's profile is not replayed under another's by accident."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root)
        recorded = {"name": "another-dataset", "version": "1", "digest": "d"}
        (root / "run" / "run.json").write_text(json.dumps({"source": str(source), "config": {"profile": recorded}}))
        with pytest.raises(ReplayRefused, match="another-dataset"):
            _checked(root)
        profile = _checked(root, allow_profile_change=True).profile
        assert profile is not None
        assert profile["recorded"] == recorded

    def test_a_run_with_no_recorded_profile_is_accepted(self, tmp_path: Path) -> None:
        """A run from before profiles: nothing to compare."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        assert _checked(root).profile is None

    def test_the_recorded_hint_is_reused_without_hints(self, tmp_path: Path) -> None:
        """No --hints reads the hint the run recorded rather than replaying with none."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        inputs = _checked(root)
        assert (inputs.hint, inputs.hint_from, inputs.changed) == (HINT_COUGH, HINT_RECORDED, [])

    def test_a_store_that_recorded_no_hint_is_refused_without_hints_or_no_hint(self, tmp_path: Path) -> None:
        """A store from before ADMIT recorded hints cannot say whether one was used."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        store = read_store(root)
        [stream] = store.entities("stream")
        for key in (HINT, HINT_DIGEST, HINT_FIELD_DIGESTS):
            del stream.attributes[key]
        with pytest.raises(ReplayRefused, match="records no hint"):
            check_replay_inputs(store, root, load_triage_config(profile=ACTIVE), hint=None, hint_given=False)
        accepted = check_replay_inputs(
            store, root, load_triage_config(profile=ACTIVE), hint=None, hint_given=False, no_hint=True
        )
        assert (accepted.hint, accepted.changed) == (None, None)

    def test_a_store_that_recorded_no_task_is_refused_without_hints_or_no_hint(self, tmp_path: Path) -> None:
        """A store from before ADMIT recorded the task: its recorded hint names none, so every branch would detect."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        store = read_store(root)
        for stream in store.entities("stream"):
            stream.attributes.pop(TASK_FROM, None)
        for activity in store.activities("ADMIT"):
            activity.parameters.pop(TASK_FROM, None)
        with pytest.raises(ReplayRefused, match="records no task"):
            check_replay_inputs(store, root, load_triage_config(profile=ACTIVE), hint=None, hint_given=False)
        given = check_replay_inputs(store, root, load_triage_config(profile=ACTIVE), hint=HINT_COUGH, hint_given=True)
        assert given.hint_from == HINT_GIVEN
        none = check_replay_inputs(
            store, root, load_triage_config(profile=ACTIVE), hint=None, hint_given=False, no_hint=True
        )
        assert none.hint is None

    def test_a_run_recording_no_identity_is_refused_unless_the_hint_declares_one(self, tmp_path: Path) -> None:
        """--no-hint would record an identity declaring nothing, which every later pass accepts (step-6 review 2)."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        store = read_store(root)
        for stream in store.entities("stream"):
            for key in (DECLARED_PARTICIPANT, DECLARED_SESSION, DECLARED_TASK_ID, DECLARED_FROM):
                stream.attributes.pop(key, None)
        for activity in store.activities("ADMIT"):
            for key in (DECLARED_PARTICIPANT, DECLARED_SESSION, DECLARED_TASK_ID, DECLARED_FROM):
                activity.parameters.pop(key, None)
        with pytest.raises(ReplayRefused, match="records no identity"):
            check_replay_inputs(
                store, root, load_triage_config(profile=ACTIVE), hint=None, hint_given=False, no_hint=True
            )
        declaring = AudioHints(metadata={"participant_id": "p01", "session_id": "s1", "task_token": "x"})
        given = check_replay_inputs(store, root, load_triage_config(profile=ACTIVE), hint=declaring, hint_given=True)
        assert given.hint_from == HINT_GIVEN

    def test_a_given_hint_that_differs_is_accepted_and_named(self, tmp_path: Path) -> None:
        """The caller's correction wins; the fields it moved are recorded."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        inputs = _checked(root, hint=HINT_RAINBOW, hint_given=True)
        assert (inputs.hint_from, inputs.changed) == (HINT_GIVEN, ["metadata.task_token"])

    def test_a_refold_refuses_a_changed_hint(self, tmp_path: Path) -> None:
        """Re-folding over branch reports made under another hint would mix the two."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        with pytest.raises(ReplayRefused, match="replay the run instead"):
            _checked(root, hint=HINT_RAINBOW, hint_given=True, allow_hint_change=False)
        assert _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False).changed == []

    def test_a_refold_refuses_a_store_that_records_no_hint(self, tmp_path: Path) -> None:
        """Nothing to compare the re-fold's hint with: the message says to replay first (review K8)."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        store = read_store(root)
        [stream] = store.entities("stream")
        for key in (HINT, HINT_DIGEST, HINT_FIELD_DIGESTS):
            del stream.attributes[key]
        config = load_triage_config(profile=ACTIVE)
        with pytest.raises(ReplayRefused, match="replay it first"):
            check_replay_inputs(store, root, config, hint=HINT_RAINBOW, hint_given=True, allow_hint_change=False)
        replayed = check_replay_inputs(store, root, config, hint=HINT_RAINBOW, hint_given=True)
        assert replayed.changed is None

    def test_a_refold_refuses_decisions_taken_under_another_profile_digest(self, tmp_path: Path) -> None:
        """Same profile name, other digest: VERDICT would re-score the old reports; a replay is accepted (C3)."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root, hint=HINT_COUGH)
        config = load_triage_config(profile=ACTIVE)
        assert config.profile is not None
        recorded = {"name": config.profile.name, "version": config.profile.version, "digest": "another-digest"}
        (root / "run" / "run.json").write_text(json.dumps({"source": str(source), "config": {"profile": recorded}}))
        with pytest.raises(ReplayRefused, match="replay the run instead"):
            _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False)
        assert _checked(root, hint=HINT_COUGH, hint_given=True).profile["recorded"] == recorded

    def test_decisions_recorded_under_the_legacy_digest_are_the_same_profile(self, tmp_path: Path) -> None:
        """A byte digest recorded before 2026-11-14 names the unchanged profile: no --allow-profile-change."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root, hint=HINT_COUGH)
        config = load_triage_config(profile=ACTIVE)
        assert config.profile is not None and config.profile.legacy_digest != config.profile.digest
        recorded = {
            "name": config.profile.name,
            "version": config.profile.version,
            "digest": config.profile.legacy_digest,
        }
        (root / "run" / "run.json").write_text(json.dumps({"source": str(source), "config": {"profile": recorded}}))
        refold = _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False)
        assert refold.profile is None, "not recorded as a profile change"
        assert _checked(root, hint=HINT_COUGH, hint_given=True).profile is None

    def test_a_refold_refuses_decisions_taken_with_other_stages_off(self, tmp_path: Path) -> None:
        """The run left REDACT off and this configuration runs it: a re-fold would fold another graph (C3)."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root, hint=HINT_COUGH)
        log = {"source": str(source), "notes": {"REDACT": OFF_IN_WORKFLOW}}
        (root / "run" / "run.json").write_text(json.dumps(log))
        with pytest.raises(ReplayRefused, match="off"):
            _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False)
        (root / "run" / "run.json").write_text(json.dumps({**log, "notes": {}}))
        assert _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False).changed == []

    def test_the_stages_and_profile_a_replay_ran_under_speak_for_its_decisions(self, tmp_path: Path) -> None:
        """A replay's marker records the stages off; the next re-fold compares against it, not run.json."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root, hint=HINT_COUGH)
        (root / "run" / "run.json").write_text(json.dumps({"source": str(source), "notes": {"REDACT": "x"}}))
        store = read_store(root)
        assert decided_under(store, {"notes": {"REDACT": OFF_IN_WORKFLOW}})[1] == frozenset({"REDACT"})
        store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={OFF_STAGES_KEY: []})
        assert decided_under(store, {"notes": {"REDACT": OFF_IN_WORKFLOW}}) == (None, frozenset())

    def test_a_marker_s_profile_speaks_for_its_decisions_over_run_json(self, tmp_path: Path) -> None:
        """The profile a marker records is the one its decisions were taken under, whatever run.json says."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        store = read_store(root)
        recorded = {"name": "p", "version": "1", "digest": "legacy"}
        log = {"config": {"profile": recorded}}
        assert decided_under(store, log)[0] == recorded
        now = {"name": "p", "version": "1", "digest": "current"}
        store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={PROFILE_KEY: now})
        assert decided_under(store, log)[0] == now
        store.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={"profile_change": None})
        assert decided_under(store, log)[0] == recorded, "a marker from before 2026-11-14 that recorded no change"

    def test_the_marker_that_ended_last_speaks_though_it_repeats_an_earlier_one(self, tmp_path: Path) -> None:
        """Satra review 2026-11-14, verifier round 2: markers P, Q, then P again: P's decisions stand.

        A marker is content-addressed, so the repeated P keeps its first place in the store; its later end decides.
        """
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        store = read_store(root)
        log = {"config": {"profile": {"name": "p", "version": "1", "digest": "run"}}}
        p_profile = {"name": "p", "version": "1", "digest": "P"}
        q_profile = {"name": "p", "version": "1", "digest": "Q"}
        p_marker = {PROFILE_KEY: p_profile, OFF_STAGES_KEY: []}
        first = store.activity(
            node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters=p_marker, ended="2026-11-14T01:00:00+00:00"
        )
        store.activity(
            node=REFOLD_NODE,
            step=REFOLD_MARKER_STEP,
            parameters={PROFILE_KEY: q_profile, OFF_STAGES_KEY: ["REDACT"]},
            ended="2026-11-14T02:00:00+00:00",
        )
        assert decided_under(store, log) == (q_profile, frozenset({"REDACT"}))
        again = store.activity(
            node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters=p_marker, ended="2026-11-14T03:00:00+00:00"
        )
        assert again == first and store.activities()[-1].id != again, "the repeat keeps its first place"
        assert decided_under(store, log) == (p_profile, frozenset())

    def test_a_legacy_run_replayed_once_takes_a_later_comment_edit_as_no_change(self, tmp_path: Path) -> None:
        """Recorded under the byte digest; one replay records the current one, so a comment edit is no change."""
        profile_root = tmp_path / "profile"
        profile_root.mkdir()
        manifest = yaml.safe_dump({"name": "extends-fixture", "version": "1", "extends": str(FIXTURE_PROFILE)})
        (profile_root / "profile.yaml").write_text(manifest)
        deactivate_profile()
        config = load_triage_config(profile=profile_root)
        assert config.profile is not None and config.profile.legacy_digest != config.profile.digest
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root, hint=HINT_COUGH)
        legacy = {"name": "extends-fixture", "version": "1", "digest": config.profile.legacy_digest}
        log = {"source": str(source), "config": {"profile": legacy}}
        (root / "run" / "run.json").write_text(json.dumps(log))
        store = read_store(root, run_id=replay_run_id(root, config.config_hash))
        inputs = check_replay_inputs(store, root, config, hint=HINT_COUGH, hint_given=True)
        assert inputs.profile is None, "the legacy digest names the unchanged profile"
        outcome = replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=root / "run",
            artifacts_dir=root / "released",
            summary_dir=root / "summary",
            inputs=inputs,
        )
        current = {"name": "extends-fixture", "version": "1", "digest": config.profile.digest}
        written = store.get_activity(outcome.marker)
        assert written.parameters[PROFILE_KEY] == current
        assert written.ended is not None, "the marker records when it ended, which orders it among the others"
        assert written.parameters[REVIEW_KEY] == outcome.review and outcome.review in REVIEW_STATES

        (profile_root / "profile.yaml").write_text("# corrected wording\n" + manifest)
        deactivate_profile()
        edited = load_triage_config(profile=profile_root)
        assert edited.profile is not None and edited.profile.legacy_digest != config.profile.legacy_digest
        assert edited.config_hash == config.config_hash
        before = read_store(root)
        with pytest.raises(ReplayRefused, match="replay the run instead"):
            check_replay_inputs(before, root, edited, hint=HINT_COUGH, hint_given=True, allow_hint_change=False)
        refold = check_replay_inputs(store, root, edited, hint=HINT_COUGH, hint_given=True, allow_hint_change=False)
        assert refold.profile is None, "no --allow-profile-change, and not recorded as a change"
        assert check_replay_inputs(store, root, edited, hint=HINT_COUGH, hint_given=True).profile is None
        assert decided_under(store, log)[0] == current

    def test_a_hint_whose_identity_cannot_be_built_is_refused_not_raised(self, tmp_path: Path) -> None:
        """A label that cannot name a folder refuses this row; the rest of the slice goes on (review K4)."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        bad = AudioHints(metadata={"participant_id": "a/b", "task_token": "voluntary-cough"})
        with pytest.raises(ReplayRefused, match="cannot name a folder"):
            _checked(root, hint=bad, hint_given=True)

    def test_a_recorded_target_speaker_is_not_reused(self, tmp_path: Path) -> None:
        """Only its digest is stored, so the hint cannot be rebuilt from the record."""
        root = tmp_path / "rec-0001_20260920-030808"
        speaker = TargetSpeakerEmbedding(vector=[0.1, 0.2], provenance=SpeakerEmbeddingProvenance(model_id="m"))
        _seed_run(root, hint=AudioHints(target_speaker=speaker))
        with pytest.raises(ReplayRefused, match="target speaker"):
            _checked(root)


class TestAChangedHintBecomesTheRunsRecord:
    """A replay with a changed hint records it, so every stage and the next pass read it."""

    def test_the_identity_is_superseded_and_the_marker_names_the_change(self, tmp_path: Path) -> None:
        """The recording stream is replaced; the marker says what moved and who was affected."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        config = load_triage_config(profile=ACTIVE)
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
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        store = read_store(root)
        inputs = check_replay_inputs(store, root, load_triage_config(profile=ACTIVE), hint=HINT_COUGH, hint_given=True)
        assert not inputs.hint_moved
        assert not record_moved(store, HINT_COUGH)

    def test_a_task_the_task_list_now_describes_otherwise_is_recorded_anew(self, tmp_path: Path) -> None:
        """The hint is unchanged but the recorded description is not the list's now: the replay records it (C1)."""
        hint = hint_for("p01", "s1", "voluntary-cough")
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=hint)
        config = load_triage_config(profile=ACTIVE)
        store = read_store(root, run_id=replay_run_id(root, config.config_hash))
        held = recording_entity(store)
        assert held is not None
        held.attributes[TASK_EXPECTS_DIGEST] = "described-otherwise"
        assert record_moved(store, hint)
        inputs = check_replay_inputs(store, root, config, hint=hint, hint_given=True)
        assert not inputs.hint_moved
        outcome = replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=root / "run",
            artifacts_dir=root / "released",
            summary_dir=root / "summary",
            inputs=inputs,
        )
        recorded = recording_entity(store)
        assert recorded is not None and recorded.id != held.id
        assert recorded.attributes[TASK_EXPECTS_DIGEST] != "described-otherwise"
        assert outcome.identity_change is None
        assert not record_moved(store, hint)

    def test_the_identity_change_is_on_the_outcome_as_on_the_marker(self, tmp_path: Path) -> None:
        """The replay log's row names what moved, not only the marker (review K1)."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        config = load_triage_config(profile=ACTIVE)
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
        marker = store.get_activity(outcome.marker).parameters
        assert outcome.identity_change == marker["identity_change"]
        assert outcome.identity_change is not None
        assert outcome.identity_change["to"]["declared_task_id"] == "rainbow-passage"
        assert marker[OFF_STAGES_KEY] == []


def _strip_identity(store: ProvStore) -> None:
    """Drop the identity ADMIT recorded, as on a store written before ADMIT recorded one."""
    for stream in store.entities("stream"):
        for key in (DECLARED_PARTICIPANT, DECLARED_SESSION, DECLARED_TASK_ID, DECLARED_FROM):
            stream.attributes.pop(key, None)
    for activity in store.activities("ADMIT"):
        for key in (DECLARED_PARTICIPANT, DECLARED_SESSION, DECLARED_TASK_ID, DECLARED_FROM):
            activity.parameters.pop(key, None)


class TestAReplayDoesNotMoveTheParticipantOrSession:
    """A hint naming another participant or session is refused unless allowed (Satra review 2026-11-14, decision 2)."""

    HINT = hint_for("p01", "s1", "voluntary-cough")

    def test_another_participant_is_refused(self, tmp_path: Path) -> None:
        """A wrong --hints directory would re-identify the run; the replay refuses and names the change."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        with pytest.raises(IdentityChangeRefused, match="--allow-identity-change") as refused:
            _checked(root, hint=hint_for("p02", "s1", "voluntary-cough"), hint_given=True)
        assert refused.value.change == {"participant": {"from": "p01", "to": "p02"}}

    def test_another_session_is_refused(self, tmp_path: Path) -> None:
        """The session is guarded as the participant is."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        with pytest.raises(IdentityChangeRefused) as refused:
            _checked(root, hint=hint_for("p01", "s2", "voluntary-cough"), hint_given=True)
        assert refused.value.change == {"session": {"from": "s1", "to": "s2"}}

    def test_an_allowed_change_is_replayed_and_recorded_as_an_identity_change(self, tmp_path: Path) -> None:
        """With the flag the new participant becomes the run's record, and the marker says what moved."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        config = load_triage_config(profile=ACTIVE)
        store = read_store(root, run_id=replay_run_id(root, config.config_hash))
        moved = hint_for("p02", "s1", "voluntary-cough")
        inputs = check_replay_inputs(store, root, config, hint=moved, hint_given=True, allow_identity_change=True)
        outcome = replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=root / "run",
            artifacts_dir=root / "released",
            summary_dir=root / "summary",
            inputs=inputs,
        )
        recorded = recorded_identity(store)
        assert recorded is not None and recorded.participant == "p02"
        marker = store.get_activity(outcome.marker).parameters
        assert marker["identity_change"]["from"][DECLARED_PARTICIPANT] == "p01"
        assert marker["identity_change"]["to"][DECLARED_PARTICIPANT] == "p02"

    def test_a_first_record_is_not_a_change(self, tmp_path: Path) -> None:
        """A run that records no identity yet (a v4-era store) records the hint's."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        store = read_store(root)
        _strip_identity(store)
        given = check_replay_inputs(
            store,
            root,
            load_triage_config(profile=ACTIVE),
            hint=hint_for("p02", "s2", "voluntary-cough"),
            hint_given=True,
        )
        assert given.hint_from == HINT_GIVEN

    def test_a_task_change_is_accepted(self, tmp_path: Path) -> None:
        """A corrected task is what a replay is for."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        inputs = _checked(root, hint=hint_for("p01", "s1", "rainbow-passage"), hint_given=True)
        assert inputs.hint_from == HINT_GIVEN and inputs.changed

    def _replayed(self, root: Path, hint: AudioHints | None, **kwargs: Any) -> ProvStore:  # noqa: ANN401
        config = load_triage_config(profile=ACTIVE)
        store = read_store(root, run_id=replay_run_id(root, config.config_hash))
        inputs = check_replay_inputs(store, root, config, hint=hint, **kwargs)
        replay_decisions(
            store,
            config,
            inputs.hint,
            run_dir=root / "run",
            artifacts_dir=root / "released",
            summary_dir=root / "summary",
            inputs=inputs,
        )
        return store

    def test_no_hint_keeps_the_recorded_participant_and_session(self, tmp_path: Path) -> None:
        """--no-hint declares nothing, which is not a change: not refused, and the identity is not cleared."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        store = self._replayed(root, None, hint_given=False, no_hint=True)
        recorded = recorded_identity(store)
        assert recorded is not None and (recorded.participant, recorded.session) == ("p01", "s1")
        assert {"participant", "session"} <= set(recorded.sources)

    def test_a_hint_declaring_only_the_task_keeps_the_participant_and_session(self, tmp_path: Path) -> None:
        """A task-only hint moves the task and leaves the recorded participant and session standing."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        store = self._replayed(root, hint_for(task="rainbow-passage"), hint_given=True)
        recorded = recorded_identity(store)
        assert recorded is not None and (recorded.participant, recorded.session) == ("p01", "s1")
        assert recorded.task_id == "rainbow-passage"
        assert (
            cli.replay_identity(
                identity_of(self.HINT), root.name, hint_for(task="rainbow-passage"), hint_given=True
            ).participant
            == "p01"
        )

    def test_a_refold_is_not_refused_by_the_identity_guard(self, tmp_path: Path) -> None:
        """A re-fold given another participant is refused by its own hint-differs rule, not as an identity change."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        with pytest.raises(ReplayRefused, match="replay the run instead") as refused:
            _checked(root, hint=hint_for("p02", "s1", "voluntary-cough"), hint_given=True, allow_hint_change=False)
        assert not isinstance(refused.value, IdentityChangeRefused)
        assert _checked(root, hint=self.HINT, hint_given=True, allow_hint_change=False).changed == []

    def test_an_empty_recorded_participant_takes_the_hints(self, tmp_path: Path) -> None:
        """A participant the run records empty is a first record when the hint declares one, not a change (Q1)."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=hint_for(session="s1", task="voluntary-cough"))
        before = recorded_identity(read_store(root))
        assert before is not None and before.participant is None
        store = self._replayed(root, self.HINT, hint_given=True)
        recorded = recorded_identity(store)
        assert recorded is not None and (recorded.participant, recorded.session) == ("p01", "s1")
        assert "participant" in recorded.sources

    def test_a_recorded_participant_still_refuses_beside_an_empty_session(self, tmp_path: Path) -> None:
        """Only a recorded value that the hint changes is refused; the empty session it fills is not named."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=hint_for("p01", task="voluntary-cough"))
        with pytest.raises(IdentityChangeRefused) as refused:
            _checked(root, hint=hint_for("p02", "s1", "voluntary-cough"), hint_given=True)
        assert refused.value.change == {"participant": {"from": "p01", "to": "p02"}}

    def test_the_driver_row_names_the_refused_change(self, tmp_path: Path) -> None:
        """The row is refused, names the change, and the store is left as it was."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        before = (root / "run" / "store.jsonl").read_text()
        moved = hint_for("p02", "s1", "voluntary-cough")
        record = cli.replay_one(
            root,
            load_triage_config(profile=ACTIVE),
            build_hint=lambda _: (moved,),
            out_root=None,
            stem=root.name,
            commit=None,
        )
        assert record["status"] == cli.REFUSED
        assert record["identity_refused"] == {"participant": {"from": "p01", "to": "p02"}}
        assert (root / "run" / "store.jsonl").read_text() == before

    def test_a_refused_replay_leaves_no_out_root_mirror(self, tmp_path: Path) -> None:
        """The check runs before the mirror is made, so nothing is created under the identity the hint would give."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        out_root = tmp_path / "replayed"
        moved = hint_for("p02", "s1", "voluntary-cough")
        record = cli.replay_one(
            root,
            load_triage_config(profile=ACTIVE),
            build_hint=lambda _: (moved,),
            out_root=out_root,
            stem=root.name,
            commit=None,
        )
        assert record["status"] == cli.REFUSED and record["identity_refused"]
        assert not (out_root / "p02").exists()
        assert not out_root.exists() or not any(out_root.rglob("*"))

    def test_an_accepted_replay_writes_its_out_root_mirror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same replay, allowed, mirrors the run under the new participant and writes its store there."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=self.HINT)
        out_root = tmp_path / "replayed"
        moved = hint_for("p02", "s1", "voluntary-cough")
        monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))
        record = cli.replay_one(
            root,
            load_triage_config(profile=ACTIVE),
            build_hint=lambda _: (moved,),
            out_root=out_root,
            stem=root.name,
            commit=None,
            allow_identity_change=True,
        )
        assert record["status"] == cli.OK
        assert Path(record["run_root"]) == out_root / "p02" / "s1" / root.name
        assert (out_root / "p02" / "s1" / root.name / "run" / "store.jsonl").is_file()

    def test_the_slice_lists_the_participants_an_identity_change_moved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Both sides of each participant or session change are listed, so the cohort pass reruns only those (check 9).

        Satra review 2026-11-14, verifier round 2: a first record counts (the corpus passes refuse a store with no
        identity, so the cohort pass sees that run for the first time); a task-only change and a change of where the
        identity came from move no run, so they list nobody.
        """
        roots = [tmp_path / f"run{n}" for n in range(6)]
        for root in roots:
            _seed_run(root)
        manifest = _manifest(tmp_path / "manifest.jsonl", roots)

        def _side(participant: str, session: str = "s1", task: str = "t", source: str = "hint") -> dict[str, str]:
            return {
                DECLARED_PARTICIPANT: participant,
                DECLARED_SESSION: session,
                DECLARED_TASK_ID: task,
                DECLARED_FROM: source,
            }

        rows = iter(
            [
                {"status": cli.OK, "identity_change": {"from": _side("p01"), "to": _side("p02")}},
                {"status": cli.OK, "identity_change": {"from": None, "to": _side("p03")}},
                {"status": cli.OK, "identity_change": {"from": _side("p04", "s1"), "to": _side("p04", "s2")}},
                {"status": cli.OK, "identity_change": {"from": _side("p05", task="a"), "to": _side("p05", task="b")}},
                {"status": cli.OK, "identity_change": {"from": _side("p06", source="x"), "to": _side("p06")}},
                {"status": cli.OK, "retired": 0},
            ]
        )
        monkeypatch.setattr(cli, "replay_one", lambda run_root, config, **kwargs: next(rows))
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)]) == 0
        [summary_path] = (tmp_path / "slices").glob("*.summary.json")
        summary = json.loads(summary_path.read_text())
        assert summary["identity_participants"] == ["p01", "p02", "p03", "p04"]
        assert len(summary["identity_changed"]) == 5, "every recorded change is still named"
        assert "participants affected: p01 p02 p03 p04" in capsys.readouterr().out

    def test_a_reread_outranks_a_refusal_in_the_exit_code_and_the_refusal_is_still_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A shard with both exits 3, as main() documents, and still names the refused run."""
        roots = [tmp_path / f"run{n}" for n in range(2)]
        for root in roots:
            _seed_run(root)
        manifest = _manifest(tmp_path / "manifest.jsonl", roots)
        rows = iter(
            [
                {"status": cli.REFUSED, "identity_refused": {"session": {"from": "s1", "to": "s2"}}},
                {"status": cli.OK, "retired": 0, "review": "needs_reread", "review_why": "x"},
            ]
        )
        monkeypatch.setattr(cli, "replay_one", lambda run_root, config, **kwargs: next(rows))
        code = cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)])
        assert code == 3
        [summary_path] = (tmp_path / "slices").glob("*.summary.json")
        assert json.loads(summary_path.read_text())["identity_refused"] == ["run0"]
        assert "REFUSED: 1 runs" in capsys.readouterr().err

    def test_the_slice_counts_refused_identity_changes_and_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The summary lists the stems refused for an identity change, and the task fails."""
        roots = [tmp_path / f"run{n}" for n in range(2)]
        for root in roots:
            _seed_run(root)
        manifest = _manifest(tmp_path / "manifest.jsonl", roots)
        rows = iter(
            [
                {"status": cli.REFUSED, "identity_refused": {"session": {"from": "s1", "to": "s2"}}},
                {"status": cli.OK, "retired": 0},
            ]
        )
        monkeypatch.setattr(cli, "replay_one", lambda run_root, config, **kwargs: next(rows))
        code = cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)])
        assert code == 1
        [summary_path] = (tmp_path / "slices").glob("*.summary.json")
        assert json.loads(summary_path.read_text())["identity_refused"] == ["run0"]
        assert "REFUSED: 1 runs" in capsys.readouterr().err

    def test_every_refused_row_s_reason_is_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Satra review 2026-11-14, verifier round 2: a refusal that is not an identity change is printed too."""
        roots = [tmp_path / f"run{n}" for n in range(2)]
        for root in roots:
            _seed_run(root)
        manifest = _manifest(tmp_path / "manifest.jsonl", roots)
        rows = iter(
            [
                {"status": cli.REFUSED, cli.DERIVATION: "the recording's digest differs from ADMIT's"},
                {"status": cli.REFUSED, cli.DERIVATION: "moves session", "identity_refused": {"session": {}}},
            ]
        )
        monkeypatch.setattr(cli, "replay_one", lambda run_root, config, **kwargs: next(rows))
        code = cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)])
        assert code == 1
        err = capsys.readouterr().err
        assert "REFUSED run0: the recording's digest differs from ADMIT's" in err
        assert "REFUSED run1: moves session" in err
        [summary_path] = (tmp_path / "slices").glob("*.summary.json")
        assert [row["stem"] for row in json.loads(summary_path.read_text())["refused"]] == ["run0", "run1"]

    def test_the_driver_passes_allow_identity_change(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--allow-identity-change reaches each row's replay."""
        root = tmp_path / "run0"
        _seed_run(root)
        manifest = _manifest(tmp_path / "manifest.jsonl", [root])
        seen: list[object] = []

        def _fake(run_root: Path, config: object, **kwargs: object) -> dict:
            seen.append(kwargs["allow_identity_change"])
            return {"status": cli.OK, "retired": 0}

        monkeypatch.setattr(cli, "replay_one", _fake)
        argv = [str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(tmp_path)]
        assert cli.main([*argv, "--allow-identity-change"]) == 0
        assert cli.main(argv) == 0
        assert seen == [True, False]


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
        assert realign_stimulus(store, load_triage_config(profile=ACTIVE), hint, run_dir=tmp_path, software="s") is None

    def test_moved_expected_speech_is_realigned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The old alignment is retired and a new one is written."""
        store = self._store_with_alignment("another")
        monkeypatch.setattr(extend_module, "stimulus_alignment", lambda *a, **k: "new-alignment")
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        assert (
            realign_stimulus(store, load_triage_config(profile=ACTIVE), hint, run_dir=tmp_path, software="s")
            == "new-alignment"
        )
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
        assert realign_stimulus(store, load_triage_config(profile=ACTIVE), hint, run_dir=tmp_path, software="s") is None
        assert find_measurement(store, STIMULUS_MEASUREMENT) is not None

    def test_a_hint_with_no_expected_speech_retires_the_alignment(self, tmp_path: Path) -> None:
        """Nothing to align against: the alignment goes, with nothing in its place."""
        store = self._store_with_alignment("another")
        assert (
            realign_stimulus(store, load_triage_config(profile=ACTIVE), AudioHints(), run_dir=tmp_path, software="s")
            is None
        )
        assert find_measurement(store, STIMULUS_MEASUREMENT) is None

    def test_an_alignment_that_recorded_no_digest_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Written before alignments recorded one: unknown, so not rebuilt."""
        store = self._store_with_alignment(None)
        monkeypatch.setattr(extend_module, "stimulus_alignment", lambda *a, **k: pytest.fail("rebuilt"))
        hint = AudioHints(expected_speech=[ExpectedSpeech(text="Hello.")])
        assert realign_stimulus(store, load_triage_config(profile=ACTIVE), hint, run_dir=tmp_path, software="s") is None


class TestTheRecordAReplayWrites:
    """The record a changed hint writes is always live, and reads the file ADMIT read."""

    def _store(self, audio_file: str, path: str) -> ProvStore:
        store = ProvStore(run_id="r")
        store.activity(node="ADMIT", step=None, parameters={"audio_file": audio_file})
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": path})
        return store

    def test_switching_back_to_an_earlier_hint_leaves_a_live_record(self) -> None:
        """Content-addressed ids must not hand back a record that was already retired."""
        store = self._store("/d/rec-0001.wav", "/d/rec-0001.wav")
        for hint in (HINT_RAINBOW, HINT_COUGH, HINT_RAINBOW):
            record_replayed_hint(store, hint, software="s")
            assert recording_entity(store) is not None
        assert declared_task_id(store) == "rainbow-passage"

    def test_a_refused_recording_gets_the_identity_on_the_replays_activity(self) -> None:
        """ADMIT wrote no stream: the replay records the identity on its own activity, and readers see it (K7)."""
        store = ProvStore(run_id="refused")
        store.activity(node="ADMIT", step=None, parameters={"audio_file": "/d/rec-0001.wav"})
        assert recorded_identity(store) is None
        assert record_replayed_hint(store, hint_for("p01", "s1", "rainbow-passage"), software="s") is not None
        identity = recorded_identity(store)
        assert identity is not None
        assert (identity.participant, identity.session, identity.task_id) == ("p01", "s1", "rainbow-passage")
        assert declared_task_id(store) == "rainbow-passage"

    def test_without_a_hint_nothing_is_read_off_the_file_admit_was_given(self) -> None:
        """A replay without a hint records an identity declaring nothing, whatever the file was called (step 6a)."""
        store = self._store("/bids/sub-01_ses-01_task-loudness-1.wav", "/exports/platform-0042.wav")
        record_replayed_hint(store, None, software="s")
        assert declared_task_id(store) == "unknown"

    def test_what_the_new_hint_no_longer_declares_is_dropped_but_a_corpus_path_id_is_kept(self) -> None:
        """Review of step 9: a dropped target speaker or declared id does not survive; a corpus-path id does."""
        from senselab.audio.data_structures import TargetSpeaker
        from senselab.audio.workflows.triage.identity import recorded_identity

        plain = AudioHints(metadata={"participant_id": "p01"})
        store = self._store("/c/p01/s2/recording.wav", "/c/p01/s2/recording.wav")
        declared = AudioHints(
            metadata={"participant_id": "p01", "recording_id": "r-1"}, target_speakers=[TargetSpeaker(id="p01:child")]
        )
        record_replayed_hint(store, declared, software="s")
        first = recorded_identity(store)
        assert first is not None and (first.target_speaker, first.recording_id) == ("p01:child", "r-1")
        record_replayed_hint(store, plain, software="s")
        dropped = recorded_identity(store)
        assert dropped is not None and (dropped.target_speaker, dropped.recording_id) == (None, None)

        flat = ProvStore(run_id="flat")
        flat.activity(node="ADMIT", step=None, parameters={"audio_file": "/c/p01/s2/recording.wav"})
        admitted = identity_of(plain, stem="recording", corpus_path="p01/s2/recording.wav").as_attributes()
        flat.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", **admitted})
        record_replayed_hint(flat, plain, software="s")
        kept = recorded_identity(flat)
        assert kept is not None and kept.id_of("recording") == "p01/s2/recording.wav"


def _substitute_keeping_stat(source: Path) -> None:
    """Change one byte of the recording, then restore its modification time: same size, same mtime, other bytes."""
    before = source.stat()
    substitute = bytearray(source.read_bytes())
    substitute[-1] ^= 0xFF
    source.write_bytes(bytes(substitute))
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = source.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def _counting_digests(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every recording the check hashes."""
    hashed: list[Path] = []
    real = extend_module.file_digest

    def counting(path: Path) -> Any:  # noqa: ANN401
        hashed.append(path)
        return real(path)

    monkeypatch.setattr(extend_module, "file_digest", counting)
    return hashed


class TestTheRecordingCheck:
    """Replay always re-hashes; the other passes take a matching size and mtime as ``stat_matched`` (#613-1)."""

    def test_replay_hashes_an_unchanged_recording(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Size and modification time as ADMIT recorded them still send the library replay check to the bytes."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root)
        hashed = _counting_digests(monkeypatch)
        assert _checked(root).audio == AUDIO_VERIFIED
        assert hashed == [source]

    def test_replay_refuses_other_bytes_of_the_same_size_and_time(self, tmp_path: Path) -> None:
        """A substitute copied with its size and modification time preserved is refused, not verified."""
        root = tmp_path / "rec-0001_20260920-030808"
        _substitute_keeping_stat(_seed_run(root))
        with pytest.raises(ReplayRefused, match="digest differs"):
            _checked(root)

    def test_a_refold_records_a_matching_stat_as_stat_matched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same check as a re-fold (allow_hint_change False) does not re-hash a recording whose stat matches."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root, hint=HINT_COUGH)
        hashed = _counting_digests(monkeypatch)
        assert _checked(root, hint=HINT_COUGH, hint_given=True, allow_hint_change=False).audio == AUDIO_STAT_MATCHED
        assert hashed == []

    def test_the_replay_driver_hashes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The driver's replay check hashes too, so the substitute's row is refused."""
        root = tmp_path / "rec-0001_20260920-030808"
        _substitute_keeping_stat(_seed_run(root))
        monkeypatch.setattr(cli, "replay_decisions", _stub_replay({}))
        record = cli.replay_one(
            root, load_triage_config(profile=ACTIVE), build_hint=None, out_root=None, stem=root.name, commit=None
        )
        assert record["status"] == cli.REFUSED
        assert "digest differs" in record[cli.DERIVATION]

    def test_another_pass_records_a_matching_stat_as_stat_matched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A corpus pass does not re-hash a recording whose stat matches, and does not call it verified."""
        root = tmp_path / "rec-0001_20260920-030808"
        _seed_run(root)
        hashed = _counting_digests(monkeypatch)
        audio, _ = check_run_inputs(read_store(root), root, load_triage_config(profile=ACTIVE))
        assert audio == AUDIO_STAT_MATCHED
        assert hashed == []

    def test_another_pass_hashes_a_touched_recording(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A changed modification time sends the check to the bytes, which then verify."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root)
        os.utime(source, ns=(1, 1))
        hashed = _counting_digests(monkeypatch)
        audio, _ = check_run_inputs(read_store(root), root, load_triage_config(profile=ACTIVE))
        assert audio == AUDIO_VERIFIED
        assert hashed == [source]

    def test_another_pass_refuses_touched_other_bytes(self, tmp_path: Path) -> None:
        """Other bytes under another modification time are hashed and refused."""
        root = tmp_path / "rec-0001_20260920-030808"
        source = _seed_run(root)
        sf.write(str(source), _samples() * 0.5, SR)
        with pytest.raises(ReplayRefused, match="digest differs"):
            check_run_inputs(read_store(root), root, load_triage_config(profile=ACTIVE))


def test_an_out_root_mirror_is_placed_where_the_replay_leaves_the_identity() -> None:
    """A given hint's identity places it; else the recorded one; else, for a run recording none, nothing declared."""
    stem = "rec-0001"
    hint = AudioHints(metadata={"participant_id": "p02", "session_id": "s5"})
    moved = cli.replay_identity(None, stem, hint, hint_given=True)
    assert (moved.participant, moved.session) == ("p02", "s5")
    assert cli.replay_identity(None, stem, None, hint_given=False) == identity_of(None)
    recorded = identity_of(hint)
    assert cli.replay_identity(recorded, stem, None, hint_given=False) == recorded


def test_a_marker_matches_only_the_corpus_readings_it_left() -> None:
    """A replay after extend_cohort reran replays again; a marker from before plan step 9 matches nothing."""
    from senselab.audio.workflows.triage.cohort_fold import COHORT_READING
    from senselab.audio.workflows.triage.extend import (
        CORPUS_INPUTS_KEY,
        REPLAY_MARKER_STEP,
        REPLAY_NODE,
        corpus_inputs_digest,
        find_replay_marker,
    )

    store = ProvStore(run_id="marker")
    left = corpus_inputs_digest(store)
    marker = store.activity(
        node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={"config_hash": "c", CORPUS_INPUTS_KEY: left}
    )
    assert find_replay_marker(store, "c", corpus_inputs=corpus_inputs_digest(store)) == marker
    store.entity(prov_type="measurement", extent=None, attributes={"name": COHORT_READING, "value": 1})
    assert find_replay_marker(store, "c", corpus_inputs=corpus_inputs_digest(store)) is None
    old = ProvStore(run_id="old")
    old.activity(node=REPLAY_NODE, step=REPLAY_MARKER_STEP, parameters={"config_hash": "c"})
    assert find_replay_marker(old, "c", corpus_inputs=corpus_inputs_digest(old)) is None


def test_a_cohort_pass_that_changes_nothing_leaves_the_marker_matching() -> None:
    """Review of step 9: rewriting the reading that stands mints nothing, so replay and cohort passes settle."""
    from senselab.audio.workflows.triage.cohort_stage import write_cohort_reading
    from senselab.audio.workflows.triage.extend import (
        CORPUS_INPUTS_KEY,
        REPLAY_MARKER_STEP,
        REPLAY_NODE,
        corpus_inputs_digest,
        find_replay_marker,
    )

    store = ProvStore(run_id="settled")
    reading = {"version": "1", "cohort_key": "k", "status": "computed", "cosine": 0.9}
    write_cohort_reading(store, reading)
    marker = store.activity(
        node=REPLAY_NODE,
        step=REPLAY_MARKER_STEP,
        parameters={"config_hash": "c", CORPUS_INPUTS_KEY: corpus_inputs_digest(store)},
    )
    write_cohort_reading(store, dict(reading))
    assert find_replay_marker(store, "c", corpus_inputs=corpus_inputs_digest(store)) == marker
    write_cohort_reading(store, {**reading, "cosine": 0.5})
    assert find_replay_marker(store, "c", corpus_inputs=corpus_inputs_digest(store)) is None
