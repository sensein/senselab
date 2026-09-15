"""The diarization extend-in-place driver: what it adds to a finished run, and what it leaves alone.

Nothing here loads pyannote. Its checkpoints are gated and a pass over one recording is seconds of
CPU, so the model spec and the diarizer are monkeypatched **on the node module**, not on this
driver: the driver calls ``preprocess.diarization``, which is where those two names are resolved.
Patching them here instead would leave the real model in the call and the tests would need a token.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.nodes import preprocess as preprocess_module
from senselab.audio.workflows.triage.nodes.common import find_measurement, path_attributes, software_agent
from senselab.audio.workflows.triage.nodes.preprocess import diarization_measurement
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_diarization.py"
_spec = importlib.util.spec_from_file_location("extend_diarization_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_diarization_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
ENHANCED_DIARIZATION = diarization_measurement("enhanced")
RESIDUAL_DIARIZATION = diarization_measurement("residual")


class _FakeModel:
    """A model spec stub carrying exactly what the block reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "b" * 40


def _diarizer(*segments: tuple[float, float, str]) -> Callable[..., list]:
    """A stand-in diarizer returning one fixed set of ``(start, end, speaker)`` segments per audio."""

    def _diarize(audios: list, **kwargs: Any) -> list:  # noqa: ANN401
        return [[ScriptLine(speaker=speaker, start=start, end=end) for start, end, speaker in segments]] * len(audios)

    return _diarize


@pytest.fixture
def stub_diarizer(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Replace the model spec and the diarizer on the node module the driver calls into."""

    def _install(diarize: Optional[Callable[..., list]] = None) -> None:
        monkeypatch.setattr(
            preprocess_module,
            "diarization_model",
            lambda config: _FakeModel(str(config.require("diarization.model"))),
        )
        monkeypatch.setattr(preprocess_module, "diarize_audios", diarize or _diarizer((0.0, 0.5, "SPEAKER_00")))

    _install()
    return _install


def _seed_run(
    root: Path, *, seconds: float = 0.5, hz: float = 120.0, streams: Sequence[str] = ("enhanced", "residual")
) -> Path:
    """Write one finished run: the named streams on disk and a store that names them.

    Both halves of the enhancement partition by default, because that is what
    ``diarization.streams`` asks for; ``streams=("enhanced",)`` is a run whose residual block
    never ran.

    Args:
        root: The run root, created if absent.
        seconds: Each stream's duration.
        hz: The enhanced stream's fundamental.
        streams: Which streams to write.

    Returns:
        The enhanced stream's path, which is what the manifest names.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    grid = np.arange(int(seconds * SR)) / SR

    store = ProvStore(run_id=root.name)
    agent = software_agent(store)
    activity = store.activity(node="PREPROCESS", step="residual", parameters={})
    store.was_associated_with(activity, agent)
    for index, name in enumerate(streams):
        wave = (0.4 / (index + 1)) * np.sin(2 * np.pi * hz * (index + 1) * grid)
        sf.write(str(run_dir / "streams" / f"{name}.flac"), wave.astype(np.float32), SR)
        entity = store.entity(
            prov_type="stream",
            extent=(0.0, seconds),
            attributes={
                "name": name,
                **path_attributes(f"streams/{name}.flac", run_dir),
                "sampling_rate": SR,
                "channels": 1,
            },
        )
        store.was_generated_by(entity, activity)
        store.was_attributed_to(entity, agent)
    store.write_jsonl(run_dir / "store.jsonl")
    return run_dir / "streams" / "enhanced.flac"


def _manifest(path: Path, roots: Sequence[Path]) -> Path:
    """Write a manifest naming one enhanced stream per run root."""
    lines = [
        json.dumps({"stem": root.name, "enhanced": str(root / "run" / "streams" / "enhanced.flac")}) for root in roots
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _store_of(root: Path) -> ProvStore:
    """Read one run's store back under its own run id."""
    return ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)


def _log_rows(log_dir: Path) -> List[dict[str, Any]]:
    """The per-recording outcome records the slice wrote, in manifest order."""
    return [json.loads(line) for line in (log_dir / "slices" / "slice-0-of-1.jsonl").read_text().splitlines()]


def _summary(log_dir: Path) -> dict[str, Any]:
    """The summary sidecar the slice wrote beside its log."""
    return json.loads((log_dir / "slices" / "slice-0-of-1.summary.json").read_text())


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[..., tuple[Path, list[Path]]]:
    """A factory for a manifest over N finished runs."""

    def _build(count: int, streams: Sequence[str] = ("enhanced", "residual")) -> tuple[Path, list[Path]]:
        roots = []
        for index in range(count):
            root = tmp_path / "corpus" / f"sub-{index:02d}_ses-1_20260915-000000"
            _seed_run(root, hz=110.0 + 10.0 * index, streams=streams)
            roots.append(root)
        return _manifest(tmp_path / "manifest.jsonl", roots), roots

    return _build


class TestTheExtendPass:
    """What one pass adds to a finished run, and what it leaves untouched."""

    def test_one_measurement_per_stream_is_merged_into_the_existing_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None]
    ) -> None:
        """Both halves of the enhancement partition land, each naming its own stream and sidecar."""
        manifest, roots = corpus(2)
        before = {root: _store_of(root).fingerprint() for root in roots}

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        for root in roots:
            store = _store_of(root)
            assert store.fingerprint() != before[root]
            for stream in ("enhanced", "residual"):
                measurement = find_measurement(store, diarization_measurement(stream))
                assert measurement is not None, stream
                assert measurement.attributes["signal"] == stream
                assert measurement.attributes["n_speakers"] == 1
                assert measurement.attributes["path"] == f"derivatives/{stream}_diarization.npz"
                assert (root / "run" / measurement.attributes["path"]).is_file()
                assert len(measurement.attributes["checksum_sha256"]) == 64

    def test_a_run_whose_residual_was_never_written_still_gets_its_enhanced_reading(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """One stream missing must not cost the other: the two are separate blocks, not one."""
        manifest, roots = corpus(1, streams=("enhanced",))
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1

        store = _store_of(roots[0])
        assert find_measurement(store, ENHANCED_DIARIZATION) is not None
        assert find_measurement(store, RESIDUAL_DIARIZATION) is None
        row = _log_rows(tmp_path)[0]
        assert row["enhanced_diarization"] == "ok"
        assert "no stream named 'residual'" in row["residual_diarization"]
        assert row["enhanced_n_speakers"] == 1

    def test_nothing_is_written_outside_the_recordings_own_run(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None]
    ) -> None:
        """The driver only touches ``run/`` and ``prov/`` under the run it was pointed at."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]

    def test_the_bep028_files_are_re_exported_from_the_merged_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None]
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])

        io_graph = json.loads((roots[0] / "prov" / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        measurement = find_measurement(_store_of(roots[0]), ENHANCED_DIARIZATION)
        assert measurement is not None
        assert any(measurement.id in identifier for identifier in entity_ids)

    def test_the_slice_log_carries_a_speaker_count_per_stream_never_one_total(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """The two counts answer different questions, so neither the row nor the summary sums them.

        A row carrying only a total could not say whether enhancement removed a voice, which is the
        whole reason the residual half is measured.
        """
        stub_diarizer(_diarizer((0.0, 0.3, "SPEAKER_00"), (0.2, 0.5, "SPEAKER_01")))
        manifest, _ = corpus(2)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])

        rows = _log_rows(tmp_path)
        assert [row["enhanced_n_speakers"] for row in rows] == [2, 2]
        assert [row["residual_n_speakers"] for row in rows] == [2, 2]
        assert all("n_speakers" not in row for row in rows), "a single total would hide which stream saw what"
        summary = _summary(tmp_path)
        assert summary["speaker_counts"] == {"enhanced": {"2": 2}, "residual": {"2": 2}}
        assert summary["streams"] == ["enhanced", "residual"]

    def test_an_empty_residual_is_zero_voices_and_still_an_ok_row(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """A residual holding no voice is the ordinary case: nothing was removed, and that is a value.

        Recording it as ``0`` rather than as an absence is what lets a reader tell "enhancement took
        nothing out" from "the residual could not be read".
        """

        def _by_stream(audios: list, **kwargs: Any) -> list:  # noqa: ANN401
            """One speaker where the signal is loud, none where it is the quiet residual."""
            loud = float(audios[0].waveform.abs().max()) > 0.3
            return [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=0.5)] if loud else []]

        stub_diarizer(_by_stream)
        manifest, roots = corpus(1)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        store = _store_of(roots[0])
        residual = find_measurement(store, RESIDUAL_DIARIZATION)
        assert residual is not None, "an empty residual is measured, not absent"
        assert residual.attributes["n_speakers"] == 0
        assert residual.attributes["n_segments"] == 0
        row = _log_rows(tmp_path)[0]
        assert row["status"] == "ok"
        assert row["enhanced_n_speakers"] == 1
        assert row["residual_n_speakers"] == 0


class TestIdempotence:
    """A task that dies mid-way restarts where it stopped, and a finished slice re-run changes nothing."""

    def test_a_second_unforced_run_writes_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None]
    ) -> None:
        """Merging is a set union, so the second pass must add no record and no file."""
        manifest, roots = corpus(2)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        after_first = {root: _store_of(root).fingerprint() for root in roots}
        sidecars = {
            root: (root / "run" / "derivatives" / f"{ENHANCED_DIARIZATION}.npz").stat().st_mtime_ns for root in roots
        }

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        for root in roots:
            store = _store_of(root)
            assert store.fingerprint() == after_first[root]
            held = [e for e in store.entities("measurement") if e.attributes["name"] == ENHANCED_DIARIZATION]
            assert len(held) == 1
            rewritten = (root / "run" / "derivatives" / f"{ENHANCED_DIARIZATION}.npz").stat().st_mtime_ns
            assert rewritten == sidecars[root], "the npz was rewritten, which would re-identify the entity"

    def test_the_second_run_reports_every_recording_as_skipped_with_its_stored_count(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """A skipped row still says what the store holds, so a re-run's log is a full census."""
        manifest, _ = corpus(2)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])

        rows = _log_rows(tmp_path)
        assert [row["status"] for row in rows] == ["skipped", "skipped"]
        assert [row["enhanced_n_speakers"] for row in rows] == [1, 1]
        assert [row["residual_n_speakers"] for row in rows] == [1, 1]

    def test_a_pass_that_adds_a_stream_diarizes_only_the_one_that_is_missing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """Skipping is per stream, so widening ``diarization.streams`` costs only the new stream."""
        enhanced_only = tmp_path / "enhanced_only.yaml"
        enhanced_only.write_text("diarization:\n  streams: [enhanced]\n")
        manifest, roots = corpus(1)
        assert (
            cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--config", str(enhanced_only)]) == 0
        )
        first = find_measurement(_store_of(roots[0]), ENHANCED_DIARIZATION)
        assert first is not None and find_measurement(_store_of(roots[0]), RESIDUAL_DIARIZATION) is None

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        store = _store_of(roots[0])
        assert find_measurement(store, RESIDUAL_DIARIZATION) is not None
        again = find_measurement(store, ENHANCED_DIARIZATION)
        assert again is not None and again.id == first.id, "the held stream was re-derived"
        row = _log_rows(tmp_path)[0]
        assert row["enhanced_diarization"] == "skipped"
        assert row["residual_diarization"] == "ok"

    def test_force_re_derives_and_retires_the_reading_it_replaces(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None]
    ) -> None:
        """The store is append-only, so the old reading is invalidated rather than removed."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        first = find_measurement(_store_of(roots[0]), ENHANCED_DIARIZATION)
        assert first is not None and first.attributes["n_speakers"] == 1

        stub_diarizer(_diarizer((0.0, 0.3, "SPEAKER_00"), (0.3, 0.5, "SPEAKER_01")))
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--force"]) == 0

        store = _store_of(roots[0])
        current = find_measurement(store, ENHANCED_DIARIZATION)
        assert current is not None and current.attributes["n_speakers"] == 2
        assert current.id != first.id
        assert store.is_invalidated(first.id), "the superseded reading is still readable as current"
        assert len([e for e in store.entities("measurement") if e.attributes["name"] == ENHANCED_DIARIZATION]) == 2


class TestOneBadRecording:
    """A recording the model or the filesystem refuses is an outcome, not the end of the slice."""

    def test_a_typed_absence_leaves_the_rest_of_the_slice_intact(
        self, corpus: Callable[..., tuple[Path, list[Path]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A host with no token records an absence per recording and exits 0; nothing is a failure."""

        def _gated(**kwargs: Any) -> Any:  # noqa: ANN401
            raise ValueError("401 Client Error: gated repo; no token on this host")

        monkeypatch.setattr(preprocess_module, "PyannoteAudioModel", _gated)
        manifest, roots = corpus(2)
        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        for root in roots:
            assert find_measurement(_store_of(root), ENHANCED_DIARIZATION) is None
            assert find_measurement(_store_of(root), RESIDUAL_DIARIZATION) is None
        rows = _log_rows(tmp_path)
        assert [row["status"] for row in rows] == ["absent", "absent"]
        assert all("SpeakerDiarizationUnavailable" in row["enhanced_diarization"] for row in rows)
        assert all("SpeakerDiarizationUnavailable" in row["residual_diarization"] for row in rows)

    def test_a_run_with_no_store_is_recorded_and_the_others_still_extend(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert find_measurement(_store_of(roots[1]), ENHANCED_DIARIZATION) is not None
        rows = _log_rows(tmp_path)
        assert rows[0]["status"] == "error"
        assert rows[1]["status"] == "ok"


class TestTheSlicing:
    """Task *i* of *n* takes a stride, so no task draws the corpus's long tail on its own."""

    def test_each_task_takes_every_nth_row(
        self, corpus: Callable[..., tuple[Path, list[Path]]], stub_diarizer: Callable[..., None], tmp_path: Path
    ) -> None:
        """Two shards of four rows partition them, and together they diarize every run once."""
        manifest, roots = corpus(4)
        for index in range(2):
            assert cli.main([str(manifest), "--slice-index", str(index), "--slice-count", "2"]) == 0
        for root in roots:
            assert find_measurement(_store_of(root), ENHANCED_DIARIZATION) is not None
        shard = json.loads((tmp_path / "slices" / "slice-0-of-2.summary.json").read_text())
        assert shard["rows"] == 2


class TestTheStreamsAreTheConfigsChoice:
    """The manifest path locates the run; ``diarization.streams`` chooses what is diarized."""

    def test_an_override_naming_another_stream_is_an_error_when_the_run_has_none(
        self,
        corpus: Callable[..., tuple[Path, list[Path]]],
        stub_diarizer: Callable[..., None],
        tmp_path: Path,
    ) -> None:
        """A manifest that does not match the configuration must be loud, not silently skipped."""
        override = tmp_path / "plain.yaml"
        override.write_text("diarization:\n  streams: [plain]\n")
        manifest, roots = corpus(1)

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1", "--config", str(override)]) == 1
        assert find_measurement(_store_of(roots[0]), ENHANCED_DIARIZATION) is None
        assert "no stream named 'plain'" in _log_rows(tmp_path)[0]["plain_diarization"]
