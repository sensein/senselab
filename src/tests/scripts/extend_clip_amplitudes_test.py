"""The clip-amplitude extend driver: what it appends to a finished run, and what it leaves alone.

Nothing here is faked. There is no model and no venv in this path — the whole pass is a decode and
a numpy scan — so the runs are real runs in miniature: a source WAV on disk, a store ADMIT wrote the
``recording`` stream into, and clip spans carrying nothing but their family and their signal, which
is the shape the completed corpus is in.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities
from senselab.audio.workflows.triage.nodes.quality import (
    CLIP_AMPLITUDE_MEASUREMENT,
    CLIP_FAMILY,
    CLIP_LEVELS,
    CONTRADICTED_CLIP,
    quality,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_clip_amplitudes.py"
_spec = importlib.util.spec_from_file_location("extend_clip_amplitudes_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_clip_amplitudes_under_test"] = cli
_spec.loader.exec_module(cli)

SR = 16000
"""The fixture's sampling rate, so a sample index and a time are one conversion apart."""


def _samples(*, clip_level: float, louder: float | None) -> np.ndarray:
    """A quiet bed with one plateau over samples 8000..8400, and optionally one louder sample.

    Args:
        clip_level: The level the plateau holds.
        louder: An amplitude to place at sample 24000, or None to leave the bed there.

    Returns:
        The recording, mono float32.
    """
    grid = np.arange(2 * SR) / SR
    out = (0.05 * np.sin(2 * np.pi * 220.0 * grid)).astype(np.float32)
    out[8000:8400] = np.float32(clip_level)
    if louder is not None:
        out[24000] = np.float32(louder)
    return out


def _seed_run(root: Path, *, clip_level: float = 0.5, louder: float | None = 0.9, spans: bool = True) -> Path:
    """Write one finished run: a source WAV, and a store with clip spans carrying no amplitudes.

    The source sits outside the run tree, which is where a corpus's originals sit: the store's
    ``recording`` stream is what names it.

    Args:
        root: The run root, created if absent.
        clip_level: The plateau's level.
        louder: An unclipped sample's amplitude, or None for a recording nothing contradicts.
        spans: Whether to write the clip span at all.

    Returns:
        The source WAV's path.
    """
    run_dir = root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    source = root.parent / f"{root.name}.wav"
    sf.write(str(source), _samples(clip_level=clip_level, louder=louder), SR)
    # The manifest names an enhanced stream, so every run must have one for its root to be derived.
    sf.write(str(run_dir / "streams" / "enhanced.flac"), _samples(clip_level=clip_level, louder=None), SR)

    store = ProvStore(run_id=root.name)
    admitted = admit(store, source, load_triage_config(), run_dir=run_dir)
    assert admitted.audio is not None
    if spans:
        agent = store.agent(agent_type="software", version="senselab test-seed")
        activity = store.activity(node="PREPROCESS", step="clip_spans", parameters={})
        store.was_associated_with(activity, agent)
        span_id = store.entity(
            prov_type="span",
            extent=(8000 / SR, 8400 / SR),
            attributes={"family": CLIP_FAMILY, "signal": "recording"},
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, agent)
    store.write_jsonl(run_dir / "store.jsonl")
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


def _store_of(root: Path) -> ProvStore:
    """Read one run's store back under its own run id.

    Args:
        root: The run root.

    Returns:
        The store.
    """
    return ProvStore.read_jsonl(root / "run" / "store.jsonl", run_id=root.name)


def _log(tmp_path: Path) -> list[dict[str, object]]:
    """The slice log the driver wrote.

    Args:
        tmp_path: The directory the manifest sits in.

    Returns:
        One record per manifest row, in order.
    """
    path = tmp_path / "slices" / "clip-amplitudes-slice-0-of-1.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def corpus(tmp_path: Path) -> Callable[..., tuple[Path, list[Path]]]:
    """A factory for a manifest over N finished runs whose clip spans carry no amplitudes.

    Returns:
        A callable taking the run count and :func:`_seed_run`'s keyword arguments.
    """

    def _build(count: int, **kwargs: object) -> tuple[Path, list[Path]]:
        roots = []
        for index in range(count):
            root = tmp_path / "corpus" / f"sub-{index:02d}_ses-1_20260912-000000"
            root.mkdir(parents=True, exist_ok=True)
            _seed_run(root, **kwargs)  # type: ignore[arg-type]
            roots.append(root)
        return _manifest(tmp_path / "manifest.jsonl", roots), roots

    return _build


class TestTheExtendPass:
    """What one pass appends to a finished run, and what it leaves untouched."""

    def test_the_measurement_is_appended_and_no_span_is_touched(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store keeps every record it had, and the clip spans keep their exact attributes."""
        manifest, roots = corpus(2)
        before = {root: [(e.id, dict(e.attributes)) for e in live_entities(_store_of(root), "span")] for root in roots}

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        for root in roots:
            store = _store_of(root)
            measurement = find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT)
            assert measurement is not None
            assert [(e.id, dict(e.attributes)) for e in live_entities(store, "span")] == before[root]
            assert set(measurement.attributes[CLIP_LEVELS]) == {span_id for span_id, _ in before[root]}
            assert measurement.attributes["edge_guard_samples"] == 3

    def test_quality_then_reaches_the_finding_it_refused_to_make(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The whole point of the pass: the store refuses before it and produces a verdict after."""
        manifest, roots = corpus(1)
        config = load_triage_config()
        with pytest.raises(LookupError, match=CLIP_AMPLITUDE_MEASUREMENT):
            quality(_store_of(roots[0]), "recording", config, run_dir=roots[0] / "run")

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0

        store = _store_of(roots[0])
        result = quality(store, "recording", config, run_dir=roots[0] / "run")
        assert result.verdict.outcome is Outcome.FLAG
        assert CONTRADICTED_CLIP in result.verdict.why
        assert store.get_entity(result.verdict_entity_id).attributes["checked_n"] == 1

    def test_nothing_is_written_outside_the_recordings_own_run(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """The store gains entities and ``prov/`` is exported; no second tree appears."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        assert sorted(entry.name for entry in roots[0].iterdir()) == ["prov", "run"]
        assert sorted(entry.name for entry in (roots[0] / "run").iterdir()) == ["store.jsonl", "streams"]

    def test_the_bep028_files_are_re_exported_from_the_merged_store(
        self, corpus: Callable[..., tuple[Path, list[Path]]]
    ) -> None:
        """``prov/`` must not disagree with the store it was exported from."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])

        io_graph = json.loads((roots[0] / "prov" / "prov-triage_io.json").read_text())
        entity_ids = {node["Id"] for key in ("Files", "prov:Entity") for node in io_graph.get(key, [])}
        measurement = find_measurement(_store_of(roots[0]), CLIP_AMPLITUDE_MEASUREMENT)
        assert measurement is not None
        assert any(measurement.id in identifier for identifier in entity_ids), measurement.id

    def test_the_host_environment_is_recorded(self, corpus: Callable[..., tuple[Path, list[Path]]]) -> None:
        """The pass ran somewhere, and the store says where; no venv is involved in this one."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        environments = _store_of(roots[0]).environments()
        assert [env.kind for env in environments] == ["host"]


class TestASecondPass:
    """A task that dies and reruns must add nothing, which is what makes the array restartable."""

    def test_a_rerun_over_the_same_corpus_changes_nothing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The stores are byte-identical and every row reports ``skipped``."""
        manifest, roots = corpus(2)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        after_first = {root: (root / "run" / "store.jsonl").read_bytes() for root in roots}

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        for root in roots:
            assert (root / "run" / "store.jsonl").read_bytes() == after_first[root]
            store = _store_of(root)
            measurements = [
                e for e in store.entities("measurement") if e.attributes["name"] == CLIP_AMPLITUDE_MEASUREMENT
            ]
            assert len(measurements) == 1
        assert [record["status"] for record in _log(tmp_path)] == ["skipped", "skipped"]

    def test_a_run_that_already_had_the_measurement_is_never_opened_for_writing(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Skipping is what keeps a completed slice cheap; the source is not even decoded."""
        manifest, roots = corpus(1)
        cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"])
        (roots[0].parent / f"{roots[0].name}.wav").unlink()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        assert _log(tmp_path)[0]["status"] == "skipped"


class TestARunWithNothingToMeasure:
    """No clip span is not a failure: nothing was asserted, so nothing needs reading against."""

    def test_a_run_with_no_clip_span_gains_no_measurement_and_is_not_rewritten(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """QUALITY passes such a store without the measurement, so writing one would be noise."""
        manifest, roots = corpus(1, spans=False)
        before = (roots[0] / "run" / "store.jsonl").read_bytes()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 0
        assert (roots[0] / "run" / "store.jsonl").read_bytes() == before
        assert find_measurement(_store_of(roots[0]), CLIP_AMPLITUDE_MEASUREMENT) is None
        assert _log(tmp_path)[0]["status"] == "skipped"
        result = quality(_store_of(roots[0]), "recording", load_triage_config(), run_dir=roots[0] / "run")
        assert result.verdict.outcome is Outcome.PASS


class TestOneBadRecording:
    """A recording the filesystem refuses is an outcome record, not the end of the slice."""

    def test_a_run_with_no_store_is_recorded_and_the_others_still_extend(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """A missing store is this recording's error; its neighbours are unaffected."""
        manifest, roots = corpus(2)
        (roots[0] / "run" / "store.jsonl").unlink()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert find_measurement(_store_of(roots[1]), CLIP_AMPLITUDE_MEASUREMENT) is not None
        assert [record["status"] for record in _log(tmp_path)] == ["error", "ok"]

    def test_a_source_recording_that_moved_is_recorded_rather_than_guessed_at(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """The store names the source; a source that is gone leaves nothing to measure."""
        manifest, roots = corpus(2)
        (roots[0].parent / f"{roots[0].name}.wav").unlink()

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert find_measurement(_store_of(roots[0]), CLIP_AMPLITUDE_MEASUREMENT) is None
        assert find_measurement(_store_of(roots[1]), CLIP_AMPLITUDE_MEASUREMENT) is not None
        assert [record["status"] for record in _log(tmp_path)] == ["error", "ok"]

    def test_a_source_recording_whose_bytes_changed_is_refused(
        self, corpus: Callable[..., tuple[Path, list[Path]]], tmp_path: Path
    ) -> None:
        """Amplitudes from other bytes would be keyed against spans detected on the originals."""
        manifest, roots = corpus(1)
        source = roots[0].parent / f"{roots[0].name}.wav"
        sf.write(str(source), _samples(clip_level=0.8, louder=0.9), SR)

        assert cli.main([str(manifest), "--slice-index", "0", "--slice-count", "1"]) == 1
        assert find_measurement(_store_of(roots[0]), CLIP_AMPLITUDE_MEASUREMENT) is None
        assert "digests to" in str(_log(tmp_path)[0]["clip_amplitude"])

    def test_a_manifest_row_naming_no_run_root_is_refused_rather_than_guessed(self) -> None:
        """A wrong guess would append a measurement to the wrong recording's store."""
        with pytest.raises(ValueError, match="no run root"):
            cli.run_root_of(Path("/corpus/enhanced.flac"))
