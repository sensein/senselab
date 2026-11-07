"""The scale ladder's trees and the resource tables generated from its results."""

import hashlib
import importlib.util
import io
import json
import platform
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import soundfile
from senselab_core.kernel.declare import REGISTRY, BackendSpec
from senselab_core.runtime import fake_backend
from senselab_core.runtime.call import Packaged, use_packaged
from senselab_core.runtime.catalogue import Catalogue
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import ResourceTable
from senselab_core.testing.workers import mit_lock_records

ROOT = Path(__file__).resolve().parents[4]


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bids_tree = _module("scale_bids_tree_under_test", ROOT / "benchmarks" / "scale" / "bids_tree.py")
generate = _module("generate_resources_under_test", ROOT / "scripts" / "generate_resources.py")

TREE_DIGEST = "74166948de67e1d029c3ef723ef61412257f2c74b65142f5c5f8a6df2d3e89d9"
"""The digest of `synthetic(root, 3, 0)`: its relative paths and bytes, the same on every host."""


def _digest(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return h.hexdigest()


@pytest.mark.logic
def test_synthetic_tree_is_byte_identical_for_a_seed(tmp_path: Path) -> None:
    """A seed gives the pinned bytes twice; another seed differs; every tone decodes at the tree's rate and lengths."""
    first, again = bids_tree.synthetic(tmp_path / "a", 3, 0), bids_tree.synthetic(tmp_path / "b", 3, 0)
    assert _digest(first) == _digest(again) == TREE_DIGEST
    assert _digest(bids_tree.synthetic(tmp_path / "c", 3, 1)) != TREE_DIGEST
    tones = sorted(first.rglob("*_audio.wav"))
    assert [t.relative_to(first).parts[0] for t in tones] == ["sub-0", "sub-1", "sub-2"]
    for tone in tones:
        info = soundfile.info(tone)
        assert info.samplerate == bids_tree.RATE and bids_tree.SECONDS[0] <= info.duration <= bids_tree.SECONDS[1]
    [long] = bids_tree.synthetic(tmp_path / "long", 1, 0, seconds=7).rglob("*_audio.wav")
    assert soundfile.info(long).duration == 7
    with pytest.raises(ValueError, match="at least one"):
        bids_tree.synthetic(tmp_path / "none", 0, 0)


@pytest.mark.logic
def test_librispeech_utterances_are_laid_out_as_bids(tmp_path: Path) -> None:
    """The first n utterances, in archive order, go to `sub-<speaker>` with a run per speaker, bytes unchanged."""
    archive = tmp_path / "test-clean.tar.gz"
    names = [
        "61/70968/61-70968-0000",
        "61/70968/61-70968-0001",
        "121/121726/121-121726-0000",
        "121/121726/121-121726-0001",
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for index, name in enumerate(names):
            data = f"flac {index}".encode()
            member = tarfile.TarInfo(f"LibriSpeech/test-clean/{name}.flac")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    root = bids_tree.lay_librispeech(tmp_path / "bids", archive, 3)
    laid = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*.flac")}
    assert laid == {
        "sub-61/ses-01/audio/sub-61_ses-01_task-read_run-1_audio.flac": b"flac 0",
        "sub-61/ses-01/audio/sub-61_ses-01_task-read_run-2_audio.flac": b"flac 1",
        "sub-121/ses-01/audio/sub-121_ses-01_task-read_run-1_audio.flac": b"flac 2",
    }
    with pytest.raises(ValueError, match="fewer than 5"):
        bids_tree.lay_librispeech(tmp_path / "short", archive, 5)


def _result(
    point: int, items: int, jobs: list[str], measure: dict[str, Any] | None, sha: str = "a" * 40
) -> dict[str, Any]:
    return {
        "format": "scale-result@1.0",
        "task": "speech_to_text",
        "point": point,
        "items": items,
        "input_s": items * 7.0,
        "backend": "transformers",
        "model": {"id": "org/m", "revision": "b" * 40},
        "resource_class": "gpu:h100",
        "sha": sha,
        "jobs": jobs,
        "measure": measure,
    }


@pytest.mark.logic
def test_resource_table_derivation_names_jobs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The largest point's measure is written with the commit and every job of the unit; a measureless one is listed."""
    small = {
        "load_s": 9.0,
        "claim_s": 0.01,
        "resident_bytes": 1,
        "peak_memory_bytes": 2,
        "s_per_input_s": 0.2,
        "input_s_per_item": 7.0,
    }
    large = {**small, "load_s": 8.0, "s_per_input_s": 0.1}
    results = tmp_path / "results"
    results.mkdir()
    for name, doc in {
        "a@1": _result(1, 1, ["101"], small),
        "a@100": _result(100, 100, ["102", "103"], large),
        "b@1": {**_result(1, 1, ["104"], None), "task": "diarize"},
    }.items():
        (results / f"{name}.json").write_text(json.dumps(doc))
    assert generate.main([str(results), "--out", str(tmp_path / "out")]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "diarize@1: no measure; skipped"
    table = ResourceTable.model_validate(load_yaml(tmp_path / "out" / "speech_to_text" / "transformers.yaml", "table"))
    [row] = table.rows
    assert (row.id, row.revision, row.download_bytes) == ("org/m", "b" * 40, None)
    assert row.classes["gpu:h100"].load_s == 8.0 and row.classes["gpu:h100"].s_per_input_s == 0.1
    measured = row.classes["gpu:h100"].measured
    assert measured.sha == "a" * 40 and measured.jobs == ("101", "102", "103")
    (results / "c@1.json").write_text(json.dumps(_result(1, 1, ["105"], small, sha="c" * 40)))
    with pytest.raises(ValueError, match="several commits"):
        generate.main([str(results), "--out", str(tmp_path / "out")])


@pytest.mark.logic
def test_flatness_covers_time_memory_files_and_bytes_against_budgets() -> None:
    """Each measure's value at 10⁴ over at 10² is judged against `flat_ratio`, its largest against its ceiling."""
    ladder = _module("scale_ladder_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")

    def point(at: int, wall: float, peak: int, files: int, written: int) -> dict[str, Any]:
        keys = ("per_item_s", "peak_rss_bytes", "files_written", "bytes_written")
        return {"task": "t", "point": at, "items": at, **dict(zip(keys, (wall, peak, files, written), strict=True))}

    results = [point(100, 1.0, 1000, 200, 10_000), point(10_000, 1.2, 1000, 30_000, 1_000_000)]
    budgets = {"flat_ratio": 1.25, "per_item": {"t": {"per_item_s": 1.1, "peak_rss_bytes": None}}}
    assert ladder.flatness(results, budgets) == [
        "t per_item_s: at 10000 / at 100 = 1.200, flat ok, ceiling EXCEEDED (1.1)",
        "t peak_rss_bytes: at 10000 / at 100 = 1.000, flat ok, ceiling unmeasured",
        "t files_per_item: at 10000 / at 100 = 1.500, flat EXCEEDED (1.25), ceiling unmeasured",
        "t bytes_per_item: at 10000 / at 100 = 1.000, flat ok, ceiling unmeasured",
    ]
    assert all("unmeasured" in line for line in ladder.flatness(results, {})[:1])


def _this_interpreter(backend: BackendSpec, profile: ExecutionProfile) -> EnvHandle:  # noqa: ARG001 - an EnvOpener
    return EnvHandle(sys.executable, platform.python_version(), "1" * 64, "2" * 64, None)


@pytest.mark.logic
def test_a_suite_with_entries_that_do_not_plan_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry is planned first: two bad entries are both named, and no point is measured or written."""
    ladder = _module("scale_ladder_refusal_under_test", ROOT / "benchmarks" / "scale" / "ladder.py")
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "cache"))
    previous = use_packaged(Packaged(Catalogue(entries=()), _this_interpreter, REGISTRY, mit_lock_records))
    assert fake_backend.FAKE in REGISTRY.backends(fake_backend.ECHO.name)
    suite = tmp_path / "bad.yaml"
    suite.write_text(
        "format: scale-suite@1.0\nname: bad\nseed: 0\nlong_s: 1\ntasks:\n"
        "- {task: fake.echo, port: payload, data: payloads, points: [1, 100]}\n"
        "- {task: fake.echo, stack: absent, port: payload, data: payloads, points: [1]}\n"
        "- {task: no.such.task, port: x, data: payloads, points: [1]}\n"
    )
    try:
        with pytest.raises(ladder.SuiteRefused, match="2 entries do not plan") as refused:
            ladder.run(suite, ExecutionProfile.load("laptop"))
    finally:
        use_packaged(previous)
    lines = str(refused.value).splitlines()[1:]
    assert [line.split(":")[0] for line in lines] == ["entry 1 (fake.echo, stack absent)", "entry 2 (no.such.task)"]
    assert "NoBackend" in lines[0] and "UnknownTask" in lines[1]
    runs = tmp_path / "cache" / "2x" / "runs"
    assert not list((tmp_path / "cache").rglob("*@*.json")) and not (runs.is_dir() and any(runs.iterdir()))
