"""The scale ladder's trees and the resource tables generated from its results."""

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import soundfile
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.resources import ResourceTable

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
    small = {"load_s": 9.0, "resident_bytes": 1, "peak_memory_bytes": 2, "s_per_input_s": 0.2, "input_s_per_item": 7.0}
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
    assert row.measured.sha == "a" * 40 and row.measured.jobs == ("101", "102", "103")
    (results / "c@1.json").write_text(json.dumps(_result(1, 1, ["105"], small, sha="c" * 40)))
    with pytest.raises(ValueError, match="several commits"):
        generate.main([str(results), "--out", str(tmp_path / "out")])
