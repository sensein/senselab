"""The task-extent SQUIM extend driver: in place, or into a mirrored tree that leaves the finished one untouched."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from typing import Sequence

import pytest
import torch
from senselab_core.runtime.provenance import ProvStore

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage import task_squim
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.common import find_measurement, path_attributes, write_stream

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_task_squim.py"
_spec = importlib.util.spec_from_file_location("extend_task_squim_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_task_squim_under_test"] = cli
_spec.loader.exec_module(cli)

RATE = 16_000
STEM = "sub-a_ses-b_task-caterpillar-passage"


def _fake(audios: Sequence[Audio]) -> list[dict[str, float]]:
    return [{"pesq": 3.0, "stoi": 0.97, "si_sdr": 15.0} for _ in audios]


def _run(tree: Path) -> Path:
    run_root = tree / "sub-a" / "ses-b" / f"{STEM}_20000101-000000"
    run_dir = run_root / "run"
    (run_dir / "streams").mkdir(parents=True)
    (run_dir / "derivatives").mkdir()
    (run_root / "summary").mkdir()
    store = ProvStore(run_id=run_root.name)
    t = torch.arange(RATE * 12, dtype=torch.float32) / RATE
    audio = Audio(waveform=(0.5 * torch.sin(2 * math.pi * 220.0 * t)).unsqueeze(0), sampling_rate=RATE)
    for name in ("recording", "plain", "enhanced"):
        relative, _ = write_stream(audio, run_dir, name)
        store.entity(
            prov_type="stream", extent=(0.0, 12.0), attributes={"name": name, **path_attributes(relative, run_dir)}
        )
    store.entity(prov_type="span", extent=(1.0, 11.0), attributes={"role": "task_extent", "branch": "SPEECH"})
    store.write_jsonl(run_dir / "store.jsonl")
    return run_root


@pytest.fixture(autouse=True)
def _no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task_squim, "squim_scorer", lambda: _fake)


def test_in_place_the_store_gains_the_measurement(tmp_path: Path) -> None:
    """The run's own store holds a task_squim measurement afterwards, and a second pass writes nothing."""
    run_root = _run(tmp_path / "out")
    row = {"stem": STEM, "enhanced": str(run_root / "run" / "streams" / "enhanced.flac")}
    [first] = cli.process([row], load_triage_config())
    assert first["status"] == "ok" and first["windows_n"] == 1
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    assert find_measurement(store, task_squim.MEASUREMENT) is not None
    [second] = cli.process([row], load_triage_config())
    assert second["status"] == "present"


def test_out_root_leaves_the_finished_store_untouched(tmp_path: Path) -> None:
    """With --out-root the store is copied and written there; everything else is a link to the finished run."""
    run_root = _run(tmp_path / "out")
    before = (run_root / "run" / "store.jsonl").read_bytes()
    row = {"stem": STEM, "enhanced": str(run_root / "run" / "streams" / "enhanced.flac")}
    [record] = cli.process([row], load_triage_config(), out_root=tmp_path / "mirror")
    assert record["status"] == "ok"
    assert (run_root / "run" / "store.jsonl").read_bytes() == before
    mirrored = tmp_path / "mirror" / "sub-a" / "ses-b" / run_root.name
    assert (mirrored / "run" / "streams").is_symlink() and (mirrored / "summary").is_symlink()
    assert not (mirrored / "run" / "store.jsonl").is_symlink()
    store = ProvStore.read_jsonl(mirrored / "run" / "store.jsonl", run_id=run_root.name)
    assert find_measurement(store, task_squim.MEASUREMENT) is not None
