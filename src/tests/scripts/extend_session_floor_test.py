"""The own-floor and session-floor extend drivers over real miniature runs grouped into one session."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import soundfile as sf

from senselab.audio.workflows.triage.extend import read_store, write_store
from senselab.audio.workflows.triage.nodes.background import (
    FLOOR_FROM_SESSION,
    OWN_FLOOR,
    SESSION_FLOOR,
    SPEECH_RESIDUAL,
)
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
RATE = 16000


def _load(name: str) -> ModuleType:
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_under_test"] = module
    spec.loader.exec_module(module)
    return module


floor_cli = _load("extend_background_floor")
session_cli = _load("extend_session_floor")


def _breaths(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(8 * RATE) * 1e-3
    for start in (1.0, 3.0, 5.0):
        n = int(0.6 * RATE)
        x[int(start * RATE) : int(start * RATE) + n] += rng.standard_normal(n) * 2e-2 * np.hanning(n)
    return x.astype(np.float32)


def _run(root: Path, stem: str, samples: np.ndarray) -> dict[str, str]:
    run_root = root / stem
    streams = run_root / "run" / "streams"
    streams.mkdir(parents=True)
    sf.write(streams / "plain.wav", samples, RATE, subtype="FLOAT")
    sf.write(streams / "enhanced.flac", samples, RATE)
    store = ProvStore(run_id=stem)
    store.entity(prov_type="stream", extent=None, attributes={"name": "plain", "path": "streams/plain.wav"})
    write_store(store, run_root)
    return {"stem": stem, "enhanced": str(streams / "enhanced.flac")}


def test_members_of_one_session_share_one_floor(tmp_path: Path) -> None:
    """Three quiet members and a fourth from another session: the three read their session's floor."""
    rows = [_run(tmp_path, f"sub-a_ses-1_task-t{i}", _breaths(i)) for i in range(3)]
    rows.append(_run(tmp_path, "sub-b_ses-9_task-t0", _breaths(7)))
    assert [r["status"] for r in floor_cli.process(rows)] == ["ok"] * 4
    assert [s for s, _ in session_cli.sessions_of(rows)] == ["sub-a_ses-1", "sub-b_ses-9"]
    out = session_cli.process_session("sub-a_ses-1", rows[:3])
    assert {r[SESSION_FLOOR] for r in out} == {FLOOR_FROM_SESSION}
    store = read_store(Path(rows[0]["enhanced"]).parents[2])
    assert find_measurement(store, OWN_FLOOR) is not None
    found = find_measurement(store, SESSION_FLOOR)
    assert found is not None and found.attributes["used_n"] == 3 and found.attributes["session"] == "sub-a_ses-1"


def test_an_own_floor_already_written_is_not_written_again(tmp_path: Path) -> None:
    """The own-floor pass skips a store that already carries one."""
    rows = [_run(tmp_path, "sub-a_ses-1_task-t0", _breaths(1))]
    floor_cli.process(rows)
    assert floor_cli.process(rows)[0]["status"] == "skipped"


def _speech_run(root: Path, stem: str, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    t = np.arange(6 * RATE) / RATE
    speech = (np.sin(2 * np.pi * 150.0 * t) * 0.05 * ((t % 2.0) < 1.0)).astype(np.float32)
    room = (rng.standard_normal(6 * RATE) * 1e-3).astype(np.float32)
    run_root = root / stem
    streams = run_root / "run" / "streams"
    streams.mkdir(parents=True)
    store = ProvStore(run_id=stem)
    for name, samples in (("plain", speech + room), ("enhanced", speech), ("residual", room)):
        sf.write(streams / f"{name}.wav", samples, RATE, subtype="FLOAT")
        store.entity(prov_type="stream", extent=None, attributes={"name": name, "path": f"streams/{name}.wav"})
    store.entity(prov_type="stream", extent=None, attributes={"name": "recording", "path": f"/data/{stem}.wav"})
    windows = [{"start": float(a), "end": float(a) + 1.0, "label_scores": [{"Speech": 0.9}]} for a in (0, 2, 4)]
    (run_root / "run" / "enhanced_yamnet.json").write_text(json.dumps(windows))
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "enhanced_yamnet_scores", "path": "enhanced_yamnet.json"},
    )
    write_store(store, run_root)
    return {"stem": stem, "enhanced": str(streams / "enhanced.wav")}


def test_the_session_pass_writes_each_speech_residual_and_the_session_background(tmp_path: Path) -> None:
    """Two speech tasks and a breath task: each store gains its reading; the session background has the two."""
    rows = [
        _speech_run(tmp_path, "sub-a_ses-1_task-rainbow-passage", 1),
        _speech_run(tmp_path, "sub-a_ses-1_task-free-speech-1", 2),
        _speech_run(tmp_path, "sub-a_ses-1_task-breath-sounds", 3),
    ]
    floor_cli.process(rows)
    out = session_cli.process_session("sub-a_ses-1", rows)
    assert {r["session_background"] for r in out} == {"speech_residual"}
    assert {r["session_background_members_n"] for r in out} == {2}
    store = read_store(Path(rows[2]["enhanced"]).parents[2])
    reading = find_measurement(store, SPEECH_RESIDUAL)
    assert reading is not None and reading.attributes["kind"] == "airway" and reading.attributes["foreground"]
    found = find_measurement(store, SESSION_FLOOR)
    assert found is not None
    assert sorted(found.attributes["session_background"]["members"]) == [
        "sub-a_ses-1_task-free-speech-1",
        "sub-a_ses-1_task-rainbow-passage",
    ]
