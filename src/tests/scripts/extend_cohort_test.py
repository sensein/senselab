"""The cohort driver over miniature finished runs: collect, quantiles, apply, resume, read-only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.cohort_fold import COHORT_READING, MEASURED
from senselab.audio.workflows.triage.extend import read_store, write_store
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

_REPO_ROOT = Path(__file__).resolve().parents[3]
RATE = 16000
DIM = 8
SESSION = "sub-a_ses-1"


def _load(name: str) -> ModuleType:
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_under_test"] = module
    spec.loader.exec_module(module)
    return module


cli = _load("extend_cohort")


def _embed(samples: np.ndarray, rate: int) -> np.ndarray | None:
    if samples.size == 0:
        return None
    vector = np.zeros(DIM)
    vector[int(round(float(np.median(samples)) * 10))] = 1.0
    vector[DIM - 1] = 0.05
    return vector / np.linalg.norm(vector)


def _run(root: Path, task: str, duration: float, speech: list[tuple[float, float, str, float]], extent) -> dict:  # noqa: ANN001
    stem = f"{SESSION}_task-{task}"
    run_root = root / f"{stem}_20261010-000000"
    streams = run_root / "run" / "streams"
    derivatives = run_root / "run" / "derivatives"
    streams.mkdir(parents=True)
    derivatives.mkdir(parents=True)
    samples = np.zeros(int(duration * RATE), dtype=np.float32)
    for start, end, _, level in speech:
        samples[int(start * RATE) : int(end * RATE)] = level
    sf.write(streams / "plain.wav", samples, RATE, subtype="FLOAT")
    sf.write(streams / "enhanced.flac", samples, RATE)
    np.savez(
        derivatives / "enhanced_diarization.npz",
        starts=np.array([s for s, _, _, _ in speech]),
        ends=np.array([e for _, e, _, _ in speech]),
        speakers=np.array([k for _, _, k, _ in speech]),
    )
    store = ProvStore(run_id=run_root.name)
    store.entity(prov_type="stream", extent=(0.0, duration), attributes={"name": "recording", "path": f"/b/{stem}.wav"})
    store.entity(prov_type="stream", extent=None, attributes={"name": "plain", "path": "streams/plain.wav"})
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "enhanced_diarization",
            "signal": "enhanced",
            "path": "derivatives/enhanced_diarization.npz",
        },
    )
    if extent is not None:
        store.entity(prov_type="span", extent=extent, attributes={"role": "task_extent", "family": "voice"})
    write_store(store, run_root)
    return {"stem": run_root.name, "enhanced": str(streams / "enhanced.flac")}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Three speech-bearing recordings and four vowels of one session, the last far longer than the rest."""
    rows = [
        _run(tmp_path, "rainbow-passage", 12.0, [(1.0, 9.0, "A", 0.1)], (1.0, 9.0)),
        _run(tmp_path, "caterpillar-passage", 12.0, [(1.0, 10.0, "A", 0.1)], (1.0, 10.0)),
        _run(
            tmp_path, "respiration-and-cough-breath-1", 10.0, [(2.0, 4.0, "A", 0.1), (6.0, 9.0, "B", 0.2)], (1.0, 5.0)
        ),
    ]
    rows += [_run(tmp_path, f"prolonged-vowel-{i}", 10.0, [], (1.0, 9.0)) for i in range(1, 4)]
    rows.append(_run(tmp_path, "prolonged-vowel-4", 50.0, [], (1.0, 9.0)))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


@pytest.fixture()
def config(tmp_path: Path) -> Path:
    """An override naming the enrollment model and the match cut."""
    path = tmp_path / "override.yaml"
    path.write_text(
        "speech:\n"
        "  enrollment_model:\n"
        "    model_id: speechbrain/spkrec-ecapa-voxceleb\n"
        f"    revision: {'0' * 40}\n"
        "  target_match_cosine: 0.231\n"
    )
    return path


def _apply(manifest: Path, config: Path, quantiles: Path, *extra: str) -> int:
    return cli.main(
        [
            "apply",
            str(manifest),
            "--quantiles",
            str(quantiles),
            "--slice-index",
            "0",
            "--slice-count",
            "1",
            "--config",
            str(config),
            "--log-dir",
            str(manifest.parent / "logs"),
            *extra,
        ]
    )


def _quantiles(manifest: Path) -> Path:
    logs = manifest.parent / "logs"
    assert cli.main(["collect", str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(logs)]) == 0
    out = manifest.parent / "cohort" / "cohort_quantiles.json"
    assert cli.main(["quantiles", "--facts", str(logs), "--out", str(out)]) == 0
    return out


def _reading(row: dict) -> dict:
    store = read_store(Path(row["enhanced"]).parents[2])
    found = find_measurement(store, COHORT_READING)
    assert found is not None
    return found.attributes


def test_the_pass_writes_every_store_and_a_rerun_is_skipped(
    corpus: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enrollment flags the breath talker, the long vowel is an outlier; a second pass skips and writes nothing."""
    monkeypatch.setattr(cli, "ecapa_embedder", lambda *a, **k: _embed)
    quantiles = _quantiles(corpus)
    artefact = json.loads(quantiles.read_text())
    assert artefact["families"]["prolonged-vowel"]["n"] == 4
    assert _apply(corpus, config, quantiles) == 0
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    breath = _reading(rows[2])
    assert breath["other_speaker"]["status"] == MEASURED
    assert breath["other_speaker"]["nonmatch_spans"] == [[6.0, 9.0]]
    assert breath["quantiles"]["sha256"] and breath["cohort_key"]
    assert _reading(rows[0])["other_speaker"]["nonmatch_n"] == 0
    assert _reading(rows[6])["checks"]["recording_duration_outlier"]["outcome"] == "review"
    assert _reading(rows[3])["checks"]["recording_duration_outlier"]["outcome"] == "pass"
    stores = [(Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows]
    assert _apply(corpus, config, quantiles) == 0
    assert [(Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows] == stores
    log = (corpus.parent / "logs" / "slices" / "cohort-slice-0-of-1.jsonl").read_text().splitlines()
    assert {json.loads(line)["status"] for line in log} == {"skipped"}


def test_readings_out_writes_no_store(
    corpus: Path, config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read-only mode leaves every store as it was and writes the session's readings beside."""
    monkeypatch.setattr(cli, "ecapa_embedder", lambda *a, **k: _embed)
    quantiles = _quantiles(corpus)
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    stores = [(Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows]
    out = tmp_path / "readings"
    assert _apply(corpus, config, quantiles, "--readings-out", str(out), "--sessions", SESSION) == 0
    assert [(Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows] == stores
    written = [json.loads(line) for line in (out / f"{SESSION}.jsonl").read_text().splitlines()]
    assert len(written) == len(rows)
    assert sum(1 for record in written if record["nonmatch_n"]) == 1


def test_an_unset_enrollment_model_leaves_the_enrollment_unavailable(corpus: Path, tmp_path: Path) -> None:
    """Without a model and cut the checks still run and the enrollment block names what is missing."""
    quantiles = _quantiles(corpus)
    override = tmp_path / "null.yaml"
    override.write_text("speech:\n  enrollment_model:\n  target_match_cosine:\n")
    assert _apply(corpus, override, quantiles) == 0
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    reading = _reading(rows[6])
    assert reading["other_speaker"]["status"] == "unavailable"
    assert reading["checks"]["recording_duration_outlier"]["outcome"] == "review"
