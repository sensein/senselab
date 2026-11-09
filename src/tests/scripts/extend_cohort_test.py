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
from senselab.audio.workflows.triage.identity import ManifestWithoutIdentity, identity_columns, identity_of
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.audio.workflows.triage.task_catalogue import task_attributes
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.conftest import hint_for
from tests.audio.workflows.triage.nodes.conftest import word_attributes

pytestmark = pytest.mark.usefixtures("triage_profile")
"""The drivers' internals these tests call read the fixture profile's data; a driver run is given ``--profile``."""

_REPO_ROOT = Path(__file__).resolve().parents[3]
RATE = 16000
DIM = 8
TALKER_WORDS = ((6.5, 7.0), (7.5, 8.0))


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


def _run(  # noqa: ANN202
    root: Path,
    stem: str,
    task_id: str,
    duration: float,
    speech: list[tuple[float, float, str, float]],
    extent,  # noqa: ANN001
    participant: str = "pa",
    session: str = "s1",
    words: tuple[tuple[float, float], ...] = (),
):
    """One finished run of ``stem``, its hint declaring ``participant``/``session`` and the task id ``task_id``."""
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
    windows = [{"start": s, "end": e, "label_scores": [{"Speech": 0.9}]} for s, e, _, _ in speech]
    (derivatives / "yamnet_scores.json").write_text(json.dumps(windows))
    store = ProvStore(run_id=run_root.name)
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "yamnet_scores", "signal": "plain", "path": "derivatives/yamnet_scores.json"},
    )
    task, _, take = task_id.rpartition("-") if task_id[-1].isdigit() else (task_id, "", "")
    hint = hint_for(participant, session, task, take=int(take) if take else None)
    identity = identity_of(hint)
    store.entity(
        prov_type="stream",
        extent=(0.0, duration),
        attributes={"name": "recording", "path": f"/b/{stem}.wav", **identity.as_attributes(), **task_attributes(hint)},
    )
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
    for index, span in enumerate(words):
        store.entity(prov_type="word", extent=span, attributes=word_attributes(f"word{index}", span, index=index))
    write_store(store, run_root)
    return {"stem": run_root.name, "enhanced": str(streams / "enhanced.flac"), **identity_columns(identity)}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Three speech-bearing recordings and four vowels of one session, the last far longer than the rest."""
    rows = [
        _run(tmp_path, "rec-0001", "rainbow-passage", 12.0, [(1.0, 9.0, "A", 0.1)], (1.0, 9.0)),
        _run(tmp_path, "rec-0002", "caterpillar-passage", 12.0, [(1.0, 10.0, "A", 0.1)], (1.0, 10.0)),
        _run(
            tmp_path,
            "rec-0003",
            "respiration-and-cough-breath-1",
            10.0,
            [(2.0, 4.0, "A", 0.1), (6.0, 9.0, "B", 0.2)],
            (1.0, 5.0),
            words=TALKER_WORDS,
        ),
    ]
    rows += [_run(tmp_path, f"rec-000{i + 3}", f"prolonged-vowel-{i}", 10.0, [], (1.0, 9.0)) for i in range(1, 4)]
    rows.append(_run(tmp_path, "rec-0007", "prolonged-vowel-4", 50.0, [], (1.0, 9.0)))
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
    assert _reading(rows[0])["other_speaker"]["nonmatch"] == []
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
    assert _apply(corpus, config, quantiles, "--readings-out", str(out), "--participants", "pa") == 0
    assert [(Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows] == stores
    written = [json.loads(line) for line in (out / "pa.jsonl").read_text().splitlines()]
    assert len(written) == len(rows)
    assert sum(1 for record in written if record["nonmatch"]) == 1


def _two_participants(root: Path) -> Path:
    """``pa`` with two sessions (the second has only a breath task, its talker on level 0.2), ``pb`` with one."""
    rows = [
        _run(root, "rec-0001", "rainbow-passage", 12.0, [(1.0, 9.0, "A", 0.1)], (1.0, 9.0)),
        _run(root, "rec-0002", "caterpillar-passage", 12.0, [(1.0, 10.0, "A", 0.1)], (1.0, 10.0)),
        _run(
            root,
            "rec-0003",
            "respiration-and-cough-breath-1",
            10.0,
            [(2.0, 4.0, "A", 0.1), (6.0, 9.0, "B", 0.2)],
            (1.0, 5.0),
            session="s2",
            words=TALKER_WORDS,
        ),
        _run(root, "rec-0004", "rainbow-passage", 12.0, [(1.0, 9.0, "A", 0.3)], (1.0, 9.0), participant="pb"),
        _run(root, "rec-0005", "caterpillar-passage", 12.0, [(1.0, 9.0, "A", 0.3)], (1.0, 9.0), participant="pb"),
    ]
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def test_apply_shards_by_participant_and_falls_back_across_sessions(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every session of a participant lands in one task; a session with no enrollment reads the other one's."""
    monkeypatch.setattr(cli, "ecapa_embedder", lambda *a, **k: _embed)
    manifest = _two_participants(tmp_path)
    grouped = cli.participants_of([json.loads(line) for line in manifest.read_text().splitlines()])
    assert [(p, [s for s, _ in sessions]) for p, sessions in grouped] == [
        ("pa", ["pa/s1", "pa/s2"]),
        ("pb", ["pb/s1"]),
    ]
    quantiles = _quantiles(manifest)
    out = tmp_path / "readings"
    for index in range(2):
        assert (
            cli.main(
                [
                    "apply",
                    str(manifest),
                    "--quantiles",
                    str(quantiles),
                    "--slice-index",
                    str(index),
                    "--slice-count",
                    "2",
                    "--config",
                    str(config),
                    "--log-dir",
                    str(tmp_path / "logs"),
                    "--readings-out",
                    str(out),
                ]
            )
            == 0
        )
    slices = [
        [
            json.loads(line)
            for line in (tmp_path / "logs" / "slices" / f"cohort-slice-{i}-of-2.jsonl").read_text().splitlines()
        ]
        for i in range(2)
    ]
    assert [sorted({r["participant"] for r in records}) for records in slices] == [["pa"], ["pb"]]
    assert sorted({r["session"] for r in slices[0]}) == ["pa/s1", "pa/s2"]
    sources = {r["session"]: r["enrollment_source"] for r in slices[0]}
    assert sources == {"pa/s1": "session", "pa/s2": "participant_other_sessions"}
    breath = next(r for r in slices[0] if r["session"] == "pa/s2")
    assert breath["nonmatch"] == ["non_task"], "the talker's words and the removed speech around them, pooled"
    assert breath["pools"]["non_task"]["pooled_s"] == 3.0 and breath["pools"]["target"]["status"] == "not_applicable"
    assert {r["consistency"]["status"] for r in slices[1]} == {"not_applicable"}
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["pa.jsonl", "pb.jsonl"]
    assert cli.main(["consistency", str(tmp_path / "logs")]) == 0


def test_lower_fence_flags_the_outlying_session() -> None:
    """Tukey's fence over the session cosines sits under the bulk and above the outlier."""
    found = cli.lower_fence([0.8, 0.82, 0.85, 0.79, 0.81, 0.83, 0.1], 1.5)
    assert found["under_fence"] == 1 and 0.1 < found["fence"] < 0.79


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


def test_a_stale_manifest_row_refuses_its_whole_participant(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row naming an identity its run does not record refuses every store of its participant, before any write."""
    monkeypatch.setattr(cli, "ecapa_embedder", lambda *a, **k: _embed)
    manifest = _two_participants(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[1] = {**rows[1], "task": "loudness"}
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    quantiles = _quantiles(manifest)
    stores = {r["stem"]: (Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows}
    _apply(manifest, config, quantiles, "--participants", "pa")
    pa = [r for r in rows if r["participant"] == "pa"]
    assert all((Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() == stores[r["stem"]] for r in pa)


def test_a_manifest_without_identity_columns_is_refused(tmp_path: Path) -> None:
    """Grouping never parses a stem: a manifest from before rows carried identity is rebuilt."""
    with pytest.raises(ManifestWithoutIdentity):
        cli.participants_of([{"stem": "rec-0001", "enhanced": "/x"}])


def test_recordings_declaring_no_session_pool_under_their_participant() -> None:
    """A participant's session-less recordings form one pool, not one session each; one declaring none is its own."""
    rows = [
        {"stem": "a", "enhanced": "/x/a", **identity_columns(identity_of(hint_for("pa", token="x")))},
        {"stem": "b", "enhanced": "/x/b", **identity_columns(identity_of(hint_for("pa", token="y")))},
        {"stem": "c", "enhanced": "/x/c", **identity_columns(identity_of(hint_for()))},
    ]
    grouped = {p: [s for s, _ in sessions] for p, sessions in cli.participants_of(rows)}
    assert grouped == {"pa": ["pa/"], "c": ["c"]}


def _made_with_another_profile(row: dict) -> None:
    """Record in the row's run.json that the run was made with a profile of another name."""
    log = Path(row["enhanced"]).parents[1] / "run.json"
    log.write_text(json.dumps({"config": {"profile": {"name": "another-dataset", "version": "1", "digest": "x"}}}))


def test_a_member_made_with_another_profile_refuses_its_whole_participant(
    tmp_path: Path, config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review J4: any member's refusal, not only a stale row, refuses the group; its other members are not enrolled."""
    monkeypatch.setattr(cli, "ecapa_embedder", lambda *a, **k: _embed)
    manifest = _two_participants(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    quantiles = _quantiles(manifest)
    _made_with_another_profile(rows[1])
    stores = {r["stem"]: (Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() for r in rows}
    assert _apply(manifest, config, quantiles, "--participants", "pa") == 1
    pa = [r for r in rows if r["participant"] == "pa"]
    assert all((Path(r["enhanced"]).parents[2] / "run" / "store.jsonl").read_bytes() == stores[r["stem"]] for r in pa)
    logged = [json.loads(line) for line in next((tmp_path / "logs" / "slices").glob("cohort-slice-*.jsonl")).open()]
    assert {r["status"] for r in logged} == {"error"} and all("refused" in r["error"] for r in logged)


def test_collect_reads_through_the_checked_store(tmp_path: Path) -> None:
    """Review A3: collect refuses, as an error row, a run made with another profile."""
    manifest = _two_participants(tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    _made_with_another_profile(rows[0])
    logs = tmp_path / "logs"
    assert cli.main(["collect", str(manifest), "--slice-index", "0", "--slice-count", "1", "--log-dir", str(logs)]) == 0
    facts = [json.loads(line) for line in next((logs / "slices").glob("cohort-facts-*.jsonl")).open()]
    assert [bool(f.get("error")) for f in facts] == [True] + [False] * (len(rows) - 1)


@pytest.mark.parametrize("command", ["collect", "quantiles", "apply", "consistency"])
def test_without_a_profile_every_phase_exits_2_naming_it(
    command: str, tmp_path: Path, without_triage_profile: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review A3: each phase activates the profile before it reads; with none it exits 2, before reading anything."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")
    argv = {
        "collect": ["collect", str(manifest), "--slice-index", "0", "--slice-count", "1"],
        "quantiles": ["quantiles", "--facts", str(tmp_path), "--out", str(tmp_path / "q.json")],
        "apply": ["apply", str(manifest), "--slice-index", "0", "--slice-count", "1"],
        "consistency": ["consistency", str(tmp_path)],
    }[command]
    assert cli.main(argv) == 2
    assert "dataset profile" in capsys.readouterr().err
    assert not (tmp_path / "q.json").exists()


def test_a_pilot_consistency_cosine_is_read_beside_its_unset_bound() -> None:
    """Review J2: a pilot's block holds the cosine at top level, with no comparison; the log reads it."""
    assert cli._consistency_cosine({"status": "unavailable", "reason": "consistency_unset", "cosine": 0.71}) == 0.71
    measured = {"comparisons": [{"name": "session_vs_other_sessions_cosine", "value": 0.6}]}
    assert cli._consistency_cosine(measured) == 0.6
    assert cli._consistency_cosine({"status": "not_applicable"}) is None


def test_enrollment_vectors_of_two_target_speakers_of_one_session_are_two_files(tmp_path: Path) -> None:
    """Review J1: a group that is its participant keeps ``<session>.npy``; a target speaker's goes under it."""
    assert cli._vector_path(tmp_path, "pa", "pa", "pa/s1") == tmp_path / "pa/s1.npy"
    child = cli._vector_path(tmp_path, "pa:child", "pa", "pa/s1")
    parent = cli._vector_path(tmp_path, "pa:parent", "pa", "pa/s1")
    assert child != parent and child == tmp_path / "pa:child" / "pa/s1.npy"
    sessions = [("pa/s1", [{"stem": "a", "participant": "pa"}, {"stem": "b", "participant": "pa"}])]
    assert cli._participants_of_group(sessions) == {"pa"}
