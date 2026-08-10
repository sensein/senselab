"""The CLI's three modes (generate / evaluate / aggregate) and every refusal between them.

No test here runs a diarization backend, builds a subprocess venv, or synthesizes audio:
`generate_corpus` is monkeypatched to write tiny real (silent) wav/rttm fixtures instead of
calling Qwen3-TTS, and `diarize_audios`/`Audio` are monkeypatched inside the `evaluate`
module the CLI actually calls into -- exactly as `speaker_ceiling_evaluate_test.py` does for
that module in isolation. The fixture writes *real* files (not just manifest entries)
because `--mode evaluate` now validates the corpus on disk (`aggregate.validate_corpus_for_counts`
opens every wav with soundfile) before any backend runs.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
import soundfile as sf

from senselab.utils.data_structures import ScriptLine

_REPO_ROOT = Path(__file__).resolve().parents[3]

# `scripts/` is deliberately not an importable package (pyproject sets
# pythonpath = ["src"]). Loading `probe_speaker_ceilings.py` by file location runs its
# own top-level `sys.path.insert(0, .../scripts/speaker_ceiling)` followed by plain
# `from derive import ...` / `from evaluate import ...` / `from generate import ...` /
# `import aggregate` -- those land in `sys.modules` under their bare names ("derive",
# "evaluate", "generate", "aggregate"), which is exactly what lets this test reach back
# into the "evaluate" module the CLI is using and monkeypatch `diarize_audios`/`Audio` there.
_CLI = _REPO_ROOT / "scripts" / "probe_speaker_ceilings.py"
_spec = importlib.util.spec_from_file_location("probe_speaker_ceilings_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"
cli = importlib.util.module_from_spec(_spec)
sys.modules["probe_speaker_ceilings_under_test"] = cli
_spec.loader.exec_module(cli)

evaluate = sys.modules["evaluate"]  # the real module the CLI's `from evaluate import ...` bound


@pytest.fixture(autouse=True)
def _no_real_hf_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construction inside `run_session` reaches the Hub unless mocked -- see the
    identical fixture in `speaker_ceiling_evaluate_test.py`.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)


class _FakeAudio:
    """Stands in for `Audio(filepath=...)`: no test here reads a real audio file through senselab."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath


def _write_real_corpus_fixture(
    out_dir: Path,
    counts: List[int],
    sessions_per_count: int,
    seed: int,
    manifest_name: str = "manifest.json",
    sample_rate: int = 16000,
) -> Path:
    """Write a tiny but real corpus: actual (silent) wav files, real rttm text, a real manifest.

    Mirrors `generate.generate_corpus`'s output shape closely enough for
    `aggregate.validate_corpus_for_counts` and `aggregate.merge_corpus_manifests` to accept
    it, without calling Qwen3-TTS. `Audio` is separately mocked (see `_install_scripted_diarizer`)
    so `run_session` never actually reads these wavs -- but validation does, directly with
    soundfile, so the files must be real.
    """
    out_dir = Path(out_dir)
    sessions = []
    for k in counts:
        k_dir = out_dir / f"k={k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        for i in range(sessions_per_count):
            wav_path = k_dir / f"session_{i}.wav"
            rttm_path = k_dir / f"session_{i}.rttm"
            sf.write(str(wav_path), np.zeros(1600, dtype="float32"), sample_rate)
            speakers = [f"voice_{j}" for j in range(k)]
            rttm_path.write_text(
                "\n".join(f"SPEAKER session_{i} 1 0.0 1.0 <NA> <NA> {name} <NA> <NA>" for name in speakers) + "\n"
            )
            sessions.append(
                {
                    "k": k,
                    "session_index": i,
                    "wav": str(wav_path.relative_to(out_dir)),
                    "rttm": str(rttm_path.relative_to(out_dir)),
                    "speakers": speakers,
                    "num_turns": k,
                    "duration_seconds": 1.0,
                }
            )
    manifest = {
        "method": "test fixture -- real tiny files, not a real TTS-composed corpus",
        "tts_model": {"path_or_uri": "fixture", "revision": None, "resolved_commit_sha": "f" * 40, "voice_pool": []},
        "session_params": {},
        "seed": seed,
        "counts": list(counts),
        "sessions_per_count": sessions_per_count,
        "sessions": sessions,
    }
    (out_dir / manifest_name).write_text(json.dumps(manifest))
    return out_dir


def _install_real_fake_generate_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `generate_corpus` with one that writes real tiny fixtures via :func:`_write_real_corpus_fixture`.

    Needed for any test exercising `--mode generate` followed by `--mode evaluate`: the
    latter now validates the corpus for real, so a manifest naming files that were never
    written would fail validation before a single backend runs.
    """

    def _fake_generate_corpus(
        out_dir: Path,
        counts: List[int],
        sessions_per_count: int,
        seed: int,
        tts_model: Optional[Any] = None,
        device: Optional[Any] = None,
        manifest_name: str = "manifest.json",
    ) -> Path:
        return _write_real_corpus_fixture(out_dir, counts, sessions_per_count, seed, manifest_name=manifest_name)

    monkeypatch.setattr(cli, "generate_corpus", _fake_generate_corpus)


def _install_scripted_diarizer(monkeypatch: pytest.MonkeyPatch, predicted_by_k: Dict[int, int]) -> None:
    """Make every session's diarization result perfectly match `predicted_by_k[k]` speakers.

    The wav path is `k=<k>/session_<i>.wav`, so the true k is recoverable from the path
    without needing the RTTM ground truth `evaluate.py` never reads.
    """
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    def _fake_diarize_audios(
        audios: List[Any], model: Any = None, device: Any = None, **kwargs: Any
    ) -> List[List[ScriptLine]]:
        (audio,) = audios
        k = int(Path(audio.filepath).parent.name.split("=")[1])
        n = predicted_by_k[k]
        return [[ScriptLine(speaker=f"S{i}", start=0.0, end=1.0) for i in range(n)]]

    monkeypatch.setattr(evaluate, "diarize_audios", _fake_diarize_audios)


# ---------------------------------------------------------------------------------------
# --mode generate
# ---------------------------------------------------------------------------------------


def test_generate_mode_writes_a_manifest_covering_every_requested_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_real_fake_generate_corpus(monkeypatch)
    corpus_dir = tmp_path / "corpus"

    rc = cli.main(
        ["--mode", "generate", "--counts", "1", "2", "--sessions", "2", "--seed", "5", "--corpus", str(corpus_dir)]
    )

    assert rc == 0
    manifest = json.loads((corpus_dir / "manifest.json").read_text())
    assert manifest["counts"] == [1, 2]
    assert manifest["seed"] == 5


def test_generate_mode_defaults_corpus_under_a_durable_artifacts_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitting --corpus in generate mode still writes somewhere durable, not a temp dir.

    Chdir's into tmp_path first so the default relative `artifacts/...` path lands there,
    never in the real repo checkout.
    """
    monkeypatch.chdir(tmp_path)
    _install_real_fake_generate_corpus(monkeypatch)

    rc = cli.main(["--mode", "generate", "--counts", "1", "--sessions", "1", "--seed", "9"])

    assert rc == 0
    expected = tmp_path / "artifacts" / "speaker_ceiling" / "corpus" / "seed-9"
    assert (expected / "manifest.json").exists()


def test_generate_mode_shard_k_writes_only_that_ks_manifest_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_real_fake_generate_corpus(monkeypatch)
    corpus_dir = tmp_path / "corpus"

    rc = cli.main(
        [
            "--mode",
            "generate",
            "--counts",
            "1",
            "2",
            "3",
            "--sessions",
            "1",
            "--seed",
            "1",
            "--corpus",
            str(corpus_dir),
            "--shard-k",
            "2",
        ]
    )

    assert rc == 0
    assert not (corpus_dir / "manifest.json").exists()
    fragment = json.loads((corpus_dir / "manifest.k2.json").read_text())
    assert fragment["counts"] == [2]
    assert not (corpus_dir / "k=1").exists()
    assert not (corpus_dir / "k=3").exists()


def test_generate_mode_uses_slurm_array_task_id_when_shard_k_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    _install_real_fake_generate_corpus(monkeypatch)
    corpus_dir = tmp_path / "corpus"

    rc = cli.main(
        ["--mode", "generate", "--counts", "1", "2", "3", "--sessions", "1", "--seed", "1", "--corpus", str(corpus_dir)]
    )

    assert rc == 0
    assert (corpus_dir / "manifest.k3.json").exists()


def test_generate_mode_shard_k_outside_counts_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_real_fake_generate_corpus(monkeypatch)
    rc = cli.main(
        [
            "--mode",
            "generate",
            "--counts",
            "1",
            "2",
            "--sessions",
            "1",
            "--seed",
            "1",
            "--corpus",
            str(tmp_path / "corpus"),
            "--shard-k",
            "5",
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------------------
# --mode evaluate
# ---------------------------------------------------------------------------------------


def test_evaluate_mode_refuses_without_corpus(tmp_path: Path) -> None:
    """Phase 2 must never silently generate its own audio -- --corpus is mandatory."""
    rc = cli.main(
        ["--mode", "evaluate", "--counts", "1", "--sessions", "1", "--out", str(tmp_path / "out"), "--backends", "pyannote"]
    )
    assert rc == 2


def test_evaluate_mode_never_calls_generate_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with a valid --corpus, evaluate mode must never regenerate audio."""

    def _explode(**kwargs: Any) -> Any:
        raise AssertionError("evaluate mode must never call generate_corpus")

    monkeypatch.setattr(cli, "generate_corpus", _explode)
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1], sessions_per_count=1, seed=1)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1})

    rc = cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "--sessions",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
        ]
    )
    assert rc == 0


def test_evaluate_mode_refuses_a_corpus_missing_a_session_wav(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1], sessions_per_count=1, seed=1)
    (corpus_dir / "k=1" / "session_0.wav").unlink()

    rc = cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "--sessions",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
        ]
    )
    assert rc == 2
    assert not (tmp_path / "out" / "cells").exists()


def test_evaluate_mode_refuses_a_corpus_at_the_wrong_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 24 kHz corpus (the TTS model's native rate) must be caught before any GPU time is spent."""
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1], sessions_per_count=1, seed=1, sample_rate=24000)

    rc = cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "--sessions",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
        ]
    )
    assert rc == 2


def test_evaluate_mode_checkpoints_cells_and_writes_no_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1, 2], sessions_per_count=2, seed=7)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "2",
            "--sessions",
            "2",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_dir),
            "--device",
            "cpu",
            "--backends",
            "pyannote",
        ]
    )

    assert rc == 0
    assert not (out_dir / "profile.json").exists()
    cell1 = json.loads((out_dir / "cells" / "pyannote__k1.json").read_text())
    cell2 = json.loads((out_dir / "cells" / "pyannote__k2.json").read_text())
    assert len(cell1["outcomes"]) == 2
    assert len(cell2["outcomes"]) == 2
    assert cell1["corpus_identity"]["seed"] == 7


def test_evaluate_mode_shard_k_evaluates_only_that_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1, 2], sessions_per_count=1, seed=1)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "2",
            "--sessions",
            "1",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_dir),
            "--device",
            "cpu",
            "--backends",
            "pyannote",
            "--shard-k",
            "2",
        ]
    )

    assert rc == 0
    assert not (out_dir / "cells" / "pyannote__k1.json").exists()
    assert (out_dir / "cells" / "pyannote__k2.json").exists()


def test_evaluate_mode_refuses_a_cached_cell_from_a_different_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale cells_dir left over from a previous --corpus must not silently contaminate a new run."""
    corpus_a = tmp_path / "corpus-a"
    corpus_b = tmp_path / "corpus-b"
    _write_real_corpus_fixture(corpus_a, counts=[1], sessions_per_count=1, seed=1)
    _write_real_corpus_fixture(corpus_b, counts=[1], sessions_per_count=1, seed=2)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1})
    out_dir = tmp_path / "out"

    common_args = ["--mode", "evaluate", "--counts", "1", "--sessions", "1", "--out", str(out_dir), "--backends", "pyannote"]
    assert cli.main([*common_args, "--corpus", str(corpus_a)]) == 0
    assert cli.main([*common_args, "--corpus", str(corpus_b)]) == 2


def test_evaluate_mode_resumed_after_a_deleted_cell_matches_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting one cell and rerunning evaluate must recompute only that cell, matching a full run."""
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1, 2], sessions_per_count=2, seed=3)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    common_args = [
        "--mode",
        "evaluate",
        "--counts",
        "1",
        "2",
        "--sessions",
        "2",
        "--corpus",
        str(corpus_dir),
        "--device",
        "cpu",
        "--backends",
        "pyannote",
    ]

    def _outcomes_without_timing(cell_path: Path) -> list:
        # elapsed_s is real wall-clock time (time.time() around run_session) and will
        # never match bit-for-bit between two separate processes -- what must match is
        # everything that actually describes the measurement.
        payload = json.loads(cell_path.read_text())
        return [{k: v for k, v in outcome.items() if k != "elapsed_s"} for outcome in payload["outcomes"]]

    reference_out = tmp_path / "reference"
    cli.main([*common_args, "--out", str(reference_out)])
    reference_cell2 = _outcomes_without_timing(reference_out / "cells" / "pyannote__k2.json")

    resumed_out = tmp_path / "resumed"
    cli.main([*common_args, "--out", str(resumed_out)])
    (resumed_out / "cells" / "pyannote__k2.json").unlink()
    cli.main([*common_args, "--out", str(resumed_out)])

    assert _outcomes_without_timing(resumed_out / "cells" / "pyannote__k2.json") == reference_cell2
    assert (resumed_out / "cells" / "pyannote__k1.json").exists()


# ---------------------------------------------------------------------------------------
# --mode aggregate
# ---------------------------------------------------------------------------------------


def test_aggregate_mode_refuses_without_corpus(tmp_path: Path) -> None:
    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "--sessions", "1", "--out", str(tmp_path / "out"), "--backends", "pyannote"]
    )
    assert rc == 2


def test_out_is_required_for_evaluate_and_aggregate_modes(tmp_path: Path) -> None:
    rc_evaluate = cli.main(
        ["--mode", "evaluate", "--counts", "1", "--sessions", "1", "--corpus", str(tmp_path / "c")]
    )
    rc_aggregate = cli.main(
        ["--mode", "aggregate", "--counts", "1", "--sessions", "1", "--corpus", str(tmp_path / "c")]
    )
    assert rc_evaluate == 2
    assert rc_aggregate == 2


def test_aggregate_refuses_a_missing_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend never evaluated at all must refuse, not silently emit a profile with a hole."""
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1, 2], sessions_per_count=2, seed=1)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    out_dir = tmp_path / "out"

    # Only k=1 gets evaluated; k=2's cell is never written.
    cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "2",
            "--sessions",
            "2",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
            "--shard-k",
            "1",
        ]
    )

    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "2", "--sessions", "2", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
    )
    assert rc == 1
    assert not (out_dir / "profile.json").exists()


def test_aggregate_refuses_when_cells_disagree_about_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell evaluated against a different corpus than the one being aggregated must refuse.

    Simulates the scenario a --corpus swap mid-sweep would leave behind: the cell (from an
    earlier evaluate call against seed=1) is left on disk, but --corpus now points at a
    manifest for seed=2.
    """
    corpus_a = tmp_path / "corpus-a"
    corpus_b = tmp_path / "corpus-b"
    _write_real_corpus_fixture(corpus_a, counts=[1], sessions_per_count=1, seed=1)
    _write_real_corpus_fixture(corpus_b, counts=[1], sessions_per_count=1, seed=2)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1})
    out_dir = tmp_path / "out"

    cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "--sessions",
            "1",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_a),
            "--backends",
            "pyannote",
        ]
    )

    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "--sessions", "1", "--out", str(out_dir), "--corpus", str(corpus_b), "--backends", "pyannote"]
    )
    assert rc == 1
    assert not (out_dir / "profile.json").exists()


def test_aggregate_refuses_when_more_sessions_are_required_than_were_evaluated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original short-cell refusal, unchanged, reached through the new aggregate mode."""
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1], sessions_per_count=2, seed=1)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1})
    out_dir = tmp_path / "out"

    cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "--sessions",
            "2",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
        ]
    )

    # Aggregate as though 3 sessions were required, but the corpus (and thus the cell) only has 2.
    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "--sessions", "3", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
    )
    assert rc == 1
    assert not (out_dir / "profile.json").exists()
    dump_path = out_dir / "refusal_outcomes.json"
    assert dump_path.exists(), "the refusal-dump fix should write outcomes even reached via --mode aggregate"


def test_aggregate_refuses_when_a_backend_produced_zero_successes_at_the_smallest_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1, 2], sessions_per_count=1, seed=1)
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    def _always_raises(audios: List[Any], model: Any = None, device: Any = None, **kwargs: Any) -> List[List[ScriptLine]]:
        (audio,) = audios
        k = int(Path(audio.filepath).parent.name.split("=")[1])
        if k == 1:
            raise RuntimeError("CUDA is required")
        return [[ScriptLine(speaker="S0", start=0.0, end=1.0), ScriptLine(speaker="S1", start=0.0, end=1.0)]]

    monkeypatch.setattr(evaluate, "diarize_audios", _always_raises)
    out_dir = tmp_path / "out"

    cli.main(
        [
            "--mode",
            "evaluate",
            "--counts",
            "1",
            "2",
            "--sessions",
            "1",
            "--out",
            str(out_dir),
            "--corpus",
            str(corpus_dir),
            "--backends",
            "pyannote",
        ]
    )

    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "2", "--sessions", "1", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
    )
    assert rc == 1
    assert not (out_dir / "profile.json").exists()
    dumped = json.loads((out_dir / "refusal_outcomes.json").read_text())
    assert dumped["pyannote"]["1"][0]["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------------------
# End to end: generate -> evaluate -> aggregate
# ---------------------------------------------------------------------------------------


def test_full_three_phase_flow_produces_a_correct_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_real_fake_generate_corpus(monkeypatch)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    corpus_dir = tmp_path / "corpus"
    out_dir = tmp_path / "out"

    assert (
        cli.main(
            ["--mode", "generate", "--counts", "1", "2", "--sessions", "2", "--seed", "5", "--corpus", str(corpus_dir), "--device", "cpu"]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "--mode",
                "evaluate",
                "--counts",
                "1",
                "2",
                "--sessions",
                "2",
                "--out",
                str(out_dir),
                "--corpus",
                str(corpus_dir),
                "--device",
                "cpu",
                "--backends",
                "pyannote",
            ]
        )
        == 0
    )
    assert not (out_dir / "profile.json").exists()
    assert (
        cli.main(
            ["--mode", "aggregate", "--counts", "1", "2", "--sessions", "2", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
        )
        == 0
    )

    profile = json.loads((out_dir / "profile.json").read_text())
    assert profile["backends"]["pyannote"]["accuracy_curve"] == {"1": 1.0, "2": 1.0}
    assert profile["backends"]["pyannote"]["ceiling"] == 2
    assert profile["corpus_manifest"]["seed"] == 5
    assert "caveat" in profile and "not a guarantee about a real recording" in profile["caveat"]


def test_sharded_generate_and_evaluate_then_aggregate_produces_the_same_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The array-friendly path: shard generation and evaluation by k, aggregate once."""
    _install_real_fake_generate_corpus(monkeypatch)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})
    corpus_dir = tmp_path / "corpus"
    out_dir = tmp_path / "out"

    for k in (1, 2):
        assert (
            cli.main(
                [
                    "--mode",
                    "generate",
                    "--counts",
                    "1",
                    "2",
                    "--sessions",
                    "2",
                    "--seed",
                    "5",
                    "--corpus",
                    str(corpus_dir),
                    "--device",
                    "cpu",
                    "--shard-k",
                    str(k),
                ]
            )
            == 0
        )
    assert not (corpus_dir / "manifest.json").exists()

    for k in (1, 2):
        assert (
            cli.main(
                [
                    "--mode",
                    "evaluate",
                    "--counts",
                    "1",
                    "2",
                    "--sessions",
                    "2",
                    "--out",
                    str(out_dir),
                    "--corpus",
                    str(corpus_dir),
                    "--device",
                    "cpu",
                    "--backends",
                    "pyannote",
                    "--shard-k",
                    str(k),
                ]
            )
            == 0
        )

    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "2", "--sessions", "2", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
    )
    assert rc == 0
    profile = json.loads((out_dir / "profile.json").read_text())
    assert profile["backends"]["pyannote"]["accuracy_curve"] == {"1": 1.0, "2": 1.0}
    assert profile["corpus_manifest"]["counts"] == [1, 2]
    assert len(profile["corpus_manifest"]["sessions"]) == 4


def test_a_wrong_count_and_a_refusal_land_in_different_confusion_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Right, wrong, and refused stay three distinguishable outcomes through the full flow."""
    corpus_dir = tmp_path / "corpus"
    _write_real_corpus_fixture(corpus_dir, counts=[1], sessions_per_count=3, seed=1)
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    calls = {"n": 0}

    def _mixed(audios: List[Any], model: Any = None, device: Any = None, **kwargs: Any) -> List[List[ScriptLine]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [[ScriptLine(speaker="S0", start=0.0, end=1.0)]]  # right: 1 speaker
        if calls["n"] == 2:
            return [[ScriptLine(speaker="S0", start=0.0, end=1.0), ScriptLine(speaker="S1", start=0.0, end=1.0)]]  # wrong: 2
        raise ValueError("refused")  # refused

    monkeypatch.setattr(evaluate, "diarize_audios", _mixed)
    out_dir = tmp_path / "out"

    cli.main(
        ["--mode", "evaluate", "--counts", "1", "--sessions", "3", "--out", str(out_dir), "--corpus", str(corpus_dir), "--device", "cpu", "--backends", "pyannote"]
    )
    rc = cli.main(
        ["--mode", "aggregate", "--counts", "1", "--sessions", "3", "--out", str(out_dir), "--corpus", str(corpus_dir), "--backends", "pyannote"]
    )

    assert rc == 0
    block = json.loads((out_dir / "profile.json").read_text())["backends"]["pyannote"]
    assert block["confusion"]["1"] == {"1": 1, "2": 1, "refused": 1}
    assert block["refusal_reasons"]["1"] == {"ValueError": 1}
    assert block["accuracy_curve"]["1"] == pytest.approx(1 / 3)
