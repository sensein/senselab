"""The CLI: corpus -> evaluation -> derivation -> profile, and the two hard refusals.

No test here runs a diarization backend, builds a subprocess venv, or synthesizes audio:
`generate_corpus` is monkeypatched to write a tiny hand-built manifest with no real speech,
and `diarize_audios`/`Audio` are monkeypatched inside the `evaluate` module the CLI
actually calls into, exactly as `speaker_ceiling_evaluate_test.py` does for that module in
isolation.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from senselab.utils.data_structures import ScriptLine

_REPO_ROOT = Path(__file__).resolve().parents[3]

# `scripts/` is deliberately not an importable package (pyproject sets
# pythonpath = ["src"]). Loading `probe_speaker_ceilings.py` by file location runs its
# own top-level `sys.path.insert(0, .../scripts/speaker_ceiling)` followed by plain
# `from derive import ...` / `from evaluate import ...` / `from generate import ...` --
# those land in `sys.modules` under their bare names ("derive", "evaluate", "generate"),
# which is exactly what lets this test reach back into the "evaluate" module the CLI is
# using and monkeypatch `diarize_audios`/`Audio` there.
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
    """Stands in for `Audio(filepath=...)`: no test here reads a real audio file."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath


def _fake_manifest(counts: List[int], sessions_per_count: int) -> Dict[str, Any]:
    """A manifest shaped like generate_corpus's, with no real audio behind it."""
    sessions = []
    for k in counts:
        for i in range(sessions_per_count):
            sessions.append(
                {
                    "k": k,
                    "session_index": i,
                    "wav": f"k={k}/session_{i}.wav",
                    "rttm": f"k={k}/session_{i}.rttm",
                    "speakers": [f"voice_{j}" for j in range(k)],
                    "num_turns": 3,
                    "duration_seconds": 5.0,
                }
            )
    return {
        "method": "test fixture -- not a real TTS-composed corpus",
        "tts_model": {"path_or_uri": "fixture", "revision": None, "resolved_commit_sha": None, "voice_pool": []},
        "session_params": {},
        "seed": 0,
        "counts": counts,
        "sessions_per_count": sessions_per_count,
        "sessions": sessions,
    }


def _install_fake_generate_corpus(monkeypatch: pytest.MonkeyPatch, counts: List[int], sessions_per_count: int) -> None:
    """Replace the CLI's bound `generate_corpus` so no TTS ever runs.

    Writes only a manifest.json -- `evaluate_backend` never opens the wav paths it lists
    because `Audio` is mocked too, so no placeholder audio files are needed either.
    """

    def _fake_generate_corpus(
        out_dir: Path,
        counts: List[int],
        sessions_per_count: int,
        seed: int,
        tts_model: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(_fake_manifest(counts, sessions_per_count)))
        return out_dir

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


def test_dry_run_emits_a_profile_with_a_curve_over_the_requested_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tiny CPU dry run from the brief's Step 3: completes and emits a real profile."""
    _install_fake_generate_corpus(monkeypatch, counts=[1, 2], sessions_per_count=2)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})  # every session exactly right

    out_dir = tmp_path / "ceiling-dry"
    rc = cli.main(
        ["--counts", "1", "2", "--sessions", "2", "--out", str(out_dir), "--device", "cpu", "--backends", "pyannote"]
    )

    assert rc == 0
    profile = json.loads((out_dir / "profile.json").read_text())
    assert profile["counts"] == [1, 2]
    assert profile["sessions_per_count"] == 2
    assert profile["threshold"] == cli.DEFAULT_ACCURACY_THRESHOLD
    assert set(profile["backends"].keys()) == {"pyannote"}

    block = profile["backends"]["pyannote"]
    assert block["accuracy_curve"] == {"1": 1.0, "2": 1.0}
    assert block["confusion"] == {"1": {"1": 2}, "2": {"2": 2}}
    assert block["ceiling"] == 2
    assert block["refusal_reasons"] == {}
    # The corpus's generation method travels with the profile, next to the curve it produced.
    assert "corpus_manifest" in profile
    assert profile["corpus_manifest"]["method"] == "test fixture -- not a real TTS-composed corpus"
    assert "caveat" in profile and "not a guarantee about a real recording" in profile["caveat"]


def test_a_wrong_count_and_a_refusal_land_in_different_confusion_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Right, wrong, and refused stay three distinguishable outcomes end to end."""
    _install_fake_generate_corpus(monkeypatch, counts=[1], sessions_per_count=3)
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    calls = {"n": 0}

    def _mixed(audios: List[Any], model: Any = None, device: Any = None, **kwargs: Any) -> List[List[ScriptLine]]:
        calls["n"] += 1
        if calls["n"] == 1:
            return [[ScriptLine(speaker="S0", start=0.0, end=1.0)]]  # right: 1 speaker
        if calls["n"] == 2:
            return [
                [ScriptLine(speaker="S0", start=0.0, end=1.0), ScriptLine(speaker="S1", start=0.0, end=1.0)]
            ]  # wrong: 2
        raise ValueError("refused")  # refused

    monkeypatch.setattr(evaluate, "diarize_audios", _mixed)

    out_dir = tmp_path / "ceiling-mixed"
    rc = cli.main(
        ["--counts", "1", "--sessions", "3", "--out", str(out_dir), "--device", "cpu", "--backends", "pyannote"]
    )

    assert rc == 0
    block = json.loads((out_dir / "profile.json").read_text())["backends"]["pyannote"]
    assert block["confusion"]["1"] == {"1": 1, "2": 1, "refused": 1}
    assert block["refusal_reasons"]["1"] == {"ValueError": 1}
    assert block["accuracy_curve"]["1"] == pytest.approx(1 / 3)


def test_refuses_when_more_sessions_are_requested_than_were_generated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requesting more sessions than the corpus has must hard-error, not emit a partial profile.

    This is the brief's own required negative test: an un-triggered refusal is an
    unverified one.
    """
    _install_fake_generate_corpus(monkeypatch, counts=[1, 2], sessions_per_count=2)
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1, 2: 2})

    out_dir = tmp_path / "ceiling-short"
    # The CLI always generates a corpus with the same --sessions it will later require, so
    # a mismatch can only come from a corpus built independently of this run's --sessions.
    # Point --corpus at one built with 2 sessions/count and then ask for 3, which is
    # exactly the brief's own required negative test: "ask for more sessions than you
    # generated."
    corpus_dir = tmp_path / "prebuilt-corpus"
    monkeypatch.setattr(
        cli, "generate_corpus", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run"))
    )
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "manifest.json").write_text(json.dumps(_fake_manifest([1, 2], sessions_per_count=2)))

    rc = cli.main(
        [
            "--counts",
            "1",
            "2",
            "--sessions",
            "3",  # more than the 2 the fixed corpus actually has
            "--out",
            str(out_dir),
            "--device",
            "cpu",
            "--backends",
            "pyannote",
            "--corpus",
            str(corpus_dir),
        ]
    )

    assert rc == 1
    assert not (out_dir / "profile.json").exists()


def test_refuses_when_a_backend_produced_zero_successes_at_the_smallest_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backend that fails every session at the smallest k measured the harness, not itself."""
    _install_fake_generate_corpus(monkeypatch, counts=[1, 2], sessions_per_count=2)
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    def _always_raises(
        audios: List[Any], model: Any = None, device: Any = None, **kwargs: Any
    ) -> List[List[ScriptLine]]:
        (audio,) = audios
        k = int(Path(audio.filepath).parent.name.split("=")[1])
        if k == 1:
            raise RuntimeError("CUDA is required")
        return [[ScriptLine(speaker="S0", start=0.0, end=1.0), ScriptLine(speaker="S1", start=0.0, end=1.0)]]

    monkeypatch.setattr(evaluate, "diarize_audios", _always_raises)

    out_dir = tmp_path / "ceiling-broken"
    rc = cli.main(
        ["--counts", "1", "2", "--sessions", "2", "--out", str(out_dir), "--device", "cpu", "--backends", "pyannote"]
    )

    assert rc == 1
    assert not (out_dir / "profile.json").exists()


def test_reuses_an_existing_corpus_without_calling_generate_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--corpus skips generation entirely, for evaluating a corpus already on disk."""
    monkeypatch.setattr(
        cli, "generate_corpus", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run"))
    )
    _install_scripted_diarizer(monkeypatch, predicted_by_k={1: 1})

    corpus_dir = tmp_path / "prebuilt-corpus"
    corpus_dir.mkdir()
    (corpus_dir / "manifest.json").write_text(json.dumps(_fake_manifest([1], sessions_per_count=1)))

    out_dir = tmp_path / "out"
    rc = cli.main(
        [
            "--counts",
            "1",
            "--sessions",
            "1",
            "--out",
            str(out_dir),
            "--device",
            "cpu",
            "--backends",
            "pyannote",
            "--corpus",
            str(corpus_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "profile.json").exists()
    assert not (out_dir / "corpus").exists()
