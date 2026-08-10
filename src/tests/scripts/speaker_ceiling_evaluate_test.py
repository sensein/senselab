"""The backend-evaluation loop: three outcomes, the full confusion, and the two refusals."""

import importlib.util
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest

from senselab.utils.data_structures import ScriptLine

# `scripts/` is deliberately not an importable package -- pyproject sets
# `pythonpath = ["src"]`, so the repo root is not on sys.path. Load by file location
# instead, the convention `speaker_ceiling_derive_test.py` already uses for the same reason.
_EVALUATE = Path(__file__).resolve().parents[3] / "scripts" / "speaker_ceiling" / "evaluate.py"
_spec = importlib.util.spec_from_file_location("speaker_ceiling_evaluate_under_test", _EVALUATE)
assert _spec is not None and _spec.loader is not None, f"could not load {_EVALUATE}"
evaluate = importlib.util.module_from_spec(_spec)
sys.modules["speaker_ceiling_evaluate_under_test"] = evaluate
_spec.loader.exec_module(evaluate)


@pytest.fixture(autouse=True)
def _no_real_hf_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file may construct a PyannoteAudioModel/HFModel via `run_session`.

    Both subclass `HFModel`, which validates against the Hub and resolves a commit SHA at
    construction -- unmocked, that reaches the network and (per this repo's own incident)
    can pull a full multi-GB snapshot. Applied to every test here rather than per-test,
    since `run_session` always builds a model before it ever reaches (the mocked)
    `diarize_audios`.
    """
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)


class _FakeAudio:
    """Stands in for `Audio(filepath=...)` so no test here reads a real file from disk."""

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath


def _line(speaker: Optional[str]) -> ScriptLine:
    return ScriptLine(speaker=speaker, start=0.0, end=1.0)


# ---------------------------------------------------------------------------------------
# _count_speakers
# ---------------------------------------------------------------------------------------


def test_count_speakers_counts_distinct_identity_labels() -> None:
    """An identity-kind backend's distinct speaker labels are the predicted count."""
    caps = evaluate.DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    lines = [_line("SPEAKER_00"), _line("SPEAKER_01"), _line("SPEAKER_00")]
    assert evaluate._count_speakers(lines, caps) == 2


def test_count_speakers_drops_overlap_for_role_backends() -> None:
    """OVERLAP marks two known talkers speaking at once, not a third one.

    capabilities.py documents this explicitly for child-adult (max_speakers=2 despite
    three label values); naively counting distinct labels here would contradict that
    documented rationale and penalize the backend for its own labeling convention
    rather than for a real miscount.
    """
    caps = evaluate.DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="role",
        labels_stable_across_files=False,
        max_speakers=2,
        honors_speaker_hints=False,
    )
    lines = [_line("CHILD"), _line("ADULT"), _line("OVERLAP")]
    assert evaluate._count_speakers(lines, caps) == 2


def test_count_speakers_ignores_none_and_empty_speaker() -> None:
    """A line with no speaker label contributes nothing to the count."""
    caps = evaluate.DiarizationCapabilities(
        populates_text=True,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    lines = [ScriptLine(text="hello", start=0.0, end=1.0), _line("S0")]
    assert evaluate._count_speakers(lines, caps) == 1


# ---------------------------------------------------------------------------------------
# run_session
# ---------------------------------------------------------------------------------------


def test_run_session_records_the_predicted_count_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend that answers gets its exact distinct-speaker count recorded."""
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)
    monkeypatch.setattr(evaluate, "diarize_audios", lambda **kwargs: [[_line("S0"), _line("S1")]])

    backend = evaluate.BACKENDS_BY_NAME["pyannote"]
    outcome = evaluate.run_session(backend, Path("/does/not/matter.wav"), device=None)

    assert outcome.predicted == 2
    assert outcome.error_type is None
    assert outcome.error_message is None


def test_run_session_records_none_and_the_exception_type_on_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal (e.g. child-adult under its 10s window) is None plus the exception type.

    Never a fabricated 0 and never a guessed count -- exact_count_accuracy already
    treats None as "not correct" without conflating it with a wrong number, and the
    exception type is what lets a reader of the profile tell a documented refusal from
    an unrelated crash.
    """
    monkeypatch.setattr(evaluate, "Audio", _FakeAudio)

    def _raise(**kwargs: Any) -> List[Any]:
        raise ValueError("clip is under the 10s minimum window")

    monkeypatch.setattr(evaluate, "diarize_audios", _raise)

    backend = evaluate.BACKENDS_BY_NAME["child_adult"]
    outcome = evaluate.run_session(backend, Path("/does/not/matter.wav"), device=None)

    assert outcome.predicted is None
    assert outcome.error_type == "ValueError"
    assert outcome.error_message is not None and "10s minimum window" in outcome.error_message


def test_run_session_never_raises_even_when_model_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated repo raising at construction is still a refusal, not a crash out of run_session."""

    def _raise_at_construction(backend: Any) -> Any:
        raise RuntimeError("gated repo, no token")

    monkeypatch.setattr(evaluate, "_build_model", _raise_at_construction)

    backend = evaluate.BACKENDS_BY_NAME["diarizen"]
    outcome = evaluate.run_session(backend, Path("/does/not/matter.wav"), device=None)

    assert outcome.predicted is None
    assert outcome.error_type == "RuntimeError"


# ---------------------------------------------------------------------------------------
# evaluate_backend
# ---------------------------------------------------------------------------------------


def test_evaluate_backend_groups_sessions_by_k_and_dispatches_each_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every session the manifest lists for a swept k gets exactly one run_session call."""
    calls: List[Path] = []

    def _fake_run_session(backend: Any, wav_path: Path, device: Any) -> Any:
        calls.append(wav_path)
        return evaluate.SessionOutcome(session=wav_path.stem, predicted=1)

    monkeypatch.setattr(evaluate, "run_session", _fake_run_session)

    manifest = {
        "sessions": [
            {"k": 1, "session_index": 1, "wav": "k=1/session_1.wav"},
            {"k": 1, "session_index": 0, "wav": "k=1/session_0.wav"},
            {"k": 2, "session_index": 0, "wav": "k=2/session_0.wav"},
            # A k not in `counts` below must not be evaluated at all.
            {"k": 5, "session_index": 0, "wav": "k=5/session_0.wav"},
        ]
    }
    backend = evaluate.BACKENDS_BY_NAME["pyannote"]
    outcomes = evaluate.evaluate_backend(backend, Path("/corpus"), manifest, counts=[1, 2], device=None)

    assert sorted(str(p) for p in calls) == [
        "/corpus/k=1/session_0.wav",
        "/corpus/k=1/session_1.wav",
        "/corpus/k=2/session_0.wav",
    ]
    # Ordered by session_index within a k, regardless of manifest order.
    assert [p.name for p in [Path(c) for c in calls if "k=1" in str(c)]] == ["session_0.wav", "session_1.wav"]
    assert list(outcomes.keys()) == [1, 2]
    assert len(outcomes[1]) == 2
    assert len(outcomes[2]) == 1


def test_evaluate_backend_returns_an_empty_list_for_a_k_with_no_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A swept k absent from the manifest comes back as an empty list, not a missing key.

    check_sweep_is_complete relies on this: an empty list still compares against
    sessions_per_count and trips the refusal, whereas a missing key could be mistaken
    for "not requested".
    """
    monkeypatch.setattr(evaluate, "run_session", lambda *a, **k: evaluate.SessionOutcome(session="x", predicted=1))
    manifest = {"sessions": [{"k": 1, "session_index": 0, "wav": "k=1/session_0.wav"}]}
    backend = evaluate.BACKENDS_BY_NAME["pyannote"]
    outcomes = evaluate.evaluate_backend(backend, Path("/corpus"), manifest, counts=[1, 3], device=None)
    assert outcomes[3] == []


def test_evaluate_backend_rejects_a_manifest_without_a_sessions_list() -> None:
    """A manifest that was not written by generate_corpus fails loudly, not silently as zero sessions."""
    backend = evaluate.BACKENDS_BY_NAME["pyannote"]
    with pytest.raises(ValueError, match="sessions"):
        evaluate.evaluate_backend(backend, Path("/corpus"), {}, counts=[1], device=None)


# ---------------------------------------------------------------------------------------
# confusion_from_outcomes / refusal_reasons_from_outcomes / curve_from_outcomes
# ---------------------------------------------------------------------------------------


def test_confusion_from_outcomes_separates_refused_from_wrong() -> None:
    """A refusal and a wrong count must render as distinguishable keys in the confusion."""
    outcomes = [
        evaluate.SessionOutcome(session="a", predicted=2),
        evaluate.SessionOutcome(session="b", predicted=2),
        evaluate.SessionOutcome(session="c", predicted=3),
        evaluate.SessionOutcome(session="d", predicted=None, error_type="ValueError"),
    ]
    assert evaluate.confusion_from_outcomes(outcomes) == {"2": 2, "3": 1, "refused": 1}


def test_refusal_reasons_from_outcomes_tallies_exception_types() -> None:
    """The exception type behind a refusal is recoverable, even though it is not a confusion key."""
    outcomes = [
        evaluate.SessionOutcome(session="a", predicted=None, error_type="ValueError"),
        evaluate.SessionOutcome(session="b", predicted=None, error_type="ValueError"),
        evaluate.SessionOutcome(session="c", predicted=None, error_type="RuntimeError"),
        evaluate.SessionOutcome(session="d", predicted=1),
    ]
    assert evaluate.refusal_reasons_from_outcomes(outcomes) == {"ValueError": 2, "RuntimeError": 1}


def test_curve_from_outcomes_matches_exact_count_accuracy() -> None:
    """The per-k accuracy is exactly what derive.exact_count_accuracy would compute by hand."""
    outcomes_by_k = {
        1: [evaluate.SessionOutcome(session="a", predicted=1), evaluate.SessionOutcome(session="b", predicted=1)],
        2: [evaluate.SessionOutcome(session="c", predicted=1), evaluate.SessionOutcome(session="d", predicted=2)],
    }
    assert evaluate.curve_from_outcomes(outcomes_by_k) == {1: 1.0, 2: 0.5}


# ---------------------------------------------------------------------------------------
# The two hard refusals
# ---------------------------------------------------------------------------------------


def test_check_sweep_is_complete_passes_when_every_cell_is_full() -> None:
    outcomes = {
        "pyannote": {
            1: [evaluate.SessionOutcome(session="a", predicted=1), evaluate.SessionOutcome(session="b", predicted=1)]
        }
    }
    evaluate.check_sweep_is_complete(outcomes, sessions_per_count=2)  # must not raise


def test_check_sweep_is_complete_names_the_short_backend_and_cell() -> None:
    """A short cell must hard-error, naming which backend and which k, per the brief.

    This is also the exact scenario a caller trips by asking for more sessions than a
    corpus was generated with.
    """
    outcomes = {"pyannote": {2: [evaluate.SessionOutcome(session="a", predicted=2)]}}
    with pytest.raises(evaluate.InsufficientMeasurementError) as excinfo:
        evaluate.check_sweep_is_complete(outcomes, sessions_per_count=2)
    message = str(excinfo.value)
    assert "pyannote" in message
    assert "k=2" in message
    assert "1/2" in message


def test_check_smallest_count_has_successes_passes_with_at_least_one_success() -> None:
    outcomes = {
        "pyannote": {
            1: [
                evaluate.SessionOutcome(session="a", predicted=None, error_type="ValueError"),
                evaluate.SessionOutcome(session="b", predicted=1),
            ]
        }
    }
    evaluate.check_smallest_count_has_successes(outcomes, smallest_k=1)  # must not raise


def test_check_smallest_count_has_successes_raises_when_a_backend_is_all_refusals() -> None:
    """Zero successes at the smallest k means the harness is broken, not the backend measured."""
    outcomes = {
        "child_adult": {
            1: [
                evaluate.SessionOutcome(session="a", predicted=None, error_type="RuntimeError"),
                evaluate.SessionOutcome(session="b", predicted=None, error_type="RuntimeError"),
            ]
        },
        "pyannote": {1: [evaluate.SessionOutcome(session="a", predicted=1)]},
    }
    with pytest.raises(evaluate.InsufficientMeasurementError) as excinfo:
        evaluate.check_smallest_count_has_successes(outcomes, smallest_k=1)
    message = str(excinfo.value)
    assert "child_adult" in message
    assert "pyannote" not in message


def test_check_smallest_count_has_successes_ignores_a_k_with_no_sessions_at_all() -> None:
    """An empty list at the smallest k (nothing generated for it) is not the same failure.

    check_sweep_is_complete already refuses on an empty/short cell; this check is only
    about a backend that ran and refused every time, so a k with literally no sessions
    must not also trip it.
    """
    outcomes: dict = {"pyannote": {1: []}}
    evaluate.check_smallest_count_has_successes(outcomes, smallest_k=1)  # must not raise
