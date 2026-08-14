"""Assembling per-cell checkpoints and per-shard manifests, and the missing-shard refusal."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# `scripts/` is deliberately not an importable package -- pyproject sets `pythonpath = ["src"]`.
# Load by file location, the convention every other test in this directory already uses.
_AGGREGATE = Path(__file__).resolve().parents[3] / "scripts" / "speaker_ceiling" / "aggregate.py"
_spec = importlib.util.spec_from_file_location("speaker_ceiling_aggregate_under_test", _AGGREGATE)
assert _spec is not None and _spec.loader is not None, f"could not load {_AGGREGATE}"
aggregate = importlib.util.module_from_spec(_spec)
sys.modules["speaker_ceiling_aggregate_under_test"] = aggregate
_spec.loader.exec_module(aggregate)

evaluate = sys.modules["evaluate"]  # aggregate.py's own `from evaluate import ...` bound this


def _write_cell(cells_dir: Path, backend: str, k: int, predicted: int) -> None:
    outcomes = [evaluate.SessionOutcome(session=f"session_{i}", predicted=predicted) for i in range(2)]
    evaluate.write_cell(evaluate.cell_path(cells_dir, backend, k), outcomes)


# ---------------------------------------------------------------------------------------
# load_cells
# ---------------------------------------------------------------------------------------


def test_load_cells_returns_every_checkpointed_outcome(tmp_path: Path) -> None:
    """Every present (backend, k) cell loads back exactly what was written."""
    _write_cell(tmp_path, "pyannote", 1, predicted=1)
    _write_cell(tmp_path, "pyannote", 2, predicted=2)

    outcomes = aggregate.load_cells(tmp_path, ["pyannote"], [1, 2])
    assert outcomes["pyannote"][1][0].predicted == 1
    assert outcomes["pyannote"][2][0].predicted == 2


def test_load_cells_refuses_when_a_cell_checkpoint_is_missing(tmp_path: Path) -> None:
    """A missing cell -- an array task that never finished -- is a hard refusal, naming the file.

    This is the brief's own scenario: a sweep that lost a task must not silently aggregate
    around the hole.
    """
    _write_cell(tmp_path, "pyannote", 1, predicted=1)
    # k=2's cell was never written -- simulating a shard that never completed.

    with pytest.raises(aggregate.MissingShardError) as excinfo:
        aggregate.load_cells(tmp_path, ["pyannote"], [1, 2])
    message = str(excinfo.value)
    assert "pyannote__k2.json" in message
    assert "pyannote__k1.json" not in message


def test_load_cells_treats_a_corrupt_checkpoint_the_same_as_a_missing_one(tmp_path: Path) -> None:
    """A checkpoint that exists but fails to parse must refuse exactly like an absent one.

    A truncated write that somehow slipped past write-then-rename must not be trusted as a
    completed cell -- see evaluate.read_cell's docstring for the same posture.
    """
    _write_cell(tmp_path, "pyannote", 1, predicted=1)
    (tmp_path / "pyannote__k2.json").write_text("{not valid json")

    with pytest.raises(aggregate.MissingShardError, match="pyannote__k2.json"):
        aggregate.load_cells(tmp_path, ["pyannote"], [1, 2])


# ---------------------------------------------------------------------------------------
# merge_corpus_manifests
# ---------------------------------------------------------------------------------------


def _manifest(counts: list, sessions_per_count: int, seed: int = 1) -> dict:
    sessions = [
        {"k": k, "session_index": i, "wav": f"k={k}/session_{i}.wav", "rttm": f"k={k}/session_{i}.rttm"}
        for k in counts
        for i in range(sessions_per_count)
    ]
    return {
        "method": "tts-composed sessions",
        "tts_model": {"path_or_uri": "fixture"},
        "session_params": {"turn_prob": 0.875},
        "seed": seed,
        "counts": counts,
        "sessions_per_count": sessions_per_count,
        "sessions": sessions,
    }


def test_merge_corpus_manifests_reads_an_unsharded_manifest_directly(tmp_path: Path) -> None:
    """A single manifest.json already covering every requested k needs no merging at all."""
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest([1, 2], sessions_per_count=2)))
    manifest = aggregate.merge_corpus_manifests(tmp_path, counts=[1, 2])
    assert manifest["counts"] == [1, 2]
    assert len(manifest["sessions"]) == 4


def test_merge_corpus_manifests_combines_per_shard_fragments(tmp_path: Path) -> None:
    """Per-shard manifest.k<k>.json fragments are merged into one manifest covering every k."""
    (tmp_path / "manifest.k1.json").write_text(json.dumps(_manifest([1], sessions_per_count=2)))
    (tmp_path / "manifest.k2.json").write_text(json.dumps(_manifest([2], sessions_per_count=2)))

    manifest = aggregate.merge_corpus_manifests(tmp_path, counts=[1, 2])
    assert manifest["counts"] == [1, 2]
    assert len(manifest["sessions"]) == 4
    assert {s["k"] for s in manifest["sessions"]} == {1, 2}


def test_merge_corpus_manifests_refuses_when_a_shard_fragment_is_missing(tmp_path: Path) -> None:
    """No manifest.json and a missing manifest.k2.json fragment must refuse, naming k=2."""
    (tmp_path / "manifest.k1.json").write_text(json.dumps(_manifest([1], sessions_per_count=2)))

    with pytest.raises(aggregate.MissingShardError, match=r"k=\[2\]"):
        aggregate.merge_corpus_manifests(tmp_path, counts=[1, 2])


def test_merge_corpus_manifests_refuses_when_fragments_disagree_on_seed(tmp_path: Path) -> None:
    """Fragments from two different seeds must never be silently merged into one profile."""
    (tmp_path / "manifest.k1.json").write_text(json.dumps(_manifest([1], sessions_per_count=2, seed=1)))
    (tmp_path / "manifest.k2.json").write_text(json.dumps(_manifest([2], sessions_per_count=2, seed=2)))

    with pytest.raises(aggregate.MissingShardError, match="disagrees"):
        aggregate.merge_corpus_manifests(tmp_path, counts=[1, 2])


def _write_wav(path: Path, sample_rate: int, num_samples: int = 1600) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(num_samples, dtype="float32"), sample_rate)


def _write_rttm(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("SPEAKER session_0 1 0.0 1.0 <NA> <NA> s0 <NA> <NA>\n")


# ---------------------------------------------------------------------------------------
# validate_corpus_for_counts
# ---------------------------------------------------------------------------------------


def test_validate_corpus_for_counts_passes_for_a_complete_correctly_rated_corpus(tmp_path: Path) -> None:
    """A corpus whose files actually exist, at the right rate, is accepted silently."""
    _write_wav(tmp_path / "k=1" / "session_0.wav", sample_rate=aggregate.CORPUS_SAMPLE_RATE)
    _write_rttm(tmp_path / "k=1" / "session_0.rttm")
    manifest = _manifest([1], sessions_per_count=1)

    aggregate.validate_corpus_for_counts(tmp_path, manifest, counts=[1])  # must not raise


def test_validate_corpus_for_counts_refuses_a_missing_wav(tmp_path: Path) -> None:
    """A session the manifest lists but whose wav is absent on disk is refused, naming the path."""
    _write_rttm(tmp_path / "k=1" / "session_0.rttm")
    manifest = _manifest([1], sessions_per_count=1)

    with pytest.raises(aggregate.CorpusValidationError, match="missing wav"):
        aggregate.validate_corpus_for_counts(tmp_path, manifest, counts=[1])


def test_validate_corpus_for_counts_refuses_the_wrong_sample_rate(tmp_path: Path) -> None:
    """A wav at the TTS model's native rate (24 kHz), not the corpus rate, is refused.

    This is the exact measured failure generate.py's docstring records: pyannote rejects a
    24 kHz file outright. Evaluation must catch it before spending any GPU time diarizing.
    """
    _write_wav(tmp_path / "k=1" / "session_0.wav", sample_rate=24000)
    _write_rttm(tmp_path / "k=1" / "session_0.rttm")
    manifest = _manifest([1], sessions_per_count=1)

    with pytest.raises(aggregate.CorpusValidationError, match="24000 Hz"):
        aggregate.validate_corpus_for_counts(tmp_path, manifest, counts=[1])


def test_validate_corpus_for_counts_refuses_a_k_with_no_sessions_at_all(tmp_path: Path) -> None:
    """A requested k entirely absent from the manifest is refused, naming that k."""
    _write_wav(tmp_path / "k=1" / "session_0.wav", sample_rate=aggregate.CORPUS_SAMPLE_RATE)
    _write_rttm(tmp_path / "k=1" / "session_0.rttm")
    manifest = _manifest([1], sessions_per_count=1)

    with pytest.raises(aggregate.CorpusValidationError, match=r"k=\[2\]"):
        aggregate.validate_corpus_for_counts(tmp_path, manifest, counts=[1, 2])


# ---------------------------------------------------------------------------------------
# check_cells_share_one_corpus
# ---------------------------------------------------------------------------------------


def test_check_cells_share_one_corpus_passes_when_every_cell_matches_the_manifest(tmp_path: Path) -> None:
    manifest = _manifest([1], sessions_per_count=1, seed=5)
    identity = evaluate.corpus_identity_from_manifest(manifest)
    evaluate.write_cell(evaluate.cell_path(tmp_path, "pyannote", 1), [], corpus_identity=identity)

    aggregate.check_cells_share_one_corpus(tmp_path, ["pyannote"], [1], manifest)  # must not raise


def test_check_cells_share_one_corpus_refuses_a_cell_from_a_different_corpus(tmp_path: Path) -> None:
    """A cell evaluated against a different seed than the corpus at hand must be refused.

    This is the check that actually enforces "every backend saw the same audio" -- without
    it, a difference in ceilings between two backends could just be a difference in what
    each one was shown.
    """
    manifest = _manifest([1], sessions_per_count=1, seed=5)
    other_identity = evaluate.corpus_identity_from_manifest(_manifest([1], sessions_per_count=1, seed=99))
    evaluate.write_cell(evaluate.cell_path(tmp_path, "pyannote", 1), [], corpus_identity=other_identity)

    with pytest.raises(aggregate.CorpusMismatchError, match="different corpus"):
        aggregate.check_cells_share_one_corpus(tmp_path, ["pyannote"], [1], manifest)


def test_check_cells_share_one_corpus_ignores_a_cell_with_no_recorded_identity(tmp_path: Path) -> None:
    """A cell written without identity tracking (e.g. an older test fixture) is not flagged.

    It was never asked to record one, so its absence is not evidence of a mismatch -- see
    evaluate.write_cell's docstring for the same distinction.
    """
    manifest = _manifest([1], sessions_per_count=1, seed=5)
    evaluate.write_cell(evaluate.cell_path(tmp_path, "pyannote", 1), [])  # no corpus_identity given

    aggregate.check_cells_share_one_corpus(tmp_path, ["pyannote"], [1], manifest)  # must not raise


def test_merge_corpus_manifests_falls_back_to_fragments_when_the_full_manifest_is_partial(tmp_path: Path) -> None:
    """A manifest.json covering fewer counts than requested is not trusted as complete.

    This is the shape a sharded run's k=1 task alone would leave if it (incorrectly) wrote
    to the unsharded name -- the merge must fall back to requiring every requested k's own
    fragment rather than trusting the stale partial manifest.json.
    """
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest([1], sessions_per_count=2)))
    (tmp_path / "manifest.k1.json").write_text(json.dumps(_manifest([1], sessions_per_count=2)))
    (tmp_path / "manifest.k2.json").write_text(json.dumps(_manifest([2], sessions_per_count=2)))

    manifest = aggregate.merge_corpus_manifests(tmp_path, counts=[1, 2])
    assert manifest["counts"] == [1, 2]
    assert len(manifest["sessions"]) == 4
