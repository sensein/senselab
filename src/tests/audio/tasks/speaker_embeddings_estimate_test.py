"""Estimate one speaker embedding from files that may contain that speaker.

No model is loaded anywhere here: the per-window extraction is monkeypatched to return controlled
vectors, so these tests exercise the aggregation, provenance and rejection logic deterministically
and without downloading a snapshot.
"""

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings import estimate_speaker_embedding_from_audios

_DIM = 48


def _audio(seconds: float = 8.0, sr: int = 16000) -> Audio:
    return Audio(waveform=torch.rand(1, int(seconds * sr)), sampling_rate=sr)


def _cone(axis: np.ndarray, n: int, spread: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = axis[None, :] + spread * rng.normal(size=(n, axis.size))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def axes() -> tuple[np.ndarray, np.ndarray]:
    """Two near-orthogonal directions: a target speaker and an intruder."""
    rng = np.random.default_rng(41)
    a = rng.normal(size=_DIM)
    a /= np.linalg.norm(a)
    b = rng.normal(size=_DIM)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    return a, b


def _patch_extraction(monkeypatch: pytest.MonkeyPatch, per_audio_vectors: list[np.ndarray]) -> None:
    """Make per-window extraction return the given vectors, one array per input audio."""
    from senselab.audio.tasks.speaker_embeddings import api as est_api

    calls = {"i": 0}

    def fake(audio, models, device=None, window_s=2.0, hop_s=1.0, **kwargs):  # noqa: ANN001, ANN003, ANN202
        vectors = per_audio_vectors[calls["i"]]
        calls["i"] += 1
        model_id = str(models[0].path_or_uri) if models else "stub/model"
        return {
            model_id: [
                est_api.WindowEmbedding(start_s=float(i) * hop_s, end_s=float(i) * hop_s + window_s, vector=v)
                for i, v in enumerate(vectors)
            ]
        }

    monkeypatch.setattr(est_api, "extract_per_window_embeddings", fake)
    monkeypatch.setattr(
        est_api, "_resolve_embedding_model", lambda model: ("speechbrain/spkrec-ecapa-voxceleb", "c" * 40, None)
    )


def test_the_estimate_is_a_unit_vector_with_provenance(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """A vector without provenance cannot be interpreted later, so both come back together."""
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()])

    assert np.linalg.norm(np.asarray(result.vector)) == pytest.approx(1.0)
    assert result.provenance.model_id == "speechbrain/spkrec-ecapa-voxceleb"
    assert result.provenance.model_commit_sha == "c" * 40
    assert result.provenance.method == "spherical_mean"
    assert result.provenance.window_s == 2.0
    assert result.provenance.hop_s == 1.0
    assert result.provenance.n_windows_used == 40
    assert result.provenance.n_windows_dropped == 0


def test_the_distribution_is_attached(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """The statistics are the whole point of the estimator.

    Without them the caller cannot judge how well-supported the centroid is.
    """
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()])

    assert result.distribution is not None
    assert result.distribution.counts.n_files == 2
    assert result.distribution.nulls.cos_sd_null == pytest.approx(1.0 / np.sqrt(_DIM))
    assert set(result.distribution.within_file) == set(result.distribution.cross_file.cos_file_centroid_to_pooled)


def test_contamination_is_visible_without_rejection(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """With the flag off nothing is dropped.

    The intruder shows up in the per-file statistics -- which is how a caller curates its input
    instead of us deciding.
    """
    target, intruder = axes
    _patch_extraction(
        monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2), _cone(intruder, 20, 0.03, 3)]
    )

    result = estimate_speaker_embedding_from_audios([_audio(), _audio(), _audio()])

    assert result.provenance.n_windows_dropped == 0
    assert result.provenance.method == "spherical_mean"
    # Narrows `Optional[EmbeddingDistribution]` for mypy: the estimator always attaches a
    # distribution, so this never actually fires, but a bare `result.distribution.x` would
    # leave the access unchecked against the field's declared type.
    assert result.distribution is not None
    lofo = result.distribution.centroid_robustness.leave_one_file_out_cos
    worst = min(lofo, key=lambda k: lofo[k])
    assert lofo[worst] < max(lofo.values())


def test_rejection_drops_the_intruder_and_records_it(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Rejection is a decision, so it must never be silent.

    The method string and the dropped count both say it happened.
    """
    target, intruder = axes
    _patch_extraction(
        monkeypatch, [_cone(target, 20, 0.03, 1), _cone(target, 20, 0.03, 2), _cone(intruder, 20, 0.03, 3)]
    )

    result = estimate_speaker_embedding_from_audios([_audio(), _audio(), _audio()], reject_contamination=True)

    assert result.provenance.method == "spherical_mean+dominant_cluster"
    assert result.provenance.n_windows_dropped == 20
    assert result.provenance.n_windows_used == 40
    assert np.asarray(result.vector) @ target > 0.9


def test_rejection_is_off_by_default(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """The default path decides nothing."""
    import inspect

    assert inspect.signature(estimate_speaker_embedding_from_audios).parameters["reject_contamination"].default is False


def test_an_empty_input_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """No files means no estimate. Returning a zero vector would look like a measurement."""
    with pytest.raises(ValueError, match="at least one"):
        estimate_speaker_embedding_from_audios([])


def test_source_files_are_recorded_when_available(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Provenance has to name what the estimate came from, or it cannot be reproduced."""
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 10, 0.03, 1)])
    audio = _audio()
    audio.metadata["source"] = "unused"
    result = estimate_speaker_embedding_from_audios([audio])
    assert isinstance(result.provenance.source_files, list)
