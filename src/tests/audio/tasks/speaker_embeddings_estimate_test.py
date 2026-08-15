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
        # `models` is a list of plain id strings -- `extract_per_window_embeddings`'s real
        # contract -- not model objects; a fake that expected `.path_or_uri` here would pass
        # even when the estimator called the real function with the wrong type, which is
        # exactly the boundary bug `test_the_real_extraction_boundary_is_crossed_correctly`
        # below exists to catch instead.
        model_id = str(models[0]) if models else "stub/model"
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


def test_the_resolved_commit_sha_is_what_gets_threaded_to_extraction(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Provenance must record the same commit that extraction was actually asked to load with.

    Regression: `estimate_speaker_embedding_from_audios` resolved `commit_sha` for provenance but
    never passed it onward, so `extract_per_window_embeddings` (before its own fix) loaded
    `revision="main"` regardless -- a caller pinning an old commit got that SHA recorded while
    current-main weights loaded. This freezes that `commit_sha` is threaded through as the
    `revision=` kwarg, so the two cannot drift apart silently again.
    """
    from senselab.audio.tasks.speaker_embeddings import api as est_api

    target, _ = axes
    captured_revisions: list[object] = []

    def fake(audio, models, device=None, window_s=2.0, hop_s=1.0, revision=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured_revisions.append(revision)
        model_id = str(models[0]) if models else "stub/model"
        vectors = _cone(target, 10, 0.03, 1)
        return {
            model_id: [
                est_api.WindowEmbedding(start_s=float(i) * hop_s, end_s=float(i) * hop_s + window_s, vector=v)
                for i, v in enumerate(vectors)
            ]
        }

    monkeypatch.setattr(est_api, "extract_per_window_embeddings", fake)
    monkeypatch.setattr(
        est_api, "_resolve_embedding_model", lambda model: ("speechbrain/spkrec-ecapa-voxceleb", "e" * 40, None)
    )

    result = estimate_speaker_embedding_from_audios([_audio()])

    assert captured_revisions == ["e" * 40]
    assert result.provenance.model_commit_sha == "e" * 40


def test_a_partial_extraction_failure_is_recorded_not_swallowed(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """One file's model failure must be visible in provenance, not just a silently thinner estimate.

    Regression: this function was the only caller of `extract_per_window_embeddings` that omitted
    `failures=`, so a model crashing on one file out of several vanished into an empty window list
    with no trace -- `n_windows_dropped` stayed 0 and the failed file still appeared unmarked in
    `source_files`.
    """
    from senselab.audio.tasks.speaker_embeddings import api as est_api

    target, _ = axes
    call_index = {"i": 0}

    def fake(audio, models, device=None, window_s=2.0, hop_s=1.0, failures=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        model_id = str(models[0])
        i = call_index["i"]
        call_index["i"] += 1
        if i == 0:
            if failures is not None:
                failures[model_id] = "model failed during extraction: RuntimeError('boom')"
            return {model_id: []}
        vectors = _cone(target, 10, 0.03, 2)
        return {
            model_id: [
                est_api.WindowEmbedding(start_s=float(j) * hop_s, end_s=float(j) * hop_s + window_s, vector=v)
                for j, v in enumerate(vectors)
            ]
        }

    monkeypatch.setattr(est_api, "extract_per_window_embeddings", fake)
    monkeypatch.setattr(
        est_api, "_resolve_embedding_model", lambda model: ("speechbrain/spkrec-ecapa-voxceleb", "f" * 40, None)
    )

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()], file_ids=["failed-file", "ok-file"])

    assert result.provenance.extraction_failures == {
        "failed-file": "model failed during extraction: RuntimeError('boom')"
    }
    assert "ok-file" not in result.provenance.extraction_failures
    assert result.provenance.n_windows_used == 10


def test_every_file_failing_raises_the_real_cause_not_a_duration_guess(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that cannot load must not be misdiagnosed as "are the inputs shorter than window_s?".

    Regression: with no `failures=` collector, a total extraction failure and a genuinely
    too-short input produced the exact same generic message, hiding which one actually happened.
    """
    from senselab.audio.tasks.speaker_embeddings import api as est_api

    def fake(audio, models, device=None, window_s=2.0, hop_s=1.0, failures=None, **kwargs):  # noqa: ANN001, ANN003, ANN202
        model_id = str(models[0])
        if failures is not None:
            failures[model_id] = "model failed during extraction: OSError('no space left on device')"
        return {model_id: []}

    monkeypatch.setattr(est_api, "extract_per_window_embeddings", fake)
    monkeypatch.setattr(
        est_api, "_resolve_embedding_model", lambda model: ("speechbrain/spkrec-ecapa-voxceleb", "f" * 40, None)
    )

    with pytest.raises(ValueError, match="no space left on device"):
        estimate_speaker_embedding_from_audios([_audio()])


def test_file_ids_replace_the_positional_placeholder(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Caller-supplied ids must key every per-file block, not the `audio-0`/`audio-1` fallback.

    Regression: `Audio.filepath()` is empty for any in-memory or preprocessed audio (this module's
    own docstring recommends `resample_audios`, which drops `filepath`), so the positional fallback
    made `source_files`, `vectors_per_file`, `within_file` and `leave_one_file_out_cos` all
    unmappable back to a real recording.
    """
    target, _ = axes
    _patch_extraction(monkeypatch, [_cone(target, 10, 0.03, 1), _cone(target, 10, 0.03, 2)])

    result = estimate_speaker_embedding_from_audios([_audio(), _audio()], file_ids=["patient-01", "patient-02"])

    assert result.provenance.source_files == ["patient-01", "patient-02"]
    assert result.distribution is not None
    assert set(result.distribution.within_file) == {"patient-01", "patient-02"}


def test_file_ids_length_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """A miscounted id list must fail loudly rather than silently mis-map files to ids."""
    with pytest.raises(ValueError, match="file_ids"):
        estimate_speaker_embedding_from_audios([_audio(), _audio()], file_ids=["only-one"])


def test_the_real_extraction_boundary_is_crossed_correctly(monkeypatch: pytest.MonkeyPatch, axes) -> None:  # noqa: ANN001
    """Regression test for a boundary bug the other tests structurally cannot see.

    Every other test in this file monkeypatches `extract_per_window_embeddings` itself, which is
    precisely the seam a call-site type mismatch lives on -- a fake standing in for the whole
    function can accept whatever the estimator happens to pass it, real contract or not. A prior
    version of the estimator passed a `SenselabModel` object where `extract_per_window_embeddings`
    declares (and its real body assumes) a plain model-id string; every one of the other tests
    still passed, because their fake never exercised that body.

    This test instead lets the real `extract_per_window_embeddings` run -- real `window_starts`,
    real `slice_audio`, real per-window loop, real `SpeechBrainModel(path_or_uri=model_id, ...)`
    construction -- and only replaces the one call inside it that would otherwise load a model:
    `extract_speaker_embeddings_from_audios`. `SpeechBrainModel` construction still runs for real,
    so a wrong-typed id would still fail pydantic's `path_or_uri: Union[str, Path]` validation
    exactly as it did in production; only the two network-touching leaf calls inside that
    construction (`check_hf_repo_exists`, `resolve_revision`) are stubbed, so no HTTP request and
    no model download happen anywhere in this test.
    """
    from senselab.audio.tasks.speaker_embeddings import windowing as windowing_module
    from senselab.utils import model_revision as model_revision_module
    from senselab.utils.data_structures import model as model_module

    monkeypatch.setattr(model_module, "check_hf_repo_exists", lambda **kwargs: True)  # noqa: ANN003
    monkeypatch.setattr(model_revision_module, "resolve_revision", lambda *a, **k: "d" * 40)  # noqa: ANN002, ANN003

    target, _ = axes
    synthetic_vector = torch.from_numpy(_cone(target, 1, 0.0, 7)[0]).float()

    def fake_extract_embeddings(audios, model=None, device=None):  # noqa: ANN001, ANN202
        # Stands in for the one call in `extract_per_window_embeddings` that would otherwise load
        # ECAPA and run real inference; everything upstream of it (windowing, model construction)
        # is real.
        return [synthetic_vector for _ in audios]

    monkeypatch.setattr(windowing_module, "extract_speaker_embeddings_from_audios", fake_extract_embeddings)

    result = estimate_speaker_embedding_from_audios([_audio()])

    assert isinstance(result.vector, list)
    assert np.linalg.norm(np.asarray(result.vector)) == pytest.approx(1.0)
    # provenance.model_id must still be the plain id string, and _resolve_embedding_model's own
    # resolution (shared with HFModel's, both routed through the same patched `resolve_revision`)
    # must still produce a real 40-hex commit rather than a ref.
    assert result.provenance.model_id == "speechbrain/spkrec-ecapa-voxceleb"
    assert result.provenance.model_commit_sha == "d" * 40
    assert result.provenance.unresolved_reason is None
