"""The per-window embedding cache: key stability, round-trip fidelity, and path coercion.

No model downloads — the extractor's model call is stubbed, so these exercise the cache
layer itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    _deserialize_windows,
    _serialize_windows,
    extract_per_window_embeddings,
)

_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def _audio(seconds: float = 6.0) -> Audio:
    return Audio(waveform=torch.zeros(1, int(seconds * 16000)), sampling_rate=16000)


@pytest.fixture
def stub_models(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Stub the model call; return a dict counting how many times it ran."""
    calls = {"n": 0}

    def _fake(*, audios: Any, model: Any, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        calls["n"] += 1
        rng = np.random.default_rng(0)
        return [torch.tensor(rng.standard_normal(192), dtype=torch.float32) for _ in audios]

    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.embeddings.extract_speaker_embeddings_from_audios",
        _fake,
    )
    return calls


def test_serialize_round_trip_is_bit_exact() -> None:
    """A cached window must equal a freshly computed one, or the cache changes results."""
    entries = [
        WindowEmbedding(start_s=0.0, end_s=2.0, vector=np.random.default_rng(0).standard_normal(192).astype(np.float32))
    ]
    restored = _deserialize_windows(_serialize_windows(entries))
    assert len(restored) == 1
    assert restored[0].start_s == entries[0].start_s
    assert restored[0].end_s == entries[0].end_s
    assert restored[0].vector.dtype == np.float32
    assert np.array_equal(restored[0].vector, entries[0].vector)


def test_second_call_hits_the_cache(tmp_path: Path, stub_models: dict[str, int]) -> None:
    """A repeat call with the same (audio, model, grid) must not re-run the model."""
    audio = _audio()
    first = extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path)
    assert stub_models["n"] == 1
    second = extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path)
    assert stub_models["n"] == 1, "model re-ran despite a warm cache"
    assert len(first[_MODEL]) == len(second[_MODEL])
    for a, b in zip(first[_MODEL], second[_MODEL]):
        assert np.array_equal(a.vector, b.vector)
        assert (a.start_s, a.end_s) == (b.start_s, b.end_s)


def test_a_different_grid_is_a_different_cache_entry(tmp_path: Path, stub_models: dict[str, int]) -> None:
    """Window/hop are part of the key, so two grids coexist instead of clobbering.

    This is what lets a coarse enrollment grid and a fine detection grid be cached side by
    side rather than each invalidating the other.
    """
    audio = _audio()
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path)
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=0.5, hop_s=0.25, cache_dir=tmp_path)
    assert stub_models["n"] == 2, "the second grid should miss, not reuse the first"
    # And each remains individually warm.
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path)
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=0.5, hop_s=0.25, cache_dir=tmp_path)
    assert stub_models["n"] == 2


def test_cache_dir_accepts_a_string(tmp_path: Path, stub_models: dict[str, int]) -> None:
    """A str cache dir must work: it arrives from CLI args and env vars in practice.

    Regression: a str reached ``cache_lookup``, which does ``cache_dir / f"{key}.json"``,
    and raised ``TypeError: unsupported operand type(s) for /``. Worse, ``build.py`` wraps
    per-file extraction in try/except, so a systematically bad cache dir silently produced
    an ``insufficient`` profile instead of surfacing the cause.
    """
    audio = _audio()
    out = extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=str(tmp_path))
    assert out[_MODEL], "extraction produced no windows"
    assert stub_models["n"] == 1
    # and the entry is reusable when passed as a str again
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=str(tmp_path))
    assert stub_models["n"] == 1


def test_failed_extraction_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient model failure must be retried next run, not replayed as an empty result."""
    calls = {"n": 0}

    def _boom(*, audios: Any, model: Any, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        calls["n"] += 1
        raise RuntimeError("model load failed")

    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.embeddings.extract_speaker_embeddings_from_audios",
        _boom,
    )
    audio = _audio()
    failures: dict[str, str] = {}
    out = extract_per_window_embeddings(
        audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path, failures=failures
    )
    assert out[_MODEL] == [] and _MODEL in failures
    extract_per_window_embeddings(audio=audio, models=[_MODEL], window_s=2.0, hop_s=1.0, cache_dir=tmp_path)
    assert calls["n"] == 2, "an empty failure result was cached and replayed"
