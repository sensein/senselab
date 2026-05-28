"""Tests for per-window speaker-embedding extraction dispatch.

Focuses on the backend-routing logic: a WavLM checkpoint id must be loaded
through the transformers WavLM backend (``TransformersWavLMModel``), while
SpeechBrain speaker models keep using ``SpeechBrainModel``. The extractor call
itself is mocked, so these run offline with no model downloads.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis import embeddings as E
from senselab.utils.data_structures import SpeechBrainModel, TransformersWavLMModel

_ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
_RESNET = "speechbrain/spkrec-resnet-voxceleb"
_WAVLM = "microsoft/wavlm-base-plus-sv"


def _audio(seconds: float = 4.0, sr: int = 16000) -> Audio:
    return Audio(waveform=torch.zeros(1, int(seconds * sr)), sampling_rate=sr)


def test_embedding_model_handle_routes_wavlm_to_transformers() -> None:
    """A ``microsoft/wavlm-*`` id resolves to the transformers WavLM handle."""
    handle = E._embedding_model_handle(_WAVLM)
    assert isinstance(handle, TransformersWavLMModel)
    assert handle.path_or_uri == _WAVLM


def test_embedding_model_handle_routes_speechbrain_by_default() -> None:
    """SpeechBrain speaker models keep the SpeechBrain handle (unchanged path)."""
    for model_id in (_ECAPA, _RESNET):
        handle = E._embedding_model_handle(model_id)
        assert isinstance(handle, SpeechBrainModel)
        assert handle.path_or_uri == model_id


def test_extract_per_window_dispatches_correct_handle_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """The consensus mix routes each model id to its matching backend handle.

    Mocks ``extract_speaker_embeddings_from_audios`` to record the handle type
    it receives, so the three-model default (ECAPA + ResNet + WavLM) exercises
    the dispatch without loading any model.
    """
    seen: dict[str, str] = {}

    def _fake(*, audios: list[Audio], model: Any, device: Any = None) -> list[torch.Tensor]:  # noqa: ANN401
        seen[model.path_or_uri] = type(model).__name__
        return [torch.ones(192) for _ in audios]

    monkeypatch.setattr(E, "extract_speaker_embeddings_from_audios", _fake)

    out = E.extract_per_window_embeddings(
        audio=_audio(),
        models=[_ECAPA, _RESNET, _WAVLM],
        window_s=2.0,
        hop_s=1.0,
    )

    assert set(out) == {_ECAPA, _RESNET, _WAVLM}
    assert seen[_ECAPA] == "SpeechBrainModel"
    assert seen[_RESNET] == "SpeechBrainModel"
    assert seen[_WAVLM] == "TransformersWavLMModel"
    # Every model produced the same window grid (one embedding per window).
    assert len({len(v) for v in out.values()}) == 1
