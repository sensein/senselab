"""AST's minimum-input guard (specs/20260909-ast-too-short-guard/).

AST's feature extractor calls ``torchaudio.compliance.kaldi.fbank``, which raises a bare
``AssertionError`` when the audio is shorter than its analysis window — 53 of 61,442 triage
recordings hit this and lost every other derivative with them, because an unclassified
``AssertionError`` is a hard failure in ``preprocess.py`` rather than a cascading absence.
``AudioTooShortForAST`` (a ``ValueError``) is raised before the pipeline runs, so the existing
``except (ValueError, LookupError)`` classification there catches it unchanged.

These tests do not load AST's weights: ``_ast_min_samples`` only reads a feature extractor's own
attributes, and the pipeline-level test stubs the pipeline builder.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import ASTFeatureExtractor

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.huggingface import (
    AudioTooShortForAST,
    HuggingFaceAudioClassifier,
    _ast_min_samples,
)
from senselab.utils.data_structures import HFModel

AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


def test_audio_too_short_for_ast_is_a_value_error() -> None:
    """A ``ValueError`` subclass, so ``preprocess.py``'s existing classification catches it."""
    assert issubclass(AudioTooShortForAST, ValueError)


def test_ast_min_samples_matches_torchaudio_kaldi_defaults() -> None:
    """400 samples at AST's own 16 kHz: 25 ms frame_length (torchaudio's own default), not a literal."""
    assert _ast_min_samples(ASTFeatureExtractor()) == 400


def test_ast_min_samples_is_none_for_a_non_ast_extractor() -> None:
    """A feature extractor that isn't AST's carries no AST-specific minimum."""
    assert _ast_min_samples(SimpleNamespace(sampling_rate=16000)) is None


class TestClassifyAudiosWithTransformersRefusesTooShortAST:
    """Only the length guard is under test here.

    The pipeline builder is stubbed, so no AST weights load.
    """

    @staticmethod
    def _model() -> HFModel:
        # ``classify_audios_with_transformers`` never reads through to the Hub here because
        # ``_get_hf_audio_classification_pipeline`` is monkeypatched below; a SimpleNamespace
        # avoids HFModel's own Hub-validation round trip.
        return cast(HFModel, SimpleNamespace(path_or_uri=AST_ID, revision="main"))

    def test_399_samples_at_16khz_raises_the_typed_absence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One sample short of AST's 400-sample minimum."""
        fake_pipe = SimpleNamespace(feature_extractor=ASTFeatureExtractor())
        monkeypatch.setattr(
            HuggingFaceAudioClassifier,
            "_get_hf_audio_classification_pipeline",
            lambda **kwargs: fake_pipe,
        )
        audio = Audio(waveform=torch.zeros(1, 399), sampling_rate=16000)

        with pytest.raises(AudioTooShortForAST, match="399 samples.*need at least 400"):
            HuggingFaceAudioClassifier.classify_audios_with_transformers([audio], model=self._model())

    def test_400_samples_at_16khz_reaches_the_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exactly AST's minimum: the guard must not refuse the boundary itself."""

        class _FakePipe:
            feature_extractor = ASTFeatureExtractor()
            calls: list[Any] = []

            def __call__(self, formatted: list, **kwargs: Any) -> list:  # noqa: ANN401
                self.calls.append(formatted)
                return [[{"label": "Speech", "score": 1.0}] for _ in formatted]

        fake_pipe = _FakePipe()
        monkeypatch.setattr(
            HuggingFaceAudioClassifier,
            "_get_hf_audio_classification_pipeline",
            lambda **kwargs: fake_pipe,
        )
        audio = Audio(waveform=torch.zeros(1, 400), sampling_rate=16000)

        results = HuggingFaceAudioClassifier.classify_audios_with_transformers([audio], model=self._model())

        assert len(results) == 1
        assert len(fake_pipe.calls) == 1 and len(fake_pipe.calls[0]) == 1
