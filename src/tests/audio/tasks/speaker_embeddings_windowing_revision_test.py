"""Regression tests: `extract_per_window_embeddings` must load the model at a resolved commit SHA.

Finding: `windowing.py` used to construct ``SpeechBrainModel(path_or_uri=model_id, revision="main")``
unconditionally, discarding any resolved commit SHA a caller (e.g.
``estimate_speaker_embedding_from_audios``) had already computed for its own provenance. A caller
pinning an old commit therefore got that SHA recorded in ``SpeakerEmbeddingProvenance`` while
current-``main`` weights actually loaded -- "confidently wrong" provenance, the exact failure
CLAUDE.md names ("Recording a SHA while loading through a ref is the one outcome worse than
recording nothing").

Neither test constructs a real ``SpeechBrainModel``: the class is stubbed out entirely so
construction is a pure capture of its kwargs, and the one network-touching leaf
(``extract_speaker_embeddings_from_audios``) is stubbed to hand back deterministic vectors.
"""

from typing import Any

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings import windowing


def _audio(seconds: float = 3.0, sr: int = 16000) -> Audio:
    return Audio(waveform=torch.rand(1, int(seconds * sr)), sampling_rate=sr)


def _stub_model_and_extraction(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
    class _FakeSpeechBrainModel:
        def __init__(self, *, path_or_uri: str, revision: str) -> None:
            captured["path_or_uri"] = path_or_uri
            captured["revision"] = revision

    monkeypatch.setattr(windowing, "SpeechBrainModel", _FakeSpeechBrainModel)
    monkeypatch.setattr(
        windowing,
        "extract_speaker_embeddings_from_audios",
        lambda audios, model, device=None: [torch.rand(8) for _ in audios],  # noqa: ANN001, ANN202
    )


def test_a_caller_supplied_revision_reaches_model_construction_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact SHA passed in must be the one the model is constructed with -- not "main".

    Uses a well-formed 40-hex SHA so `resolve_revision`'s own short-circuit for an already-resolved
    ref fires with no network call; nothing about revision resolution needs mocking here.
    """
    captured: dict[str, Any] = {}
    _stub_model_and_extraction(monkeypatch, captured)

    sha = "a" * 40
    windowing.extract_per_window_embeddings(
        audio=_audio(), models=["some/model"], window_s=1.0, hop_s=0.5, revision=sha
    )

    assert captured["revision"] == sha
    assert captured["path_or_uri"] == "some/model"


def test_a_stale_revision_does_not_get_silently_replaced_by_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug this guards: an old pinned SHA used to be discarded in favor of a hardcoded "main".

    Two distinct SHAs stand in for "the commit the caller resolved a while ago" and "what main
    would resolve to today" -- the fix must load the former, never silently upgrade to the latter.
    """
    captured: dict[str, Any] = {}
    _stub_model_and_extraction(monkeypatch, captured)

    old_pinned_sha = "1" * 40
    windowing.extract_per_window_embeddings(
        audio=_audio(), models=["some/model"], window_s=1.0, hop_s=0.5, revision=old_pinned_sha
    )

    assert captured["revision"] == old_pinned_sha
    assert captured["revision"] != "main"


def test_no_revision_supplied_falls_back_to_resolving_main_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing callers that pass no `revision` (both `audio_analysis` callers) must keep working.

    `None` resolves each model's own "main" -- through `resolve_revision`, never a bare literal
    handed straight to `SpeechBrainModel`, which is what made the ref-load bug possible in the
    first place: a ref that never touches a resolver cannot become a SHA anywhere on the path.
    """
    from senselab.utils import model_revision as model_revision_module

    calls: list[tuple[str, str]] = []

    def fake_resolve_revision(repo_id: str, ref: str = "main", **kwargs: Any) -> str:  # noqa: ANN003, ANN401
        calls.append((repo_id, ref))
        return "b" * 40

    monkeypatch.setattr(model_revision_module, "resolve_revision", fake_resolve_revision)

    captured: dict[str, Any] = {}
    _stub_model_and_extraction(monkeypatch, captured)

    windowing.extract_per_window_embeddings(audio=_audio(), models=["some/model"], window_s=1.0, hop_s=0.5)

    assert calls == [("some/model", "main")]
    assert captured["revision"] == "b" * 40
