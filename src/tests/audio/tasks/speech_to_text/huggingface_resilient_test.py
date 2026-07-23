"""Behavior test: the HF ASR pipeline loads via the SHA-pinning resilient loader."""

import pytest

from senselab.utils.data_structures import DeviceType, HFModel


def test_asr_pipeline_loads_via_load_hf_resilient(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_hf_asr_pipeline routes loading through load_hf_resilient, not a bare pipeline() with a mutable revision."""
    import senselab.audio.tasks.speech_to_text.huggingface as hf_asr

    # Avoid Hub validation when constructing the HFModel.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)

    captured: dict = {}

    def fake_load(loader: object, *args: object, repo_id: str, revision: str = "main", **kwargs: object) -> str:
        captured["loader"] = loader
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        captured["kwargs"] = kwargs
        return "PIPE"

    monkeypatch.setattr(hf_asr, "load_hf_resilient", fake_load)
    hf_asr.HuggingFaceASR._pipelines.clear()

    model: HFModel = HFModel(path_or_uri="openai/whisper-small", revision="main")
    pipe = hf_asr.HuggingFaceASR._get_hf_asr_pipeline(
        model,
        return_timestamps=False,
        max_new_tokens=10,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )

    assert pipe == "PIPE"
    assert captured["loader"] is hf_asr.pipeline
    assert captured["repo_id"] == "openai/whisper-small"
    assert captured["revision"] == "main"
    # The mutable revision must NOT go straight to pipeline(); load_hf_resilient injects the resolved SHA.
    assert "revision" not in captured["kwargs"]
    assert captured["kwargs"]["model"] == "openai/whisper-small"
    assert captured["kwargs"]["task"] == "automatic-speech-recognition"
