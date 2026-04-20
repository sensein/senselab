"""Tests for the speech to text task."""

from unittest.mock import Mock

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text import huggingface as huggingface_asr_module
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.audio.tasks.speech_to_text.huggingface import HuggingFaceASR
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine


def test_scriptline_from_dict() -> None:
    """Test creating ScriptLine from dict."""
    data = {
        "text": "Hello world",
        "chunks": [{"text": "Hello", "timestamps": [0.0, 1.0]}, {"text": "world", "timestamps": [1.0, 2.0]}],
    }
    scriptline = ScriptLine.from_dict(data)

    # Ensure chunks is not None before using it
    assert scriptline.chunks is not None
    assert len(scriptline.chunks) == 2
    assert scriptline.chunks[0].text == "Hello"
    assert scriptline.chunks[0].get_timestamps()[0] == 0.0
    assert scriptline.chunks[0].get_timestamps()[1] == 1.0

    assert scriptline.chunks[1].text == "world"
    assert scriptline.chunks[1].get_timestamps()[0] == 1.0
    assert scriptline.chunks[1].get_timestamps()[1] == 2.0


def test_transcribe_audios_clamps_negative_timestamp_artifacts(
    monkeypatch: pytest.MonkeyPatch, resampled_mono_audio_sample: Audio
) -> None:
    """Tiny negative Whisper timestamps should be clamped before ScriptLine validation."""
    fake_pipeline = Mock()
    fake_pipeline.feature_extractor = Mock(sampling_rate=resampled_mono_audio_sample.sampling_rate)
    fake_pipeline.return_value = [
        {
            "text": "hello world",
            "chunks": [
                {"text": "hello", "timestamp": [-0.02, 0.31]},
                {"text": "world", "timestamp": [0.31, 0.72]},
            ],
        }
    ]

    monkeypatch.setattr(HuggingFaceASR, "_get_hf_asr_pipeline", Mock(return_value=fake_pipeline))

    transcripts = transcribe_audios(
        audios=[resampled_mono_audio_sample],
        model=HFModel.model_construct(path_or_uri="openai/whisper-tiny", revision="main", info=None),
        device=DeviceType.CPU,
    )

    assert transcripts[0].start == 0.0
    assert transcripts[0].end == 0.72
    assert transcripts[0].chunks is not None
    assert transcripts[0].chunks[0].start == 0.0
    assert transcripts[0].chunks[0].end == 0.31


def test_hf_asr_pipeline_forwards_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ASR pipeline should forward the Hugging Face token to transformers."""
    monkeypatch.setattr(HuggingFaceASR, "_pipelines", {})
    pipeline_mock = Mock()
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(huggingface_asr_module, "pipeline", pipeline_mock)

    HuggingFaceASR._get_hf_asr_pipeline(
        model=HFModel.model_construct(path_or_uri="openai/whisper-tiny", revision="main", info=None),
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=DeviceType.CPU,
    )

    assert pipeline_mock.call_args.kwargs["token"] == "hf_test_token"


@pytest.fixture
def hf_model() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="openai/whisper-tiny")


@pytest.fixture
def hf_model2() -> HFModel:
    """Fixture for Hugging Face model."""
    return HFModel(path_or_uri="facebook/seamless-m4t-unity-small")


def test_hf_asr_pipeline_factory(hf_model: HFModel, any_device: DeviceType) -> None:
    """Test ASR pipeline factory."""
    pipeline1 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=any_device,
    )
    pipeline2 = HuggingFaceASR._get_hf_asr_pipeline(
        model=hf_model,
        return_timestamps="word",
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        device=any_device,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned (this is the case for serial execution)


@pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
def test_transcribe_audios(
    resampled_mono_audio_sample: Audio, resampled_mono_audio_sample_x2: Audio, hf_model: HFModel
) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=hf_model
    )
    assert len(transcripts) == 2
    assert isinstance(transcripts[0], ScriptLine)
    assert transcripts[0].text is not None and "This is Peter. This is Johnny. Kenny." in transcripts[0].text


@pytest.mark.parametrize("hf_model", ["hf_model", "hf_model2"], indirect=True)
def test_transcribe_audios_with_params(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    hf_model: HFModel,
    any_device: DeviceType,
) -> None:
    """Test transcribing audios."""
    transcripts = transcribe_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=hf_model,
        language=Language(language_code="English"),
        return_timestamps=False,
        device=any_device,
    )
    assert len(transcripts) == 2
    assert isinstance(transcripts[0], ScriptLine)
    # Note: we don't check the transcript because we have noticed that by specifying the language,
    # the transcript is not correct with our sample audio


def test_transcribe_audios_with_unsupported_params(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    hf_model: HFModel,
    any_device: DeviceType,
) -> None:
    """Test transcribing audios with an unsupported param."""
    with pytest.raises(TypeError, match="got an unexpected keyword argument"):
        transcribe_audios(
            audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
            model=hf_model,
            device=any_device,
            unsupported_param="unsupported_param",
        )


def test_transcribe_stereo_audio(
    resampled_stereo_audio_sample: Audio, hf_model: HFModel, any_device: DeviceType
) -> None:
    """Test transcribing stereo audio."""
    # Create a mock stereo audio sample
    with pytest.raises(ValueError, match="Stereo audio is not supported"):
        transcribe_audios(audios=[resampled_stereo_audio_sample], model=hf_model, device=any_device)


def test_transcribe_audio_with_wrong_sampling_rate(
    mono_audio_sample: Audio, hf_model: HFModel, any_device: DeviceType
) -> None:
    """Test transcribing stereo audio."""
    # Create a mock stereo audio sample
    with pytest.raises(ValueError, match="Incorrect sampling rate."):
        transcribe_audios(audios=[mono_audio_sample], model=hf_model, device=any_device)
