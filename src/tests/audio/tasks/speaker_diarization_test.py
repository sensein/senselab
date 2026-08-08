"""Tests for speaker diarization."""

from unittest.mock import Mock

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization import pyannote as pyannote_module
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine
from senselab.utils.data_structures.docker import docker_is_running
from senselab.utils.data_structures.model import model_for_task

if docker_is_running():
    DOCKER_AVAILABLE = True
else:
    DOCKER_AVAILABLE = False


@pytest.fixture
def pyannote_model() -> PyannoteAudioModel:
    """Fixture for Pyannote model."""
    return PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1")


@pytest.mark.skip(reason="pyannote-audio is a core dependency and always installed in test environment")
def test_pyannote_not_installed(pyannote_model: PyannoteAudioModel) -> None:
    """Test Pyannote not installed."""
    with pytest.raises(ModuleNotFoundError):
        _ = diarize_audios(audios=[Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)], model=pyannote_model)


def test_diarize_audios(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios."""
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=pyannote_model, device=cpu_cuda_device)
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


@pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker is not available",
)
@pytest.mark.skip(reason="This test takes too long, especially on CI")
def test_diarize_audios_with_nvidia_sortformer(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with NVIDIA Sortformer."""
    model: HFModel = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
    results = diarize_audios(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])
    # Optionally, check that at least one segment is returned
    assert len(results[0]) > 0


def test_diarize_audios_with_pyannote(
    resampled_mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with Pyannote."""
    results = diarize_audios_with_pyannote(
        audios=[resampled_mono_audio_sample], model=pyannote_model, device=cpu_cuda_device, num_speakers=2
    )
    assert len(results) == 1
    assert isinstance(results[0][0], ScriptLine)


def test_pyannote_pipeline_factory(pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType) -> None:
    """Test Pyannote pipeline factory."""
    pipeline1 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=cpu_cuda_device,
    )
    pipeline2 = PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=pyannote_model,
        device=cpu_cuda_device,
    )
    assert pipeline1 is pipeline2  # Check if the same instance is returned


def test_pyannote_pipeline_factory_forwards_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pyannote factory should forward the Hugging Face token to from_pretrained."""
    monkeypatch.setattr(PyannoteDiarization, "_pipelines", {})
    from_pretrained_mock = Mock()
    pipeline_mock = Mock()
    pipeline_mock.to.return_value = pipeline_mock
    from_pretrained_mock.return_value = pipeline_mock

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(pyannote_module.Pipeline, "from_pretrained", from_pretrained_mock)

    PyannoteDiarization._get_pyannote_diarization_pipeline(
        model=PyannoteAudioModel.model_construct(
            path_or_uri="pyannote/speaker-diarization-community-1",
            revision="main",
            info=None,
        ),
        device=DeviceType.CPU,
    )

    assert from_pretrained_mock.call_args.kwargs["token"] == "hf_test_token"


def test_diarize_audios_with_pyannote_invalid_sampling_rate(
    mono_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with unsupported sampling_rate."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[mono_audio_sample], model=pyannote_model, device=cpu_cuda_device)


def test_diarize_stereo_audios_with_pyannote_invalid(
    resampled_stereo_audio_sample: Audio, pyannote_model: PyannoteAudioModel, cpu_cuda_device: DeviceType
) -> None:
    """Test diarizing audios with unsupported number of channels."""
    with pytest.raises(ValueError):
        diarize_audios(audios=[resampled_stereo_audio_sample], model=pyannote_model, device=cpu_cuda_device)


@pytest.mark.parametrize(
    "model_id",
    [
        "microsoft/VibeVoice-ASR-HF",
        "AlexXu811/whisper-child-adult",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "BUT-FIT/diarizen-wavlm-large-s80-md",
        "nvidia/diar_sortformer_4spk-v1",
    ],
)
def test_model_for_task_resolves_new_diarizers_to_hfmodel(model_id: str) -> None:
    """The four new backends and Sortformer are HF-hosted, not Pyannote-hosted.

    Resolving them to PyannoteAudioModel would send them through pyannote's
    pipeline loader and fail with an opaque config error rather than dispatching.
    """
    assert isinstance(model_for_task(model_id, task="diarization"), HFModel)


def test_model_for_task_still_defaults_to_pyannote() -> None:
    """Anything not matched by a prefix stays on the Pyannote path."""
    assert isinstance(
        model_for_task("pyannote/speaker-diarization-3.1", task="diarization"),
        PyannoteAudioModel,
    )


def test_vibevoice_prefix_does_not_capture_the_tts_checkpoints() -> None:
    """`microsoft/VibeVoice-1.5B` is a TTS model, not the ASR diarizer.

    A bare `microsoft/VibeVoice` prefix would route it to
    VibeVoiceAsrForConditionalGeneration.from_pretrained and fail opaquely.
    """
    assert isinstance(
        model_for_task("microsoft/VibeVoice-1.5B", task="diarization"),
        PyannoteAudioModel,
    )


def test_speaker_hints_warn_when_the_backend_ignores_them(caplog: pytest.LogCaptureFixture) -> None:
    """Only Pyannote honours num_speakers.

    Silently dropping the hint on the other backends makes a misconfigured run
    indistinguishable from a working one.
    """
    import logging

    from senselab.audio.tasks.speaker_diarization.api import _warn_if_speaker_hints_ignored

    with caplog.at_level(logging.WARNING):
        _warn_if_speaker_hints_ignored(
            backend_name="DiariZen",
            num_speakers=2,
            min_speakers=None,
            max_speakers=None,
        )
    assert any("num_speakers" in r.message for r in caplog.records)
