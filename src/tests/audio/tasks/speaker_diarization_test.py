"""Tests for speaker diarization."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import api as diarization_api
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_diarization import pyannote as pyannote_module
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization, diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine
from senselab.utils.data_structures.docker import docker_is_running
from senselab.utils.data_structures.model import HFModel

if docker_is_running():
    DOCKER_AVAILABLE = True
else:
    DOCKER_AVAILABLE = False

_CHILD_ADULT_VENV_ROOT = Path.home() / ".cache" / "senselab" / "venvs" / "child-adult-diarization"
child_adult_venv_present = _CHILD_ADULT_VENV_ROOT.exists()


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


def test_diarize_audios_dispatches_to_vibevoice(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes a microsoft/VibeVoice* model id to the VibeVoice backend."""
    sentinel = [[ScriptLine(speaker="0", start=0.0, end=1.0, text="hi")]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_vibevoice", mock_fn)

    model = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_child_adult(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes the AlexXu811/whisper-child-adult model id to the child-adult backend."""
    sentinel = [[ScriptLine(speaker="ADULT", start=0.0, end=1.0)]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_child_adult", mock_fn)

    model = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_moss(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes an OpenMOSS-Team/MOSS-Transcribe-Diarize model id to the MOSS backend."""
    sentinel = [[ScriptLine(speaker="S01", start=0.0, end=1.0, text="hi")]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_moss", mock_fn)

    model = HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


def test_diarize_audios_dispatches_to_diarizen(monkeypatch: pytest.MonkeyPatch) -> None:
    """diarize_audios routes a BUT-FIT/diarizen model id to the DiariZen backend."""
    sentinel = [[ScriptLine(speaker="SPEAKER_00", start=0.0, end=1.0)]]
    mock_fn = Mock(return_value=sentinel)
    monkeypatch.setattr(diarization_api, "diarize_audios_with_diarizen", mock_fn)

    model = HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")
    result = diarize_audios(audios=[], model=model)

    assert result is sentinel
    mock_fn.assert_called_once()


@pytest.mark.skip(reason="Downloads a 7B model; run manually on a GPU machine")
def test_diarize_audios_with_vibevoice(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with VibeVoice-ASR-HF."""
    from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice

    model = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")
    results = diarize_audios_with_vibevoice(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skip(reason="Provisions a dedicated venv and downloads a model; run manually")
def test_diarize_audios_with_moss(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with MOSS-Transcribe-Diarize."""
    from senselab.audio.tasks.speaker_diarization.moss import diarize_audios_with_moss

    model = HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    results = diarize_audios_with_moss(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skip(reason="Provisions a dedicated venv (forked pyannote-audio) and downloads a model; run manually")
def test_diarize_audios_with_diarizen(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with DiariZen."""
    from senselab.audio.tasks.speaker_diarization.diarizen import diarize_audios_with_diarizen

    model = HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")
    results = diarize_audios_with_diarizen(audios=[resampled_mono_audio_sample], model=model)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])


@pytest.mark.skipif(
    not child_adult_venv_present,
    reason=f"child-adult-diarization venv not provisioned at {_CHILD_ADULT_VENV_ROOT}",
)
def test_diarize_audios_with_child_adult(resampled_mono_audio_sample: Audio) -> None:
    """Test diarizing audios with the USC-SAIL child-adult classifier (requires CUDA)."""
    from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult

    model = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    results = diarize_audios_with_child_adult(audios=[resampled_mono_audio_sample], model=model, device=DeviceType.CUDA)
    assert len(results) == 1
    assert all(isinstance(line, ScriptLine) for line in results[0])
    assert all(line.speaker in ("CHILD", "ADULT", "OVERLAP") for line in results[0])


def test_diarize_audios_with_child_adult_requires_cuda() -> None:
    """diarize_audios_with_child_adult raises a clear error when CUDA isn't available/requested."""
    from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult

    model = HFModel(path_or_uri="AlexXu811/whisper-child-adult")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        diarize_audios_with_child_adult(audios=[], model=model, device=DeviceType.CPU)
