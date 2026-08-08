"""Tests for speaker diarization."""

from unittest.mock import Mock

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization import api as api_module
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
def test_model_for_task_resolves_new_diarizers_to_hfmodel(model_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The four new backends and Sortformer are HF-hosted, not Pyannote-hosted.

    Resolving them to PyannoteAudioModel would send them through pyannote's
    pipeline loader and fail with an opaque config error rather than dispatching.
    """
    # Avoid Hub validation when constructing the HFModel: the real check does a full
    # snapshot_download, not a HEAD request, and would pull multi-GB weights on every
    # cold CI run for a test that only asserts a routing decision.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    assert isinstance(model_for_task(model_id, task="diarization"), HFModel)


def test_model_for_task_still_defaults_to_pyannote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anything not matched by a prefix stays on the Pyannote path."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    assert isinstance(
        model_for_task("pyannote/speaker-diarization-3.1", task="diarization"),
        PyannoteAudioModel,
    )


def test_vibevoice_prefix_does_not_capture_the_tts_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """`microsoft/VibeVoice-1.5B` is a TTS model, not the ASR diarizer.

    A bare `microsoft/VibeVoice` prefix would route it to
    VibeVoiceAsrForConditionalGeneration.from_pretrained and fail opaquely.
    """
    # Same rationale as above: PyannoteAudioModel is an HFModel subclass, so
    # constructing one for this (deliberately non-matching) id would otherwise
    # trigger a real snapshot_download of the 4.4 GB VibeVoice-1.5B checkpoint.
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
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


_ALL_BACKEND_ATTRS = (
    "diarize_audios_with_nvidia_sortformer",
    "diarize_audios_with_vibevoice",
    "diarize_audios_with_child_adult",
    "diarize_audios_with_moss",
    "diarize_audios_with_diarizen",
)


@pytest.mark.parametrize(
    ("model_id", "backend_attr", "expects_max_new_tokens"),
    [
        ("microsoft/VibeVoice-ASR-HF", "diarize_audios_with_vibevoice", True),
        ("AlexXu811/whisper-child-adult", "diarize_audios_with_child_adult", False),
        ("OpenMOSS-Team/MOSS-Transcribe-Diarize", "diarize_audios_with_moss", True),
        ("BUT-FIT/diarizen-wavlm-large-s80-md", "diarize_audios_with_diarizen", False),
    ],
)
def test_diarize_audios_dispatches_to_the_right_backend(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    backend_attr: str,
    expects_max_new_tokens: bool,
) -> None:
    """Each new prefix must reach its own backend, and no sibling's.

    `model.py`'s prefix list and this module's `elif` chain are two independently
    maintained copies of the same routing table (`model.py` says so itself). Nothing
    short of exercising `diarize_audios` itself would catch e.g. the MOSS prefix
    silently falling into the child-adult branch. Every candidate backend is patched
    so exactly one of them is asserted to have been called; the rest must stay untouched.
    """
    # Avoid Hub validation when constructing the HFModel (see rationale on the
    # model_for_task tests above).
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)

    mocks = {name: Mock(return_value=[[]]) for name in _ALL_BACKEND_ATTRS}
    for name, mock in mocks.items():
        monkeypatch.setattr(api_module, name, mock)

    model = HFModel(path_or_uri=model_id)
    audios = [Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)]

    result = diarize_audios(audios=audios, model=model, max_new_tokens=222)

    target_mock = mocks[backend_attr]
    target_mock.assert_called_once()
    assert result == [[]]
    for name, mock in mocks.items():
        if name != backend_attr:
            mock.assert_not_called()

    call_kwargs = target_mock.call_args.kwargs
    assert call_kwargs["model"] is model
    assert call_kwargs["audios"] is audios
    if expects_max_new_tokens:
        assert call_kwargs.get("max_new_tokens") == 222
    else:
        assert "max_new_tokens" not in call_kwargs
