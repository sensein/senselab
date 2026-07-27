"""Tests for speaker_embeddings.py."""

import pytest
from torch import Tensor

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios
from senselab.utils.data_structures import DeviceType, SenselabModel, SpeechBrainModel, TransformersWavLMModel


@pytest.fixture
def ecapa_model() -> SpeechBrainModel:
    """Fixture for the ECAPA-TDNN model."""
    return SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")


@pytest.fixture
def xvector_model() -> SpeechBrainModel:
    """Fixture for the xvector model."""
    return SpeechBrainModel(path_or_uri="speechbrain/spkrec-xvect-voxceleb", revision="main")


@pytest.fixture
def resnet_model() -> SpeechBrainModel:
    """Fixture for the ResNet model."""
    return SpeechBrainModel(path_or_uri="speechbrain/spkrec-resnet-voxceleb", revision="main")


def test_extract_speaker_embeddings_from_empty_audio_list(
    ecapa_model: SpeechBrainModel, cpu_cuda_device: DeviceType
) -> None:
    """Test extracting speaker embeddings from an empty audio list returns an empty list."""
    embeddings = extract_speaker_embeddings_from_audios(audios=[], model=ecapa_model, device=cpu_cuda_device)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


def test_extract_speaker_embeddings_from_audio(
    resampled_mono_audio_sample: Audio,
    ecapa_model: SpeechBrainModel,
    xvector_model: SpeechBrainModel,
    resnet_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test extracting speaker embeddings from audio."""
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample], model=ecapa_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 192 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample], model=xvector_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 512 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample], model=resnet_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 256 for embedding in embeddings)


def test_extract_speaker_embeddings_from_multiple_audios(
    resampled_mono_audio_sample: Audio,
    ecapa_model: SpeechBrainModel,
    xvector_model: SpeechBrainModel,
    resnet_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test extracting speaker embeddings from multiple audios."""
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=ecapa_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 192 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=xvector_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 512 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample], model=resnet_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 256 for embedding in embeddings)


def test_extract_speaker_embeddings_from_multiple_audios_different_sizes(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    ecapa_model: SpeechBrainModel,
    xvector_model: SpeechBrainModel,
    resnet_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test extracting speaker embeddings from multiple audios of differing lengths."""
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=ecapa_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 192 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=xvector_model,
        device=cpu_cuda_device,
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 512 for embedding in embeddings)

    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2], model=resnet_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(embedding, Tensor) for embedding in embeddings)
    assert all(embedding.size(0) == 256 for embedding in embeddings)


def test_error_wrong_model(resampled_mono_audio_sample: Audio) -> None:
    """Test raising error when using a non-existent model."""
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample], model=SpeechBrainModel(path_or_uri="nonexistent-repo")
        )
    with pytest.raises(NotImplementedError):
        extract_speaker_embeddings_from_audios(
            audios=[resampled_mono_audio_sample],
            model=SenselabModel(path_or_uri="nonexistent-repo"),  # type: ignore
        )


def test_extract_speechbrain_speaker_embeddings_from_audio_resampled(
    mono_audio_sample: Audio,
    ecapa_model: SpeechBrainModel,
    xvector_model: SpeechBrainModel,
    resnet_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test extracting speaker embeddings from audio."""
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=ecapa_model, device=cpu_cuda_device)
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=xvector_model, device=cpu_cuda_device)
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=resnet_model, device=cpu_cuda_device)


def test_extract_speechbrain_speaker_embeddings_from_stereo_audio(
    stereo_audio_sample: Audio,
    ecapa_model: SpeechBrainModel,
    xvector_model: SpeechBrainModel,
    resnet_model: SpeechBrainModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Test extracting speaker embeddings from audio."""
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=ecapa_model, device=cpu_cuda_device)
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(
            audios=[stereo_audio_sample], model=xvector_model, device=cpu_cuda_device
        )
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=resnet_model, device=cpu_cuda_device)


# ─── WavLM backend (FR-019) ─────────────────────────────────────────────


@pytest.fixture
def wavlm_model() -> TransformersWavLMModel:
    """Fixture for the default WavLM SV model."""
    return TransformersWavLMModel(path_or_uri="microsoft/wavlm-base-plus-sv", revision="main")


def test_extract_wavlm_speaker_embeddings_from_empty_audio_list(
    wavlm_model: TransformersWavLMModel, cpu_cuda_device: DeviceType
) -> None:
    """Empty input → empty output (mirrors SpeechBrain backend)."""
    embeddings = extract_speaker_embeddings_from_audios(audios=[], model=wavlm_model, device=cpu_cuda_device)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 0


def test_extract_wavlm_speaker_embeddings_from_audio(
    resampled_mono_audio_sample: Audio,
    wavlm_model: TransformersWavLMModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Single mono 16 kHz audio → one 1-D embedding tensor (512-D for base-plus-sv)."""
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample], model=wavlm_model, device=cpu_cuda_device
    )
    assert isinstance(embeddings, list) and all(isinstance(e, Tensor) for e in embeddings)
    assert len(embeddings) == 1
    # wavlm-base-plus-sv embeds at 512-D
    assert embeddings[0].dim() == 1 and embeddings[0].size(0) == 512


def test_extract_wavlm_speaker_embeddings_from_multiple_audios(
    resampled_mono_audio_sample: Audio,
    resampled_mono_audio_sample_x2: Audio,
    wavlm_model: TransformersWavLMModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """Batch of varying-length audios produces one embedding per input."""
    embeddings = extract_speaker_embeddings_from_audios(
        audios=[resampled_mono_audio_sample, resampled_mono_audio_sample_x2],
        model=wavlm_model,
        device=cpu_cuda_device,
    )
    assert isinstance(embeddings, list) and len(embeddings) == 2
    assert all(isinstance(e, Tensor) and e.dim() == 1 and e.size(0) == 512 for e in embeddings)


def test_extract_wavlm_raises_on_non_16khz(
    mono_audio_sample: Audio,
    wavlm_model: TransformersWavLMModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """WavLM backend rejects non-16 kHz audio explicitly (same contract as SpeechBrain)."""
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[mono_audio_sample], model=wavlm_model, device=cpu_cuda_device)


def test_extract_wavlm_raises_on_stereo(
    stereo_audio_sample: Audio,
    wavlm_model: TransformersWavLMModel,
    cpu_cuda_device: DeviceType,
) -> None:
    """WavLM backend rejects non-mono audio explicitly."""
    with pytest.raises(ValueError):
        extract_speaker_embeddings_from_audios(audios=[stereo_audio_sample], model=wavlm_model, device=cpu_cuda_device)


def test_unsupported_model_type_raises_not_implemented() -> None:
    """``SenselabModel`` (neither SpeechBrain nor WavLM) → NotImplementedError."""

    class _Stub(SenselabModel):
        pass

    with pytest.raises(NotImplementedError):
        extract_speaker_embeddings_from_audios(audios=[], model=_Stub(path_or_uri="stub"))  # type: ignore[arg-type]
