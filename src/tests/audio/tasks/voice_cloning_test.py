"""This script is for testing the voice cloning API."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_cloning import clone_voices
from senselab.utils.data_structures import CoquiTTSModel

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    from TTS.api import TTS

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False


@pytest.fixture
def vc_model() -> CoquiTTSModel:
    """Fixture for Coqui TTS model."""
    return CoquiTTSModel(path_or_uri="voice_conversion_models/multilingual/multi-dataset/knnvc")


@pytest.mark.skipif(TTS_AVAILABLE, reason="TTS is available")
def test_clone_voices_tts_not_available() -> None:
    """Test when TTS is not available."""
    with pytest.raises(ModuleNotFoundError):
        CoquiTTSModel(path_or_uri="voice_conversion_models/multilingual/multi-dataset/knnvc")


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not TTS_AVAILABLE, reason="torchaudio or TTS are not available")
def test_clone_voices_length_mismatch(resampled_mono_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test length mismatch in source and target audios."""
    source_audios = [resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

    with pytest.raises(ValueError, match="The list of source and target audios must have the same length"):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not TTS_AVAILABLE, reason="torchaudio or TTS are not available")
def test_clone_voices_valid_input(resampled_mono_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test cloning voices with valid input."""
    source_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]

    try:
        cloned_output = clone_voices(
            source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None
        )
        assert isinstance(cloned_output, list), "Output must be a list."
        assert len(cloned_output) == 2, "Output list should contain exactly two audio samples."
        assert isinstance(cloned_output[0], Audio), "Each item in the output list should be an instance of Audio."
        source_duration = source_audios[0].waveform.shape[1]
        cloned_duration = cloned_output[0].waveform.shape[1]

        # Set tolerance to 1% of source duration
        tolerance = 0.01 * source_duration

        # Check if the absolute difference is within the tolerance
        assert abs(source_duration - cloned_duration) <= tolerance, (
            f"Cloned audio duration is not within acceptable range. Source: {source_duration}, "
            f"Cloned: {cloned_duration}"
        )
    except Exception as e:
        pytest.fail(f"An unexpected exception occurred: {e}")


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not TTS_AVAILABLE, reason="torchaudio or TTS are not available")
def test_clone_voices_unsupported_model(resampled_mono_audio_sample: Audio) -> None:
    """Test unsupported model."""
    source_audios = [resampled_mono_audio_sample]
    target_audios = [resampled_mono_audio_sample]
    # this uri doesn't exist
    with pytest.raises(ValueError, match="Model sensein/senselab not found. Available models:"):
        unsupported_model = CoquiTTSModel(path_or_uri="sensein/senselab")
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=unsupported_model, device=None)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not TTS_AVAILABLE, reason="torchaudio or TTS are not available")
def test_clone_voices_stereo_audio(resampled_stereo_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test unsupported stereo audio."""
    source_audios = [resampled_stereo_audio_sample]
    target_audios = [resampled_stereo_audio_sample]

    with pytest.raises(ValueError, match="Only mono audio files are supported."):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE or not TTS_AVAILABLE, reason="torchaudio or TTS are not available")
def test_clone_voices_invalid_sampling_rate(mono_audio_sample: Audio, vc_model: CoquiTTSModel) -> None:
    """Test unsupported sampling rate."""
    source_audios = [mono_audio_sample]
    target_audios = [mono_audio_sample]

    with pytest.raises(ValueError, match="Expected input sample rate"):
        clone_voices(source_audios=source_audios, target_audios=target_audios, model=vc_model, device=None)
