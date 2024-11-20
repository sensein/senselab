"""Tests for the language identification task."""

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.language_identification.api import identify_languages
from senselab.audio.tasks.language_identification.speechbrain import SpeechBrainLanguageIdentifier
from senselab.utils.data_structures import DeviceType, Language, SpeechBrainModel


@pytest.fixture
def speechbrain_lang_model() -> SpeechBrainModel:
    """Fixture for SpeechBrain language identification model."""
    return SpeechBrainModel(path_or_uri="speechbrain/lang-id-voxlingua107-ecapa")


def test_identify_languages_single_audio(
    resampled_mono_audio_sample: Audio, speechbrain_lang_model: SpeechBrainModel
) -> None:
    """Test identifying the language of a single audio sample."""
    languages = identify_languages(
        audios=[resampled_mono_audio_sample], model=speechbrain_lang_model, device=DeviceType.CPU
    )
    assert len(languages) == 1
    assert isinstance(languages[0], Language)
    assert languages[0].name is not None
    assert languages[0].alpha_3 is not None


def test_identify_languages_multiple_audios(
    resampled_mono_audio_sample: Audio, speechbrain_lang_model: SpeechBrainModel
) -> None:
    """Test identifying the language of multiple audio samples."""
    audios = [resampled_mono_audio_sample, resampled_mono_audio_sample]
    languages = identify_languages(audios=audios, model=speechbrain_lang_model, device=DeviceType.CPU)
    assert len(languages) == 2
    for language in languages:
        assert isinstance(language, Language)
        assert language.name is not None
        assert language.alpha_3 is not None


def test_identify_languages_stereo_audio(stereo_audio_sample: Audio, speechbrain_lang_model: SpeechBrainModel) -> None:
    """Test that identifying the language of a stereo audio raises a ValueError."""
    with pytest.raises(ValueError, match="Audio waveform must be mono"):
        identify_languages(audios=[stereo_audio_sample], model=speechbrain_lang_model, device=DeviceType.CPU)


def test_identify_languages_wrong_sampling_rate(
    mono_audio_sample: Audio, speechbrain_lang_model: SpeechBrainModel
) -> None:
    """Test that identifying the language of an audio with incorrect sampling rate raises a ValueError."""
    with pytest.raises(ValueError):
        identify_languages(audios=[mono_audio_sample], model=speechbrain_lang_model, device=DeviceType.CPU)


def test_model_caching(resampled_mono_audio_sample: Audio, speechbrain_lang_model: SpeechBrainModel) -> None:
    """Test model caching by identifying languages with the same model multiple times."""
    SpeechBrainLanguageIdentifier.identify_languages(audios=[resampled_mono_audio_sample], model=speechbrain_lang_model)
    assert len(SpeechBrainLanguageIdentifier._models.keys()) == 1
    SpeechBrainLanguageIdentifier.identify_languages(audios=[resampled_mono_audio_sample], model=speechbrain_lang_model)
    assert len(SpeechBrainLanguageIdentifier._models.keys()) == 1
