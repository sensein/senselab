"""This script contains unit tests for the features extraction tasks."""

from pathlib import Path

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.opensmile import extract_opensmile_features_from_audios
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    extract_audio_duration,
    extract_cpp_descriptors,
    extract_harmonicity_descriptors,
    extract_intensity_descriptors,
    extract_jitter,
    extract_pitch_descriptors,
    extract_pitch_values,
    extract_shimmer,
    extract_slope_tilt,
    extract_spectral_moments,
    extract_speech_rate,
    get_sound,
    measure_f1f2_formants_bandwidths,
)
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_filter_bank_from_audios,
    extract_mel_spectrogram_from_audios,
    extract_mfcc_from_audios,
    extract_pitch_from_audios,
    extract_spectrogram_from_audios,
    extract_torchaudio_features_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
    extract_subjective_quality_features_from_audios,
)

try:
    import opensmile

    OPENSMILE_AVAILABLE = True
except ModuleNotFoundError:
    OPENSMILE_AVAILABLE = False

try:
    import parselmouth

    PARSELMOUTH_AVAILABLE = True
except ModuleNotFoundError:
    PARSELMOUTH_AVAILABLE = False

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(OPENSMILE_AVAILABLE, reason="openSMILE is installed.")
def test_missing_opensmile_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when openSMILE is not installed."""
    with pytest.raises(ModuleNotFoundError):
        from senselab.audio.tasks.features_extraction.opensmile import OpenSmileFeatureExtractorFactory

        OpenSmileFeatureExtractorFactory.get_opensmile_extractor("eGeMAPSv02", "Functionals")


@pytest.mark.skipif(PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is installed.")
def test_missing_parselmouth_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when Praat-Parselmouth is not installed."""
    with pytest.raises(ModuleNotFoundError):
        get_sound(audio=Path("path/to/audio.wav"))


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is installed.")
def test_missing_torchaudio_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when torchaudio is not installed."""
    with pytest.raises(ModuleNotFoundError):
        extract_torchaudio_features_from_audios([Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)])


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_audio_duration(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of audio durations."""
    result = extract_audio_duration(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    assert "duration" in result
    assert isinstance(result["duration"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_speech_rate(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of speech rate features."""
    result = extract_speech_rate(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    expected_keys = ["speaking_rate", "articulation_rate", "phonation_ratio", "pause_rate", "mean_pause_dur"]
    assert all(key in result for key in expected_keys), f"Missing keys: {set(expected_keys) - set(result.keys())}"
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_pitch_values(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch values."""
    result = extract_pitch_values(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    assert "pitch_floor" in result
    assert "pitch_ceiling" in result

    assert isinstance(result["pitch_floor"], float)
    assert isinstance(result["pitch_ceiling"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_pitch_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch features."""
    result = extract_pitch_descriptors(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert all(key in result for key in ["mean_f0_hertz", "stdev_f0_hertz"])
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_intensity_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of intensity features."""
    result = extract_intensity_descriptors(resampled_mono_audio_sample, floor=75.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "mean_db" in result
    assert "range_db_ratio" in result
    assert isinstance(result["mean_db"], float)
    assert isinstance(result["range_db_ratio"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_harmonicity_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of harmonicity features."""
    result = extract_harmonicity_descriptors(resampled_mono_audio_sample, floor=75.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "hnr_db_mean" in result
    assert "hnr_db_std_dev" in result
    assert isinstance(result["hnr_db_mean"], float)
    assert isinstance(result["hnr_db_std_dev"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_slope_tilt(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of spectral slope and tilt features."""
    result = extract_slope_tilt(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert "spectral_slope" in result
    assert "spectral_tilt" in result
    assert isinstance(result["spectral_slope"], float)
    assert isinstance(result["spectral_tilt"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_cpp_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of cepstral peak prominence (CPP) features."""
    result = extract_cpp_descriptors(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "mean_cpp" in result
    assert isinstance(result["mean_cpp"], float)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_measure_f1f2_formants_bandwidths(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of formant frequency features."""
    result = measure_f1f2_formants_bandwidths(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["f1_mean", "f1_std", "b1_mean", "b1_std", "f2_mean", "f2_std", "b2_mean", "b2_std"]
    )
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_spectral_moments(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of spectral moments."""
    result = extract_spectral_moments(
        resampled_mono_audio_sample, floor=75.0, ceiling=500.0, window_size=0.025, frame_shift=0.01
    )
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["spectral_gravity", "spectral_std_dev", "spectral_skewness", "spectral_kurtosis"]
    )
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_jitter(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of jitter descriptors."""
    result = extract_jitter(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["local_jitter", "localabsolute_jitter", "rap_jitter", "ppq5_jitter", "ddp_jitter"]
    )
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not PARSELMOUTH_AVAILABLE, reason="Praat-Parselmouth is not installed.")
def test_extract_shimmer(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of shimmer descriptors."""
    result = extract_shimmer(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in ["local_shimmer", "localDB_shimmer", "apq3_shimmer", "apq5_shimmer", "apq11_shimmer", "dda_shimmer"]
    )
    assert all(isinstance(result[key], float) for key in result)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_spectrogram_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of spectrogram from audio."""
    result = extract_spectrogram_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert all("spectrogram" in spec for spec in result)
    assert all(isinstance(spec["spectrogram"], torch.Tensor) for spec in result)
    # Spectrogram shape is (freq, time)
    assert all(spec["spectrogram"].dim() == 2 for spec in result)
    assert all(spec["spectrogram"].shape[0] == 513 for spec in result)

@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_spectrogram_from_audios_specify_n_fft(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of spectrogram from audio."""
    n_fft = 400
    result = extract_spectrogram_from_audios([resampled_mono_audio_sample], n_fft)
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert all("spectrogram" in spec for spec in result)
    assert all(isinstance(spec["spectrogram"], torch.Tensor) for spec in result)
    # Spectrogram shape is (freq, time)
    assert all(spec["spectrogram"].dim() == 2 for spec in result)
    assert all(spec["spectrogram"].shape[0] == 201 for spec in result)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_mel_spectrogram_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of mel spectrogram from audio."""
    result = extract_mel_spectrogram_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert all("mel_spectrogram" in spec for spec in result)
    assert all(isinstance(spec["mel_spectrogram"], torch.Tensor) for spec in result)
    # Mel spectrogram shape is (n_mels, time)
    assert all(spec["mel_spectrogram"].dim() == 2 for spec in result)
    assert all(spec["mel_spectrogram"].shape[0] == 128 for spec in result)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_mfcc_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of MFCC from audio."""
    result = extract_mfcc_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(mfcc, dict) for mfcc in result)
    assert all("mfcc" in mfcc for mfcc in result)
    assert all(isinstance(mfcc["mfcc"], torch.Tensor) for mfcc in result)
    # MFCC shape is (n_mfcc, time)
    assert all(mfcc["mfcc"].dim() == 2 for mfcc in result)
    assert all(mfcc["mfcc"].shape[0] == 40 for mfcc in result)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_mel_filter_bank(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of mel filter bank from audio."""
    result = extract_mel_filter_bank_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(mel_fb, dict) for mel_fb in result)
    assert all("mel_filter_bank" in mel_fb for mel_fb in result)
    assert all(isinstance(mel_fb["mel_filter_bank"], torch.Tensor) for mel_fb in result)
    # Mel filter bank shape is (n_mels, time)
    assert all(mel_fb["mel_filter_bank"].dim() == 2 for mel_fb in result)
    assert all(mel_fb["mel_filter_bank"].shape[0] == 128 for mel_fb in result)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_pitch_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch from audio."""
    result = extract_pitch_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(pitch, dict) for pitch in result)
    assert all("pitch" in pitch for pitch in result)
    assert all(isinstance(pitch["pitch"], torch.Tensor) for pitch in result)
    # Pitch shape is (time)
    assert all(pitch["pitch"].dim() == 1 for pitch in result)


@pytest.mark.skipif(not OPENSMILE_AVAILABLE, reason="openSMILE is not installed.")
def test_extract_opensmile_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of openSMILE features from audio."""
    # Perform eGeMAPSv02 and Functionals features extraction
    result = extract_opensmile_features_from_audios([resampled_mono_audio_sample], plugin="cf")

    # Assert the result is a list of dictionaries, and check each dictionary
    assert isinstance(result, list)
    assert all(isinstance(features, dict) for features in result)

    # Ensure that each dictionary contains the expected keys (e.g., certain features from eGeMAPS)
    expected_keys = {"F0semitoneFrom27.5Hz_sma3nz_amean", "jitterLocal_sma3nz_amean", "shimmerLocaldB_sma3nz_amean"}
    for features in result:
        assert set(map(str.lower, features.keys())).issuperset(map(str.lower, expected_keys))

    # Check the types of the values to ensure they are either floats or integers
    for features in result:
        assert all(isinstance(value, (float, int)) for value in features.values())


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_objective_quality_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of objective quality features from audio."""
    result = extract_objective_quality_features_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "stoi" in result[0]
    assert "pesq" in result[0]
    assert "si_sdr" in result[0]
    assert isinstance(result[0]["stoi"], float)
    assert isinstance(result[0]["pesq"], float)
    assert isinstance(result[0]["si_sdr"], float)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_objective_quality_features_from_audios_invalid_audio(mono_audio_sample: Audio) -> None:
    """Test extraction of objective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_objective_quality_features_from_audios([mono_audio_sample])


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_subjective_quality_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of subjective quality features from audio."""
    result = extract_subjective_quality_features_from_audios(
        audios=[resampled_mono_audio_sample], non_matching_references=[resampled_mono_audio_sample]
    )
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "mos" in result[0]
    assert isinstance(result[0]["mos"], float)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_extract_subjective_quality_features_invalid_audio(mono_audio_sample: Audio) -> None:
    """Test extraction of subjective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_subjective_quality_features_from_audios(
            audios=[mono_audio_sample], non_matching_references=[mono_audio_sample]
        )
