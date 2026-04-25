"""This script contains unit tests for the features extraction tasks."""

from pathlib import Path

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.features_extraction.opensmile import extract_opensmile_features_from_audios
from senselab.audio.tasks.features_extraction.ppg import (
    extract_mean_phoneme_durations,
    extract_ppg_segments,
    extract_ppgs_from_audios,
    plot_ppg_phoneme_timeline,
    to_frame_major_posteriorgram,
)
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    extract_audio_duration,
    extract_cpp_descriptors,
    extract_harmonicity_descriptors,
    extract_intensity_descriptors,
    extract_jitter,
    extract_pitch_descriptors,
    extract_pitch_values,
    extract_praat_parselmouth_features_from_audios,
    extract_shimmer,
    extract_slope_tilt,
    extract_spectral_moments,
    extract_speech_rate,
    get_sound,
    measure_f1f2_formants_bandwidths,
)
from senselab.audio.tasks.features_extraction.sparc import SparcFeatureExtractor
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
    import ppgs

    PPGS_AVAILABLE = True
except ModuleNotFoundError:
    PPGS_AVAILABLE = False


@pytest.mark.skip(reason="opensmile is a core dependency and always installed; missing-dep path cannot be tested")
def test_missing_opensmile_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when openSMILE is not installed."""
    with pytest.raises(ModuleNotFoundError):
        from senselab.audio.tasks.features_extraction.opensmile import OpenSmileFeatureExtractorFactory

        OpenSmileFeatureExtractorFactory.get_opensmile_extractor("eGeMAPSv02", "Functionals")


def test_extract_opensmile_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of openSMILE features from audio."""
    # Perform eGeMAPSv02 and Functionals features extraction
    result = extract_opensmile_features_from_audios([resampled_mono_audio_sample])
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


@pytest.mark.skip(reason="torchaudio is a core dependency and always installed; missing-dep path cannot be tested")
def test_missing_torchaudio_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when torchaudio is not installed."""
    with pytest.raises(ModuleNotFoundError):
        extract_torchaudio_features_from_audios([Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)])


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


def test_extract_torchaudio_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of torchaudio features from audio."""
    result = extract_torchaudio_features_from_audios([resampled_mono_audio_sample])
    print(result[0].keys())
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert set(["pitch", "mel_filter_bank", "mfcc", "mel_spectrogram", "spectrogram"]).issubset(set(result[0].keys()))


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


def test_extract_pitch_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch from audio."""
    result = extract_pitch_from_audios([resampled_mono_audio_sample])
    assert isinstance(result, list)
    assert all(isinstance(pitch, dict) for pitch in result)
    assert all("pitch" in pitch for pitch in result)
    assert all(isinstance(pitch["pitch"], torch.Tensor) for pitch in result)
    # Pitch shape is (time)
    assert all(pitch["pitch"].dim() == 1 for pitch in result)


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


def test_extract_objective_quality_features_from_audios_invalid_audio(mono_audio_sample: Audio) -> None:
    """Test extraction of objective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_objective_quality_features_from_audios([mono_audio_sample])


def test_extract_subjective_quality_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of subjective quality features from audio."""
    result = extract_subjective_quality_features_from_audios(
        audios=[resampled_mono_audio_sample], non_matching_references=[resampled_mono_audio_sample]
    )
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert "mos" in result[0]
    assert isinstance(result[0]["mos"], float)


def test_extract_subjective_quality_features_invalid_audio(mono_audio_sample: Audio) -> None:
    """Test extraction of subjective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_subjective_quality_features_from_audios(
            audios=[mono_audio_sample], non_matching_references=[mono_audio_sample]
        )


@pytest.mark.skip(reason="parselmouth is a core dependency and always installed; missing-dep path cannot be tested")
def test_missing_parselmouth_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when Praat-Parselmouth is not installed."""
    with pytest.raises(ModuleNotFoundError):
        get_sound(audio=Path("path/to/audio.wav"))


def test_extract_audio_duration(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of audio durations."""
    result = extract_audio_duration(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    assert "duration" in result
    assert isinstance(result["duration"], float)


def test_extract_speech_rate(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of speech rate features."""
    result = extract_speech_rate(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    expected_keys = ["speaking_rate", "articulation_rate", "phonation_ratio", "pause_rate", "mean_pause_dur"]
    assert all(key in result for key in expected_keys), f"Missing keys: {set(expected_keys) - set(result.keys())}"
    assert all(isinstance(result[key], float) for key in result)


def test_extract_pitch_values(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch values."""
    result = extract_pitch_values(resampled_mono_audio_sample)
    assert isinstance(result, dict)
    assert "pitch_floor" in result
    assert "pitch_ceiling" in result

    assert isinstance(result["pitch_floor"], float)
    assert isinstance(result["pitch_ceiling"], float)


def test_extract_pitch_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of pitch features."""
    result = extract_pitch_descriptors(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert all(key in result for key in ["mean_f0_hertz", "stdev_f0_hertz"])
    assert all(isinstance(result[key], float) for key in result)


def test_extract_intensity_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of intensity features."""
    result = extract_intensity_descriptors(resampled_mono_audio_sample, floor=75.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "mean_db" in result
    assert "range_db_ratio" in result
    assert isinstance(result["mean_db"], float)
    assert isinstance(result["range_db_ratio"], float)


def test_extract_harmonicity_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of harmonicity features."""
    result = extract_harmonicity_descriptors(resampled_mono_audio_sample, floor=75.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "hnr_db_mean" in result
    assert "hnr_db_std_dev" in result
    assert isinstance(result["hnr_db_mean"], float)
    assert isinstance(result["hnr_db_std_dev"], float)


def test_extract_slope_tilt(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of spectral slope and tilt features."""
    result = extract_slope_tilt(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert "spectral_slope" in result
    assert "spectral_tilt" in result
    assert isinstance(result["spectral_slope"], float)
    assert isinstance(result["spectral_tilt"], float)


def test_extract_cpp_descriptors(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of cepstral peak prominence (CPP) features."""
    result = extract_cpp_descriptors(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert "mean_cpp" in result
    assert isinstance(result["mean_cpp"], float)


def test_measure_f1f2_formants_bandwidths(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of formant frequency features."""
    result = measure_f1f2_formants_bandwidths(resampled_mono_audio_sample, floor=75.0, ceiling=500.0, frame_shift=0.01)
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["f1_mean", "f1_std", "b1_mean", "b1_std", "f2_mean", "f2_std", "b2_mean", "b2_std"]
    )
    assert all(isinstance(result[key], float) for key in result)


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


def test_extract_jitter(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of jitter descriptors."""
    result = extract_jitter(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert all(
        key in result for key in ["local_jitter", "localabsolute_jitter", "rap_jitter", "ppq5_jitter", "ddp_jitter"]
    )
    assert all(isinstance(result[key], float) for key in result)


def test_extract_shimmer(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of shimmer descriptors."""
    result = extract_shimmer(resampled_mono_audio_sample, floor=75.0, ceiling=500.0)
    assert isinstance(result, dict)
    assert all(
        key in result
        for key in ["local_shimmer", "localDB_shimmer", "apq3_shimmer", "apq5_shimmer", "apq11_shimmer", "dda_shimmer"]
    )
    assert all(isinstance(result[key], float) for key in result)


def test_extract_praat_parselmouth_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of openSMILE features from audio."""
    # Extract Praat-Parselmouth features
    result = extract_praat_parselmouth_features_from_audios([resampled_mono_audio_sample])
    # Assert the result is a list of dictionaries, and check each dictionary
    assert isinstance(result, list)
    assert all(isinstance(features, dict) for features in result)


@pytest.mark.skip(reason="ppgs is now auto-provisioned via subprocess venv — no missing dep scenario")
def test_missing_ppg_dependency() -> None:
    """Test is obsolete — ppgs runs in isolated subprocess venv."""
    pass


def test_extract_ppgs_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of ppgs from audio."""
    result = extract_ppgs_from_audios([resampled_mono_audio_sample])
    # Assert the result is a list of tensors
    assert isinstance(result, list)
    assert all(isinstance(features, torch.Tensor) for features in result)


@pytest.mark.skip(reason="sparc runs in subprocess venv; missing-dep path cannot be tested")
def test_missing_sparc_dependency() -> None:
    """Test that a ModuleNotFoundError is raised when sparc is not installed."""
    with pytest.raises(ModuleNotFoundError):
        SparcFeatureExtractor.extract_sparc_features([Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)])


def test_extract_sparc_features(resampled_mono_audio_sample: Audio) -> None:
    """Test extraction of sparc from audio."""
    result = SparcFeatureExtractor.extract_sparc_features([resampled_mono_audio_sample])
    # Assert the result is a list of dicts
    assert isinstance(result, list)
    for features in result:
        assert isinstance(features, dict)
        assert all(
            key in features for key in ["ema", "loudness", "pitch", "periodicity", "pitch_stats", "spk_emb", "ft_len"]
        )


def test_extract_sparc_features_resample() -> None:
    """Test extraction of sparc from audio."""
    result = SparcFeatureExtractor.extract_sparc_features(
        [Audio(waveform=torch.rand(1, 44100), sampling_rate=44100)], resample=True
    )
    # Assert the result is a list of dicts
    assert isinstance(result, list)
    for features in result:
        assert isinstance(features, dict)
        assert all(
            key in features for key in ["ema", "loudness", "pitch", "periodicity", "pitch_stats", "spk_emb", "ft_len"]
        )


def test_extract_sparc_features_wrong_sample_rate() -> None:
    """Test that a ValueError is raised when sparc has wrong sampling rate."""
    with pytest.raises(ValueError):
        SparcFeatureExtractor.extract_sparc_features([Audio(waveform=torch.rand(1, 44100), sampling_rate=44100)])


# ---------------------------------------------------------------------------
# PPG phoneme duration analysis tests (use synthetic posteriorgrams)
# ---------------------------------------------------------------------------


def _make_synthetic_ppg(
    phoneme_pattern: list[int],
    num_phonemes: int = 40,
) -> torch.Tensor:
    """Build a synthetic PPG tensor with a known argmax sequence.

    Args:
        phoneme_pattern: Per-frame dominant phoneme index.
        num_phonemes: Total number of phoneme classes.

    Returns:
        Tensor of shape ``(1, num_phonemes, num_frames)``.
    """
    num_frames = len(phoneme_pattern)
    # Start with a uniform-ish floor and set the argmax phoneme high
    ppg = torch.full((num_phonemes, num_frames), 0.01)
    for frame_idx, phoneme_idx in enumerate(phoneme_pattern):
        ppg[phoneme_idx, frame_idx] = 0.95
    return ppg.unsqueeze(0)  # (1, phonemes, frames)


def test_to_frame_major_posteriorgram_3d() -> None:
    """Test to_frame_major_posteriorgram with a (1, phonemes, frames) input."""
    ppg = torch.rand(1, 40, 100)
    result = to_frame_major_posteriorgram(ppg)
    assert result.shape == (100, 40)


def test_to_frame_major_posteriorgram_2d() -> None:
    """Test to_frame_major_posteriorgram with a (phonemes, frames) input."""
    ppg = torch.rand(40, 100)
    result = to_frame_major_posteriorgram(ppg)
    assert result.shape == (100, 40)


def test_to_frame_major_posteriorgram_already_frame_major() -> None:
    """Test to_frame_major_posteriorgram when input is already (frames, phonemes)."""
    ppg = torch.rand(100, 40)
    result = to_frame_major_posteriorgram(ppg)
    assert result.shape == (100, 40)


def test_to_frame_major_posteriorgram_too_few_dims() -> None:
    """Test that a 1-D tensor raises ValueError."""
    ppg = torch.rand(40)
    with pytest.raises(ValueError, match="Expected at least a 2-D"):
        to_frame_major_posteriorgram(ppg)


def test_extract_ppg_segments_basic() -> None:
    """Test segment extraction with a simple known pattern."""
    # Pattern: 10 frames of phoneme 0, then 5 frames of phoneme 1
    pattern = [0] * 10 + [1] * 5
    ppg = _make_synthetic_ppg(pattern)
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    frame_major = to_frame_major_posteriorgram(ppg)
    segments = extract_ppg_segments(audio, frame_major)

    assert len(segments) == 2
    assert segments[0]["phoneme_index"] == 0
    assert segments[0]["frame_count"] == 10
    assert segments[1]["phoneme_index"] == 1
    assert segments[1]["frame_count"] == 5
    # Total frame count
    assert segments[0]["frame_count"] + segments[1]["frame_count"] == 15
    # Duration should sum to total audio duration
    total_dur = sum(s["duration_seconds"] for s in segments)
    expected_dur = 16000 / 16000  # 1 second
    assert abs(total_dur - expected_dur) < 1e-6


def test_extract_ppg_segments_single_phoneme() -> None:
    """Test segment extraction when only one phoneme is active."""
    pattern = [5] * 20
    ppg = _make_synthetic_ppg(pattern)
    audio = Audio(waveform=torch.rand(1, 32000), sampling_rate=16000)
    frame_major = to_frame_major_posteriorgram(ppg)
    segments = extract_ppg_segments(audio, frame_major)

    assert len(segments) == 1
    assert segments[0]["phoneme_index"] == 5
    assert segments[0]["frame_count"] == 20


def test_extract_ppg_segments_empty() -> None:
    """Test segment extraction with zero frames."""
    ppg = torch.rand(40, 0)
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    segments = extract_ppg_segments(audio, ppg)
    assert segments == []


def test_extract_mean_phoneme_durations_basic() -> None:
    """Test mean phoneme duration analysis with a known pattern."""
    # 20 frames of phoneme 0, 10 frames of phoneme 1, 20 frames of phoneme 0
    pattern = [0] * 20 + [1] * 10 + [0] * 20
    ppg = _make_synthetic_ppg(pattern)
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)

    result = extract_mean_phoneme_durations(audio, ppg)

    assert result["frame_count"] == 50
    assert result["phoneme_count"] == 40
    assert abs(result["analysis_duration_seconds"] - 1.0) < 1e-6

    # Phoneme "aa" (index 0) should appear twice, phoneme "ae" (index 1) once
    dur_map = {d["phoneme"]: d for d in result["phoneme_durations"]}
    assert "aa" in dur_map
    assert dur_map["aa"]["count"] == 2
    assert "ae" in dur_map
    assert dur_map["ae"]["count"] == 1

    # Mean segment duration: 3 segments, total 1 second
    assert abs(result["mean_segment_duration_seconds"] - 1.0 / 3) < 1e-6


def test_extract_mean_phoneme_durations_nan() -> None:
    """Test that a NaN posteriorgram returns an empty dict."""
    ppg = torch.tensor(float("nan"))
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    result = extract_mean_phoneme_durations(audio, ppg)
    assert result == {}


def test_extract_mean_phoneme_durations_empty() -> None:
    """Test that an empty posteriorgram returns an empty dict."""
    ppg = torch.empty(0)
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    result = extract_mean_phoneme_durations(audio, ppg)
    assert result == {}


def test_extract_mean_phoneme_durations_total_duration() -> None:
    """Verify that per-phoneme total durations sum to audio duration."""
    pattern = [2] * 10 + [5] * 15 + [2] * 5 + [10] * 20
    ppg = _make_synthetic_ppg(pattern)
    audio = Audio(waveform=torch.rand(1, 48000), sampling_rate=16000)  # 3 seconds

    result = extract_mean_phoneme_durations(audio, ppg)
    total_phoneme_dur = sum(d["total_duration_seconds"] for d in result["phoneme_durations"])
    assert abs(total_phoneme_dur - 3.0) < 1e-6


def test_plot_ppg_phoneme_timeline_returns_figure() -> None:
    """Test that plot_ppg_phoneme_timeline returns a matplotlib Figure."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # non-interactive backend

    pattern = [0] * 10 + [1] * 5 + [2] * 10
    ppg = _make_synthetic_ppg(pattern)
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)

    fig = plot_ppg_phoneme_timeline(audio, ppg, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_ppg_phoneme_timeline_nan_raises() -> None:
    """Test that NaN posteriorgram raises ValueError in plotting."""
    ppg = torch.tensor(float("nan"))
    audio = Audio(waveform=torch.rand(1, 16000), sampling_rate=16000)
    with pytest.raises(ValueError, match="empty or NaN"):
        plot_ppg_phoneme_timeline(audio, ppg, show=False)


def test_extract_features_from_audios(resampled_mono_audio_sample: Audio) -> None:
    """Simple test for extract_features_from_audios.

    This test verifies that given a valid list of audio samples,
    the extract_features_from_audios function returns a list of dictionaries (one per audio)
    containing non-empty feature data.
    """
    audios = [resampled_mono_audio_sample]
    features = extract_features_from_audios(
        audios=audios, opensmile=True, parselmouth=True, torchaudio=True, torchaudio_squim=True, ppgs=True, sparc=True
    )

    # Check that the output is a list and that it has one feature dict per audio.
    assert isinstance(features, list)
    assert len(features) == len(audios)

    # Check that each feature extraction result is a non-empty dictionary.
    for feat in features:
        assert isinstance(feat, dict)
        assert feat, "The feature dictionary should not be empty."
