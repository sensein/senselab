"""This script contains unit tests for the features extraction tasks."""
from typing import List

import pytest
import torch

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    get_audios_durations,
    get_audios_f0_descriptors,
    get_audios_harmonicity_descriptors,
    get_audios_jitter_descriptors,
    get_audios_shimmer_descriptors,
)
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_filter_bank_from_audios,
    extract_mel_spectrogram_from_audios,
    extract_mfcc_from_audios,
    extract_pitch_from_audios,
    extract_spectrogram_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
    extract_subjective_quality_features_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios, select_channel_from_audios


@pytest.fixture
def sample_audios() -> List[Audio]:
    """Sample audios fixture."""
    audio = Audio.from_filepath("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
    return [audio]

def test_extract_spectrogram_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of spectrogram from audio."""
    result = extract_spectrogram_from_audios(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert all('spectrogram' in spec for spec in result)
    assert all(isinstance(spec['spectrogram'], torch.Tensor) for spec in result)
    # Spectrogram shape is (audios, freq, time)
    assert all(spec['spectrogram'].dim() == 3 for spec in result)  
    assert all(spec['spectrogram'].shape[1] == 201 for spec in result)

def test_extract_mel_spectrogram_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of mel spectrogram from audio."""
    result = extract_mel_spectrogram_from_audios(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(spec, dict) for spec in result)
    assert all('mel_spectrogram' in spec for spec in result)
    assert all(isinstance(spec['mel_spectrogram'], torch.Tensor) for spec in result)
    # Mel spectrogram shape is (audios, n_mels, time)
    assert all(spec['mel_spectrogram'].dim() == 3 for spec in result)  
    assert all(spec['mel_spectrogram'].shape[1] == 128 for spec in result)

def test_extract_mfcc_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of MFCC from audio."""
    result = extract_mfcc_from_audios(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(mfcc, dict) for mfcc in result)
    assert all('mfcc' in mfcc for mfcc in result)
    assert all(isinstance(mfcc['mfcc'], torch.Tensor) for mfcc in result)
    # MFCC shape is (audios, n_mfcc, time)
    assert all(mfcc['mfcc'].dim() == 3 for mfcc in result)
    assert all(mfcc['mfcc'].shape[1] == 40 for mfcc in result)

def test_extract_mel_filter_bank(sample_audios: List[Audio]) -> None:
    """Test extraction of mel filter bank from audio."""
    result = extract_mel_filter_bank_from_audios(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(mel_fb, dict) for mel_fb in result)
    assert all('mel_filter_bank' in mel_fb for mel_fb in result)
    assert all(isinstance(mel_fb['mel_filter_bank'], torch.Tensor) for mel_fb in result)
    # Mel filter bank shape is (audios, n_mels, time)
    assert all(mel_fb['mel_filter_bank'].dim() == 3 for mel_fb in result)
    assert all(mel_fb['mel_filter_bank'].shape[1] == 128 for mel_fb in result)

def test_extract_pitch_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of pitch from audio."""
    result = extract_pitch_from_audios(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(pitch, dict) for pitch in result)
    assert all('pitch' in pitch for pitch in result)
    assert all(isinstance(pitch['pitch'], torch.Tensor) for pitch in result)
    # Pitch shape is (audios, time)
    assert all(pitch['pitch'].dim() == 2 for pitch in result)  
    
def test_get_audios_durations(sample_audios: List[Audio]) -> None:
    """Test extraction of audio durations."""
    result = get_audios_durations(sample_audios)
    assert isinstance(result, list)
    assert all(isinstance(duration, dict) for duration in result)
    assert all('duration' in duration for duration in result)
    assert all(isinstance(duration['duration'], float) for duration in result)

def test_get_audios_f0_descriptors(sample_audios: List[Audio]) -> None:
    """Test extraction of fundamental frequency descriptors from audio."""
    result = get_audios_f0_descriptors(sample_audios, f0min=75.0, f0max=500.0)
    assert isinstance(result, list)
    assert all(isinstance(f0, dict) for f0 in result)
    assert all('f0_mean_Hertz' in f0 for f0 in result)
    assert all('f0_std_dev_Hertz' in f0 for f0 in result)
    assert all(isinstance(f0['f0_mean_Hertz'], float) for f0 in result)
    assert all(isinstance(f0['f0_std_dev_Hertz'], float) for f0 in result)

def test_get_audios_harmonicity_descriptors(sample_audios: List[Audio]) -> None:
    """Test extraction of harmonicity descriptors from audio."""
    result = get_audios_harmonicity_descriptors(sample_audios, f0min=75.0)
    assert isinstance(result, list)
    assert all(isinstance(harmonicity, dict) for harmonicity in result)
    assert all('harmonicity_mean' in harmonicity for harmonicity in result)
    assert all('harmonicity_std_dev' in harmonicity for harmonicity in result)
    assert all(isinstance(harmonicity['harmonicity_mean'], float) for harmonicity in result)
    assert all(isinstance(harmonicity['harmonicity_std_dev'], float) for harmonicity in result)

def test_get_audios_jitter_descriptors(sample_audios: List[Audio]) -> None:
    """Test extraction of jitter descriptors from audio."""
    result = get_audios_jitter_descriptors(sample_audios, f0min=75.0, f0max=500.0)
    assert isinstance(result, list)
    assert all(isinstance(jitter, dict) for jitter in result)
    assert all('local_jitter' in jitter for jitter in result)
    assert all('localabsolute_jitter' in jitter for jitter in result)
    assert all('rap_jitter' in jitter for jitter in result)
    assert all('ppq5_jitter' in jitter for jitter in result)
    assert all('ddp_jitter' in jitter for jitter in result)
    assert all(isinstance(jitter['local_jitter'], float) for jitter in result)
    assert all(isinstance(jitter['localabsolute_jitter'], float) for jitter in result)
    assert all(isinstance(jitter['rap_jitter'], float) for jitter in result)
    assert all(isinstance(jitter['ppq5_jitter'], float) for jitter in result)
    assert all(isinstance(jitter['ddp_jitter'], float) for jitter in result)

def test_get_audios_shimmer_descriptors(sample_audios: List[Audio]) -> None:
    """Test extraction of shimmer descriptors from audio."""
    result = get_audios_shimmer_descriptors(sample_audios, f0min=75.0, f0max=500.0)
    assert isinstance(result, list)
    assert all(isinstance(shimmer, dict) for shimmer in result)
    assert all('local_shimmer' in shimmer for shimmer in result)
    assert all('localDB_shimmer' in shimmer for shimmer in result)
    assert all('apq3_shimmer' in shimmer for shimmer in result)
    assert all('apq5_shimmer' in shimmer for shimmer in result)
    assert all('apq11_shimmer' in shimmer for shimmer in result)
    assert all('dda_shimmer' in shimmer for shimmer in result)
    assert all(isinstance(shimmer['local_shimmer'], float) for shimmer in result)
    assert all(isinstance(shimmer['localDB_shimmer'], float) for shimmer in result)
    assert all(isinstance(shimmer['apq3_shimmer'], float) for shimmer in result)
    assert all(isinstance(shimmer['apq5_shimmer'], float) for shimmer in result)
    assert all(isinstance(shimmer['apq11_shimmer'], float) for shimmer in result)
    assert all(isinstance(shimmer['dda_shimmer'], float) for shimmer in result)

def test_extract_objective_quality_features_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of objective quality features from audio."""
    # Ensure the sample audios are in the required format for the test
    sample_audios = select_channel_from_audios(sample_audios, 0) # Convert to mono
    sample_audios = resample_audios(sample_audios, 16000) # Set to 16000 Hz

    result = extract_objective_quality_features_from_audios(sample_audios)
    assert isinstance(result, dict)
    assert 'stoi' in result
    assert 'pesq' in result
    assert 'si_sdr' in result
    assert all(isinstance(feature, float) for feature in result['stoi'])
    assert all(isinstance(feature, float) for feature in result['pesq'])
    assert all(isinstance(feature, float) for feature in result['si_sdr'])

def test_extract_objective_quality_features_from_audios_invalid_audio(sample_audios: List[Audio]) -> None:
    """Test extraction of objective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_objective_quality_features_from_audios(sample_audios)

def test_extract_subjective_quality_features_from_audios(sample_audios: List[Audio]) -> None:
    """Test extraction of subjective quality features from audio."""
    # Ensure the sample audios are in the required format for the test
    sample_audios = select_channel_from_audios(sample_audios, 0) # Convert to mono
    sample_audios = resample_audios(sample_audios, 16000) # Set to 16000 Hz

    result = extract_subjective_quality_features_from_audios(sample_audios, sample_audios)
    assert isinstance(result, dict)
    assert 'mos' in result
    assert all(isinstance(feature, float) for feature in result['mos'])

def test_extract_subjective_quality_features_invalid_audio(sample_audios: List[Audio]) -> None:
    """Test extraction of subjective quality features from invalid audio."""
    with pytest.raises(ValueError, match="Only 16000 Hz sampling rate is supported by Torchaudio-Squim model."):
        extract_subjective_quality_features_from_audios(sample_audios, sample_audios)
