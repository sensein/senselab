"""Test audio classification APIs."""

from unittest.mock import patch

import pytest
import torch

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.api import classify_audios_in_windows, scene_results_to_segments
from senselab.audio.tasks.classification.speech_emotion_recognition import (
    classify_emotions_from_speech,
)
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, HFModel
from tests.audio.conftest import MONO_AUDIO_PATH


def test_speech_emotion_recognition_continuous(cpu_cuda_device: DeviceType) -> None:
    """Tests continuous speech emotion recognition (Tier 2: ~661MB model)."""
    audio_dataset = [Audio(filepath=MONO_AUDIO_PATH)]
    resampled = resample_audios(audio_dataset, 16000)

    result = classify_emotions_from_speech(
        resampled, HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"), device=cpu_cuda_device
    )
    labels = result[0].get_labels()

    for label in labels:
        assert label in ["arousal", "valence", "dominance"], (
            "No emotion here but rather is one of arousal, valence, or dominance"
        )


def test_speech_emotion_recognition_discrete(gpu_device: DeviceType) -> None:
    """Tests discrete speech emotion recognition (Tier 3: ~1.27GB model)."""
    audio_dataset = [Audio(filepath=MONO_AUDIO_PATH)]
    resampled = resample_audios(audio_dataset, 16000)

    result = classify_emotions_from_speech(
        resampled, HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"), device=gpu_device
    )
    emotions = result[0].get_labels()
    rav_emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    for emotion in emotions:
        assert emotion in rav_emotions


def _make_dummy_result() -> AudioClassificationResult:
    """Create a dummy classification result for mocking."""
    return AudioClassificationResult(
        labels=["Speech", "Music", "Silence", "Dog", "Cat"],
        scores=[0.5, 0.3, 0.1, 0.05, 0.05],
    )


def test_classify_audios_in_windows_basic() -> None:
    """Test windowed classification returns correct window count and structure."""
    # 2 seconds of mono audio at 16 kHz
    audio = Audio(waveform=torch.randn(1, 32000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")

    with patch(
        "senselab.audio.tasks.classification.api.classify_audios",
        side_effect=lambda audios, **kw: [_make_dummy_result() for _ in audios],
    ):
        results = classify_audios_in_windows([audio], model=model, window_size=1.0, hop_size=0.5, top_k=3)

    # Outer list: one entry per audio
    assert len(results) == 1
    windows = results[0]

    # With 2s audio, 1s window, 0.5s hop: windows at 0-1, 0.5-1.5, 1.0-2.0 = 3 windows
    assert len(windows) == 3

    # Verify timestamps
    expected_times = [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0)]
    for win, (exp_start, exp_end) in zip(windows, expected_times):
        assert win["start"] == pytest.approx(exp_start, abs=1e-6)
        assert win["end"] == pytest.approx(exp_end, abs=1e-6)
        assert "labels" in win
        assert "scores" in win
        assert len(win["labels"]) == 3  # top_k=3
        assert len(win["scores"]) == 3


def test_classify_audios_in_windows_short_audio() -> None:
    """Test that audio shorter than window produces a single window."""
    # 0.5 seconds of mono audio at 16 kHz (shorter than 1s window)
    audio = Audio(waveform=torch.randn(1, 8000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")

    with patch(
        "senselab.audio.tasks.classification.api.classify_audios",
        side_effect=lambda audios, **kw: [_make_dummy_result() for _ in audios],
    ):
        results = classify_audios_in_windows([audio], model=model, window_size=1.0, hop_size=0.5, top_k=5)

    assert len(results) == 1
    windows = results[0]

    # Short audio -> single window covering the full duration
    assert len(windows) == 1
    assert windows[0]["start"] == 0.0
    assert windows[0]["end"] == pytest.approx(0.5, abs=1e-6)
    assert len(windows[0]["labels"]) == 5
    assert len(windows[0]["scores"]) == 5


def test_scene_results_to_segments() -> None:
    """Test conversion of windowed results to segment format."""
    window_results = [
        {"start": 0.0, "end": 1.0, "labels": ["Speech", "Music"], "scores": [0.8, 0.2]},
        {"start": 0.5, "end": 1.5, "labels": ["Music", "Speech"], "scores": [0.6, 0.4]},
    ]
    segments = scene_results_to_segments(window_results)

    assert len(segments) == 2
    assert segments[0] == {"label": "Speech", "start": 0.0, "end": 1.0}
    assert segments[1] == {"label": "Music", "start": 0.5, "end": 1.5}
