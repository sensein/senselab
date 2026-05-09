"""Test audio classification APIs."""

from unittest.mock import patch

import pytest
import torch

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.api import classify_audios, scene_results_to_segments
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
    scores = result[0].get_scores()

    expected = {"arousal", "valence", "dominance"}
    assert set(labels) == expected, f"Expected {expected}, got {set(labels)}"

    # Scores should be bounded continuous values in [0, 1].
    for s in scores:
        assert 0.0 <= s <= 1.0, f"Continuous SER score out of [0,1]: {s}"


def test_speech_emotion_recognition_discrete(gpu_device: DeviceType) -> None:
    """Tests discrete speech emotion recognition (Tier 3: ~1.27GB model)."""
    audio_dataset = [Audio(filepath=MONO_AUDIO_PATH)]
    resampled = resample_audios(audio_dataset, 16000)

    result = classify_emotions_from_speech(
        resampled, HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"), device=gpu_device
    )
    emotions = result[0].get_labels()
    scores = result[0].get_scores()
    rav_emotions = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    for emotion in emotions:
        assert emotion in rav_emotions

    # Regression guard: ehcalabres ships its head as classifier.{dense,output}.* but
    # declares the standard Wav2Vec2ForSequenceClassification architecture, so the
    # default transformers loader randomly initializes the head and produces
    # ~uniform softmax (every score ≈ 1/8). After the fix the head is loaded from
    # the checkpoint and at least one class clears 0.2.
    assert max(scores) > 0.2, f"Discrete SER scores look randomly initialized: {dict(zip(emotions, scores))}"
    # Probabilities should be a softmax distribution summing to 1.
    assert abs(sum(scores) - 1.0) < 1e-3


def _make_dummy_result() -> AudioClassificationResult:
    """Create a dummy classification result for mocking."""
    return AudioClassificationResult(
        labels=["Speech", "Music", "Silence", "Dog", "Cat"],
        scores=[0.5, 0.3, 0.1, 0.05, 0.05],
    )


def test_classify_audios_windowed_basic() -> None:
    """Test windowed classification returns correct window count and structure."""
    audio = Audio(waveform=torch.randn(1, 32000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")

    with patch(
        "senselab.audio.tasks.classification.api._classify_whole",
        side_effect=lambda audios, **kw: [_make_dummy_result() for _ in audios],
    ):
        results = classify_audios([audio], model=model, win_length=1.0, hop_length=0.5, top_k=3)

    assert len(results) == 1
    windows = results[0]

    # 2s audio, 1s window, 0.5s hop → 4 windows (last is partial via window_generator)
    assert len(windows) == 4

    expected_times = [(0.0, 1.0), (0.5, 1.5), (1.0, 2.0), (1.5, 2.0)]
    for win, (exp_start, exp_end) in zip(windows, expected_times):
        assert win["start"] == pytest.approx(exp_start, abs=1e-6)
        assert win["end"] == pytest.approx(exp_end, abs=1e-6)
        assert len(win["labels"]) == 3
        assert len(win["scores"]) == 3
        # Provenance captured
        assert win["win_length"] == 1.0
        assert win["hop_length"] == 0.5


def test_classify_audios_windowed_short_audio() -> None:
    """Test that audio shorter than window produces a single window."""
    audio = Audio(waveform=torch.randn(1, 8000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")

    with patch(
        "senselab.audio.tasks.classification.api._classify_whole",
        side_effect=lambda audios, **kw: [_make_dummy_result() for _ in audios],
    ):
        results = classify_audios([audio], model=model, win_length=1.0)

    assert len(results) == 1
    windows = results[0]
    assert len(windows) == 1
    assert windows[0]["start"] == 0.0
    assert windows[0]["end"] == pytest.approx(0.5, abs=1e-6)
    # Default hop_length = win_length / 2
    assert windows[0]["hop_length"] == 0.5


def test_classify_audios_windowed_default_hop() -> None:
    """Test that hop_length defaults to win_length / 2."""
    audio = Audio(waveform=torch.randn(1, 48000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="MIT/ast-finetuned-audioset-10-10-0.4593")

    with patch(
        "senselab.audio.tasks.classification.api._classify_whole",
        side_effect=lambda audios, **kw: [_make_dummy_result() for _ in audios],
    ):
        results = classify_audios([audio], model=model, win_length=2.0)

    windows = results[0]
    # 3s audio, 2s window, 1s hop → 3 windows (last partial via window_generator)
    assert len(windows) == 3
    assert windows[0]["hop_length"] == 1.0


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
