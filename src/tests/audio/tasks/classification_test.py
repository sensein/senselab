"""Test audio classification APIs."""

from typing import cast
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


@pytest.fixture(autouse=True)
def _reset_ser_module_caches() -> None:
    """Reset per-process SER caches between tests.

    Several tests monkeypatch ``transformers.AutoConfig.from_pretrained`` to return
    a fake config for ``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim``. The
    module-level ``_config_memo`` introduced in PR #515 would otherwise hand back the
    cached fake from a prior test, silently bypassing the monkeypatch in the next
    one. Same risk for ``_wav2vec2_emotion_models`` (the constructed model cache),
    which existing tests already reset by hand — centralising that here keeps the
    invariant in one place.
    """
    from senselab.audio.tasks.classification.speech_emotion_recognition import api as ser_api

    ser_api._config_memo.clear()
    ser_api._wav2vec2_emotion_models.clear()


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


@pytest.mark.parametrize(
    ("loading_info", "expected_substr"),
    [
        ({"missing_keys": {"classifier.dense.weight"}, "mismatched_keys": set()}, "missing"),
        (
            {"missing_keys": set(), "mismatched_keys": {("classifier.out_proj.weight", (3, 1024), (5, 1024))}},
            "shape-mismatched",
        ),
    ],
)
def test_wav2vec2_speech_cls_ser_raises_on_random_head(
    loading_info: dict, expected_substr: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 1 guard: silent random/mismatched-shape head loads must raise.

    Mocks ``EmotionModel.from_pretrained`` to return a model with a deliberately
    "broken" load result. The classifier shapes alone are correct, so the
    shape-sanity check passes and we exercise only the missing/mismatched-keys
    branch of the guard.
    """
    from unittest.mock import MagicMock

    from senselab.audio.tasks.classification.speech_emotion_recognition import api as ser_api

    fake_model = MagicMock()
    # The shape sanity-check expects out_features matching config.num_labels.
    fake_model.classifier.out_proj.out_features = 3
    fake_model.to.return_value = fake_model

    fake_config = MagicMock()
    fake_config.num_labels = 3
    fake_config.id2label = {0: "arousal", 1: "valence", 2: "dominance"}

    monkeypatch.setattr(
        ser_api,
        "_make_emotion_model_class",
        lambda model_type, head: MagicMock(from_pretrained=MagicMock(return_value=(fake_model, loading_info))),
    )
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", MagicMock(return_value=fake_config))
    monkeypatch.setattr(
        "transformers.Wav2Vec2FeatureExtractor.from_pretrained",
        MagicMock(return_value=MagicMock(sampling_rate=16000)),
    )

    audios = [Audio(waveform=torch.randn(1, 16000), sampling_rate=16000)]
    with pytest.raises(RuntimeError, match=expected_substr):
        ser_api._classify_wav2vec2_speech_cls_ser(
            audios,
            HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
            device=DeviceType.CPU,
        )


def test_wav2vec2_speech_cls_ser_raises_on_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase 1 guard: classifier loaded but final-layer out_features != num_labels."""
    from unittest.mock import MagicMock

    from senselab.audio.tasks.classification.speech_emotion_recognition import api as ser_api

    fake_model = MagicMock()
    fake_model.classifier.out_proj.out_features = 5  # config says 3
    fake_model.to.return_value = fake_model

    fake_config = MagicMock()
    fake_config.num_labels = 3
    fake_config.id2label = {0: "arousal", 1: "valence", 2: "dominance"}

    monkeypatch.setattr(
        ser_api,
        "_make_emotion_model_class",
        lambda model_type, head: MagicMock(
            from_pretrained=MagicMock(return_value=(fake_model, {"missing_keys": set(), "mismatched_keys": set()}))
        ),
    )
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", MagicMock(return_value=fake_config))
    monkeypatch.setattr(
        "transformers.Wav2Vec2FeatureExtractor.from_pretrained",
        MagicMock(return_value=MagicMock(sampling_rate=16000)),
    )

    audios = [Audio(waveform=torch.randn(1, 16000), sampling_rate=16000)]
    with pytest.raises(RuntimeError, match="out_features=5"):
        ser_api._classify_wav2vec2_speech_cls_ser(
            audios,
            HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
            device=DeviceType.CPU,
        )


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


@pytest.mark.parametrize(
    "loading_info,kind",
    [
        ({"missing_keys": {"classifier.weight"}, "mismatched_keys": set()}, "missing"),
        (
            {"missing_keys": set(), "mismatched_keys": {("classifier.weight", (8, 1024), (3, 1024))}},
            "mismatched",
        ),
        (
            {"missing_keys": {"head.weight"}, "mismatched_keys": set()},
            "missing-head",
        ),
        (
            {"missing_keys": {"score.weight"}, "mismatched_keys": set()},
            "missing-score",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_check_head_loaded_cleanly_raises_strict_by_default(
    monkeypatch: pytest.MonkeyPatch, loading_info: dict, kind: str
) -> None:
    """Phase 2 default: any suspect head-prefix miss/mismatch raises RuntimeError."""
    from senselab.audio.tasks.classification.huggingface import _check_head_loaded_cleanly

    monkeypatch.delenv("SENSELAB_STRICT_HEAD_LOAD", raising=False)
    # ``_check_head_loaded_cleanly`` only reads ``path_or_uri`` / ``revision`` for the
    # error message; using a SimpleNamespace avoids HFModel's hub-validation roundtrip.
    from types import SimpleNamespace

    model: HFModel = cast(HFModel, SimpleNamespace(path_or_uri="org/example", revision="main"))
    with pytest.raises(RuntimeError, match="suspect weights"):
        _check_head_loaded_cleanly(loading_info, model)


def test_check_head_loaded_cleanly_warns_when_lax(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """SENSELAB_STRICT_HEAD_LOAD=0 demotes the failure to a warning."""
    import logging

    from senselab.audio.tasks.classification.huggingface import _check_head_loaded_cleanly

    monkeypatch.setenv("SENSELAB_STRICT_HEAD_LOAD", "0")
    # ``_check_head_loaded_cleanly`` only reads ``path_or_uri`` / ``revision`` for the
    # error message; using a SimpleNamespace avoids HFModel's hub-validation roundtrip.
    from types import SimpleNamespace

    model: HFModel = cast(HFModel, SimpleNamespace(path_or_uri="org/example", revision="main"))
    with caplog.at_level(logging.WARNING, logger="senselab"):
        _check_head_loaded_cleanly({"missing_keys": {"classifier.weight"}, "mismatched_keys": set()}, model)
    assert any("suspect weights" in rec.message for rec in caplog.records)


def test_check_head_loaded_cleanly_silent_on_clean_load() -> None:
    """No suspect keys → returns silently, no log, no raise."""
    # ``_check_head_loaded_cleanly`` only reads ``path_or_uri`` / ``revision`` for the
    # error message; using a SimpleNamespace avoids HFModel's hub-validation roundtrip.
    from types import SimpleNamespace

    from senselab.audio.tasks.classification.huggingface import _check_head_loaded_cleanly

    model: HFModel = cast(HFModel, SimpleNamespace(path_or_uri="org/example", revision="main"))
    # Encoder weight misses (e.g. masked_spec_embed) and unrelated keys must not trip the guard.
    _check_head_loaded_cleanly(
        {"missing_keys": {"wav2vec2.masked_spec_embed", "encoder.something"}, "mismatched_keys": set()},
        model,
    )


@pytest.mark.parametrize(
    ("model_type", "expected_base_name", "expected_encoder_name", "expected_attr"),
    [
        ("wav2vec2", "Wav2Vec2PreTrainedModel", "Wav2Vec2Model", "wav2vec2"),
        ("hubert", "HubertPreTrainedModel", "HubertModel", "hubert"),
        ("wavlm", "WavLMPreTrainedModel", "WavLMModel", "wavlm"),
    ],
)
def test_resolve_base_returns_real_transformers_classes(
    model_type: str, expected_base_name: str, expected_encoder_name: str, expected_attr: str
) -> None:
    """Smoke test: every ``_BASE_REGISTRY`` entry must resolve to real transformers classes.

    A typo in the registry (e.g. ``"Wav2Vec2BasePreTrainedModel"`` instead of
    ``"Wav2Vec2PreTrainedModel"``) would silently route through the standard pipeline
    fallback and silently random-initialize heads. This test catches such typos at
    test-collection time without downloading any model.
    """
    from senselab.audio.tasks.classification.speech_emotion_recognition.api import _resolve_base

    resolved = _resolve_base(model_type)
    assert resolved is not None, f"_BASE_REGISTRY['{model_type}'] failed to resolve"
    base_cls, encoder_cls, attr_name = resolved
    assert base_cls.__name__ == expected_base_name
    assert encoder_cls.__name__ == expected_encoder_name
    assert attr_name == expected_attr


def test_resolve_base_unknown_model_type_returns_none() -> None:
    """Unknown ``model_type`` → ``None`` so callers can dispatch to the standard pipeline."""
    from senselab.audio.tasks.classification.speech_emotion_recognition.api import _resolve_base

    assert _resolve_base("not-a-real-family") is None


def test_load_config_cached_round_trips_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_load_config_cached`` must hit ``AutoConfig.from_pretrained`` at most once per (path, revision).

    Mocks ``AutoConfig.from_pretrained`` with a call counter, invokes the cache four
    times with the same model handle, and asserts the underlying call happened once.
    """
    from unittest.mock import MagicMock

    from senselab.audio.tasks.classification.speech_emotion_recognition import api as ser_api

    # Autouse fixture should have cleared the memo before us — sanity check.
    assert ser_api._config_memo == {}, "_config_memo leaked from a prior test"
    fake_config = MagicMock(model_type="wav2vec2", architectures=["X"], id2label={0: "a"})
    call_counter = MagicMock(return_value=fake_config)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", call_counter)

    model: HFModel = HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", revision="main")
    for _ in range(4):
        typed, raw = ser_api._load_config_cached(model)
        assert typed is fake_config
        assert raw is None
    assert call_counter.call_count == 1, f"Expected 1 AutoConfig call, got {call_counter.call_count}"


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
