"""Tests for the audio-understanding task.

No weights are downloaded: the Audio Flamingo model is ~8B parameters and under a
noncommercial licence, so every test here exercises the routing, input validation and
audio normalization around the load rather than the load itself.
"""

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.audio_understanding import describe_audios
from senselab.audio.tasks.audio_understanding.audio_flamingo import (
    MAX_AUDIO_SECONDS,
    TARGET_SAMPLING_RATE,
    AudioFlamingoUnderstanding,
)
from senselab.utils.data_structures import DeviceType, HFModel


def _audio(seconds: float = 1.0, sampling_rate: int = TARGET_SAMPLING_RATE, channels: int = 1) -> Audio:
    """Build a silent Audio of the requested shape."""
    return Audio(
        waveform=torch.zeros((channels, int(seconds * sampling_rate))),
        sampling_rate=sampling_rate,
    )


class TestThePromptIsRequired:
    """A prompt is what the model answers, so an empty one is a caller error."""

    @pytest.mark.parametrize("prompt", ["", "   "])
    def test_an_empty_prompt_is_rejected(self, prompt: str) -> None:
        """An empty or whitespace-only prompt raises rather than reaching the model."""
        with pytest.raises(ValueError, match="non-empty"):
            AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt=prompt)

    def test_no_audios_returns_no_results_without_loading(self) -> None:
        """An empty batch short-circuits before any weight load."""
        assert AudioFlamingoUnderstanding.describe_with_audio_flamingo([], prompt="describe") == []


class TestAudioIsNormalisedBeforeTheModelSeesIt:
    """The model wants 16 kHz mono; callers are not required to supply it."""

    def test_a_resampled_clip_reaches_the_target_rate(self) -> None:
        """A 44.1 kHz clip is resampled to 16 kHz."""
        prepared = AudioFlamingoUnderstanding._prepare(_audio(sampling_rate=44100))
        assert prepared.sampling_rate == TARGET_SAMPLING_RATE

    def test_a_stereo_clip_is_downmixed(self) -> None:
        """A two-channel clip is reduced to one."""
        prepared = AudioFlamingoUnderstanding._prepare(_audio(channels=2))
        assert prepared.waveform.shape[0] == 1

    def test_a_conforming_clip_is_left_alone(self) -> None:
        """16 kHz mono passes through unchanged."""
        original = _audio()
        prepared = AudioFlamingoUnderstanding._prepare(original)
        assert prepared.sampling_rate == TARGET_SAMPLING_RATE
        assert prepared.waveform.shape[0] == 1


class TestTheLengthLimitIsEnforcedLocally:
    """The card caps input at 10 minutes; exceeding it should fail before the model runs."""

    def test_an_over_long_clip_raises(self) -> None:
        """A clip past the documented ceiling raises, naming the limit."""
        with pytest.raises(ValueError, match="at most 600s"):
            AudioFlamingoUnderstanding._prepare(_audio(seconds=MAX_AUDIO_SECONDS + 1))

    def test_a_clip_at_the_limit_is_accepted(self) -> None:
        """Exactly the limit is allowed — the check is strict, not off-by-one."""
        assert AudioFlamingoUnderstanding._prepare(_audio(seconds=MAX_AUDIO_SECONDS)) is not None


class TestAttentionSelectionFallsBackSafely:
    """flash-attention is optional, so its absence must not break the load."""

    def test_cpu_always_uses_sdpa(self) -> None:
        """flash-attention is CUDA-only, so CPU gets sdpa regardless of what is installed."""
        assert AudioFlamingoUnderstanding._attention_implementation(DeviceType.CPU) == "sdpa"

    def test_cuda_without_flash_attention_uses_sdpa(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A CUDA host lacking flash-attn falls back rather than raising at load time."""
        import builtins

        real_import = builtins.__import__

        def _no_flash(name: str, *args: object, **kwargs: object) -> object:
            if name == "flash_attn":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _no_flash)
        assert AudioFlamingoUnderstanding._attention_implementation(DeviceType.CUDA) == "sdpa"


class TestTheApiRoutesOnlySupportedModels:
    """An unsupported model id should fail loudly rather than loading the default."""

    def test_an_unknown_model_is_refused(self) -> None:
        """A non-Flamingo model id raises NotImplementedError naming what is supported."""
        with pytest.raises(NotImplementedError, match="audio-flamingo"):
            describe_audios([_audio()], prompt="describe", model=HFModel(path_or_uri="openai/whisper-tiny"))
