"""Tests for the audio-understanding task.

No weights are downloaded: the Audio Flamingo model is ~8B parameters and under a
noncommercial licence, so every test here exercises the routing, input validation and
audio normalization around the load rather than the load itself.
"""

from typing import Dict, List

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature

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


class _FakeProcessor:
    """Records how ``apply_chat_template`` was called and returns a real ``BatchFeature``.

    Returning the genuine container matters: the dtype handling under test lives in
    ``BatchFeature.to``, so a hand-rolled stand-in would assert nothing about it.
    """

    def __init__(self) -> None:
        self.template_calls: List[Dict[str, object]] = []
        self.decode_calls: List[Dict[str, object]] = []

    def apply_chat_template(self, conversation: object, **kwargs: object) -> BatchFeature:
        """Record the keyword arguments and return a mixed-dtype batch."""
        self.template_calls.append(kwargs)
        return BatchFeature(
            data={
                "input_features": torch.zeros((1, 8), dtype=torch.float32),
                "input_ids": torch.ones((1, 3), dtype=torch.int64),
                "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            }
        )

    def decode(self, sequence: object, **kwargs: object) -> str:
        """Record the decode keywords and return one placeholder generation.

        ``decode``, not ``batch_decode``: ``strip_prefix`` is only exposed on the
        per-sequence call, so decoding the batch would silently drop it.
        """
        self.decode_calls.append(kwargs)
        return "a caption"


class _FakeModel:
    """Records the dtype of every tensor handed to ``generate``."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.seen: Dict[str, torch.dtype] = {}

    def generate(self, max_new_tokens: int = 0, **inputs: object) -> torch.Tensor:
        """Capture input dtypes and return a token sequence longer than the prompt."""
        self.seen = {k: v.dtype for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        return torch.ones((1, 5), dtype=torch.int64)


@pytest.fixture()
def _wired(monkeypatch: pytest.MonkeyPatch) -> tuple[_FakeProcessor, _FakeModel]:
    """Seed the weight cache so a call runs end to end without loading anything."""
    processor, mdl = _FakeProcessor(), _FakeModel()
    cache_key = "nvidia/audio-flamingo-3-hf@cpu@sdpa@think=False"
    monkeypatch.setattr(AudioFlamingoUnderstanding, "_cache", {cache_key: (processor, mdl)})
    return processor, mdl


class TestProcessorOutputIsCastToTheModelDtype:
    """The processor emits float32; bf16 weights reject it.

    On CUDA this surfaced as ``RuntimeError: Input type (float) and bias type
    (c10::BFloat16) should be the same`` and made the backend unusable on GPU.
    """

    def test_floating_inputs_are_cast_to_the_model_dtype(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """``input_features`` reaches the model in the weights' own dtype."""
        _, mdl = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert mdl.seen["input_features"] == torch.bfloat16

    @pytest.mark.parametrize("key", ["input_ids", "attention_mask"])
    def test_integer_inputs_keep_their_dtype(self, _wired: tuple[_FakeProcessor, _FakeModel], key: str) -> None:
        """Index tensors stay integral — casting them to bf16 would break embedding lookup."""
        _, mdl = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert mdl.seen[key] == torch.int64

    def test_a_float32_model_leaves_features_alone(
        self, monkeypatch: pytest.MonkeyPatch, _wired: tuple[_FakeProcessor, _FakeModel]
    ) -> None:
        """The cast follows the weights rather than hard-coding bf16."""
        processor, _ = _wired
        mdl = _FakeModel(dtype=torch.float32)
        monkeypatch.setattr(
            AudioFlamingoUnderstanding, "_cache", {"nvidia/audio-flamingo-3-hf@cpu@sdpa@think=False": (processor, mdl)}
        )
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert mdl.seen["input_features"] == torch.float32


class TestTheSamplingRateReachesTheProcessor:
    """transformers routes processor arguments through ``processor_kwargs``.

    Passing them as bare ``**kwargs`` warns and replaces the whole ``processor_kwargs``
    dict, so anything else intended for the processor would be discarded.
    """

    def test_the_sampling_rate_is_passed_in_processor_kwargs(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """The rate arrives in the dict transformers reads it from."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert processor.template_calls[0]["processor_kwargs"] == {"sampling_rate": TARGET_SAMPLING_RATE}

    def test_the_sampling_rate_is_not_passed_as_a_bare_kwarg(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """It must not also ride in ``**kwargs``, which triggers the clobbering path."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert "sampling_rate" not in processor.template_calls[0]


class TestTheTranscriptionPrefixCanBeStripped:
    """``strip_prefix`` lives on ``decode`` alone, so ``batch_decode`` would swallow it."""

    def test_strip_prefix_reaches_the_decoder(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """The flag is forwarded, so the canned wrapper can actually be removed."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_audio()], prompt="transcribe", device=DeviceType.CPU, strip_prefix=True
        )
        assert processor.decode_calls[0]["strip_prefix"] is True

    def test_it_defaults_to_leaving_the_answer_untouched(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Captioning answers carry no prefix, so stripping is opt-in."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert processor.decode_calls[0]["strip_prefix"] is False


class TestTheThinkAdapterIsASeparateVariant:
    """AF-Think is a PEFT adapter in the repo's ``think`` subfolder, not the base checkpoint.

    Asking the base weights to follow a reasoning prompt is a variant mismatch, so the
    two must not share a cache entry.
    """

    def test_think_and_base_do_not_share_a_cache_entry(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """``think=True`` misses the base entry rather than silently reusing it."""
        calls: List[tuple[str, str]] = []

        def _fake_load(model_name: str, revision: str, **kwargs: object) -> tuple[object, object]:
            calls.append((model_name, revision))
            return _FakeProcessor(), _FakeModel()

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(AudioFlamingoUnderstanding, "_load_think", _fake_load)
            AudioFlamingoUnderstanding.describe_with_audio_flamingo(
                [_audio()], prompt="describe", device=DeviceType.CPU, think=True
            )
        assert calls == [("nvidia/audio-flamingo-3-hf", "main")]

    def test_the_base_path_never_loads_the_adapter(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Without ``think`` the adapter loader is not reached at all."""

        def _boom(*args: object, **kwargs: object) -> tuple[object, object]:
            raise AssertionError("_load_think must not run when think is False")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(AudioFlamingoUnderstanding, "_load_think", _boom)
            AudioFlamingoUnderstanding.describe_with_audio_flamingo(
                [_audio()], prompt="describe", device=DeviceType.CPU
            )


class TestTheApiRoutesOnlySupportedModels:
    """An unsupported model id should fail loudly rather than loading the default."""

    def test_an_unknown_model_is_refused(self) -> None:
        """A non-Flamingo model id raises NotImplementedError naming what is supported."""
        with pytest.raises(NotImplementedError, match="audio-flamingo"):
            describe_audios([_audio()], prompt="describe", model=HFModel(path_or_uri="openai/whisper-tiny"))

    def test_think_and_strip_prefix_reach_the_backend(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """The public wrapper forwards both flags rather than dropping them."""
        processor, _ = _wired
        seen: Dict[str, object] = {}

        def _capture(**kwargs: object) -> List[str]:
            seen.update(kwargs)
            return ["ok"]

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(AudioFlamingoUnderstanding, "describe_with_audio_flamingo", _capture)
            describe_audios([_audio()], prompt="describe", think=True, strip_prefix=True)
        assert seen["think"] is True
        assert seen["strip_prefix"] is True
