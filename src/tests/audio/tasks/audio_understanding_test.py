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


def _marked_audio(marker: int, seconds: float = 1.0) -> Audio:
    """Build a 16 kHz mono clip whose every sample equals ``marker``.

    The marker survives the processor and generation in the fakes below, so a test can
    tell which answer belongs to which input rather than trusting positional luck.
    """
    return Audio(
        waveform=torch.full((1, int(seconds * TARGET_SAMPLING_RATE)), float(marker)),
        sampling_rate=TARGET_SAMPLING_RATE,
    )


class _FakeProcessor:
    """Records how ``apply_chat_template`` was called and returns a real ``BatchFeature``.

    Returning the genuine container matters: the dtype handling under test lives in
    ``BatchFeature.to``, so a hand-rolled stand-in would assert nothing about it.

    Each conversation's audio marker is carried into ``input_ids`` so that ordering is
    observable end to end.
    """

    def __init__(self) -> None:
        self.template_calls: List[Dict[str, object]] = []
        self.decode_calls: List[Dict[str, object]] = []
        self.batches: List[List[int]] = []

    def apply_chat_template(self, conversation: object, **kwargs: object) -> BatchFeature:
        """Record the call and return one row per conversation, tagged by audio marker."""
        assert isinstance(conversation, list), "a batched call passes a list of conversations"
        markers = [int(turn[0]["content"][1]["audio"][0]) for turn in conversation]  # type: ignore[index]
        self.template_calls.append(kwargs)
        self.batches.append(markers)
        n = len(markers)
        return BatchFeature(
            data={
                "input_features": torch.zeros((n, 8), dtype=torch.float32),
                "input_ids": torch.tensor([[0, 0, m] for m in markers], dtype=torch.int64),
                "attention_mask": torch.ones((n, 3), dtype=torch.int64),
            }
        )

    def batch_decode(self, sequences: torch.Tensor, **kwargs: object) -> List[str]:
        """Record the decode keywords and name each row by the marker it carries.

        The real ``AudioFlamingo3Processor.batch_decode`` forwards to ``decode``, which
        applies ``strip_prefix`` per string only when handed a 2-D batch; a 1-D sequence
        makes it iterate characters instead.
        """
        assert sequences.ndim == 2, "batch_decode needs a 2-D batch for strip_prefix to apply per row"
        self.decode_calls.append(kwargs)
        return [f"caption for {int(row[-1])}" for row in sequences]


class _FakeModel:
    """Records the dtype of every tensor handed to ``generate`` and counts the calls."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.seen: Dict[str, torch.dtype] = {}
        self.generate_calls = 0
        self.batch_rows: List[int] = []

    def generate(self, max_new_tokens: int = 0, **inputs: object) -> torch.Tensor:
        """Capture input dtypes and continue each row, preserving its marker."""
        self.seen = {k: v.dtype for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        self.generate_calls += 1
        ids = inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        self.batch_rows.append(int(ids.shape[0]))
        marker = ids[:, -1:]
        return torch.cat([ids, torch.zeros_like(marker), marker], dim=1)


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
        assert processor.template_calls[0]["processor_kwargs"]["sampling_rate"] == TARGET_SAMPLING_RATE

    def test_the_sampling_rate_is_not_passed_as_a_bare_kwarg(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """It must not also ride in ``**kwargs``, which triggers the clobbering path."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo([_audio()], prompt="describe", device=DeviceType.CPU)
        assert "sampling_rate" not in processor.template_calls[0]


class TestGenerationIsBatched:
    """The clips are sent to the model together, not one per ``generate`` call.

    A serial loop returns the same strings, so a test that only checks the answers proves
    nothing about batching; these count the calls the loop makes.
    """

    def test_one_generate_call_serves_the_whole_batch(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Four clips inside one batch reach the model in a single call."""
        processor, mdl = _wired
        audios = [_marked_audio(i) for i in range(1, 5)]
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            audios, prompt="describe", device=DeviceType.CPU, batch_size=4
        )
        assert mdl.generate_calls == 1
        assert len(processor.template_calls) == 1
        assert mdl.batch_rows == [4]

    def test_the_batch_is_chunked_by_batch_size(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Five clips at batch_size 2 make three calls of 2, 2 and 1 — not five."""
        _, mdl = _wired
        audios = [_marked_audio(i) for i in range(1, 6)]
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            audios, prompt="describe", device=DeviceType.CPU, batch_size=2
        )
        assert mdl.generate_calls == 3
        assert mdl.batch_rows == [2, 2, 1]

    def test_batch_size_one_still_works(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Per-clip generation stays available, and still decodes through the 2-D path."""
        _, mdl = _wired
        audios = [_marked_audio(i) for i in range(1, 4)]
        answers = AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            audios, prompt="describe", device=DeviceType.CPU, batch_size=1
        )
        assert mdl.batch_rows == [1, 1, 1]
        assert answers == ["caption for 1", "caption for 2", "caption for 3"]

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_a_meaningless_batch_size_is_refused(self, batch_size: int) -> None:
        """Zero or negative would silently produce no batches at all."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            AudioFlamingoUnderstanding.describe_with_audio_flamingo(
                [_audio()], prompt="describe", device=DeviceType.CPU, batch_size=batch_size
            )


class TestAnswersKeepInputOrder:
    """One answer per clip, in the order the clips were given."""

    def test_answers_follow_input_order_within_a_batch(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Distinct clips come back matched to their own input, not shuffled."""
        audios = [_marked_audio(m) for m in (7, 3, 9, 1)]
        answers = AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            audios, prompt="describe", device=DeviceType.CPU, batch_size=4
        )
        assert answers == ["caption for 7", "caption for 3", "caption for 9", "caption for 1"]

    def test_answers_follow_input_order_across_chunks(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """Chunk boundaries do not reorder or drop answers."""
        markers = [4, 8, 2, 6, 5]
        answers = AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_marked_audio(m) for m in markers], prompt="describe", device=DeviceType.CPU, batch_size=2
        )
        assert answers == [f"caption for {m}" for m in markers]

    def test_every_clip_gets_exactly_one_answer(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """The count matches the input even when it straddles chunks unevenly."""
        audios = [_marked_audio(i) for i in range(1, 8)]
        answers = AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            audios, prompt="describe", device=DeviceType.CPU, batch_size=3
        )
        assert len(answers) == len(audios)


class TestThePromptIsSentWithEveryClip:
    """Batching must not attach the prompt to only the first conversation."""

    def test_each_conversation_in_the_batch_carries_the_prompt(
        self, _wired: tuple[_FakeProcessor, _FakeModel], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All four conversations hold the instruction, not just the leading one."""
        processor, _ = _wired
        seen: List[object] = []
        original = processor.apply_chat_template

        def _capture(conversation: object, **kwargs: object) -> BatchFeature:
            seen.append(conversation)
            return original(conversation, **kwargs)

        monkeypatch.setattr(processor, "apply_chat_template", _capture)
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_marked_audio(i) for i in range(1, 5)], prompt="describe it", device=DeviceType.CPU, batch_size=4
        )
        conversations = seen[0]
        assert isinstance(conversations, list) and len(conversations) == 4
        for turn in conversations:
            assert turn[0]["content"][0] == {"type": "text", "text": "describe it"}


class TestPaddingIsLeftAlignedForGeneration:
    """Decoder-only generation needs left padding, or continuations start mid-prompt.

    The processor's own default is left, but relying on an upstream default silently
    breaks generation the day it changes, so the value is passed explicitly.
    """

    def test_the_padding_side_is_requested_explicitly(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """``padding_side`` reaches the processor rather than being left to a default."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_marked_audio(1), _marked_audio(2)], prompt="describe", device=DeviceType.CPU, batch_size=2
        )
        assert processor.template_calls[0]["processor_kwargs"]["padding_side"] == "left"


class TestTheTranscriptionPrefixCanBeStripped:
    """``strip_prefix`` is forwarded through ``batch_decode``, which applies it per row.

    ``AudioFlamingo3Processor.batch_decode`` delegates to ``decode``, so the flag survives;
    what does not survive is a 1-D sequence, where the same code iterates the decoded
    string's characters instead of a list of answers.
    """

    def test_strip_prefix_reaches_the_decoder(self, _wired: tuple[_FakeProcessor, _FakeModel]) -> None:
        """The flag is forwarded, so the canned wrapper can actually be removed."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_audio()], prompt="transcribe", device=DeviceType.CPU, strip_prefix=True
        )
        assert processor.decode_calls[0]["strip_prefix"] is True

    def test_strip_prefix_is_forwarded_once_per_batch_not_per_clip(
        self, _wired: tuple[_FakeProcessor, _FakeModel]
    ) -> None:
        """One batched decode carries the flag for every row in the batch."""
        processor, _ = _wired
        AudioFlamingoUnderstanding.describe_with_audio_flamingo(
            [_marked_audio(i) for i in range(1, 5)],
            prompt="transcribe",
            device=DeviceType.CPU,
            strip_prefix=True,
            batch_size=4,
        )
        assert len(processor.decode_calls) == 1
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
            describe_audios([_audio()], prompt="describe", think=True, strip_prefix=True, batch_size=4)
        assert seen["think"] is True
        assert seen["strip_prefix"] is True
        assert seen["batch_size"] == 4
