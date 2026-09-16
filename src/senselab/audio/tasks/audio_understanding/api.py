"""Audio understanding: generate text about an audio clip from a free-text prompt."""

from __future__ import annotations

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.audio_understanding.audio_flamingo import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_ID,
    AudioFlamingoUnderstanding,
)
from senselab.utils.data_structures import DeviceType, HFModel


def describe_audios(
    audios: List[Audio],
    prompt: str,
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    max_new_tokens: int = 500,
    think: bool = False,
    strip_prefix: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[str]:
    """Answer ``prompt`` about each audio with an audio-language model.

    Covers captioning, sound-event description and open-ended audio question
    answering, since the response is whatever the prompt asks for.

    Args:
        audios: Audio clips to describe.
        prompt: The instruction sent alongside each clip.
        model: HF model spec. Defaults to ``nvidia/audio-flamingo-3-hf``, whose weights
            are **NVIDIA OneWay Noncommercial License — non-commercial research only**.
        device: CPU or CUDA. CUDA strongly recommended.
        max_new_tokens: Generation cap.
        think: Load the AF-Think adapter. Set this for prompts that ask the model to
            reason before answering; the base checkpoint is not a reasoning model.
        strip_prefix: Drop the canned transcription wrapper from the answer.
        batch_size: Clips sent to the model per generate call. Larger values raise
            throughput and peak memory together; 1 restores per-clip generation.

    Returns:
        One generated string per input audio, in input order.

    Raises:
        NotImplementedError: If ``model`` is not a supported audio-language model.
    """
    model_id = str(model.path_or_uri) if model is not None else DEFAULT_MODEL_ID
    if "audio-flamingo" not in model_id:
        raise NotImplementedError(f"No audio-understanding backend for {model_id!r}. Supported: {DEFAULT_MODEL_ID!r}.")
    return AudioFlamingoUnderstanding.describe_with_audio_flamingo(
        audios=audios,
        prompt=prompt,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        think=think,
        strip_prefix=strip_prefix,
        batch_size=batch_size,
    )
