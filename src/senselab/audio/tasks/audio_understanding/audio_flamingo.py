"""Audio Flamingo 3 — in-process audio-language backend.

**Licence: NVIDIA OneWay Noncommercial License — non-commercial research use
only.** Portions of the training-data generation are additionally subject to the
Qwen Research License and OpenAI's Terms of Use. This is stricter than most
weights senselab loads; do not use this backend in a commercial product.
See <https://huggingface.co/nvidia/audio-flamingo-3-hf>.

Audio Flamingo 3 takes an audio clip plus a free-text prompt and generates a
textual answer, so it covers captioning, description and open-ended audio
question answering as well as transcription. ``transformers`` ships
``AudioFlamingo3ForConditionalGeneration`` natively, and senselab already pins
``transformers>=5.3``, so this loads in-process with no subprocess venv.

The model consumes audio in 30 s windows internally and accepts clips up to
10 minutes; longer input must be split by the caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype

DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
TARGET_SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 600.0


class AudioFlamingoUnderstanding:
    """Audio Flamingo 3 captioning / description / audio QA via ``transformers``.

    Weights are cached per ``(model, device, attention)`` so repeated calls with the
    same settings reuse them; the model is ~8B parameters and loading it is expensive.
    """

    _cache: Dict[str, Any] = {}

    @classmethod
    def describe_with_audio_flamingo(
        cls,
        audios: List[Audio],
        prompt: str,
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
        max_new_tokens: int = 500,
        think: bool = False,
        strip_prefix: bool = False,
    ) -> List[str]:
        """Generate a textual response to ``prompt`` for each audio.

        Args:
            audios: Audio clips. Resampled to 16 kHz and downmixed to mono if needed.
                Each must be at most ``MAX_AUDIO_SECONDS`` long.
            prompt: The instruction sent alongside the audio.
            model: HF model spec (default: ``nvidia/audio-flamingo-3-hf``).
            device: CPU or CUDA. CUDA strongly recommended.
            max_new_tokens: Generation cap.
            think: Load the AF-Think PEFT adapter from the repository's ``think``
                subfolder. Required for prompts that ask the model to reason before
                answering; the base checkpoint is not a reasoning model.
            strip_prefix: Drop the canned ``The spoken content of the audio is "..."``
                wrapper the transcription checkpoints prepend.

        Returns:
            One generated string per input audio, in input order.

        Raises:
            ValueError: If ``prompt`` is empty or an audio exceeds ``MAX_AUDIO_SECONDS``.
            FileNotFoundError: If ``think`` is set but the adapter is absent from the snapshot.
        """
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not audios:
            return []

        import torch
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        from senselab.utils.dependencies import load_hf_resilient

        model_name = str(model.path_or_uri) if model is not None else DEFAULT_MODEL_ID
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]
        dtype = torch.bfloat16 if device_type == DeviceType.CUDA else torch.float32
        attention = cls._attention_implementation(device_type)

        cache_key = f"{model_name}@{device_type.value}@{attention}@think={think}"
        if cache_key not in cls._cache:
            revision = (model.revision if model is not None else None) or "main"
            device_map = "auto" if device_type == DeviceType.CUDA else "cpu"
            if think:
                cls._cache[cache_key] = cls._load_think(
                    model_name, revision, dtype=dtype, attention=attention, device_map=device_map
                )
            else:
                processor = load_hf_resilient(
                    AutoProcessor.from_pretrained,
                    model_name,
                    repo_id=model_name,
                    revision=revision,
                )
                mdl = load_hf_resilient(
                    AudioFlamingo3ForConditionalGeneration.from_pretrained,
                    model_name,
                    repo_id=model_name,
                    revision=revision,
                    dtype=dtype,
                    attn_implementation=attention,
                    device_map=device_map,
                )
                cls._cache[cache_key] = (processor, mdl)
        processor, mdl = cls._cache[cache_key]

        results: List[str] = []
        for audio in audios:
            prepared = cls._prepare(audio)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": prepared.waveform.squeeze(0).numpy()},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                processor_kwargs={"sampling_rate": TARGET_SAMPLING_RATE},
            ).to(device=mdl.device, dtype=mdl.dtype)
            with torch.no_grad():
                outputs = mdl.generate(**inputs, max_new_tokens=max_new_tokens)
            # decode, not batch_decode: strip_prefix is only exposed on the per-sequence call.
            decoded = processor.decode(
                outputs[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
                strip_prefix=strip_prefix,
            )
            results.append(decoded.strip())
        return results

    @classmethod
    def _load_think(
        cls,
        model_name: str,
        revision: str,
        *,
        dtype: "torch.dtype",
        attention: str,
        device_map: str,
    ) -> Tuple[object, object]:
        """Load the base checkpoint, then its AF-Think extra trainables and LoRA adapter.

        Args:
            model_name: Repository holding both the base weights and the ``think`` subfolder.
            revision: Ref or SHA; resolved to an immutable SHA before anything is read.
            dtype: Weight dtype.
            attention: ``flash_attention_2`` or ``sdpa``.
            device_map: ``auto`` on CUDA, ``cpu`` otherwise.

        Returns:
            A ``(processor, model)`` pair whose model is the PEFT-wrapped reasoning variant.

        Raises:
            FileNotFoundError: If the snapshot holds no ``think/non_lora_trainables.bin``.
        """
        from pathlib import Path

        import torch
        from peft import PeftModel
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        from senselab.utils.dependencies import resolve_model

        _, snapshot = resolve_model(model_name, revision)
        extras = Path(snapshot) / "think" / "non_lora_trainables.bin"
        if not extras.is_file():
            raise FileNotFoundError(
                f"{model_name} has no think/non_lora_trainables.bin in its snapshot at {snapshot}; "
                "the AF-Think adapter is unavailable for this revision."
            )

        processor = AutoProcessor.from_pretrained(snapshot)
        mdl = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            snapshot,
            dtype=dtype,
            attn_implementation=attention,
            device_map=device_map,
        )
        mdl.load_state_dict(torch.load(extras, map_location="cpu", weights_only=True), strict=False)
        return processor, PeftModel.from_pretrained(mdl, str(snapshot), subfolder="think")

    @staticmethod
    def _attention_implementation(device_type: DeviceType) -> str:
        """Return ``flash_attention_2`` when it is installed and usable, else ``sdpa``."""
        if device_type != DeviceType.CUDA:
            return "sdpa"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            return "sdpa"
        return "flash_attention_2"

    @staticmethod
    def _prepare(audio: Audio) -> Audio:
        """Return ``audio`` as 16 kHz mono, raising if it exceeds the model's length limit.

        Args:
            audio: The clip to normalize.

        Returns:
            A 16 kHz mono clip.

        Raises:
            ValueError: If the clip is longer than ``MAX_AUDIO_SECONDS``.
        """
        duration = audio.waveform.shape[-1] / audio.sampling_rate
        if duration > MAX_AUDIO_SECONDS:
            raise ValueError(
                f"Audio Flamingo 3 accepts at most {MAX_AUDIO_SECONDS:.0f}s of audio; got {duration:.1f}s. "
                "Split the recording before calling this backend."
            )
        if audio.sampling_rate != TARGET_SAMPLING_RATE:
            from senselab.audio.tasks.preprocessing import resample_audios

            audio = resample_audios([audio], resample_rate=TARGET_SAMPLING_RATE)[0]
        if audio.waveform.shape[0] > 1:
            from senselab.audio.tasks.preprocessing import downmix_audios_to_mono

            audio = downmix_audios_to_mono([audio])[0]
        return audio
