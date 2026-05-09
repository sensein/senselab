"""IBM Granite Speech 3.3 ASR — in-process backend.

Granite Speech is a multimodal LLM that takes an audio + text prompt and
generates a textual answer. Unlike standard HF ASR pipelines, it uses
``GraniteSpeechProcessor`` with a ``(text, audio, device)`` signature
(not the ``{array, sampling_rate}`` audio dict the
``AutomaticSpeechRecognitionPipeline`` passes in), so it cannot be
driven through the HF ASR pipeline. We load the processor and model
directly in-process — no subprocess venv needed, since
``transformers``, ``torch``, ``torchaudio``, and ``peft`` are all
already in senselab's core install (peft is required for Granite's
LoRA adapter; loading without it produces gibberish).

Granite is text-only (no native timestamps); the analyze_audio script's
auto-align stage adds per-segment timestamps via the multilingual MMS
forced-aligner downstream when this backend is used through
``transcribe_audios``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype


class GraniteSpeechASR:
    """IBM Granite Speech 3.3 transcription via direct ``transformers`` use.

    Routed automatically by ``senselab.audio.tasks.speech_to_text.api`` when
    the model id matches the ``ibm-granite/granite-speech-`` prefix.
    Returns text-only ScriptLines (Granite does not emit native
    timestamps); pair with the multilingual forced-aligner downstream to
    add per-segment timing.

    Pipelines are cached per ``(model, device)`` so repeated calls with
    the same settings reuse the loaded weights. Loading an 8B-parameter
    model is expensive; this avoids paying the cost on every invocation.
    """

    _cache: Dict[str, Any] = {}

    @classmethod
    def transcribe_with_granite(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with Granite Speech.

        Args:
            audios: Audio clips to transcribe (mono, 16 kHz expected).
            model: HF model id (default: ``ibm-granite/granite-speech-3.3-8b``).
            device: CPU or CUDA. CUDA strongly recommended; CPU works but
                is very slow for an 8B-parameter model.

        Returns:
            One ``ScriptLine`` per input audio with ``text`` populated.
            ``start``, ``end``, and ``chunks`` are intentionally None /
            empty — Granite does not produce native timestamps.
            Downstream auto-alignment (MMS) adds per-segment timing.
        """
        # Imports inside the function so module import doesn't pay the
        # transformers / torch import cost when Granite is never used.
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        model_name = model.path_or_uri if model is not None else "ibm-granite/granite-speech-3.3-8b"
        device_type = (
            device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU])[0]
        )

        # bfloat16 on accelerators (CUDA + MPS), float32 on CPU. CPU
        # bfloat16 *runs* but ~40x slower than fp32 because PyTorch's
        # CPU kernels lack a native bf16 SIMD path on most platforms.
        # Verified on torch 2.11: matmul/softmax/silu/layer_norm/cumsum
        # all run in bf16 on MPS without falling back to CPU.
        dtype = torch.bfloat16 if device_type in (DeviceType.CUDA, DeviceType.MPS) else torch.float32

        cache_key = f"{model_name}@{device_type.value}"
        if cache_key not in cls._cache:
            processor = AutoProcessor.from_pretrained(model_name)
            mdl = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=dtype)
            if device_type == DeviceType.CUDA and torch.cuda.is_available():
                mdl = mdl.cuda()
            elif device_type == DeviceType.MPS and torch.backends.mps.is_available():
                mdl = mdl.to("mps")
            cls._cache[cache_key] = (processor, mdl)
        processor, mdl = cls._cache[cache_key]

        device_str = device_type.value
        results: List[ScriptLine] = []

        for audio in audios:
            # Granite expects 16 kHz mono. Normalize defensively; callers
            # almost always preprocess upstream but we don't assume.
            if audio.sampling_rate != 16000:
                from senselab.audio.tasks.preprocessing import resample_audios

                audio = resample_audios([audio], resample_rate=16000)[0]
            if audio.waveform.shape[0] > 1:
                from senselab.audio.tasks.preprocessing import downmix_audios_to_mono

                audio = downmix_audios_to_mono([audio])[0]

            chat = [
                {
                    "role": "system",
                    "content": (
                        "Knowledge Cutoff Date: April 2024.\n"
                        "You are Granite, a multimodal AI assistant developed by IBM."
                    ),
                },
                {
                    "role": "user",
                    "content": "<|audio|>can you transcribe the speech into a written format?",
                },
            ]
            prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            # Granite's feature extractor expects a 1D tensor (or list of
            # them); senselab Audio.waveform is shape [channels, T]. We
            # already downmixed to mono above, so [1, T] → squeeze to [T].
            wav_1d = audio.waveform.squeeze(0)
            # ``return_tensors="pt"`` is critical: without it Granite's
            # processor returns ``input_ids`` / ``attention_mask`` as
            # nested Python lists, which the underlying generate() then
            # tries to call ``.shape`` on.
            inputs = processor(prompt, audio=wav_1d, device=device_str, return_tensors="pt")
            inputs = {k: (v.to(device_str) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                generated = mdl.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                )
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            text = processor.decode(generated[0, input_len:], skip_special_tokens=True)
            results.append(ScriptLine(text=text.strip()))

        return results
