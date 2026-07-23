"""This module implements the VibeVoice-ASR-HF diarization backend."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token

try:
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    VIBEVOICE_AVAILABLE = True
except ImportError:
    VIBEVOICE_AVAILABLE = False


class VibeVoiceDiarization:
    """Factory for creating and caching **VibeVoice-ASR-HF** processor/model pairs.

    Pairs are cached per *(model.path_or_uri, revision, device)*, so repeated calls
    with the same configuration reuse the initialized model.

    Guidance:
        - VibeVoice-ASR-HF is a 7B-parameter unified ASR+diarization+timestamping
          model; CUDA is strongly recommended.
        - Supported devices: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _models: Dict[str, Tuple["AutoProcessor", "VibeVoiceAsrForConditionalGeneration"]] = {}

    @classmethod
    def _get_vibevoice_model(
        cls,
        model: HFModel,
        device: Optional[DeviceType],
    ) -> Tuple["AutoProcessor", "VibeVoiceAsrForConditionalGeneration"]:
        """Get or create a VibeVoice-ASR-HF processor/model pair.

        Args:
            model (HFModel): The VibeVoice-ASR-HF model.
            device (DeviceType | None): The device to run the model on.

        Returns:
            Tuple[AutoProcessor, VibeVoiceAsrForConditionalGeneration]: The processor and model.
        """
        if not VIBEVOICE_AVAILABLE:
            raise ModuleNotFoundError(
                "VibeVoice-ASR-HF requires `transformers>=5.3`. "
                "Please install/upgrade senselab audio dependencies using `pip install senselab`."
            )

        resolved_device, dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{resolved_device}"
        if key not in cls._models:
            token = get_huggingface_token()
            processor = AutoProcessor.from_pretrained(model.path_or_uri, revision=model.revision, token=token)
            vv_model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                model.path_or_uri, revision=model.revision, token=token, dtype=dtype
            )
            vv_model = vv_model.to(torch.device(resolved_device.value))  # type: ignore[arg-type]
            vv_model.eval()
            cls._models[key] = (processor, vv_model)
        return cls._models[key]


def diarize_audios_with_vibevoice(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    max_new_tokens: int = 4096,
) -> List[List[ScriptLine]]:
    """Diarize audios with **VibeVoice-ASR-HF**; returns per-speaker segments per audio.

    VibeVoice-ASR-HF is a unified ASR+diarization+timestamping foundation model
    (`microsoft/VibeVoice-ASR-HF`, natively supported by ``transformers>=5.3`` via
    ``AutoProcessor``/``VibeVoiceAsrForConditionalGeneration`` — no custom repo code
    or ``trust_remote_code`` needed). Each audio is transcribed in a single pass and
    the model's own diarization is parsed out of its structured JSON-shaped output.

    Args:
        audios (list[Audio]):
            Audio clips to diarize.
        model (HFModel | None):
            VibeVoice model. Defaults to ``HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")``.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``). CUDA is
            strongly recommended — this is a 7B-parameter model.
        max_new_tokens (int):
            Generation budget. Defaults to 4096; raise this for longer recordings.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker`, `start`, `end`, and `text`. Empty for an audio whose output didn't
        parse into structured segments (logged as a warning, not raised).

    Raises:
        ModuleNotFoundError: If `transformers>=5.3` is not installed.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> lines = diarize_audios_with_vibevoice([a1], device=DeviceType.CPU)
        >>> len(lines) == 1
        True
    """
    if model is None:
        model = HFModel(path_or_uri="microsoft/VibeVoice-ASR-HF")

    processor, vv_model = VibeVoiceDiarization._get_vibevoice_model(model, device)

    results: List[List[ScriptLine]] = []
    for audio in audios:
        with tempfile.TemporaryDirectory(prefix="senselab-vibevoice-") as tmpdir:
            wav_path = str(Path(tmpdir) / "audio.wav")
            audio.save_to_file(wav_path)

            inputs = processor.apply_transcription_request(audio=wav_path)  # type: ignore[attr-defined]
            inputs = {k: v.to(vv_model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = vv_model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[misc]

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]

            try:
                segments = processor.decode(generated_ids, return_format="parsed")  # type: ignore[attr-defined]
            except json.JSONDecodeError as exc:
                # extract_speaker_dict() doesn't always catch malformed JSON itself
                # (it only guards a handful of shape checks) — surface this the same
                # way as its documented "return original text on failure" contract.
                logger.warning(f"VibeVoice-ASR-HF produced unparsable output: {exc}")
                segments = []

            if isinstance(segments, str) or not segments:
                if isinstance(segments, str):
                    logger.warning("VibeVoice-ASR-HF output did not parse into structured segments.")
                segments = []

            script_lines = [
                ScriptLine(
                    speaker=str(seg.get("Speaker")),
                    start=float(seg["Start"]),
                    end=float(seg["End"]),
                    text=seg.get("Content"),
                )
                for seg in segments
                if isinstance(seg, dict) and seg.get("Start") is not None and seg.get("End") is not None
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

    return results
