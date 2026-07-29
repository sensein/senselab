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
from senselab.utils.dependencies import load_hf_resilient

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

        resolved_device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{resolved_device}"
        if key not in cls._models:
            # load_hf_resilient resolves (repo_id, revision) to an immutable commit
            # SHA once (download-once via the cross-process heartbeat lock) and
            # injects revision=<sha>, so a cached model makes no per-call Hub
            # version check — the 429 source under parallel batch load.
            token = get_huggingface_token()
            repo_id = str(model.path_or_uri)
            revision = model.revision or "main"
            processor = load_hf_resilient(
                AutoProcessor.from_pretrained, repo_id, repo_id=repo_id, revision=revision, token=token
            )
            # dtype="auto" (not the shared device-selector's fp16 default): this
            # checkpoint's config declares bfloat16, and every upstream example
            # loads it with dtype="auto" — forcing fp16 on a bf16-trained model
            # loses exponent range (overflow -> NaN), matching the model's own
            # trained precision instead. diarize_audios_with_vibevoice casts
            # inputs to whatever dtype actually loads, so this isn't tied to fp16.
            vv_model = load_hf_resilient(
                VibeVoiceAsrForConditionalGeneration.from_pretrained,
                repo_id,
                repo_id=repo_id,
                revision=revision,
                token=token,
                dtype="auto",
            )
            vv_model = vv_model.to(torch.device(resolved_device.value))  # type: ignore[arg-type]
            vv_model.eval()
            cls._models[key] = (processor, vv_model)
        return cls._models[key]

    @classmethod
    def release_all(cls) -> None:
        """Drop all cached processor/model pairs and free GPU memory.

        VibeVoice-ASR-HF is a 7B-parameter model held in a class-level cache with
        no natural eviction point — unlike the other three new diarization
        backends, which are subprocess-hosted and free their memory automatically
        on process exit. Call this once a caller is done running VibeVoice for a
        pass so its ~14GB+ footprint doesn't stay resident alongside another large
        in-process model for the rest of the run.
        """
        cls._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
            model_dtype = next(vv_model.parameters()).dtype
            inputs = {
                k: (
                    v.to(device=vv_model.device, dtype=model_dtype)
                    if v.is_floating_point()
                    else v.to(device=vv_model.device)
                )
                if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

            with torch.no_grad():
                output_ids = vv_model.generate(**inputs, max_new_tokens=max_new_tokens)  # type: ignore[misc]

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]

            if generated_ids.shape[-1] >= max_new_tokens:
                # generate() only stops before max_new_tokens if it hit an EOS token;
                # reaching the budget means the output is very likely cut off mid-JSON,
                # which is a different failure mode than "no speech in this audio" —
                # without this, a caller can't tell the two apart from an empty result.
                logger.warning(
                    f"VibeVoice-ASR-HF hit max_new_tokens={max_new_tokens} without generating "
                    "an end token; output is likely truncated. Pass a higher max_new_tokens "
                    "for longer recordings."
                )

            try:
                segments = processor.decode(generated_ids, return_format="parsed")  # type: ignore[attr-defined]
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
                # extract_speaker_dict() doesn't always catch malformed JSON itself
                # (it only guards a handful of shape checks) — surface this the same
                # way as its documented "return original text on failure" contract,
                # without letting a shape error abort the whole batch.
                logger.warning(f"VibeVoice-ASR-HF produced unparsable output: {exc}")
                segments = []

            if isinstance(segments, str) or not segments:
                if isinstance(segments, str):
                    logger.warning("VibeVoice-ASR-HF output did not parse into structured segments.")
                segments = []

            script_lines = []
            for seg in segments:
                if not isinstance(seg, dict) or seg.get("Start") is None or seg.get("End") is None:
                    continue
                try:
                    start = float(seg["Start"])
                    end = float(seg["End"])
                except (TypeError, ValueError) as exc:
                    # A single malformed segment shouldn't abort the whole batch
                    # the same way a decode()-level parse failure wouldn't (see
                    # the except clause above) — skip just this segment.
                    logger.warning(f"VibeVoice-ASR-HF produced a segment with unparsable Start/End ({exc}); skipping.")
                    continue
                script_lines.append(
                    ScriptLine(speaker=str(seg.get("Speaker")), start=start, end=end, text=seg.get("Content"))
                )
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

    return results
