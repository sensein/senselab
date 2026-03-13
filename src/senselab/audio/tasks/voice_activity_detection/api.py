"""Voice Activity Detection (VAD) via diarization backends.

This module exposes a simple VAD API that *reuses speaker-diarization models*
and relabels all detected segments as `"VOICE"`. It supports:

- **Pyannote** (default, via Hugging Face),
- **NVIDIA Sortformer** (via Hugging Face).

Both models expect **mono, 16 kHz** audio objects.
Output is a list per input audio; each inner list contains `ScriptLine` entries
with `(start, end)` and `speaker="VOICE"`.

Notes:
    - This function operates on in-memory `Audio` objects (no file I/O).
    - For Pyannote, resample/downmix upstream as needed (e.g., mono @ 16 kHz).
    - For Sortformer, resample/downmix upstream as needed (e.g., mono @ 16 kHz).
"""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel


def detect_human_voice_activity_in_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Detect human voice activity (VAD) and return time segments labeled `"VOICE"`.

    Under the hood, this calls a supported **diarization** backend to obtain
    speech segments and then sets `speaker="VOICE"` for each segment.

    Supported backends:
        - `PyannoteAudioModel` → Pyannote diarization (**default** if `model=None`).
        - `HFModel` whose `path_or_uri` starts with `"nvidia/diar_sortformer"`.

    Args:
        audios (list[Audio]):
            Audio clips to analyze. Ensure backend-specific requirements are met
            (e.g., mono and correct sampling rate).
        model (SenselabModel | None):
            Backend selector. If ``None``, defaults to
            ``PyannoteAudioModel("pyannote/speaker-diarization-community-1", "main")``.
        device (DeviceType | None):
            Preferred device for inference (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``).

    Returns:
        list[list[ScriptLine]]:
            One list per input audio; each inner list contains `ScriptLine` entries
            with `(start, end)` and `speaker="VOICE"`.

    Raises:
        NotImplementedError:
            If `model` is neither a `PyannoteAudioModel` nor a supported NVIDIA
            Sortformer `HFModel` (i.e., `path_or_uri` doesn’t start with
            `"nvidia/diar_sortformer"`).

    Examples:
        Pyannote (default model, CPU):
            >>> from pathlib import Path
            >>> from senselab.audio.data_structures import Audio
            >>> from senselab.utils.data_structures import DeviceType
            >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
            >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
            >>> vad = detect_human_voice_activity_in_audios([a1, a2], device=DeviceType.CPU)

        NVIDIA Sortformer (HF), CUDA:
            >>> from pathlib import Path
            >>> from senselab.audio.data_structures import Audio
            >>> from senselab.utils.data_structures import HFModel, DeviceType
            >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
            >>> hf = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
            >>> vad = detect_human_voice_activity_in_audios([a1], model=hf, device=DeviceType.CUDA)
    """
    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")

    if isinstance(model, PyannoteAudioModel):
        results = diarize_audios_with_pyannote(audios=audios, model=model, device=device)
        for sample in results:
            for chunk in sample:
                chunk.speaker = "VOICE"
        return results
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith("nvidia/diar_sortformer"):
        result = diarize_audios_with_nvidia_sortformer(
            audios=audios,
            model=model,
            device=device,
        )
        for sample in result:
            for chunk in sample:
                chunk.speaker = "VOICE"
        return result
    else:
        raise NotImplementedError("Only Pyannote and NVIDIA Sortformer models are supported for now.")
