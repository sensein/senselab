"""Voice Activity Detection (VAD) via dedicated and diarization-based backends.

This module exposes a simple VAD API with multiple backends:

- **Pyannote dedicated VAD** (default when a VAD-specific model ID is passed),
- **Pyannote diarization** (relabels diarization segments as ``"VOICE"``),
- **NVIDIA Sortformer** (via Hugging Face, relabels diarization segments).

All backends expect **mono, 16 kHz** audio objects.
Output is a list per input audio; each inner list contains `ScriptLine` entries
with `(start, end)` and `speaker="VOICE"`.

Notes:
    - This function operates on in-memory `Audio` objects (no file I/O).
    - For all backends, resample/downmix upstream as needed (e.g., mono @ 16 kHz).
"""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.audio.tasks.voice_activity_detection.pyannote_vad import PyannoteVAD
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel

# Pyannote model IDs that indicate a dedicated VAD pipeline
_PYANNOTE_VAD_PREFIXES = ("pyannote/voice-activity-detection",)


def _is_pyannote_vad_model(model: PyannoteAudioModel) -> bool:
    """Check if a PyannoteAudioModel refers to a dedicated VAD model."""
    return str(model.path_or_uri).startswith(_PYANNOTE_VAD_PREFIXES)


@requires_compatibility("audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios")
def detect_human_voice_activity_in_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Detect human voice activity (VAD) and return time segments labeled ``"VOICE"``.

    Under the hood, this routes to one of three backends:

    1. **Pyannote dedicated VAD** -- when a ``PyannoteAudioModel`` with a
       VAD-specific model ID (e.g., ``pyannote/voice-activity-detection``) is
       passed. This uses a lightweight segmentation model purpose-built for
       speech/non-speech detection.
    2. **Pyannote diarization** -- when a ``PyannoteAudioModel`` with a
       diarization model ID is passed (or ``model=None``). Diarization
       segments are relabeled as ``"VOICE"``.
    3. **NVIDIA Sortformer** -- when an ``HFModel`` whose ``path_or_uri``
       starts with ``"nvidia/diar_sortformer"`` is passed. Diarization
       segments are relabeled as ``"VOICE"``.

    Args:
        audios (list[Audio]):
            Audio clips to analyze. Ensure backend-specific requirements are met
            (e.g., mono and correct sampling rate).
        model (SenselabModel | None):
            Backend selector:

            - ``None`` defaults to Pyannote diarization
              (``pyannote/speaker-diarization-community-1``).
            - ``PyannoteAudioModel("pyannote/voice-activity-detection")`` uses
              the dedicated Pyannote VAD pipeline.
            - ``PyannoteAudioModel("pyannote/speaker-diarization-community-1")``
              uses the Pyannote diarization pipeline.
            - ``HFModel("nvidia/diar_sortformer_4spk-v1")`` uses the NVIDIA
              Sortformer diarization pipeline.
        device (DeviceType | None):
            Preferred device for inference (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``).

    Returns:
        list[list[ScriptLine]]:
            One list per input audio; each inner list contains ``ScriptLine``
            entries with ``(start, end)`` and ``speaker="VOICE"``.

    Raises:
        NotImplementedError:
            If ``model`` is not a supported type.

    Examples:
        Pyannote dedicated VAD:
            >>> from pathlib import Path
            >>> from senselab.audio.data_structures import Audio
            >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
            >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
            >>> mdl = PyannoteAudioModel(path_or_uri="pyannote/voice-activity-detection")
            >>> vad = detect_human_voice_activity_in_audios([a1], model=mdl, device=DeviceType.CPU)

        Pyannote diarization (default model, CPU):
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

    if isinstance(model, PyannoteAudioModel) and _is_pyannote_vad_model(model):
        # Dedicated VAD pipeline — segments already labeled "VOICE"
        return PyannoteVAD.detect_voice_activity(audios=audios, model=model, device=device)
    elif isinstance(model, PyannoteAudioModel):
        # Diarization-based VAD — relabel speaker segments as "VOICE"
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
        raise NotImplementedError(
            "Only Pyannote (VAD or diarization) and NVIDIA Sortformer models are supported for now."
        )
