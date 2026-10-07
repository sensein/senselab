"""Voice Activity Detection (VAD) over diarization backends.

Two backends, both of which relabel diarization segments as ``"VOICE"``:

- **Pyannote diarization** (default, ``pyannote/speaker-diarization-community-1``),
- **NVIDIA Sortformer** (via Hugging Face).

There is no dedicated-VAD backend; see
``specs/20260818-093000-drop-pre-4x-pyannote/decision.md``.

All backends expect **mono, 16 kHz** audio objects. Output is a list per input audio; each
inner list contains `ScriptLine` entries with `(start, end)` and `speaker="VOICE"`. These
functions operate on in-memory `Audio` objects (no file I/O), so resample and downmix
upstream as needed.
"""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel


@requires_compatibility("audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios")
def detect_human_voice_activity_in_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Detect human voice activity (VAD) and return time segments labeled ``"VOICE"``.

    Under the hood, this routes to one of two backends, each of which relabels its
    diarization segments as ``"VOICE"``:

    1. **Pyannote diarization** -- when a ``PyannoteAudioModel`` is passed, or
       ``model=None``.
    2. **NVIDIA Sortformer** -- when an ``HFModel`` whose ``path_or_uri`` starts with
       ``"nvidia/diar_sortformer"`` is passed.

    Args:
        audios (list[Audio]):
            Audio clips to analyze. Ensure backend-specific requirements are met
            (e.g., mono and correct sampling rate).
        model (SenselabModel | None):
            Backend selector:

            - ``None`` defaults to Pyannote diarization
              (``pyannote/speaker-diarization-community-1``).
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

    if isinstance(model, PyannoteAudioModel):
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
        raise NotImplementedError("Only Pyannote diarization and NVIDIA Sortformer models are supported for now.")
