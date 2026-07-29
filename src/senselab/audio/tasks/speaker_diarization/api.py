"""This module implements some utilities for the speaker diarization task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult
from senselab.audio.tasks.speaker_diarization.diarizen import diarize_audios_with_diarizen
from senselab.audio.tasks.speaker_diarization.moss import diarize_audios_with_moss
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel

_VIBEVOICE_PREFIXES = ("microsoft/VibeVoice",)
_CHILD_ADULT_PREFIXES = ("AlexXu811/whisper-child-adult",)
_MOSS_PREFIXES = ("OpenMOSS-Team/MOSS-Transcribe-Diarize",)
_DIARIZEN_PREFIXES = ("BUT-FIT/diarizen",)


@requires_compatibility("audio.tasks.speaker_diarization.diarize_audios")
def diarize_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[DeviceType] = None,
    max_new_tokens: Optional[int] = None,
) -> List[List[ScriptLine]]:
    """Diarize a batch of `Audio` objects, returning per-speaker time segments.

    Supports **Pyannote** (default), **NVIDIA Sortformer**, **VibeVoice-ASR-HF**, and
    **USC-SAIL child-adult** (HF-identified) backends:
    - If `model` is a `PyannoteAudioModel`, uses Pyannote (typically expects **mono, 16 kHz**).
      Optional `num_speakers` or (`min_speakers`, `max_speakers`) are honored.
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"nvidia/diar_sortformer"`,
      uses NVIDIA Sortformer via an isolated subprocess venv (nvidia/diar_sortformer_4spk-v1
      detects max **4 speakers**).
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"microsoft/VibeVoice"`,
      uses VibeVoice-ASR-HF in-process (``transformers>=5.3``'s
      ``VibeVoiceAsrForConditionalGeneration``).
    - If `model` is an `HFModel` and `model.path_or_uri` starts with
      `"AlexXu811/whisper-child-adult"`, uses the USC-SAIL child-adult classifier via an
      isolated subprocess venv (speaker labels are `"CHILD"`/`"ADULT"`/`"OVERLAP"`/`"SILENCE"`
      rather than speaker identities; **CUDA only**, see ``child_adult.py``).
    - If `model` is an `HFModel` and `model.path_or_uri` starts with
      `"OpenMOSS-Team/MOSS-Transcribe-Diarize"`, uses MOSS-Transcribe-Diarize (0.9B, Apache 2.0)
      via an isolated subprocess venv (needs ``transformers>=5.6``, kept out of the core
      environment — see ``moss.py``).
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"BUT-FIT/diarizen"`,
      uses DiariZen (WavLM-Conformer EEND + VBx clustering; diarization only, no
      transcription) via an isolated subprocess venv that installs DiariZen's own
      forked pyannote-audio — code MIT, but **model weights are CC BY-NC 4.0
      (non-commercial only)**; see ``diarizen.py``.

    Args:
        audios (list[Audio]):
            Audio objects to diarize.
        model (SenselabModel | None):
            Diarization backend:
              * ``PyannoteAudioModel(...)`` → Pyannote (default if ``None``).
              * ``HFModel(path_or_uri="nvidia/diar_sortformer...")`` → NVIDIA Sortformer.
              * ``HFModel(path_or_uri="microsoft/VibeVoice...")`` → VibeVoice-ASR-HF.
              * ``HFModel(path_or_uri="AlexXu811/whisper-child-adult")`` → USC-SAIL child-adult.
              * ``HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")`` → MOSS-Transcribe-Diarize.
              * ``HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")`` → DiariZen.
        num_speakers (int | None):
            If known, fix the number of speakers (Pyannote only).
        min_speakers (int | None):
            Lower bound when estimating number of speakers (Pyannote only).
        max_speakers (int | None):
            Upper bound when estimating number of speakers (Pyannote only).
            NVIDIA Sortformer is limited to 4 speakers.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``).
        max_new_tokens (int | None):
            Generation budget for the joint ASR+diarization backends
            (VibeVoice-ASR-HF, MOSS-Transcribe-Diarize only — ignored otherwise).
            ``None`` uses each backend's own default (4096). Raise this for
            recordings long enough to risk truncated generation.

    Returns:
        list[list[ScriptLine]]: One list per input audio; each `ScriptLine` carries
        `speaker`, `start`, and `end`.

    Raises:
        NotImplementedError: If an unsupported model type is passed.

    Example (Pyannote, default model, CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> lines = diarize_audios([a1, a2], device=DeviceType.CPU)
        >>> len(lines) == 2
        True

    Example (NVIDIA Sortformer via HF, CUDA):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, HFModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> hf = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
        >>> lines = diarize_audios([a1], model=hf, device=DeviceType.CUDA)
        >>> isinstance(lines[0], list)
        True
    """
    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")

    if isinstance(model, PyannoteAudioModel):
        return diarize_audios_with_pyannote(
            audios=audios,
            model=model,
            device=device,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith("nvidia/diar"):
        return diarize_audios_with_nvidia_sortformer(
            audios=audios,
            model=model,
            device=device,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_VIBEVOICE_PREFIXES):
        vibevoice_kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return diarize_audios_with_vibevoice(
            audios=audios,
            model=model,
            device=device,
            **vibevoice_kwargs,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_CHILD_ADULT_PREFIXES):
        return diarize_audios_with_child_adult(
            audios=audios,
            model=model,
            device=device,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_MOSS_PREFIXES):
        moss_kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return diarize_audios_with_moss(
            audios=audios,
            model=model,
            device=device,
            **moss_kwargs,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_DIARIZEN_PREFIXES):
        return diarize_audios_with_diarizen(
            audios=audios,
            model=model,
            device=device,
        )
    else:
        raise NotImplementedError(
            "Only Pyannote, NVIDIA Sortformer, VibeVoice-ASR-HF, the USC-SAIL child-adult "
            "classifier, MOSS-Transcribe-Diarize, and DiariZen (from HuggingFace) models are "
            "supported for now."
        )
