"""This module implements some utilities for the speaker diarization task."""

from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.audio.tasks.speaker_diarization.child_adult import CAPABILITIES as _CHILD_ADULT_CAPS
from senselab.audio.tasks.speaker_diarization.child_adult import diarize_audios_with_child_adult
from senselab.audio.tasks.speaker_diarization.diarizen import CAPABILITIES as _DIARIZEN_CAPS
from senselab.audio.tasks.speaker_diarization.diarizen import diarize_audios_with_diarizen
from senselab.audio.tasks.speaker_diarization.moss import CAPABILITIES as _MOSS_CAPS
from senselab.audio.tasks.speaker_diarization.moss import diarize_audios_with_moss
from senselab.audio.tasks.speaker_diarization.nvidia import CAPABILITIES as _SORTFORMER_CAPS
from senselab.audio.tasks.speaker_diarization.nvidia import diarize_audios_with_nvidia_sortformer
from senselab.audio.tasks.speaker_diarization.pyannote import CAPABILITIES as _PYANNOTE_CAPS
from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote
from senselab.audio.tasks.speaker_diarization.vibevoice import CAPABILITIES as _VIBEVOICE_CAPS
from senselab.audio.tasks.speaker_diarization.vibevoice import diarize_audios_with_vibevoice
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, HFModel, PyannoteAudioModel, ScriptLine, SenselabModel
from senselab.utils.data_structures.logging import logger

# NOTE: this is deliberately "nvidia/diar", not "nvidia/diar_sortformer" — it is
# the literal the dispatch branch below already used before this constant existed.
# Narrowing it to "_sortformer" would change dispatch for any future
# "nvidia/diar*" checkpoint that isn't a Sortformer build; keep it exact so the
# constant cannot silently change what already ships.
#
# This deliberately diverges from `model.py`'s `model_for_task` (task="diarization"),
# which matches the narrower "nvidia/diar_sortformer" literal. That module's own
# docstring asserts the two dispatch tables "must be kept in sync by hand" — they
# are provably not, right here, and this divergence is why. Don't trust that
# assertion; check both tables directly.
_SORTFORMER_PREFIXES = ("nvidia/diar",)
_VIBEVOICE_PREFIXES = ("microsoft/VibeVoice-ASR",)
_CHILD_ADULT_PREFIXES = ("AlexXu811/whisper-child-adult",)
_MOSS_PREFIXES = ("OpenMOSS-Team/MOSS-Transcribe-Diarize",)
_DIARIZEN_PREFIXES = ("BUT-FIT/diarizen",)

# Backends dispatched above whose "speaker" label is a role (e.g. CHILD/ADULT/
# OVERLAP), not a speaker identity. Single source of truth for this distinction:
# once ported from PR #537 (see the module docstrings' "Not wired into
# audio_analysis" sections), clustering.py/identity.py/presence.py will import
# it from here (rather than each restating the literal) so adding a second
# role-only backend here can't silently miss excluding it from identity
# clustering / presence voting too. No consumer imports this yet on this
# branch — it is not dead code, it is pre-wired for that port.
ROLE_LABEL_ONLY_PREFIXES = _CHILD_ADULT_PREFIXES


def _warn_if_speaker_hints_ignored(
    backend_name: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
) -> None:
    """Warn when num_speakers/min_speakers/max_speakers were passed to a backend that ignores them."""
    if num_speakers is not None or min_speakers is not None or max_speakers is not None:
        logger.warning(
            f"num_speakers/min_speakers/max_speakers are ignored by {backend_name} "
            "(Pyannote-only hints) and will have no effect on this call."
        )


@requires_compatibility("audio.tasks.speaker_diarization.diarize_audios")
def diarize_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    device: Optional[DeviceType] = None,
    exclusive: bool = True,
    max_new_tokens: Optional[int] = None,
) -> List[List[ScriptLine]]:
    """Diarize a batch of `Audio` objects, returning per-speaker time segments.

    Supports **Pyannote** (default), **NVIDIA Sortformer**, **VibeVoice-ASR-HF**,
    **USC-SAIL child-adult**, **MOSS-Transcribe-Diarize**, and **DiariZen**
    (HF-identified) backends:
    - If `model` is a `PyannoteAudioModel`, uses Pyannote (typically expects **mono, 16 kHz**).
      Optional `num_speakers` or (`min_speakers`, `max_speakers`) are honored.
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"nvidia/diar_sortformer"`,
      uses NVIDIA Sortformer via an isolated subprocess venv (nvidia/diar_sortformer_4spk-v1
      detects max **4 speakers**).
    - If `model` is an `HFModel` and `model.path_or_uri` starts with `"microsoft/VibeVoice-ASR"`,
      uses VibeVoice-ASR-HF in-process (``transformers>=5.3``'s
      ``VibeVoiceAsrForConditionalGeneration``). Narrower than the bare `"microsoft/VibeVoice"`
      prefix on purpose — `microsoft/VibeVoice-1.5B`/`-Large` are TTS checkpoints, not ASR.
    - If `model` is an `HFModel` and `model.path_or_uri` starts with
      `"AlexXu811/whisper-child-adult"`, uses the USC-SAIL child-adult classifier via an
      isolated subprocess venv (speaker labels are `"CHILD"`/`"ADULT"`/`"OVERLAP"` — frames
      classified as silence produce no segment — rather than speaker identities;
      **CUDA only**, see ``child_adult.py``).
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
              * ``HFModel(path_or_uri="microsoft/VibeVoice-ASR...")`` → VibeVoice-ASR-HF.
              * ``HFModel(path_or_uri="AlexXu811/whisper-child-adult")`` → USC-SAIL child-adult.
              * ``HFModel(path_or_uri="OpenMOSS-Team/MOSS-Transcribe-Diarize")`` → MOSS-Transcribe-Diarize.
              * ``HFModel(path_or_uri="BUT-FIT/diarizen-wavlm-large-s80-md")`` → DiariZen.
        num_speakers (int | None):
            If known, fix the number of speakers (Pyannote only).
        min_speakers (int | None):
            Lower bound when estimating number of speakers (Pyannote only).
        max_speakers (int | None):
            Upper bound when estimating number of speakers (Pyannote only; ignored,
            with a warning, by every other backend). NVIDIA Sortformer detects at
            most 4 speakers regardless of this argument — that's a fixed model
            limit, not something `max_speakers` raises or lowers.
        device (DeviceType | None):
            Preferred device (e.g., ``DeviceType.CPU``, ``DeviceType.CUDA``).
        exclusive (bool):
            Pyannote only. ``True`` (default) returns a partition, where concurrent speech has been
            resolved away and no consumer can detect overlap. ``False`` returns the overlapping
            view. Ignored by every other backend, none of which return an exclusive partition
            regardless of this argument: NVIDIA Sortformer emits per-speaker activity and always
            preserves concurrency; VibeVoice-ASR-HF and MOSS-Transcribe-Diarize emit whatever
            segments their own transcript-parsing produces, with no partition step; DiariZen's
            EEND + VBx pipeline emits genuinely overlapping turns by design; and the USC-SAIL
            child-adult classifier emits an explicit ``OVERLAP`` label for frames it can't assign
            to a single role. Unlike the speaker-count hints above, no warning is raised when
            ``exclusive`` is passed to one of these — it defaults to ``True``, so there is no way
            to tell an explicit pass from the default.
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
            exclusive=exclusive,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_SORTFORMER_PREFIXES):
        _warn_if_speaker_hints_ignored("NVIDIA Sortformer", num_speakers, min_speakers, max_speakers)
        return diarize_audios_with_nvidia_sortformer(
            audios=audios,
            model=model,
            device=device,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_VIBEVOICE_PREFIXES):
        _warn_if_speaker_hints_ignored("VibeVoice-ASR-HF", num_speakers, min_speakers, max_speakers)
        vibevoice_kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return diarize_audios_with_vibevoice(
            audios=audios,
            model=model,
            device=device,
            **vibevoice_kwargs,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_CHILD_ADULT_PREFIXES):
        _warn_if_speaker_hints_ignored("the USC-SAIL child-adult classifier", num_speakers, min_speakers, max_speakers)
        return diarize_audios_with_child_adult(
            audios=audios,
            model=model,
            device=device,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_MOSS_PREFIXES):
        _warn_if_speaker_hints_ignored("MOSS-Transcribe-Diarize", num_speakers, min_speakers, max_speakers)
        moss_kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        return diarize_audios_with_moss(
            audios=audios,
            model=model,
            device=device,
            **moss_kwargs,
        )
    elif isinstance(model, HFModel) and str(model.path_or_uri).startswith(_DIARIZEN_PREFIXES):
        _warn_if_speaker_hints_ignored("DiariZen", num_speakers, min_speakers, max_speakers)
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


_CAPABILITIES_BY_PREFIX: tuple[tuple[tuple[str, ...], DiarizationCapabilities], ...] = (
    (_SORTFORMER_PREFIXES, _SORTFORMER_CAPS),
    (_VIBEVOICE_PREFIXES, _VIBEVOICE_CAPS),
    (_CHILD_ADULT_PREFIXES, _CHILD_ADULT_CAPS),
    (_MOSS_PREFIXES, _MOSS_CAPS),
    (_DIARIZEN_PREFIXES, _DIARIZEN_CAPS),
)


def capabilities_for(model_id: str) -> DiarizationCapabilities:
    """Return what the backend handling ``model_id`` provides.

    Mirrors :func:`~senselab.utils.data_structures.model.model_for_task`'s dispatch
    (task="diarization"), including its fallback: an id matching no prefix resolves
    to Pyannote, because that is the backend ``model_for_task`` would wrap it for.
    Returning ``None`` instead would make every caller write the same check for a
    case that router treats as ordinary.

    This function answers by **id string**, not by the backend :func:`diarize_audios`
    would actually dispatch to for a given call. `diarize_audios` itself does not
    fall back to Pyannote for an unmatched `HFModel` — it raises `NotImplementedError`
    — Pyannote is only reached there via an `isinstance(model, PyannoteAudioModel)`
    check that this string-taking function cannot perform. So this reports DiariZen
    for a DiariZen id even if a caller hands `diarize_audios` a `PyannoteAudioModel`
    wrapping that same id — that call would run Pyannote, not DiariZen.
    """
    for prefixes, caps in _CAPABILITIES_BY_PREFIX:
        if any(model_id.startswith(p) for p in prefixes):
            return caps
    return _PYANNOTE_CAPS
