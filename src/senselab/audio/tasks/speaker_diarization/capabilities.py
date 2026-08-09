"""What each diarization backend actually provides.

Six backends reach :func:`diarize_audios` and share a return type while disagreeing
about almost everything else. Measured across three recordings on an H100: ``text`` is
populated by exactly two of six; ``speaker`` denotes an identity for five and a *role*
for the USC-SAIL child-adult classifier; DiariZen's VBx clustering numbers speakers per
audio, so the same run produced ``['1','2']`` for one file and ``['0','0','1','0']`` for
another. None of that was discoverable without running the model.

This module declares it instead. The record is static rather than returned per call
because the question a caller needs answered — "can this give me more than two
speakers?" — has to be answerable *before* paying for a 16 GB download and a GPU minute.

``ScriptLine`` deliberately does not change. It already provides a uniform key set, and
it is shared by ASR, forced alignment and the workflow's harvesters, so reshaping it for
a diarization-specific gap would be the wrong blast radius.

This module imports no backend, so it stays cheap to import from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, get_args

SpeakerLabelKind = Literal["identity", "role"]

_VALID_LABEL_KINDS = get_args(SpeakerLabelKind)


@dataclass(frozen=True)
class DiarizationCapabilities:
    """What one diarization backend populates, and what its labels mean.

    Attributes:
        populates_text: Whether the backend fills ``ScriptLine.text``. Without this a
            consumer cannot tell "this backend does not transcribe" from "this segment
            had no words" — both look like ``text=None``.
        speaker_label_kind: ``"identity"`` when ``speaker`` names *who* is talking,
            ``"role"`` when it names *what kind* of talker (child-adult emits
            CHILD/ADULT/OVERLAP). Role labels must not reach embedding clustering: a
            per-role centroid blends distinct speakers under one label.
        labels_stable_across_files: Whether label ``"1"`` in one file denotes the same
            speaker as ``"1"`` in another. False for any backend that numbers per audio.
        max_speakers: The backend's ceiling, or ``None`` when nobody has measured it.
            ``None`` does **not** mean unlimited. Counts distinguishable *speakers*,
            not distinct ``speaker`` label values: the child-adult classifier emits
            three label values (``CHILD``, ``ADULT``, ``OVERLAP``) but declares
            ``max_speakers=2``, because ``OVERLAP`` marks two of the two known
            talkers speaking at once, not a third talker.
        honors_speaker_hints: Whether ``num_speakers``/``min_speakers``/``max_speakers``
            passed to :func:`diarize_audios` do anything. Five of six ignore them.
    """

    populates_text: bool
    speaker_label_kind: SpeakerLabelKind
    labels_stable_across_files: bool
    max_speakers: Optional[int]
    honors_speaker_hints: bool

    def __post_init__(self) -> None:
        """Reject declarations that cannot describe a real backend."""
        if self.speaker_label_kind not in _VALID_LABEL_KINDS:
            raise ValueError(
                f"speaker_label_kind must be one of {_VALID_LABEL_KINDS}, got {self.speaker_label_kind!r}. "
                "The distinction decides whether these labels may reach embedding clustering."
            )
        if self.max_speakers is not None and self.max_speakers < 1:
            raise ValueError(
                f"max_speakers must be >= 1 or None (unmeasured), got {self.max_speakers!r}. "
                "None means nobody has measured the ceiling; it does not mean unlimited."
            )
