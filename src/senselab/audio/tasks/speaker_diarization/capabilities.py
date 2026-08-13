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

# The canonical `max_speakers_evidence` value for "nobody has run a probe against this
# backend yet". Spelled out as a constant rather than repeated as a string literal at
# every call site so a typo (e.g. "unmeasured " with a trailing space) cannot silently
# create a third, unintended provenance state.
UNMEASURED = "unmeasured"

# Every "measured: ..." evidence string must start with this, so a reader — or code —
# can tell "unmeasured" apart from a real measurement with a single prefix check rather
# than parsing free text. See `DiarizationCapabilities.max_speakers_evidence`.
_MEASURED_PREFIX = "measured:"


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
        max_speakers: The largest speaker count the backend has been observed to *emit*
            — a structural ceiling, not a performance one. This is deliberately narrower
            than an earlier reading of this field: a backend's counting *accuracy*
            (does it report the right count?) degrades independently of whether it can
            structurally emit a large count at all, and folding both into one integer
            hid that distinction. See ``max_speakers_evidence`` for how a given value
            was established.

            ``None`` means *no structural ceiling was found* within whatever was tested
            — it does **not** mean unlimited, and it does **not** by itself distinguish
            "nobody has looked" from "we looked and found no cap" (``max_speakers_evidence``
            carries that distinction; see its docstring). Counts distinguishable
            *speakers*, not distinct ``speaker`` label values: the child-adult classifier
            emits three label values (``CHILD``, ``ADULT``, ``OVERLAP``) but declares
            ``max_speakers=2``, because ``OVERLAP`` marks two of the two known talkers
            speaking at once, not a third talker.

            The seed-17 speaker-ceiling probe (a 160-session, TTS-composed corpus swept
            over true speaker counts 1..8; see
            ``specs/20260809-112417-speaker-ceiling-probe/``) measured all six backends:
            Sortformer and child-adult both plateau at a fixed count (4 and 2) regardless
            of how many speakers are actually present — a real structural cap, confirming
            what their names/architecture already implied. Pyannote, VibeVoice, MOSS and
            DiariZen never plateaued across k=1..8 on that corpus; their predictions kept
            tracking (or overshooting) the true count instead of collapsing to one value,
            so they declare ``None`` here — "no structural ceiling observed", a measured
            claim, not a guess. Every number traces to one corpus and one seed, TTS-composed
            with no room acoustics or channel variation — the probe's own profile carries
            that caveat and this field points at it rather than restating it loosely; do not
            read a probe result as a guarantee about real recordings.
        max_speakers_evidence: How ``max_speakers`` was established, machine-readable
            rather than only prose in a comment. Exactly :data:`UNMEASURED` when nobody
            has run a probe against this backend — the same fact ``max_speakers=None``
            alone used to carry before this field existed. Otherwise a string starting
            ``"measured:"``, e.g. ``"measured: saturates at 4 on 20/20 k=8 sessions
            (probe seed-17)"`` when a real ceiling was found, or ``"measured: no
            saturation, emits up to 8 (probe seed-17)"`` when it was not. Checking
            ``evidence == UNMEASURED`` vs. ``evidence.startswith("measured:")`` is enough
            to recover which case applies without parsing the rest of the string — that
            two-state check is what makes this field machine-readable rather than a
            comment a reader has to trust. Rejected at construction (see
            ``__post_init__``) if it claims :data:`UNMEASURED` while ``max_speakers``
            holds a number: a number with no measurement behind it is exactly the
            unfitted literal this repo's conventions warn against.

            Counting *accuracy* — a related but separate fact from the structural bound
            this field is evidence for — is not carried here. It lives with the probe
            (``specs/20260809-112417-speaker-ceiling-probe/``) and, summarized, in
            ``model_registry.yaml``'s ``recommended_for`` for each backend.
        honors_speaker_hints: Whether ``num_speakers``/``min_speakers``/``max_speakers``
            passed to :func:`diarize_audios` do anything. Five of six ignore them.
    """

    populates_text: bool
    speaker_label_kind: SpeakerLabelKind
    labels_stable_across_files: bool
    max_speakers: Optional[int]
    max_speakers_evidence: str
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
                f"max_speakers must be >= 1 or None (no structural ceiling observed), got "
                f"{self.max_speakers!r}. None does not mean unlimited."
            )
        if self.max_speakers_evidence != UNMEASURED and not self.max_speakers_evidence.startswith(_MEASURED_PREFIX):
            raise ValueError(
                f"max_speakers_evidence must be {UNMEASURED!r} or start with {_MEASURED_PREFIX!r}, got "
                f"{self.max_speakers_evidence!r}. That two-state convention is what makes the field "
                "machine-readable instead of prose a reader has to trust."
            )
        if self.max_speakers_evidence == UNMEASURED and self.max_speakers is not None:
            raise ValueError(
                f"max_speakers={self.max_speakers!r} but max_speakers_evidence is {UNMEASURED!r} — a "
                "number with no measurement behind it is exactly the unfitted literal this repo's "
                "conventions warn against. Either supply the evidence that produced it or set "
                "max_speakers back to None."
            )
