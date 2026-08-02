"""Cross-signal physical support — a measured weight, not a declared one.

A signal's influence must come from quantities, not from its name in a config. The declared
``embedding_silhouette: derived`` gate was inferred from one recording and is demonstrably
wrong on another: on a 4.9 s group introduction the two "independent" diarizers merged four
named speakers into one, while the down-weighted clusterer recovered all five, aligned to the
names as they were spoken. Any gate calibrated on a single example will do this.

What *can* be measured without ground truth is whether a signal's claims are physically
supported. A diarizer placing a speaker where independent, non-diarizer evidence reports
silence or non-speech background has made a claim the audio does not carry, and that is a
quantity — no example needed.

Three properties are deliberate:

**Other diarizers are not evidence.** A bad diarizer can always say "one speaker", so
agreement among diarizers is not physical support; three models claiming a speaker in silence
are all wrong together, not mutually validated. Only signals that observe speech speech_presence
directly (frame posteriors, scene-classifier speech mass) count.

**Support measures precision, not recall.** It can see a claim the audio does not support; it
cannot see a speaker a signal *failed* to claim. A diarizer that under-counts is invisible to
this measure — that failure has to surface through the count posterior's disagreement instead.
Support must therefore never be read as a quality score.

**Absent evidence is not evidence of absence.** With no independent speech_presence signal in the
run, no signal is penalised. Otherwise a missing model would look like a wrong one.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT

__all__ = [
    "MIN_EVIDENCE_SPREAD",
    "informative_evidence",
    "evidence_signal_names",
    "SUPPORT_FLOOR",
    "signal_support",
    "CORROBORATION_POOLING",
    "EVIDENCE_WEIGHT_MAP",
    "presence_probability",
    "bucket_corroboration",
    "evidence_weight_from_corroboration",
]

SUPPORT_FLOOR = MIN_EVIDENCE_WEIGHT
"""Floor on support, so an unsupported signal is attenuated rather than silenced — it may be
the only source that noticed something. The number and its derivation live in
:data:`~senselab.audio.workflows.audio_analysis.floors.MIN_EVIDENCE_WEIGHT`."""


def _claims_speech(entry: Any) -> bool | None:  # noqa: ANN401 — vote entries are duck-typed
    """Whether this vote entry asserts speech in its bucket, or ``None`` if it says nothing."""
    if not isinstance(entry, Mapping):
        return None
    if "speaks" in entry:
        speaks = entry.get("speaks")
        return bool(speaks) if speaks is not None else None
    return None


def _evidence_value(entry: Any) -> float | None:  # noqa: ANN401
    """P(speech) an entry states outright, for entries that never took a direction."""
    if not isinstance(entry, Mapping):
        return None
    for field in ("p_speech", "p_voice", "value", "native_confidence"):
        v = entry.get(field)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return max(0.0, min(1.0, float(v)))
    return None


CORROBORATION_POOLING = "max"
"""How several independent evidence signals in one bucket become one number.

Named because ``max`` is a choice, and the same choice ``signal_support`` already makes: the
measure only ever *removes* weight, so it must discount a claim solely when *nothing* credible
supports it. One signal reporting speech is enough to make the claim supportable.
"""

EVIDENCE_WEIGHT_MAP = "identity_floored"
"""How a corroboration measurement becomes a weight, named so it can be replaced without re-running
a model. ``identity_floored`` is ``max(floor, corroboration)``.

Identity, because any other shape — a fixed multiplier, an exponent, a sigmoid — inserts a constant
nobody measured. The claim is "this source asserts speech here"; the independent evidence for that
same event is already a probability in ``[0, 1]``; that probability *is* how far the assertion
carries. The only free parameter left is the floor, which is named, shared and justified.
"""


def presence_probability(entry: Any) -> float | None:  # noqa: ANN401 — vote entries are duck-typed
    """``P(speech)`` carried by one presence vote, read in the direction the voter cast it.

    ``native_confidence`` is the voter's confidence in *its own* ``speaks`` direction (see
    ``speech_presence_link._directed``), so a voter reporting ``speaks=False`` at confidence 0.8 is
    asserting ``P(speech) = 0.2``. Reading that field raw turns every negative vote into a positive
    one — the difference between "no speech here" and "confident speech here", for exactly the
    voters whose job is to say no. It also silently made every directed voter fail
    :func:`informative_evidence`'s "willing to say no" screen, because a directed confidence is
    ``max(p, 1 − p)`` and can never fall below 0.5.

    Falls back to :func:`_evidence_value` for entries that state a probability outright and never
    took a direction.
    """
    if not isinstance(entry, Mapping):
        return None
    if "speaks" in entry and entry.get("speaks") is not None:
        raw = entry.get("native_confidence")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            confidence = max(0.0, min(1.0, float(raw)))
            return confidence if entry.get("speaks") else 1.0 - confidence
        return 1.0 if entry.get("speaks") else 0.0
    return _evidence_value(entry)


def bucket_corroboration(
    votes: Mapping[str, Any],
    *,
    evidence_signals: Sequence[str],
) -> float | None:
    """Strongest independent indication of speech in one bucket, or ``None`` if none was measured.

    This is :func:`signal_support`'s measure at the resolution of a single bucket, the same way
    ``rounds.regional_weights`` is ``reliability``'s global weight at regional resolution. It is
    leave-one-out by construction: ``evidence_signals`` only ever contains voters that observe
    speech presence *directly*, so a claimant can never be in its own evidence pool and attenuating
    the claimant cannot move the number that measured it.

    Args:
        votes: The bucket's active votes, ``{signal → payload}``.
        evidence_signals: Names admitted as independent presence evidence, derived per run via
            :func:`evidence_signal_names` + :func:`informative_evidence`.

    Returns:
        ``max`` over the evidence signals present in this bucket (see :data:`CORROBORATION_POOLING`),
        or ``None`` when no evidence signal reported a usable value here. ``None`` means *unmeasured*
        and must never be read as zero.
    """
    if not isinstance(votes, Mapping):
        return None
    values = [
        p for name in evidence_signals if name in votes and (p := presence_probability(votes.get(name))) is not None
    ]
    return max(values) if values else None


def evidence_weight_from_corroboration(
    corroboration: float,
    *,
    floor: float = MIN_EVIDENCE_WEIGHT,
) -> float:
    """``max(floor, clamp01(corroboration))`` — the weight *is* the measurement.

    Args:
        corroboration: Independent evidence for the claim, in ``[0, 1]``.
        floor: Minimum weight. See :data:`EVIDENCE_WEIGHT_MAP` for why the map above the floor is
            the identity.

    Returns:
        The weight to apply to the claimant's vote.

    Raises:
        ValueError: If ``floor`` is not strictly positive. A zero floor re-introduces erasure
            through the back door, because the presence fold drops voters at ``weight <= 0`` — a
            floor that can be configured to zero is not a floor.
    """
    if float(floor) <= 0.0:
        raise ValueError(
            f"evidence-weight floor must be > 0; got {floor}. A zero floor deletes the vote from "
            "aggregation instead of attenuating it, which is the erasure this map exists to avoid."
        )
    return max(float(floor), max(0.0, min(1.0, float(corroboration))))


def signal_support(
    speech_presence_buckets: Sequence[Mapping[str, Any]],
    *,
    evidence_signals: Sequence[str],
    floor: float = SUPPORT_FLOOR,
) -> dict[str, float]:
    """How far each signal's speech claims are corroborated by independent evidence.

    Args:
        speech_presence_buckets: Per-bucket speech_presence votes, as harvested. Each bucket's ``votes``
            maps signal name to its entry; claim entries carry ``speaks``, evidence entries
            carry a continuous P(speech).
        evidence_signals: Names treated as independent speech_presence evidence. These are scored
            against nobody and score nobody but the claimants.
        floor: Minimum support, so an unsupported signal is attenuated not erased.

    Returns:
        ``{signal → support in [floor, 1]}``, containing only signals that made at least one
        claim in a bucket where independent evidence was available. A signal absent from the
        result was not measured and should keep its default weight — that is different from
        being measured as unsupported.
    """
    evidence_names = set(evidence_signals)
    claimed: dict[str, list[float]] = {}

    for bucket in speech_presence_buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        votes = bucket.get("votes") or {}
        if not isinstance(votes, Mapping):
            continue

        # Pooled evidence for this bucket: the strongest independent indication of speech.
        # Max rather than mean because the measure only ever *removes* weight — it should
        # discount a claim solely when nothing credible supports it, and one signal
        # reporting speech is enough to make the claim supportable.
        available = [presence_probability(votes.get(name)) for name in evidence_names if name in votes]
        present = [v for v in available if v is not None]
        if not present:
            continue
        evidence = max(present)

        for name, entry in votes.items():
            if name in evidence_names:
                continue
            if _claims_speech(entry) is True:
                claimed.setdefault(str(name), []).append(evidence)

    return {name: max(float(floor), sum(values) / len(values)) for name, values in sorted(claimed.items()) if values}


# Voters that observe speech speech_presence *directly*. Diarizers and ASR are excluded on
# principle rather than by name: both infer speech_presence from a decision that already
# presupposes a speaker, so using them as corroboration would let a wrong presupposition
# validate itself.
_FRAME_VOTER_PREFIX = "frame_"
_ACOUSTIC_VOTER_PREFIX = "acoustic_"


def evidence_signal_names(speech_presence_buckets: Sequence[Mapping[str, Any]]) -> set[str]:
    """Derive the independent-evidence voter set from the harvest itself.

    Read structurally rather than from a configured list, which would drift the moment a
    voter is renamed or added — and a stale evidence list fails silently, by quietly
    measuring support against nothing.

    Frame posteriors (``frame_*``) and acoustic proxies (``acoustic_*``) are identified by
    the prefix the harvester already assigns them; scene classifiers are read from the
    ``__sources__`` bookkeeping entry, which already lists exactly which classifiers ran.
    """
    names: set[str] = set()
    for bucket in speech_presence_buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        votes = bucket.get("votes") or {}
        if not isinstance(votes, Mapping):
            continue
        for name, entry in votes.items():
            text = str(name)
            if text.startswith("__"):
                continue
            if text.startswith((_FRAME_VOTER_PREFIX, _ACOUSTIC_VOTER_PREFIX)) and isinstance(entry, Mapping):
                names.add(text)
        sources = votes.get("__sources__")
        if isinstance(sources, Mapping):
            for classifier in sources.get("classifiers") or []:
                if str(classifier) in votes:
                    names.add(str(classifier))
    return names


MIN_EVIDENCE_SPREAD = 0.15
"""Minimum range (max − min) before a signal can be considered informative at all."""

EVIDENCE_LOW_THRESHOLD = 0.20
"""Below this, an evidence signal is reporting "no speech here"."""

MIN_LOW_FRACTION = 0.02
"""An evidence signal must report "no speech" in at least this fraction of buckets.

This, not the range, is the criterion that matters. Support only ever *removes* weight, so it
runs entirely on negative evidence: a signal that never says "no speech" cannot withhold
support from anything, and including it makes the whole measure inert.

Measured over 697 buckets of a real recording, four of seven candidate evidence signals never
once fell below 0.20 — ``acoustic_hnr`` (median 0.500), ``acoustic_loudness`` (0.897),
``acoustic_spectral_activity`` (0.940) and ``ast`` (0.728). Pooled alongside genuine VAD they
held support at 0.996 for every claimant. The two purpose-built voice detectors behaved as
detectors should: ``frame_segmentation`` reported no speech in 503 of 697 buckets and
``frame_brouhaha_vad`` in 601.

Range alone would not have caught this: ``acoustic_loudness`` swings 0.500 and ``ast`` 0.242
while neither ever reaches a negative verdict. Willingness to say no is the property, and it
is measurable on the run with no per-model judgement.

Caveat on those figures: they were taken while the screen read ``native_confidence`` undirected,
which cannot fall below 0.5 for any voter that took a direction — so part of what they measured was
the reading, not the voter. The screen now uses :func:`presence_probability`. The thresholds are
unchanged because the *property* they test is unchanged, but the per-voter verdicts above must be
re-measured before they are cited again."""


def informative_evidence(
    speech_presence_buckets: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
    *,
    min_spread: float = MIN_EVIDENCE_SPREAD,
    low_threshold: float = EVIDENCE_LOW_THRESHOLD,
    min_low_fraction: float = MIN_LOW_FRACTION,
) -> set[str]:
    """Keep only evidence signals capable of withholding support.

    A signal reporting the same value everywhere cannot say *where* speech is, so admitting
    it into the pool guarantees every claim looks supported. Measured on a real run: support
    came out between 0.967 and 1.000 for every claimant — inert — because acoustic proxies
    reporting ~0.57 in silence as well as in speech were pooled by max alongside real VAD.

    Discrimination is a property of the signal, measurable on the run itself with no example
    and no per-model judgement, which is the same standard the weights themselves are held to.

    Args:
        speech_presence_buckets: Per-bucket speech_presence votes.
        candidates: Evidence signal names to screen.
        min_spread: Required max − min range.
        low_threshold: Value below which the signal is reporting "no speech".
        min_low_fraction: Fraction of buckets the signal must report as "no speech".

    Returns:
        The subset that varies enough to locate speech in time.
    """
    series: dict[str, list[float]] = {}
    for bucket in speech_presence_buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        votes = bucket.get("votes") or {}
        if not isinstance(votes, Mapping):
            continue
        for name in candidates:
            value = presence_probability(votes.get(name))
            if value is not None:
                series.setdefault(str(name), []).append(value)

    keep: set[str] = set()
    for name, values in series.items():
        if len(values) < 4:
            # Too few observations to judge variation; keep it rather than discard evidence
            # on the strength of a measurement that could not be made.
            keep.add(name)
            continue
        if max(values) - min(values) < float(min_spread):
            continue
        low = sum(1 for v in values if v < float(low_threshold))
        if low / len(values) >= float(min_low_fraction):
            keep.add(name)
    return keep
