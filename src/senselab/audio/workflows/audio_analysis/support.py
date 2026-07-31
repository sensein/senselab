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
are all wrong together, not mutually validated. Only signals that observe speech presence
directly (frame posteriors, scene-classifier speech mass) count.

**Support measures precision, not recall.** It can see a claim the audio does not support; it
cannot see a speaker a signal *failed* to claim. A diarizer that under-counts is invisible to
this measure — that failure has to surface through the count posterior's disagreement instead.
Support must therefore never be read as a quality score.

**Absent evidence is not evidence of absence.** With no independent presence signal in the
run, no signal is penalised. Otherwise a missing model would look like a wrong one.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "evidence_signal_names",
    "SUPPORT_FLOOR",
    "signal_support",
]

SUPPORT_FLOOR = 0.05
"""Floor on support, so an unsupported signal is attenuated rather than silenced — it may be
the only source that noticed something. Mirrors the perturbation-reliability floor."""


def _claims_speech(entry: Any) -> bool | None:  # noqa: ANN401 — vote entries are duck-typed
    """Whether this vote entry asserts speech in its bucket, or ``None`` if it says nothing."""
    if not isinstance(entry, Mapping):
        return None
    if "speaks" in entry:
        speaks = entry.get("speaks")
        return bool(speaks) if speaks is not None else None
    return None


def _evidence_value(entry: Any) -> float | None:  # noqa: ANN401
    """Continuous P(speech) carried by an evidence entry, if any."""
    if not isinstance(entry, Mapping):
        return None
    for field in ("p_speech", "p_voice", "value", "native_confidence"):
        v = entry.get(field)
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
    return None


def signal_support(
    presence_buckets: Sequence[Mapping[str, Any]],
    *,
    evidence_signals: Sequence[str],
    floor: float = SUPPORT_FLOOR,
) -> dict[str, float]:
    """How far each signal's speech claims are corroborated by independent evidence.

    Args:
        presence_buckets: Per-bucket presence votes, as harvested. Each bucket's ``votes``
            maps signal name to its entry; claim entries carry ``speaks``, evidence entries
            carry a continuous P(speech).
        evidence_signals: Names treated as independent presence evidence. These are scored
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

    for bucket in presence_buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        votes = bucket.get("votes") or {}
        if not isinstance(votes, Mapping):
            continue

        # Pooled evidence for this bucket: the strongest independent indication of speech.
        # Max rather than mean because the measure only ever *removes* weight — it should
        # discount a claim solely when nothing credible supports it, and one signal
        # reporting speech is enough to make the claim supportable.
        available = [_evidence_value(votes.get(name)) for name in evidence_names if name in votes]
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


# Voters that observe speech presence *directly*. Diarizers and ASR are excluded on
# principle rather than by name: both infer presence from a decision that already
# presupposes a speaker, so using them as corroboration would let a wrong presupposition
# validate itself.
_FRAME_VOTER_PREFIX = "frame_"
_ACOUSTIC_VOTER_PREFIX = "acoustic_"


def evidence_signal_names(presence_buckets: Sequence[Mapping[str, Any]]) -> set[str]:
    """Derive the independent-evidence voter set from the harvest itself.

    Read structurally rather than from a configured list, which would drift the moment a
    voter is renamed or added — and a stale evidence list fails silently, by quietly
    measuring support against nothing.

    Frame posteriors (``frame_*``) and acoustic proxies (``acoustic_*``) are identified by
    the prefix the harvester already assigns them; scene classifiers are read from the
    ``__sources__`` bookkeeping entry, which already lists exactly which classifiers ran.
    """
    names: set[str] = set()
    for bucket in presence_buckets or []:
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
