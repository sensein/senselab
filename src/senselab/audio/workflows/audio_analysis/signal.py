"""The L1 signal envelope: a measurement plus the provenance needed to interpret it.

L1 reports what a tool measured. It does not threshold, normalise, or rescale — every defect
found in this feature traced to L1 doing one of those, and each chosen reduction saturated
independently of the input (see ``specs/.../l1-signal-contract.md`` for the measured symptoms).

Two fields carry most of the weight.

**``units``.** Without it, L2 cannot know whether ``0.7`` is a probability, a dB value, or a
within-file rank — which is precisely how a percentile rank came to be aggregated as though it
were a probability. Declaring units also makes the value checkable: a probability outside
``[0, 1]`` is a caught error rather than a number that quietly propagates.

**``reduction``.** It records what L1 *did* do to the tool's raw output. A saturating reduction
then appears in the output instead of being findable only by rendering a figure and looking at
it, which is how six such defects were actually found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "UNITS",
    "SignalProvenance",
    "measurement",
]

UNITS: dict[str, str] = {
    "probability": "P(event) in [0, 1]; calibratable against ground truth.",
    "dB": "A ratio in decibels. Absolute — independent of input gain.",
    "dBFS": "Level relative to digital full scale; 0 dBFS is the maximum representable.",
    "LUFS": "BS.1770 gated loudness. Absolute: the same level reads the same across recordings.",
    "seconds": "A duration or an instant on the recording's timeline.",
    "hertz": "A frequency.",
    "count": "A non-negative integer tally.",
    "proportion": "A measured fraction of a whole in [0, 1]. Not a belief about anything.",
    "cosine_distance": "1 - cosine similarity, in [0, 2]; 0 is identical direction.",
    "arbitrary": "Uncalibrated model output. Comparable within one model and one recording only.",
}
"""Recognised units and what each means.

Free-text units cannot be checked, so a typo would silently disable the very validation the
declaration exists to provide. ``arbitrary`` is deliberately available and deliberately
unflattering: a signal whose output has no absolute meaning should have to say so, rather than
being rescaled to ``[0, 1]`` and thereby *looking* like a probability."""

_BOUNDED = {"probability": (0.0, 1.0), "proportion": (0.0, 1.0)}


@dataclass(frozen=True)
class SignalProvenance:
    """What a consumer needs in order to interpret one L1 signal.

    Frozen: a consumer must not be able to relabel a signal's units after the fact, because
    every downstream check would then be validating against the wrong claim.
    """

    signal: str
    model: str
    units: str
    revision: str | None = None
    resolution_s: float | None = None
    window_s: float | None = None
    reduction: str | None = None
    backend: str | None = None
    status: str = "ok"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise for the parquet sidecar and the run summary."""
        return {
            "signal": self.signal,
            "model": self.model,
            "revision": self.revision,
            "units": self.units,
            "resolution_s": self.resolution_s,
            "window_s": self.window_s,
            "reduction": self.reduction,
            "backend": self.backend,
            "status": self.status,
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


def measurement(
    value: float | None,
    *,
    units: str,
    signal: str,
    model: str,
    revision: str | None = None,
    resolution_s: float | None = None,
    window_s: float | None = None,
    reduction: str | None = None,
    backend: str | None = None,
    status: str = "ok",
    # Deliberately Any: ``extra`` carries model-specific provenance whose shape differs per
    # signal (a channel index, a label set, a venv name). Narrowing it would mean enumerating
    # every model's metadata here, which is the coupling the envelope exists to avoid.
    **extra: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Wrap one measured value with the provenance needed to interpret it.

    Args:
        value: The measured value in ``units``, or ``None`` when the signal did not produce one.
        units: One of :data:`UNITS`.
        signal: Signal name.
        model: Model or extractor that produced it.
        revision: Model revision, when pinned.
        resolution_s: How often the signal decides. Recorded separately from ``window_s``
            because they differ — openSMILE HNR steps every 10 ms over a 60 ms window, and a
            consumer assuming hop equals window would treat overlapping frames as independent.
        window_s: The span each decision covers.
        reduction: What L1 did to the tool's raw output, if anything.
        backend: Library or subprocess venv behind it.
        status: ``"ok"``, or a failure state.
        **extra: Anything model-specific worth recording.

    Returns:
        ``{"value": ..., "provenance": {...}}``.

    Raises:
        ValueError: If ``units`` is unrecognised, if a bounded unit's value is out of range, or
            if ``value`` is ``None`` while ``status`` is ``"ok"`` — "measured, and the answer is
            nothing" is not a state, it is a missing status.
    """
    if units not in UNITS:
        raise ValueError(f"unrecognised units {units!r}; expected one of {sorted(UNITS)}")
    if value is None and status == "ok":
        raise ValueError(f"{signal!r}: value is None but status is 'ok'; set a failure status instead")
    if value is not None and units in _BOUNDED:
        low, high = _BOUNDED[units]
        if not low <= float(value) <= high:
            raise ValueError(f"{signal!r}: {units} value {value} outside [{low}, {high}]")
    provenance = SignalProvenance(
        signal=signal,
        model=model,
        units=units,
        revision=revision,
        resolution_s=resolution_s,
        window_s=window_s,
        reduction=reduction,
        backend=backend,
        status=status,
        extra=dict(extra),
    )
    return {"value": None if value is None else float(value), "provenance": provenance.to_json()}
