"""The uncertainty axes, declared once (D-17).

There were three declarations and twenty-two literal tuples. ``types.UncertaintyAxis``,
``adaptive.types.AxisName`` and ``adaptive.belief.AXES`` each listed *three* axes; ``fuse``
produced a fourth, ``background_mask``, from the mask's own per-region confidence; and every
consumer that iterated ``("speech_presence", "speaker", "asr")`` therefore skipped it. The mask
axis was fused, written to ``estimates/``, drawn on the timeline — and then absent from region
proposal, from convergence marking and from the convergence report, so a run could converge on
"nothing left to do" while the fourth axis had never been asked.

**Any list of three axes is wrong.** That is the reason this module exists: one declaration, and
everything else derived from it, so a list cannot be short. What each consumer needs is a
*property* of an axis rather than its name — is it harvested from an ensemble, may an
uncorroborated speech claim discount it — so the properties are declared beside the name and the
subsets are computed.

Adding ``task`` is one edit: an :data:`AXES` entry. Nothing else in the pipeline enumerates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "ATTENUATED_AXES",
    "AXIS_GRIDS",
    "AXES",
    "AXIS_NAMES",
    "AXIS_PRIORITY",
    "CALIBRATED_AXES",
    "HARVESTED_AXES",
    "OVERLAP_INFORMED_AXES",
    "Axis",
    "AxisName",
    "axis",
]

AxisName = str
"""An axis is a plain ``str``, not a ``Literal``.

The same reason a perturbation is: the set is **open**. ``task`` is declared-but-punted, a fifth
may follow, and a type that enumerates the members is a promise the pipeline is not allowed to
keep — the three-member ``Literal`` is exactly what made ``background_mask`` unrepresentable in
the code that was supposed to act on it. What an axis *is* lives in :data:`AXES`, where the
properties travel with the name; a caller that wants only the harvested ones asks for
:data:`HARVESTED_AXES` rather than narrowing a type.
"""


@dataclass(frozen=True)
class Axis:
    """One uncertainty axis: the question it answers, and how the pipeline may treat it.

    Attributes:
        name: The axis id — the ``estimates/<name>.parquet`` filename and the ``axis`` column.
        question: What a high value on this axis means a reader does not know.
        harvested: Does an *ensemble* vote on it? ``background_mask`` does not: it is one derived
            judgement per region that reports its own confidence, so it has evidence without
            having voters. That distinction is what ``harvest`` needs and nothing else does.
        attenuable: May an uncorroborated speech claim discount this axis? Evidence that nobody
            spoke here says nothing about *which* speaker it was, so carrying the discount onto
            the speaker axis would be an unmeasured leap; and it says nothing about whether a
            region is free of target activity, which is the mask's question rather than speech's.
        overlap_informed: Does an overlapped-speech posterior say anything about this axis? Two
            people talking at once is evidence about *who* spoke and about *what was said*; it is
            not evidence about whether anyone spoke at all (they did), nor about whether a region
            is free of target activity (that is the mask's question).
        calibrated: Does this axis's aggregator take a calibration temperature? Only the axes
            whose sub-signals are combined through a softmax-like fold have one to take.
        grid: What this axis's rows are indexed by (D-24). ``"time_0.1s"`` for the axes whose
            evidence resamples or projects onto a time grid; ``"word"`` for ``asr``, whose evidence
            is a transcript. A transcript has no natural per-bucket value, so bucketing it is the
            ``REDUCE`` that :class:`~.shapes.GridRelation` names — performed on the *finest*
            evidence in the run, to match three coarser axes. Declared here rather than assumed,
            because a consumer that took time buckets for granted would silently mis-join ``asr``
            against the other three.
        rank: Tiebreak order when two regions carry the same uncertainty — lower comes first.
            A *judgement* about which axis a reader should look at first, so it is declared
            rather than derived from the order the axes happen to be written in.
        active: ``False`` for an axis the design names but does not yet produce. It still appears
            here, because "declared and not yet built" and "not thought of" are different states
            and only the first can be checked.
    """

    name: str
    question: str
    harvested: bool
    attenuable: bool
    overlap_informed: bool
    calibrated: bool
    rank: int
    grid: str = "time_0.1s"
    active: bool = True


AXES: Final[tuple[Axis, ...]] = (
    Axis(
        name="speech_presence",
        question="was there a speaker here at all?",
        harvested=True,
        attenuable=True,
        # Two people talking at once does not make it less certain that someone did.
        overlap_informed=False,
        calibrated=True,
        rank=2,
    ),
    Axis(
        name="speaker",
        question="was it the same speaker as before?",
        harvested=True,
        # Evidence that no one spoke here is silent about *which* speaker it was; discounting
        # this axis on it would be an unmeasured leap.
        attenuable=False,
        overlap_informed=True,
        calibrated=False,
        rank=1,
    ),
    Axis(
        name="asr",
        question="what was said?",
        harvested=True,
        attenuable=True,
        overlap_informed=True,
        calibrated=True,
        # First: a reader triaging a run wants the words before the speaker, and the speaker
        # before whether anyone spoke at all.
        rank=0,
        # One row per word, with an onset estimate and its variance. Every recognizer and every
        # aligner gives a word a different onset, and that spread *is* the disagreement this axis
        # measures — averaging it into a 0.1 s bucket before the axis sees it discards the
        # distinction between "a word all models agree on but time differently" and "a word models
        # disagree about", which call for different interventions.
        grid="word",
    ),
    Axis(
        name="background_mask",
        question="is this region free of target activity?",
        # No ensemble: the mask is one derived judgement per region. It reports how sure it is,
        # and ``1 - confidence`` is that judgement's uncertainty in the units the others use, so
        # it has evidence without having voters.
        harvested=False,
        # The mask's question is about target activity, not about speech, and an uncorroborated
        # speech claim is evidence about the latter only.
        attenuable=False,
        overlap_informed=False,
        calibrated=False,
        rank=3,
    ),
    Axis(
        name="task",
        question="was the requested task performed?",
        harvested=False,
        attenuable=False,
        overlap_informed=False,
        calibrated=False,
        rank=4,
        # Declared and not built. Naming it here is what makes "the fifth axis is missing" a
        # checkable statement rather than an omission nobody can see.
        active=False,
    ),
)
"""Every axis the design names, active or not. The one place the set is written down."""

AXIS_NAMES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active)
"""Every axis a run produces. What the loop iterates, what convergence is judged over."""

HARVESTED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.harvested)
"""Axes with a vote harvest — the ones an ensemble votes on (FR-001/FR-002)."""

ATTENUATED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.attenuable)
"""Axes an uncorroborated speech claim may be attenuated on.

Derived from ``attenuable``, so the justification for each exclusion sits on the axis it excludes
rather than in a comment beside a hand-written tuple that no longer had to agree with it.
"""

OVERLAP_INFORMED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.overlap_informed)
"""Axes an overlapped-speech posterior is evidence about."""

CALIBRATED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.calibrated)
"""Axes whose aggregator takes a calibration temperature."""

AXIS_PRIORITY: Final[dict[str, int]] = {a.name: a.rank for a in AXES if a.active}
"""``{axis → tiebreak rank}`` for ranking across axes; lower comes first.

Used by the disagreements index, where two regions at equal uncertainty need a stable order. It
was a hand-written dict of three, so the fourth axis fell to the ``.get(axis, 99)`` default and
sorted last by accident rather than by decision.
"""


def axis(name: str) -> Axis:
    """The declared axis called ``name``.

    Raises:
        KeyError: For an axis nothing declared — which is the point: a typo in an axis name used
            to produce an empty result set and read as "this axis had nothing to say".
    """
    for declared in AXES:
        if declared.name == name:
            return declared
    raise KeyError(f"no axis named {name!r}; declare it in axes.AXES first")


AXIS_GRIDS: Final[dict[str, str]] = {a.name: a.grid for a in AXES if a.active}
""" ``{axis → what its rows are indexed by}``.

Read this before joining two axes' rows. Three share ``time_0.1s`` and join trivially; ``asr`` is on
``word`` and joining it against them is a **projection**, which is a named derivative that records
which direction it went and what it did with a word spanning two buckets. Today that join is implicit
in the cross-axis ranking, which is how an asr finding and a presence finding could be ranked against
each other with nothing stating how their spans were reconciled.
"""
