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
from typing import Final, Literal

__all__ = [
    "ATTENUATED_AXES",
    "AXIS_GRIDS",
    "DEFAULT_TIME_GRID",
    "GridKind",
    "AXES",
    "AXIS_NAMES",
    "AXIS_PRIORITY",
    "CALIBRATED_AXES",
    "HARVESTED_AXES",
    "HARVEST_SOURCES",
    "HarvestSource",
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


GridKind = Literal["time", "word"]
"""What an axis's rows are indexed by: uniform time buckets, or one row per word."""

DEFAULT_TIME_GRID: Final[tuple[float, float]] = (0.1, 0.1)
"""``(win_length, hop_length)`` in seconds for every ``"time"``-gridded axis. **Configurable** — a
downstream need for finer or coarser buckets changes this, or the run's params override it.

**Window equals hop, so the buckets do not overlap**, and that is the point rather than a coincidence.
The run that motivated this used a 0.1 s window at a 0.02 s hop: adjacent rows shared 80% of their
audio, so 1070 rows were not 1070 independent measurements and nothing told a consumer so. A fine
*resolution* is what the question justifies; reporting five near-duplicate rows per window is not the
same thing, and the near-duplication was invisible in the output.

100 ms is sufficient for the downstream needs known today — speech and target-activity onsets are
resolved at it, and speaker turns and mask regions are much longer.
"""


@dataclass(frozen=True)
class HarvestSource:
    """Where one axis's per-bucket evidence sits on a ``PassHarvest``, and what that field holds.

    Declared because *three* readers need it — ``votes.link_pass`` (the L1→L2 link),
    ``fuse.write_final_uncertainty`` (the run's fold) and
    ``adaptive.belief.VoteStore.from_harvests`` (the loop's ingest) — and each wrote its own
    answer. Two agreed on four axes and the third enumerated three in a literal tuple, so
    ``background_mask`` was fused into 1070 buckets by the first two and rebuilt from one vote per
    mask *region* by the third: an axis with one bucket has nowhere to be uncertain, and it went
    from 1070 rows at round 0 to 1 by round 4 without anything reporting a loss.

    ``reliability._AXIS_SIGNALS`` still spells the same field names out for a *different* question
    — which key inside a bucket holds the per-signal mapping, for enumerating signal names — and is
    the one remaining copy. It already covers all four axes, so it is not part of this defect;
    folding it in here is a follow-up, not a fix.

    Attributes:
        field: The ``PassHarvest`` attribute carrying this axis's per-bucket rows.
        holds: ``"votes"`` when the field already carries one statement per source, or
            ``"measurements"`` when it carries L1 readings in native units that a *link* has to
            read under a policy first (``speech_presence``). Declared rather than inferred from
            the field name: the difference decides whether a reader may use the rows as they are,
            and guessing it is how an unlinked measurement reaches a fold.
    """

    field: str
    holds: Literal["votes", "measurements"]


@dataclass(frozen=True)
class Axis:
    """One uncertainty axis: the question it answers, and how the pipeline may treat it.

    Attributes:
        name: The axis id — the ``estimates/<name>.parquet`` filename and the ``axis`` column.
        question: What a high value on this axis means a reader does not know.
        harvested: Does an *ensemble* vote on it? Every active axis does. Kept as a declared property
            rather than assumed of all axes, because a future axis may genuinely have a single
            producer — and because it decides two things at once: whether ``harvest`` gathers votes,
            and whether the axis appears in the disagreements index.
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
        grid: What this axis's rows are indexed by (D-24). ``"time"`` for the axes whose evidence
            resamples or projects onto uniform buckets — they share :data:`DEFAULT_TIME_GRID`, so
            joining them needs no projection, which the mask's derivation from presence requires.
            ``"word"`` for ``asr``, whose evidence is a transcript: it has no natural per-bucket
            value, so bucketing it is the ``REDUCE`` that :class:`~.shapes.GridRelation` names,
            performed on the *finest* evidence in the run. Declared rather than assumed, because a
            consumer that took time buckets for granted would silently mis-join ``asr`` against the
            others.
        rank: Tiebreak order when two regions carry the same uncertainty — lower comes first.
            A *judgement* about which axis a reader should look at first, so it is declared
            rather than derived from the order the axes happen to be written in.
        active: ``False`` for an axis the design names but does not yet produce. It still appears
            here, because "declared and not yet built" and "not thought of" are different states
            and only the first can be checked.
        harvest: Where this axis's evidence lives on a ``PassHarvest`` (:class:`HarvestSource`).
            Required of every ``harvested`` axis and unread for the rest — an axis with
            ``harvested=True`` and no ``harvest`` is one the pipeline claims an ensemble votes on
            while no reader can find the votes, which is exactly the state ``background_mask`` was
            in for the adaptive loop. :data:`HARVEST_SOURCES` omits it, and every reader raises on
            it rather than folding an axis out of nothing.
    """

    name: str
    question: str
    harvested: bool
    attenuable: bool
    overlap_informed: bool
    calibrated: bool
    rank: int
    grid: GridKind = "time"
    active: bool = True
    harvest: HarvestSource | None = None


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
        # Measurements, not votes: the harvest holds ``covered_fraction`` / ``word_overlap_s`` /
        # ``frame_mean`` in their own units and every threshold that turns them into a statement is
        # L2's (``speech_presence_link``). A reader that took these for votes would fold unlinked
        # readings.
        harvest=HarvestSource(field="speech_presence_evidence", holds="measurements"),
    ),
    Axis(
        name="speaker",
        question="who is speaking here?",
        # Was "was it the same speaker as before?". That framing asked a *change* question at the
        # grid rate and validated it per (diar × embedder) pair against embeddings windowed ten
        # times coarser, so it read 0.666 on a clean two-speaker conversation whose count posterior
        # was 2 at 0.978 and whose per-speaker presence doubt averaged 0.168. The axis is composed
        # from ``attribution``'s three terms now — per-speaker presence, word location, target
        # activity. See ``speaker-axis-attribution-design.md``.
        harvested=True,
        # Evidence that no one spoke here is silent about *which* speaker it was; discounting
        # this axis on it would be an unmeasured leap.
        attenuable=False,
        overlap_informed=True,
        calibrated=False,
        rank=1,
        harvest=HarvestSource(field="speaker_votes", holds="votes"),
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
        harvest=HarvestSource(field="asr_votes", holds="votes"),
    ),
    Axis(
        name="background_mask",
        question="is this region free of target activity?",
        # An ensemble: VAD, ASR and the diarizers all bear on whether the target was active here,
        # so the mask's uncertainty is cross-source disagreement rather than one judgement's
        # self-reported confidence. It was ``False`` while a single derived judgement produced the
        # mask — which read as a property of the mask when it was a property of there being one
        # producer.
        #
        # **What each contributes depends on ``--task-type``**, which is why these are votes and
        # not a formula. In a speech task, VAD / words / speaker spans indicate target activity. In
        # a breathing task the target is the breath, speech detection is *silent* through it, and a
        # speech vote therefore indicates target **absence** — the case that made a mask built from
        # voice activity alone report the collected signal as a background source.
        #
        # ``True`` since the harvest exists: ``mask_harvest.harvest_background_mask_evidence`` fills
        # ``PassHarvest.background_mask_evidence``, and ``reliability._AXIS_SIGNALS`` names it. The flag
        # was held at ``False`` until both were in place — flipping it early puts the axis into
        # ``HARVESTED_AXES`` and every consumer then asks for evidence nothing produces.
        #
        # This is also what puts the mask into ``disagreements.json``. The index builds from
        # ``HARVESTED_AXES``, so the axis was fused, written to ``estimates/`` and drawn on the
        # timeline while being absent from the ranking that decides what a reader looks at.
        harvested=True,
        # The mask's question is about target activity, not about speech, and an uncorroborated
        # speech claim is evidence about the latter only.
        attenuable=False,
        overlap_informed=False,
        calibrated=False,
        rank=3,
        # The same declaration the other harvested axes make, and the one the adaptive store had no
        # way to read: it enumerated three axes in a literal tuple, so the loop rebuilt this axis
        # from one vote per mask *region* while L2 folded it per bucket. Declared here, all three
        # readers find it.
        harvest=HarvestSource(field="background_mask_evidence", holds="votes"),
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

HARVEST_SOURCES: Final[dict[str, HarvestSource]] = {
    a.name: a.harvest for a in AXES if a.active and a.harvested and a.harvest is not None
}
"""``{axis → where its evidence lives on a PassHarvest}`` — the one answer all three readers use.

Deliberately *not* the same set as :data:`HARVESTED_AXES`: this one is what a reader can actually
find, so an axis declared ``harvested`` without a :class:`HarvestSource` is absent here and raises
at the reader instead of quietly folding to nothing. The two sets agreeing is a checkable property
(``axes_test``) rather than an assumption, and it is the property that failed — the flag said
harvested, the loop's ingest enumerated three axes, and nothing compared them.
"""

COUPLING_IS_A_GATE: Final[frozenset[str]] = frozenset({"speaker"})
"""Axes for which another axis's value bounds *where the question applies*, never answers it.

The speaker axis asks **who** is speaking. ``speech_presence``, ``asr`` and ``background_mask``
answer *whether there is anything here to attribute* — a different question, and one whose answer
cannot be evidence about identity. Coupling them in as weighted voters was measured on a clean
two-speaker conversation: round 0 read 0.0487 from the diarizers' own agreement, round 1 read 0.1601
once ``axis::asr`` / ``axis::speech_presence`` / ``axis::background_mask`` were injected, and round 2
read 0.3601. The rise is entirely those three, and none of them had looked at a speaker.

It is the same error the ``asr_location`` voter was: a quantity that legitimately *bounds* where
attribution is a live question, used instead as an answer to it. Word timing became
``attribution.word_coverage``, a gate; these become nothing at all, because the gating work is
already done — and done better — at harvest time:

- ``attribution.word_coverage`` nulls a bucket no recognized word reaches.
- ``attribution.target_activity_doubt`` nulls a bucket the mask confidently calls ``target_free``.

Both read a **claim** (a word is there; the mask's ``state``). What a coupled row carries is the
other axis's *doubt*, which is not a presence claim at all: ``speech_presence`` doubt near zero means
that axis is confident, not that speech is present. Gating on it would read "confidently silent" and
"confidently speaking" as the same value. So a presence-probability gate, if one is ever wanted, has
to read ``speech_presence_confidence`` / ``p_voice`` — and would need a threshold nobody has measured
yet, which is why it is not here.

Axes absent from this set are unaffected: ``speech_presence``, ``asr`` and ``background_mask`` still
couple to each other as voters, where the coupled quantity and the receiving question are the same
kind of thing.
"""

ATTENUATED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.attenuable)
"""Axes an uncorroborated speech claim may be attenuated on.

Derived from ``attenuable``, so the justification for each exclusion sits on the axis it excludes
rather than in a comment beside a hand-written tuple that no longer had to agree with it.
"""

OVERLAP_INFORMED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.overlap_informed)
"""Axes an overlapped-speech posterior is evidence about."""

CALIBRATED_AXES: Final[tuple[str, ...]] = tuple(a.name for a in AXES if a.active and a.calibrated)
"""Axes whose fold *would* take a calibration temperature.

Declared and unconsumed today, and named as such rather than left to look wired: the profile's
``temperature`` block reached ``aggregate.aggregate_asr`` and
``aggregate.aggregate_speech_presence``, both of which had no production caller and are deleted.
``fuse.fuse_axis`` is the one fold and takes no temperature. See ``calibration.py``.
"""

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


AXIS_GRIDS: Final[dict[str, GridKind]] = {a.name: a.grid for a in AXES if a.active}
""" ``{axis → what its rows are indexed by}``.

Read this before joining two axes' rows. Three share ``"time"`` at :data:`DEFAULT_TIME_GRID` and join
trivially — non-overlapping and identical, so row *i* of one is row *i* of another. ``asr`` is on
``word`` and joining it against them is a **projection**, which is a named derivative that records
which direction it went and what it did with a word spanning two buckets. Today that join is implicit
in the cross-axis ranking, which is how an asr finding and a presence finding could be ranked against
each other with nothing stating how their spans were reconciled.
"""
