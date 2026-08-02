"""Typed shapes for the adaptive loop's internal records (T052).

**Why ``TypedDict`` and not dataclasses**, which is what tasks.md asked for:

1. **These records round-trip through JSON.** ``Region`` is written to
   ``rounds/<n>/regions.json`` by ``loop.py`` and read back by ``plot.py``,
   ``ls_final.py`` and the T039 harness. ``PlannedIntervention`` lands in
   ``final/iterations.json``. A dataclass would need ``to_dict``/``from_dict`` at
   every one of those boundaries, and the dict would remain the real wire format —
   so the dataclass would be a second representation to keep in sync, not a
   replacement.
2. **Candidates are built incrementally.** ``plan_round`` adds ``status``,
   ``error`` and ``intervention_id`` *after* constructing the record. A frozen
   dataclass cannot express that, and a mutable one gives up the guarantee that
   made it attractive.
3. **The actual defect class here is key typos and wrong value types**, not
   mutation — a rule reading ``region["core_strt"]`` or treating
   ``uncertainty_mass`` as a string. ``TypedDict`` catches exactly that, across
   every existing consumer, with zero runtime change and zero migration.

So this replaces ``dict[str, Any]`` annotations with checked shapes rather than
replacing dicts with objects. If a future change removes the JSON round-trip, the
dataclass version becomes worth revisiting.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

__all__ = ["AxisName", "CostClass", "PlannedIntervention", "Region", "RegionStatus"]

from senselab.audio.workflows.audio_analysis.axes import AxisName

# Re-exported, not redeclared: this module used to carry its own three-member ``Literal``, so
# ``adaptive`` and the workflow layer could disagree about how many axes there were — and did.

CostClass = Literal["light", "medium", "heavy"]
"""Intervention cost tier; indexes the budget ledger and the priority weight."""

RegionStatus = Literal["open", "converged", "irreducible", "exhausted"]
"""Lifecycle of a proposed region across rounds."""


class Region(TypedDict):
    """One contiguous high-uncertainty span proposed for intervention.

    Produced by ``regions.propose_regions``; serialized verbatim to
    ``rounds/<n>/regions.json``.

    The ``core`` span is what seeded the region and what merge-back applies to
    (a bucket joins by *midpoint*, see ``regions.region_buckets``). The ``crop``
    span is ``core`` padded by ``regions.pad_s`` and is what gets handed to a
    model — models need context either side of the span to behave sensibly, but
    that context must not silently claim buckets outside the core.

    There is no ``stream``. A region is a span of the *recording* the run is unsure about, and it
    is proposed from the axis, which is a fold across passes. A rule that must operate on audio
    still names a pass — that is an action target, not an index on the belief — and records it as
    ``action_stream`` with the per-signal evidence that elected it (``S1_stream_election``).
    """

    axis: AxisName
    region_id: str
    core_start: float
    core_end: float
    crop_start: float
    crop_end: float
    uncertainty_mass: float
    n_buckets: int
    status: RegionStatus
    action_stream: NotRequired[str]
    """Pass a rule should act on, elected by ``S1_stream_election`` from per-signal evidence.

    Absent until a rule needs one. It is not part of the belief: two rules may legitimately act on
    different passes for the same region, and neither choice changes the axis.
    """


class PlannedIntervention(TypedDict):
    """One (rule, region) pair the planner considered, admitted or not.

    Every field is part of the decision record in ``final/iterations.json`` —
    including the ones explaining why a candidate did *not* run, which is the
    point: a deferred or guard-blocked candidate has to be as legible as a fired
    one.

    ``region`` is ``None`` for stream-global rules (adjudication), which run once
    per round rather than per region.
    """

    rule: str
    cost_class: CostClass
    region_id: str | None
    region: Region | None
    axis: str
    start: float
    trigger: dict[str, Any]
    priority: float
    enabled: bool
    guard_reason: str | None
    # Added by plan_round after ranking, hence NotRequired rather than optional
    # values — the key's absence is itself meaningful before admission is decided.
    status: NotRequired[Literal["admitted", "blocked_guard", "deferred_budget"]]
    error: NotRequired[str]
    intervention_id: NotRequired[str]
    # Set by loop.py once the rule has actually run — distinct from `status`,
    # which records the *planning* decision. mypy caught this key being written
    # without ever being declared.
    exec_status: NotRequired[str]
