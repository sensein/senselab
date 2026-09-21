"""The gates, their task groups, and the readings each one is read against.

A **gate** says what reading is good enough; an **instrument setting** says how to take a reading.
Every gate's bound lives in ``verdict.gates``, keyed by the :class:`Pattern` the task's expectation
row declares, and a group that names no value for a gate does not apply it. Instrument settings
stay in ``branch:`` and in PREPROCESS.

VERDICT applies the conformance gates :data:`CONFORMANCE_GATES` names for the group; the remaining
gates are applied where the finding they produce is located, which is inside the reporting node.
A gate whose reading is absent yields :data:`UNDETERMINED`.

See ``specs/20260921-gates-in-verdict/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, UnknownConfigKey

UNDETERMINED: Literal["UNDETERMINED"] = "UNDETERMINED"
"""What a gate answers when its reading is absent, and what a branch answers about a task."""

_ABSENT = object()
"""Sentinel separating a group the packaged file does not spell from one it spells empty."""

GATE_SECTION = "verdict.gates"
"""The config section every gate's bound is read from, one sub-mapping per task group."""


class Pattern(Enum):
    """The kinds of expected pattern an instruction can ask for, and the gate table's key."""

    ORDERED_TOKENS = "ordered_tokens"
    FREE_RESPONSE = "free_response"
    ITEM_LIST = "item_list"
    SUSTAINED = "sustained"
    GLIDE = "glide"
    EFFORT = "effort"
    PER_SENTENCE = "per_sentence"
    EVENT_SERIES = "event_series"
    EVENT_ALTERNATION = "event_alternation"
    SOUND_COVERAGE = "sound_coverage"
    SYLLABLE_TRAIN = "syllable_train"
    SYLLABLE_SEQUENCE = "syllable_sequence"


AT_LEAST = "at_least"
"""A reading at or above the bound passes."""

AT_MOST = "at_most"
"""A reading at or below the bound passes."""


@dataclass(frozen=True)
class GateSpec:
    """One gate: what it reads, which way it compares, and how its bound is typed.

    Attributes:
        reading: The ``measure`` finding whose ``value`` VERDICT compares, or None for a gate whose
            finding is located and is therefore applied by the reporting node.
        op: :data:`AT_LEAST` or :data:`AT_MOST`.
        read_as: How the bound is typed when it is read from the config.
    """

    reading: str | None
    op: str
    read_as: Callable[[Any], Any]


GATE_SPECS: dict[str, GateSpec] = {
    "production_min_s": GateSpec("carrier_duration_s", AT_LEAST, float),
    "voiced_fraction_min": GateSpec("carrier_voiced_fraction", AT_LEAST, float),
    "f0_spread_max_semitones": GateSpec("carrier_f0_spread_semitones", AT_MOST, float),
    "continuity_min": GateSpec("carrier_continuity", AT_LEAST, float),
    "dominant_segment_min_fraction": GateSpec("sweep_dominant_fraction", AT_LEAST, float),
    "monotone_tolerance_semitones": GateSpec("sweep_monotone_reversal_semitones", AT_MOST, float),
    "expected_tokens_matched_min": GateSpec("expected_tokens_matched", AT_LEAST, int),
    "omissions_max": GateSpec("expected_tokens_omitted", AT_MOST, int),
    "response_min_s": GateSpec("response_duration_s", AT_LEAST, float),
    "coverage_min": GateSpec("source_content_coverage", AT_LEAST, float),
    "items_min": GateSpec("items_produced", AT_LEAST, int),
    "events_min": GateSpec("airway_events_found", AT_LEAST, int),
    "repetitions_min": GateSpec("ddk_repetitions_found", AT_LEAST, int),
    "repeat_overlap_min": GateSpec(None, AT_LEAST, float),
    "echo_overlap_max": GateSpec(None, AT_MOST, float),
    "verbatim_overlap_max": GateSpec(None, AT_MOST, float),
    "gap_off_task_min_s": GateSpec(None, AT_LEAST, float),
    "interval_max_s": GateSpec(None, AT_MOST, float),
    "score_min": GateSpec(None, AT_LEAST, float),
    "train_min_s": GateSpec(None, AT_LEAST, float),
    "rate_prominence_min": GateSpec(None, AT_LEAST, float),
}
"""Every gate name, and what it reads. The one declaration of the set.

A ``reading`` of None marks a gate whose finding carries an extent — a rejected carrier, a located
deviation, a per-event count — and which is therefore applied where that extent is known, inside
the reporting node, against the same bound this table names.
"""

GATE_KEYS = tuple(GATE_SPECS)
"""The gate names, in declaration order. ``config_test`` pins them against the packaged section."""

CONFORMANCE_GATES: dict[Pattern, tuple[str, ...]] = {
    Pattern.SUSTAINED: ("production_min_s", "voiced_fraction_min", "f0_spread_max_semitones", "continuity_min"),
    Pattern.GLIDE: (
        "production_min_s",
        "voiced_fraction_min",
        "monotone_tolerance_semitones",
        "dominant_segment_min_fraction",
    ),
    Pattern.ORDERED_TOKENS: ("expected_tokens_matched_min", "omissions_max"),
    Pattern.FREE_RESPONSE: ("response_min_s",),
    Pattern.ITEM_LIST: ("items_min",),
    Pattern.EVENT_SERIES: ("events_min",),
    Pattern.EVENT_ALTERNATION: ("events_min",),
    Pattern.SOUND_COVERAGE: (),
    Pattern.SYLLABLE_TRAIN: ("repetitions_min",),
    Pattern.SYLLABLE_SEQUENCE: ("repetitions_min",),
    Pattern.PER_SENTENCE: (),
    Pattern.EFFORT: (),
}
"""Which of the group's gates VERDICT reads as the task's conformance.

A group with none answers :data:`UNDETERMINED`, which is what a group whose instruction states no
conformable expectation should answer.
"""

RECALL_CONFORMANCE_GATES: tuple[str, ...] = ("coverage_min",)
"""The conformance gates of a ``FREE_RESPONSE`` row whose anti-pattern is ``verbatim_source``.

A recall's conformance term is how much of the source it realised, not how long it ran.
"""

VERBATIM_SOURCE = "verbatim_source"
"""The anti-pattern that reassigns a free response's conformance term to source coverage."""


def conformance_gate_names(pattern: Pattern, *, anti_pattern: str | None = None) -> tuple[str, ...]:
    """Which gates decide this task's conformance.

    Args:
        pattern: The group the expectation row declares.
        anti_pattern: The row's anti-pattern, when it declares one.

    Returns:
        The gate names, in the order they are recorded.
    """
    if pattern is Pattern.FREE_RESPONSE and anti_pattern == VERBATIM_SOURCE:
        return RECALL_CONFORMANCE_GATES
    return CONFORMANCE_GATES[pattern]


@dataclass(frozen=True)
class GateBounds:
    """One task group's gates, as the configuration resolved them.

    Attributes:
        group: The group these bounds were read for.
        bounds: Gate name to its bound. A gate the group does not name is absent; a gate the group
            names as null is present with a None value.
    """

    group: Pattern
    bounds: Mapping[str, Any]

    def bound(self, name: str) -> Any:  # noqa: ANN401 — each gate's own type
        """One gate's bound.

        Args:
            name: The gate's name.

        Returns:
            The bound, or None both when this group does not name the gate and when it names it
            null. :meth:`names` separates the two.

        Raises:
            KeyError: If the name is not a gate.
        """
        if name not in GATE_SPECS:
            raise KeyError(f"{name!r} is not a gate; check it against GATE_KEYS")
        return self.bounds.get(name)

    def names(self, name: str) -> bool:
        """Whether this group configures the gate at all.

        Args:
            name: The gate's name.

        Returns:
            True when the group's mapping spells the gate, whatever its value.
        """
        return name in self.bounds

    def record(self) -> dict[str, Any]:
        """This group's gate table, as the verdict records it.

        Returns:
            The group's name and every bound it carries.
        """
        return {"group": self.group.value, "bounds": {name: self.bounds[name] for name in sorted(self.bounds)}}


def load_gate_bounds(config: TriageConfig, group: Pattern) -> GateBounds:
    """One task group's gates, from the configuration.

    Args:
        config: The resolved triage configuration.
        group: The group to read.

    Returns:
        The bounds. A gate the group does not spell is absent from them.

    Raises:
        UnknownConfigKey: If the packaged file spells no such group.
        ValueError: If the group spells a name that is not a gate.
    """
    table = config.get(f"{GATE_SECTION}.{group.name}", _ABSENT)
    if table is _ABSENT:
        raise UnknownConfigKey(f"{GATE_SECTION}.{group.name} is not a configured task group")
    entries = dict(table or {})
    unknown = sorted(set(entries) - set(GATE_SPECS))
    if unknown:
        raise ValueError(f"{GATE_SECTION}.{group.name} names {unknown}, which are not gates; check GATE_KEYS")
    return GateBounds(group=group, bounds={name: _typed(name, value) for name, value in entries.items()})


def _typed(name: str, value: Any) -> Any:  # noqa: ANN401 — each gate's own type
    """One bound in the type its gate declares.

    Args:
        name: The gate's name.
        value: The packaged or overridden value.

    Returns:
        The value, typed, or None when nobody has measured it.
    """
    return None if value is None else GATE_SPECS[name].read_as(value)


@dataclass(frozen=True)
class AppliedGate:
    """One gate VERDICT applied, and what it made of the reading.

    Attributes:
        name: The gate's name.
        group: The task group it was configured under.
        reading: The measurement's name.
        value: What the reporting node read, or None when nothing carried the reading.
        bound: What it was read against, or None when nobody has measured the bound.
        op: :data:`AT_LEAST` or :data:`AT_MOST`.
        passed: True, False, or :data:`UNDETERMINED` where either side was absent.
    """

    name: str
    group: str
    reading: str
    value: Any
    bound: Any
    op: str
    passed: bool | Literal["UNDETERMINED"]

    def record(self) -> dict[str, Any]:
        """This application, as the verdict records it.

        Returns:
            Every field, JSON-ready.
        """
        return {
            "gate": self.name,
            "group": self.group,
            "reading": self.reading,
            "value": self.value,
            "bound": self.bound,
            "op": self.op,
            "passed": self.passed,
        }


def _passes(value: Any, bound: Any, op: str) -> bool:  # noqa: ANN401 — each gate's own type
    """Whether one reading clears one bound.

    Args:
        value: The reading.
        bound: The bound.
        op: :data:`AT_LEAST` or :data:`AT_MOST`.

    Returns:
        The comparison's result.
    """
    return float(value) >= float(bound) if op == AT_LEAST else float(value) <= float(bound)


def apply_gates(
    names: Sequence[str], bounds: GateBounds, readings: Mapping[str, Any]
) -> tuple[bool | Literal["UNDETERMINED"], list[AppliedGate]]:
    """Apply a group's conformance gates to what the reporting node read.

    Args:
        names: The gates to apply, from :func:`conformance_gate_names`.
        bounds: The group's configured bounds.
        readings: Reading name to the value the reporting node wrote, absent where it wrote none.

    Returns:
        The conformance and one record per gate applied. :data:`UNDETERMINED` when no gate was
        applied at all, and whenever any applied gate could not be answered — a reading nobody
        took, or a bound nobody has measured, is never a failure.
    """
    applied: list[AppliedGate] = []
    for name in names:
        if not bounds.names(name):
            continue
        spec = GATE_SPECS[name]
        if spec.reading is None:
            raise ValueError(f"gate {name!r} carries no reading and cannot decide a conformance")
        bound = bounds.bound(name)
        value = readings.get(spec.reading)
        passed: bool | Literal["UNDETERMINED"] = (
            UNDETERMINED if value is None or bound is None else _passes(value, bound, spec.op)
        )
        applied.append(
            AppliedGate(
                name=name,
                group=bounds.group.value,
                reading=spec.reading,
                value=value,
                bound=bound,
                op=spec.op,
                passed=passed,
            )
        )
    if not applied or any(gate.passed == UNDETERMINED for gate in applied):
        return UNDETERMINED, applied
    return all(gate.passed is True for gate in applied), applied
