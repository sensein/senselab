"""The gates, their task groups, and the readings each one is read against.

A **gate** says what reading is good enough; an **instrument setting** says how to take a reading.
Every gate's bound lives in ``verdict.gates``, and a layer that names no value for a gate does not
apply it. Instrument settings stay in ``branch:`` and in PREPROCESS.

**A bound resolves most-specific-first: family, then group, then default**, and a family overrides
its group key by key rather than wholesale, so a family naming one gate inherits the group's
others. A task group alone is too coarse — ``SYLLABLE_TRAIN`` holds both ``diadochokinesis-pa``,
which asks for a count of ten, and ``diadochokinesis-v2-puh``, which asks for five seconds and
names no count. Which layer supplied a bound travels with it, so a verdict says whether a
recording was judged by a special case or by an inherited one.

VERDICT applies the conformance gates :data:`CONFORMANCE_GATES` names for the group; the remaining
gates are applied where the finding they produce is located, which is inside the reporting node.
A gate whose reading is absent answers by the reason the caller gives: a failure where nothing was
produced, :data:`NOT_APPLICABLE` where nothing could be asked, else :data:`UNDETERMINED`.

See ``specs/20260921-gates-in-verdict/design.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, UnknownConfigKey

UNDETERMINED: Literal["UNDETERMINED"] = "UNDETERMINED"
"""What a gate answers when it cannot be decided, and what a branch answers about a task."""

NOT_APPLICABLE: Literal["NOT_APPLICABLE"] = "NOT_APPLICABLE"
"""What a gate answers when the recording holds nothing it could be asked of: no speech for a speaker gate."""

NO_CARRIER = "no_carrier"
"""The owning branch looked for a production and found none: nothing was produced, which fails the gate."""

NO_SPEECH = "no_speech"
"""The recording carries no lexical word, so a gate about who spoke has nothing to read."""

NOT_COMPUTED = "absent_not_computed"
"""The branch had what it needed and wrote no reading: a defect, never a pass or a fail."""

INSTRUMENT_ABSENT = "instrument_absent"
"""A derivative the reading is taken off never reached the store."""

NULL_NO_OVERLAP = "null_no_overlap"
"""The reading was written as null because no diarized segment intersects the task extent."""

NULL_VALUE = "null_value"
"""The reading was written as null for a reason the branch did not name."""

BOUND_UNMEASURED = "bound_unmeasured"
"""The winning layer names the gate's bound as null: nobody has measured it."""

NO_OWNER_REPORT = "no_owner_report"
"""The branch owning the declared family left no in-family report, so no conformance gate was applied."""

GATE_REASONS = (
    NO_CARRIER,
    NO_SPEECH,
    NOT_COMPUTED,
    INSTRUMENT_ABSENT,
    NULL_NO_OVERLAP,
    NULL_VALUE,
    BOUND_UNMEASURED,
    NO_OWNER_REPORT,
)
"""Why a gate answered something other than a comparison of its reading with its bound."""

UNCOMPUTED_REASONS = frozenset({NOT_COMPUTED, INSTRUMENT_ABSENT})
"""The reasons that say a reading the task is judged on should exist and does not."""

FAIL = "fail"
"""A disposition: the absence is itself the answer, and the gate fails."""

UNDECIDED = "undetermined"
"""A disposition: the absence leaves the gate undecided."""

INAPPLICABLE = "not_applicable"
"""A disposition: the gate has nothing to be asked of."""

_ABSENT = object()
"""Sentinel separating a layer the packaged file does not spell from one it spells empty."""

GATE_SECTION = "verdict.gates"
"""The config section every gate's bound is read from, in three layers."""

DEFAULT_LAYER = "default"
"""The layer every task falls back to. It carries only what is genuinely universal."""

GROUP_LAYER = "by_group"
"""The layer keyed by :class:`Pattern`."""

FAMILY_LAYER = "by_family"
"""The layer keyed by declared family, which overrides its group key by key."""

LAYERS = (FAMILY_LAYER, GROUP_LAYER, DEFAULT_LAYER)
"""The layers, most specific first. A bound is taken from the first one that names its gate."""


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


REQUIRED_COUNT = "required_count"
"""The count finding a row's ``RequiredCount`` writes. Gateable once a tolerance is derived."""

TYPICAL_COUNT = "typical_count"
"""The count finding a row's ``TypicalCount`` writes. Never gateable."""

UNGATEABLE_READINGS = frozenset({TYPICAL_COUNT})
"""Readings no gate may be bound to. :func:`_gate_specs` refuses a table naming one, at import.

See ``specs/20260921-required-and-typical-counts/design.md``.
"""


def _gate_specs(specs: dict[str, GateSpec]) -> dict[str, GateSpec]:
    """The gate table, refused if any gate reads something nothing may be judged against.

    Args:
        specs: The table as written.

    Returns:
        The same table.

    Raises:
        ValueError: If a gate names a reading in :data:`UNGATEABLE_READINGS`.
    """
    refused = sorted(name for name, spec in specs.items() if spec.reading in UNGATEABLE_READINGS)
    if refused:
        raise ValueError(
            f"gates {refused} read one of {sorted(UNGATEABLE_READINGS)}, which no gate may be bound to: "
            "the reading is a measured central tendency nobody asked the participant for"
        )
    return specs


GATE_SPECS: dict[str, GateSpec] = _gate_specs(
    {
        "production_min_s": GateSpec("carrier_duration_s", AT_LEAST, float),
        "voiced_fraction_min": GateSpec("carrier_voiced_fraction", AT_LEAST, float),
        "f0_spread_max_semitones": GateSpec("carrier_f0_spread_semitones", AT_MOST, float),
        "continuity_min": GateSpec("carrier_continuity", AT_LEAST, float),
        "dominant_segment_min_fraction": GateSpec("sweep_dominant_fraction", AT_LEAST, float),
        "glide_extent_min_semitones": GateSpec("glide_extent_semitones", AT_LEAST, float),
        "declared_duration_min_fraction": GateSpec("production_declared_fraction", AT_LEAST, float),
        "expected_tokens_matched_min": GateSpec("expected_tokens_matched", AT_LEAST, int),
        "content_omission_fraction_max": GateSpec("expected_content_omitted_fraction", AT_MOST, float),
        "response_min_s": GateSpec("response_duration_s", AT_LEAST, float),
        "dominant_speaker_share_min": GateSpec("extent_dominant_speaker_share", AT_LEAST, float),
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
)
"""Every gate name, and what it reads. The one declaration of the set.

A ``reading`` of None marks a gate whose finding carries an extent — a rejected carrier, a located
deviation, a per-event count — and which is therefore applied where that extent is known, inside
the reporting node, against the same bound this table names.
"""

GATE_KEYS = tuple(GATE_SPECS)
"""The gate names, in declaration order. ``config_test`` pins them against the packaged section."""

CONFORMANCE_GATES: dict[Pattern, tuple[str, ...]] = {
    Pattern.SUSTAINED: ("production_min_s", "declared_duration_min_fraction"),
    Pattern.GLIDE: ("production_min_s", "dominant_segment_min_fraction", "glide_extent_min_semitones"),
    Pattern.ORDERED_TOKENS: ("expected_tokens_matched_min", "content_omission_fraction_max"),
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

FLAG_GATES: dict[str, str] = {
    "dominant_speaker_share_min": "another speaker holds part of the task extent",
    "voiced_fraction_min": "the production is voiced for too little of its carrier",
    "f0_spread_max_semitones": "the held production's pitch spreads more than the bound",
    "continuity_min": "the held production's spectrum is not continuous",
}
"""Gates whose failure is a flag ground of its own, and the ground each one names.

A flag gate is never a term in the task's conformance: it says something about the recording's
circumstances, not about whether the participant performed the instruction. VERDICT applies every
one the group configures, beside :data:`CONFORMANCE_GATES`, and a group that names none applies
none.
"""

_shared = sorted(set(FLAG_GATES) & {name for names in CONFORMANCE_GATES.values() for name in names})
if _shared:
    raise ValueError(f"gates {_shared} are both a conformance term and a flag ground; a gate is one or the other")


def conformance_gate_names(pattern: Pattern) -> tuple[str, ...]:
    """Which gates decide this task's conformance.

    The group is the whole of it: no expectation row reassigns its group's term.

    Args:
        pattern: The group the expectation row declares.

    Returns:
        The gate names, in the order they are recorded.
    """
    return CONFORMANCE_GATES[pattern]


@dataclass(frozen=True)
class GateBounds:
    """One task's gates, as the three layers resolved them.

    Attributes:
        group: The task group these bounds were resolved for.
        family: The declared family they were resolved for, or None for the out-of-family mode,
            which declares none.
        bounds: Gate name to its bound. A gate no layer names is absent; a gate the winning layer
            names as null is present with a None value.
        layers: Gate name to the layer that supplied it — one of :data:`LAYERS`.
    """

    group: Pattern
    family: str | None
    bounds: Mapping[str, Any]
    layers: Mapping[str, str]

    def bound(self, name: str) -> Any:  # noqa: ANN401 — each gate's own type
        """One gate's bound, from the most specific layer that names it.

        Args:
            name: The gate's name.

        Returns:
            The bound, or None both when no layer names the gate and when the winning layer names
            it null. :meth:`names` separates the two.

        Raises:
            KeyError: If the name is not a gate.
        """
        if name not in GATE_SPECS:
            raise KeyError(f"{name!r} is not a gate; check it against GATE_KEYS")
        return self.bounds.get(name)

    def names(self, name: str) -> bool:
        """Whether any layer configures the gate at all.

        Args:
            name: The gate's name.

        Returns:
            True when one of the three layers spells the gate, whatever its value.
        """
        return name in self.bounds

    def layer(self, name: str) -> str | None:
        """Which layer supplied a gate's bound.

        Args:
            name: The gate's name.

        Returns:
            One of :data:`LAYERS`, or None when no layer names the gate.
        """
        return self.layers.get(name)

    def keyed_under(self, name: str) -> str:
        """The key the winning layer configured this gate under.

        Args:
            name: The gate's name.

        Returns:
            The group's name for the group layer, the family for the family layer, and the empty
            string for the default layer, which is keyed by nothing.
        """
        layer = self.layers.get(name)
        if layer == FAMILY_LAYER:
            return str(self.family)
        return self.group.name if layer == GROUP_LAYER else ""

    def record(self) -> dict[str, Any]:
        """The resolved table, as the verdict records it.

        Returns:
            The group, the family, every bound, and the layer each came from — so a reader can
            tell a family-specific bound from an inherited one without opening the config.
        """
        return {
            "group": self.group.value,
            "family": self.family,
            "bounds": {name: self.bounds[name] for name in sorted(self.bounds)},
            "layers": {name: self.layers[name] for name in sorted(self.layers)},
        }


def _layer_entries(config: TriageConfig, path: str, what: str) -> dict[str, Any]:
    """One layer's gate mapping, checked against the gate vocabulary.

    Args:
        config: The resolved triage configuration.
        path: The layer's dotted path.
        what: What the layer is, for the message.

    Returns:
        Gate name to its typed bound. Empty for a layer that names nothing.

    Raises:
        UnknownConfigKey: If the packaged file spells no such layer.
        ValueError: If the layer spells a name that is not a gate.
    """
    table = config.get(path, _ABSENT)
    if table is _ABSENT:
        raise UnknownConfigKey(f"{path} is not a configured {what}")
    entries = dict(table or {})
    unknown = sorted(set(entries) - set(GATE_SPECS))
    if unknown:
        raise ValueError(f"{path} names {unknown}, which are not gates; check GATE_KEYS")
    return {name: _typed(name, value) for name, value in entries.items()}


def load_gate_bounds(config: TriageConfig, group: Pattern, family: str | None = None) -> GateBounds:
    """One task's gates, resolved family-first, then group, then default.

    A family entry overrides its group's entry **key by key**: a family naming one gate inherits
    every other gate its group names.

    Args:
        config: The resolved triage configuration.
        group: The task group, from the expectation row's ``Pattern``.
        family: The declared family, or None for the out-of-family mode, which declares none and
            therefore reads only the group and default layers.

    Returns:
        The resolved bounds, each carrying the layer it came from.

    Raises:
        UnknownConfigKey: If the packaged file spells no such group or layer.
        ValueError: If a layer spells a name that is not a gate.
    """
    resolved: dict[str, Any] = {}
    layers: dict[str, str] = {}
    by_family = config.get(f"{GATE_SECTION}.{FAMILY_LAYER}") or {}
    supplied = {
        # A family the layer does not name contributes nothing, and so does the out-of-family
        # mode, whose family is None and is therefore never one of the layer's keys.
        FAMILY_LAYER: (
            _layer_entries(config, f"{GATE_SECTION}.{FAMILY_LAYER}.{family}", "declared family")
            if family in by_family
            else {}
        ),
        GROUP_LAYER: _layer_entries(config, f"{GATE_SECTION}.{GROUP_LAYER}.{group.name}", "task group"),
        DEFAULT_LAYER: _layer_entries(config, f"{GATE_SECTION}.{DEFAULT_LAYER}", "layer"),
    }
    for layer in LAYERS:
        for name, value in supplied[layer].items():
            if name not in resolved:
                resolved[name] = value
                layers[name] = layer
    return GateBounds(group=group, family=family, bounds=resolved, layers=layers)


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
        layer: Which of :data:`LAYERS` supplied the bound.
        keyed_under: The key that layer configured it under — the group's name, the family, or
            the empty string for the default layer, which is keyed by nothing.
        op: :data:`AT_LEAST` or :data:`AT_MOST`.
        passed: True, False, :data:`UNDETERMINED` where it could not be decided, or
            :data:`NOT_APPLICABLE` where the recording holds nothing it could be asked of.
        reason: One of :data:`GATE_REASONS` wherever ``passed`` is not a comparison of a reading
            with a bound, else None.
    """

    name: str
    group: str
    reading: str
    value: Any
    bound: Any
    layer: str
    keyed_under: str
    op: str
    passed: bool | Literal["UNDETERMINED", "NOT_APPLICABLE"]
    reason: str | None = None

    @property
    def ground(self) -> str | None:
        """The flag ground this gate names, or None when it is a conformance term.

        Returns:
            The :data:`FLAG_GATES` entry for this gate's name, or None.
        """
        return FLAG_GATES.get(self.name)

    def record(self) -> dict[str, Any]:
        """This application, as the verdict records it.

        Returns:
            Every field, JSON-ready. ``ground`` is present only for a flag gate.
        """
        return {
            "gate": self.name,
            "group": self.group,
            "reading": self.reading,
            "value": self.value,
            "bound": self.bound,
            "layer": self.layer,
            "keyed_under": self.keyed_under,
            "op": self.op,
            "passed": self.passed,
            **({"reason": self.reason} if self.reason is not None else {}),
            **({"ground": self.ground} if self.ground is not None else {}),
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


def _answer(
    value: Any,  # noqa: ANN401 — each gate's own type
    bound: Any,  # noqa: ANN401 — each gate's own type
    op: str,
    absence: tuple[str, str] | None,
) -> tuple[bool | Literal["UNDETERMINED", "NOT_APPLICABLE"], str | None]:
    """One gate's answer and, where it is not a plain comparison, why.

    Args:
        value: The reading, or None when nothing carried one.
        bound: The bound, or None when nobody has measured it.
        op: :data:`AT_LEAST` or :data:`AT_MOST`.
        absence: ``(disposition, reason)`` for an absent reading, from the caller, which alone can
            say why the reading is absent. None reads as :data:`NOT_COMPUTED`.

    Returns:
        The answer and its reason.
    """
    if value is not None:
        if bound is None:
            return UNDETERMINED, BOUND_UNMEASURED
        return _passes(value, bound, op), None
    disposition, reason = absence if absence is not None else (UNDECIDED, NOT_COMPUTED)
    if disposition == INAPPLICABLE:
        return NOT_APPLICABLE, reason
    if disposition == FAIL:
        return False, reason
    return UNDETERMINED, reason


def apply_gates(
    names: Sequence[str],
    bounds: GateBounds,
    readings: Mapping[str, Any],
    absences: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[bool | Literal["UNDETERMINED"], list[AppliedGate]]:
    """Apply a group's conformance gates to what the reporting node read.

    Args:
        names: The gates to apply, from :func:`conformance_gate_names`.
        bounds: The group's configured bounds.
        readings: Reading name to the value the reporting node wrote, absent where it wrote none.
        absences: Reading name to ``(disposition, reason)`` for a reading that is absent, one of
            :data:`FAIL`, :data:`UNDECIDED` or :data:`INAPPLICABLE` with one of
            :data:`GATE_REASONS`.

    Returns:
        The conformance and one record per gate applied. A gate that is not applicable takes no
        part. Any failed gate decides it False; otherwise any undecided gate leaves it
        :data:`UNDETERMINED`, as does a group with no applicable gate at all.
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
        passed, reason = _answer(value, bound, spec.op, (absences or {}).get(spec.reading))
        applied.append(
            AppliedGate(
                name=name,
                group=bounds.group.value,
                reading=spec.reading,
                value=value,
                bound=bound,
                layer=str(bounds.layer(name)),
                keyed_under=bounds.keyed_under(name),
                op=spec.op,
                passed=passed,
                reason=reason,
            )
        )
    answered = [gate for gate in applied if gate.passed != NOT_APPLICABLE]
    if any(gate.passed is False for gate in answered):
        return False, applied
    if not answered or any(gate.passed == UNDETERMINED for gate in answered):
        return UNDETERMINED, applied
    return True, applied


def apply_flag_gates(
    bounds: GateBounds,
    readings: Mapping[str, Any],
    absences: Mapping[str, tuple[str, str]] | None = None,
    exempt: Sequence[str] = (),
) -> list[AppliedGate]:
    """Apply the flag gates this group configures, which decide no conformance.

    Args:
        bounds: The group's configured bounds.
        readings: Reading name to the value the reporting node wrote.
        absences: As :func:`apply_gates`.
        exempt: Flag gates the declared family's own instruction makes inapplicable; they are not
            applied at all.

    Returns:
        One record per flag gate applied, in :data:`FLAG_GATES` order. Empty when the group names
        none.
    """
    names = tuple(name for name in FLAG_GATES if name not in set(exempt))
    _, applied = apply_gates(names, bounds, readings, absences)
    return applied
