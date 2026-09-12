"""Scoring every candidate detector against a stated reference standard, and naming the misses.

:data:`REFERENCE_STANDARDS` holds the standards; each carries ``is_proxy``, which is False only for
``agreed_asr``. Disagreements are enumerated per family by :func:`disagreements` rather than
reduced to a rate. What the standards are worth is written up in
``specs/20260910-taxonomy-routing-evidence/measurements.md``.

Two criteria are reported over the same sweep and neither replaces the other. Youden's J weights a
missed recording and a spuriously routed one equally; :func:`recall_at_budgets` weights them the way
a router does, reporting the recall reachable while the false-positive rate over the negatives stays
inside a stated over-routing budget. A reference standard may name families that its positive set
excludes by construction although their content is what the branch is for; those are held out of the
population rather than counted as errors. Both are derived in
``specs/20260911-recall-first-thresholds/design.md``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from senselab.audio.workflows.triage.routing_analysis.detectors import (
    CONSOLIDATION_FLOOR,
    DETECTORS,
    Detector,
    detector_value,
    sweep_points,
)
from senselab.audio.workflows.triage.routing_analysis.families import DECLARED_KIND, SYLLABLE_REPETITION
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.labels import CLASSIFIERS, TRACKED_LABELS, peak_key

DISAGREEMENT_CAP = 200
"""How many individual disagreements are listed per family and direction."""

PRIMARY_DETECTORS: tuple[str, ...] = (
    "speech.words_agreement",
    "speech.words_lexical",
    "speech.ast_peak.plain",
    "airway.residual_energy_fraction",
    "airway.residual_energy_fraction+hear>=0.2",
    "airway.yamnet_peak.plain",
    "airway.hear_peak.plain",
    "voice.longest_amplitude_span",
    "voice.yamnet_peak.plain",
    "voice.yamnet_chant_peak.plain",
    "voice.yamnet_singing_union.plain",
    "cough.amplitude_peak_over_floor_db_max",
    "cough.yamnet_cough_minus_breath.plain",
    "cough.squim_si_sdr_iqr",
    "glide.yamnet_singing_union.plain",
    "glide.squim_stoi_max",
    "glide.amplitude_duty_fraction",
)
"""The detectors also swept within each task family, where the imbalance is visible."""

LABEL_PREVALENCE_STREAMS: tuple[str, ...] = ("plain", "residual")
"""The streams the per-family label prevalence is reported over."""

LABEL_PREVALENCE_FLOOR = 0.2
"""The peak a label must reach to count as firing in the per-family label prevalence."""


@dataclass(frozen=True)
class Confusion:
    """One 2x2 table.

    Attributes:
        tp: Detector fired and the reference is positive.
        fp: Detector fired and the reference is negative.
        tn: Detector did not fire and the reference is negative.
        fn: Detector did not fire and the reference is positive.
    """

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def sensitivity(self) -> float | None:
        """Fraction of reference positives the detector fires on, or None with no positives."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else None

    @property
    def specificity(self) -> float | None:
        """Fraction of reference negatives the detector is silent on, or None with no negatives."""
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else None

    @property
    def false_positive_rate(self) -> float | None:
        """Fraction of reference negatives the detector fires on, or None with no negatives."""
        negatives = self.tn + self.fp
        return self.fp / negatives if negatives else None

    @property
    def youden(self) -> float | None:
        """Sensitivity plus specificity minus one, or None when either is undefined."""
        if self.sensitivity is None or self.specificity is None:
            return None
        return self.sensitivity + self.specificity - 1.0

    def as_json(self) -> dict[str, Any]:
        """This table and its derived rates, for the machine-readable output.

        Returns:
            The four counts plus sensitivity, specificity, the false-positive rate and Youden's J.
        """
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "false_positive_rate": self.false_positive_rate,
            "youden": self.youden,
        }


@dataclass(frozen=True)
class ReferenceStandard:
    """What a detector is scored against.

    Attributes:
        name: The standard's id.
        kind: The branch the standard is about.
        is_proxy: Whether it is a declaration read as a prior rather than an observation.
        description: What positive means, in one sentence.
        predicate: Whether one recording is a reference positive.
        population: Which recordings are scored at all, or None for the whole corpus. A standard
            that separates two declared families scores only within their union.
        population_description: What the restricted population is, in one phrase.
        excluded_by_construction: Families whose content is what the branch is for but which the
            positive set leaves out by construction. They are neither positives nor negatives:
            counting them as negatives charges the detector for firing correctly. Every function
            here drops them from the population rather than moving them across it.
        exclusion_description: What the excluded families are, in one phrase.
    """

    name: str
    kind: str
    is_proxy: bool
    description: str
    predicate: Callable[[RecordingFeatures], bool]
    population: Callable[[RecordingFeatures], bool] | None = None
    population_description: str = "the whole corpus"
    excluded_by_construction: frozenset[str] = frozenset()
    exclusion_description: str = "nothing"

    def excludes(self, record: RecordingFeatures) -> bool:
        """Whether one recording's family is excluded from this standard by construction.

        Args:
            record: The recording.

        Returns:
            True when the family is in :attr:`excluded_by_construction`.
        """
        return record.family in self.excluded_by_construction

    def scored_population(self) -> str:
        """The population phrase, naming the construction exclusion when there is one.

        Returns:
            The population description, extended with what is held out of it.
        """
        if not self.excluded_by_construction:
            return self.population_description
        return f"{self.population_description}, {self.exclusion_description} held out"


def _declared(kind: str) -> Callable[[RecordingFeatures], bool]:
    """A predicate reading the task declaration for one kind.

    Args:
        kind: ``speech``, ``airway`` or ``voice``.

    Returns:
        A predicate true when the recording's family is one that kind's instructions elicit.
    """

    def predicate(features: RecordingFeatures) -> bool:
        return features.family in DECLARED_KIND[kind]

    return predicate


REFERENCE_STANDARDS: tuple[ReferenceStandard, ...] = (
    ReferenceStandard(
        name="agreed_asr",
        kind="speech",
        is_proxy=False,
        description="at least one live consensus word whose outcome is agreement",
        predicate=lambda features: features.words.get("agreement", 0) >= 1,
    ),
    ReferenceStandard(
        name="declared_speech",
        kind="speech",
        is_proxy=True,
        description="the declared task family asks the participant to say words",
        predicate=_declared("speech"),
    ),
    ReferenceStandard(
        name="declared_lexical_speech",
        kind="speech",
        is_proxy=True,
        description="the declared task family asks for words, excluding syllable-repetition tasks",
        predicate=_declared("lexical_speech"),
        excluded_by_construction=SYLLABLE_REPETITION,
        exclusion_description="the syllable-repetition families",
    ),
    ReferenceStandard(
        name="declared_airway",
        kind="airway",
        is_proxy=True,
        description="the declared task family asks for a breath, cough or throat manoeuvre",
        predicate=_declared("airway"),
    ),
    ReferenceStandard(
        name="declared_voice",
        kind="voice",
        is_proxy=True,
        description="the declared task family asks for sustained or glided phonation",
        predicate=_declared("voice"),
    ),
    ReferenceStandard(
        name="declared_cough_vs_breath",
        kind="cough",
        is_proxy=True,
        description="the declared task family asks for a cough rather than for breathing",
        predicate=_declared("cough"),
        population=lambda features: features.family in DECLARED_KIND["airway"],
        population_description="the declared airway families only",
    ),
    ReferenceStandard(
        name="declared_glide",
        kind="glide",
        is_proxy=True,
        description="the declared task family asks for a continuous pitch sweep",
        predicate=_declared("glide"),
    ),
    ReferenceStandard(
        name="declared_glide_within_voice",
        kind="glide",
        is_proxy=True,
        description="the declared task family asks for a pitch sweep rather than a held vowel",
        predicate=_declared("glide"),
        population=lambda features: features.family in DECLARED_KIND["voice"],
        population_description="the declared voice families only",
    ),
)
"""The reference standards every detector of the matching kind is scored against."""

ROUTING_KINDS: tuple[str, ...] = ("speech", "airway", "voice")
"""The kinds TAXONOMY itself carries a state for; ``cough`` and ``glide`` are analysis-only."""


OVER_ROUTING_BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
"""The over-routing budgets :func:`recall_at_budgets` reports at, as false-positive rates.

A viewing parameter, not a decision: nothing downstream reads a budget, and every caller may pass
its own. Four points spanning an order of magnitude are enough to say whether a detector's recall is
bought with routing volume or comes for free.
"""

LIMIT_BUDGET = "budget"
"""The recall at this budget would rise if the budget did: the budget is what bounds it."""

LIMIT_AVAILABILITY = "availability"
"""The recall at this budget is every positive the detector can read: a wider budget buys nothing."""

ABOVE = "above"
"""The polarity whose loosest threshold is its lowest."""


@dataclass(frozen=True)
class BudgetPoint:
    """The most a detector recalls without spending more than one over-routing budget.

    Attributes:
        budget: The false-positive rate over the negatives that the point may not exceed. A point
            whose rate equals the budget exactly is inside it.
        threshold: The operating point: the loosest cut the budget allows, tightened back to the
            strictest cut reaching the same recall, because loosening past the last positive buys
            over-routing and nothing else. None when nothing inside the budget fires at all.
        confusion: The 2x2 there, over the whole population — a recording whose evidence could not
            be read counts as a non-firing rather than being dropped, because a router that cannot
            read a recording does not route it.
        limit: :data:`LIMIT_BUDGET`, :data:`LIMIT_AVAILABILITY`, or None when the population holds
            no reference positive and the question does not arise.
    """

    budget: float
    threshold: float | None
    confusion: Confusion
    limit: str | None

    @property
    def recall(self) -> float | None:
        """Fraction of reference positives routed here, or None with no positives."""
        return self.confusion.sensitivity

    @property
    def false_positive_rate(self) -> float | None:
        """Fraction of reference negatives routed here, or None with no negatives."""
        return self.confusion.false_positive_rate

    def as_json(self) -> dict[str, Any]:
        """This point, for the machine-readable output.

        Returns:
            The budget, the threshold, the limit and the 2x2 with its derived rates.
        """
        return {
            "budget": self.budget,
            "threshold": self.threshold,
            "limit": self.limit,
            "recall": self.recall,
            **self.confusion.as_json(),
        }


@dataclass(frozen=True)
class RecallCurve:
    """What one detector recalls at each over-routing budget, and what bounds it there.

    Attributes:
        name: The detector or gate the curve is for.
        reference: The reference standard positives are taken from.
        population: What was scored, in one phrase, naming any construction exclusion.
        polarity: ``above`` fires at or over the threshold, ``below`` at or under it.
        n_scored: Recordings in the population, readable or not.
        n_positive: Reference positives among them.
        n_negative: Reference negatives among them.
        n_unreadable: Recordings whose evidence the detector could not read.
        n_positive_unreadable: Reference positives among those.
        availability: Fraction of the population the detector can read.
        recall_ceiling: The recall at the loosest operating point there is, which is the fraction
            of positives the detector can read. No budget reaches past it.
        points: One :class:`BudgetPoint` per budget, in the order the budgets were given.
    """

    name: str
    reference: str
    population: str
    polarity: str
    n_scored: int
    n_positive: int
    n_negative: int
    n_unreadable: int
    n_positive_unreadable: int
    availability: float | None
    recall_ceiling: float | None
    points: tuple[BudgetPoint, ...]

    def point(self, budget: float) -> BudgetPoint:
        """The point at one budget.

        Args:
            budget: A budget the curve was computed at.

        Returns:
            That point.

        Raises:
            KeyError: When the curve carries no point at that budget.
        """
        for point in self.points:
            if point.budget == budget:
                return point
        raise KeyError(f"{self.name} carries no point at budget {budget!r}")

    def as_json(self) -> dict[str, Any]:
        """The curve, for the machine-readable output.

        Returns:
            The counts, the availability, the ceiling and one entry per budget.
        """
        return {
            "name": self.name,
            "reference": self.reference,
            "population": self.population,
            "polarity": self.polarity,
            "n_scored": self.n_scored,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_unreadable": self.n_unreadable,
            "n_positive_unreadable": self.n_positive_unreadable,
            "availability": self.availability,
            "recall_ceiling": self.recall_ceiling,
            "points": [point.as_json() for point in self.points],
        }


def _operating_points(readable: list[tuple[float, bool]], polarity: str) -> list[tuple[float | None, int, int]]:
    """Every distinct operating point a detector has on one population, strictest first.

    Args:
        readable: The recordings the detector could read, as ``(value, is_reference_positive)``.
        polarity: ``above`` or ``below``.

    Returns:
        ``(threshold, tp, fp)`` per point, opening with ``(None, 0, 0)`` — the point that fires on
        nothing — and then one per distinct value, loosening. ``tp`` and ``fp`` are cumulative, so
        both rise monotonically down the list.
    """
    readable.sort(key=lambda item: item[0], reverse=polarity == ABOVE)
    points: list[tuple[float | None, int, int]] = [(None, 0, 0)]
    true_positives = false_positives = 0
    index = 0
    while index < len(readable):
        value = readable[index][0]
        while index < len(readable) and readable[index][0] == value:
            if readable[index][1]:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        points.append((value, true_positives, false_positives))
    return points


def recall_at_budgets(
    observations: Iterable[tuple[float | None, bool]],
    *,
    name: str,
    reference: str,
    polarity: str = ABOVE,
    budgets: Sequence[float] = OVER_ROUTING_BUDGETS,
    population: str = "the whole corpus",
) -> RecallCurve:
    """The recall each over-routing budget buys, and whether the budget or availability bounds it.

    A point is inside its budget when its false-positive rate over the negatives is at or under it;
    equality is inside. Among the points inside, the one reported is the loosest, tightened back to
    the strictest cut reaching the same recall.

    Thresholds are the values the population actually carries rather than a written grid: the
    question is what the detector can do, and a grid can only answer it worse. Unreadable evidence
    is a non-firing at every threshold, so a detector whose feature is absent on most recordings
    reports the low ceiling it has rather than a high recall over the few it can read.

    Args:
        observations: One ``(value, is_reference_positive)`` per recording in the population, the
            value being None where the detector could not read the recording.
        name: The detector or gate the curve is for.
        reference: The reference standard positives were taken from.
        polarity: ``above`` fires at or over the threshold, ``below`` at or under it.
        budgets: The false-positive rates over the negatives that each point may not exceed. A
            point whose rate equals its budget exactly is inside it.
        population: What was scored, in one phrase.

    Returns:
        The curve, carrying one point per budget in the order the budgets were given.
    """
    readable: list[tuple[float, bool]] = []
    n_positive = n_negative = 0
    n_positive_unreadable = n_negative_unreadable = 0
    for value, positive in observations:
        if positive:
            n_positive += 1
        else:
            n_negative += 1
        if value is None:
            if positive:
                n_positive_unreadable += 1
            else:
                n_negative_unreadable += 1
            continue
        readable.append((float(value), positive))

    n_scored = n_positive + n_negative
    n_unreadable = n_positive_unreadable + n_negative_unreadable
    points = _operating_points(readable, polarity)
    ceiling = (n_positive - n_positive_unreadable) / n_positive if n_positive else None

    chosen: list[BudgetPoint] = []
    for budget in budgets:
        loosest = 0
        for index, (_, _, misfires) in enumerate(points):
            if n_negative and misfires / n_negative > budget:
                break
            loosest = index
        while loosest > 0 and points[loosest - 1][1] == points[loosest][1]:
            loosest -= 1
        threshold, true_positives, false_positives = points[loosest]
        table = Confusion(
            tp=true_positives,
            fp=false_positives,
            tn=n_negative - false_positives,
            fn=n_positive - true_positives,
        )
        limit: str | None
        if ceiling is None or table.sensitivity is None:
            limit = None
        elif table.sensitivity >= ceiling:
            limit = LIMIT_AVAILABILITY
        else:
            limit = LIMIT_BUDGET
        chosen.append(BudgetPoint(budget=budget, threshold=threshold, confusion=table, limit=limit))

    return RecallCurve(
        name=name,
        reference=reference,
        population=population,
        polarity=polarity,
        n_scored=n_scored,
        n_positive=n_positive,
        n_negative=n_negative,
        n_unreadable=n_unreadable,
        n_positive_unreadable=n_positive_unreadable,
        availability=(n_scored - n_unreadable) / n_scored if n_scored else None,
        recall_ceiling=ceiling,
        points=tuple(chosen),
    )


def detector_recall_at_budgets(
    records: Sequence[RecordingFeatures],
    detector: Detector,
    reference: ReferenceStandard,
    *,
    budgets: Sequence[float] = OVER_ROUTING_BUDGETS,
    family: str | None = None,
) -> RecallCurve:
    """One detector's recall at each over-routing budget, against one reference standard.

    Args:
        records: The recordings to score over.
        detector: The detector.
        reference: The standard positives are taken from. Families it excludes by construction are
            dropped from the population rather than counted as negatives.
        budgets: The false-positive rates over the negatives that each point may not exceed.
        family: Restrict to one task family, or None for the whole corpus.

    Returns:
        The curve.
    """
    observations: list[tuple[float | None, bool]] = []
    for record in records:
        if family is not None and record.family != family:
            continue
        if reference.population is not None and not reference.population(record):
            continue
        if reference.excludes(record):
            continue
        observations.append((detector_value(record, detector), reference.predicate(record)))
    return recall_at_budgets(
        observations,
        name=detector.name,
        reference=reference.name,
        polarity=detector.polarity,
        budgets=budgets,
        population=reference.scored_population(),
    )


def score_detector(
    records: Sequence[RecordingFeatures],
    detector: Detector,
    reference: ReferenceStandard,
    *,
    family: str | None = None,
    budgets: Sequence[float] = OVER_ROUTING_BUDGETS,
) -> dict[str, Any]:
    """One detector's 2x2 table at every threshold in its grid, and its recall at each budget.

    Args:
        records: The recordings to score over.
        detector: The detector.
        reference: The standard positives are taken from.
        family: Restrict to one task family, or None for the whole corpus.
        budgets: The over-routing budgets the recall curve is reported at.

    Returns:
        The detector, the reference, how many recordings were scored and how many were excluded
        because the evidence was absent, one row per threshold, and the recall curve. A row carries
        the plain table and, where the standard names families it excludes by construction, the
        same table with those families dropped from the population.
    """
    values: list[tuple[float, bool, bool]] = []
    unavailable = 0
    for record in records:
        if family is not None and record.family != family:
            continue
        if reference.population is not None and not reference.population(record):
            continue
        value = detector_value(record, detector)
        if value is None:
            unavailable += 1
            continue
        values.append((value, reference.predicate(record), reference.excludes(record)))
    rows = []
    for threshold in sweep_points(detector):
        counts = [0, 0, 0, 0]
        kept = [0, 0, 0, 0]
        for value, positive, excluded in values:
            fired = value <= threshold if detector.polarity == "below" else value >= threshold
            slot = (0 if positive else 1) if fired else (3 if positive else 2)
            counts[slot] += 1
            if not excluded:
                kept[slot] += 1
        row: dict[str, Any] = {"threshold": threshold, **Confusion(*counts).as_json()}
        if reference.excluded_by_construction:
            row["excluding_construction"] = Confusion(*kept).as_json()
        if detector.unit == "score" and math.isclose(threshold, CONSOLIDATION_FLOOR):
            row["marker"] = "taxonomy.consolidation_floor"
        rows.append(row)
    curve = detector_recall_at_budgets(records, detector, reference, budgets=budgets, family=family)
    return {
        "detector": detector.name,
        "kind": detector.kind,
        "unit": detector.unit,
        "polarity": detector.polarity,
        "reader": list(detector.reader),
        "reference": reference.name,
        "reference_is_proxy": reference.is_proxy,
        "population": reference.population_description,
        "excluded_by_construction": reference.exclusion_description if reference.excluded_by_construction else None,
        "family": family,
        "n_scored": len(values),
        "n_unavailable": unavailable,
        "n_reference_positive": sum(1 for _, positive, _ in values if positive),
        "recall_at_budget": curve.as_json(),
        "rows": rows,
    }


def _evidence_of(record: RecordingFeatures) -> dict[str, Any]:
    """The evidence lines a reader needs to judge one enumerated disagreement.

    Args:
        record: The recording.

    Returns:
        The word counts, the residual scalars, the longest spans and the speech-family peaks.
    """
    return {
        "stem": record.stem,
        "family": record.family,
        "task_id": record.task_id,
        "duration_s": record.duration_s,
        "transcript": record.transcript,
        "words": dict(record.words),
        "residual_energy_fraction": record.residual.get("energy_fraction"),
        "residual_speech_coverage_fraction": record.residual.get("speech_coverage_fraction"),
        "longest_amplitude_span_s": record.span_longest_s.get("amplitude"),
        "longest_continuity_span_s": record.span_longest_s.get("continuity"),
        "peaks": {key: value for key, value in sorted(record.peaks.items()) if value >= 0.05},
        "kind_state": dict(record.kind_state),
    }


def disagreements(records: Iterable[RecordingFeatures]) -> dict[str, Any]:
    """Every recording where agreed-ASR presence and the task declaration disagree.

    Args:
        records: The recordings.

    Returns:
        ``spoke_in_non_speech_task`` and ``silent_in_speech_task``, each a mapping from family to
        the count and up to :data:`DISAGREEMENT_CAP` individual cases with their evidence.
    """
    spoke: dict[str, dict[str, Any]] = {}
    silent: dict[str, dict[str, Any]] = {}
    for record in records:
        agreed = record.words.get("agreement", 0) >= 1
        declared_speech = record.family in DECLARED_KIND["speech"]
        if agreed and not declared_speech:
            bucket = spoke.setdefault(record.family, {"n": 0, "cases": []})
        elif declared_speech and not agreed:
            bucket = silent.setdefault(record.family, {"n": 0, "cases": []})
        else:
            continue
        bucket["n"] = int(bucket["n"]) + 1
        cases = bucket["cases"]
        if len(cases) < DISAGREEMENT_CAP:
            cases.append(_evidence_of(record))
    return {
        "cap_per_family": DISAGREEMENT_CAP,
        "spoke_in_non_speech_task": dict(sorted(spoke.items())),
        "silent_in_speech_task": dict(sorted(silent.items())),
    }


def prevalence(records: Sequence[RecordingFeatures]) -> dict[str, Any]:
    """Per-family counts, and how often each reference standard is positive within it.

    Args:
        records: The recordings.

    Returns:
        ``{family: {n, agreed_asr, declared_speech, declared_airway, declared_voice}}`` plus the
        corpus total.
    """
    per_family: dict[str, dict[str, int]] = {}
    for record in records:
        row = per_family.setdefault(record.family, {"n": 0})
        row["n"] += 1
        for reference in REFERENCE_STANDARDS:
            row[reference.name] = row.get(reference.name, 0) + int(reference.predicate(record))
    return {"n_recordings": len(records), "families": dict(sorted(per_family.items()))}


def label_prevalence(records: Sequence[RecordingFeatures]) -> dict[str, Any]:
    """How often each tracked label fires, per task family, per classifier and per stream.

    Args:
        records: The recordings.

    Returns:
        ``{stream|classifier|label: {family: {n, n_over_floor, fraction}}}`` over
        :data:`LABEL_PREVALENCE_STREAMS`, the floor being :data:`LABEL_PREVALENCE_FLOOR`.
    """
    counts: dict[str, dict[str, list[int]]] = {}
    for record in records:
        for stream in LABEL_PREVALENCE_STREAMS:
            for classifier in CLASSIFIERS:
                if f"{stream}|{classifier}" not in record.classifier_streams:
                    continue
                for label in TRACKED_LABELS[classifier]:
                    key = peak_key(stream, classifier, label)
                    slot = counts.setdefault(key, {}).setdefault(record.family, [0, 0])
                    slot[0] += 1
                    if record.peaks.get(key, 0.0) >= LABEL_PREVALENCE_FLOOR:
                        slot[1] += 1
    return {
        "floor": LABEL_PREVALENCE_FLOOR,
        "labels": {
            key: {
                family: {"n": pair[0], "n_over_floor": pair[1], "fraction": pair[1] / pair[0] if pair[0] else None}
                for family, pair in sorted(families.items())
            }
            for key, families in sorted(counts.items())
        },
    }


def taxonomy_as_run(records: Sequence[RecordingFeatures]) -> dict[str, Any]:
    """What TAXONOMY actually decided on this corpus, as the baseline every candidate improves on.

    Args:
        records: The recordings.

    Returns:
        The state counts per kind, and the 2x2 table of "the branch would run" against each kind's
        reference standard under ROUTING's rule that anything but ``absent`` runs the branch.
    """
    states: dict[str, dict[str, int]] = {}
    tables: dict[str, dict[str, Any]] = {}
    for reference in REFERENCE_STANDARDS:
        kind = reference.kind
        if kind not in ROUTING_KINDS:
            continue
        tp = fp = tn = fn = 0
        for record in records:
            state = record.kind_state.get(kind, "missing")
            states.setdefault(kind, {}).setdefault(state, 0)
            fired = state != "absent"
            positive = reference.predicate(record)
            if fired and positive:
                tp += 1
            elif fired:
                fp += 1
            elif positive:
                fn += 1
            else:
                tn += 1
        tables[reference.name] = Confusion(tp, fp, tn, fn).as_json()
    for record in records:
        for kind, state in record.kind_state.items():
            bucket = states.setdefault(kind, {})
            bucket[state] = bucket.get(state, 0) + 1
    return {"kind_states": states, "routing_would_run": tables}


@dataclass(frozen=True)
class RoutingRule:
    """One detector at one operating point, as a rule set uses it.

    Attributes:
        kind: The branch it would route to.
        detector: The detector's name.
        threshold: The operating point. Carried from the brief being answered; **not** a fitted
            floor, and nothing here proposes one.
    """

    kind: str
    detector: str
    threshold: float


DETECTOR_BY_NAME: dict[str, Detector] = {detector.name: detector for detector in DETECTORS}
"""Every detector in the catalogue, by name, so a rule set can name one."""

BASELINE_RULES: tuple[RoutingRule, ...] = (
    RoutingRule("speech", "speech.words_lexical", 2.0),
    RoutingRule("airway", "airway.residual_energy_fraction", 0.1),
    RoutingRule("airway", "airway.yamnet_peak.plain", 0.3),
    RoutingRule("voice", "voice.yamnet_singing_union.plain", 0.2),
)
"""The four rules whose 4.9% fall-through this analysis is asked to move."""

AUGMENTATION_TOP_FAMILIES = 12
"""How many families the fall-through table lists."""


def _rule_fires(record: RecordingFeatures, rule: RoutingRule) -> bool:
    """Whether one rule sends one recording to its branch.

    Args:
        record: The recording.
        rule: The rule.

    Returns:
        True when the detector's value is on the firing side of the threshold. Absent evidence is
        not a firing.
    """
    detector = DETECTOR_BY_NAME[rule.detector]
    value = detector_value(record, detector)
    if value is None:
        return False
    return value <= rule.threshold if detector.polarity == "below" else value >= rule.threshold


def bucket_coverage(records: Sequence[RecordingFeatures], rules: Sequence[RoutingRule]) -> dict[str, Any]:
    """How many recordings one rule set sends to no branch at all, and which families they are.

    Args:
        records: The recordings.
        rules: The rule set.

    Returns:
        The corpus counts, the per-kind firing counts and the per-family fall-through, worst first.
    """
    per_kind: dict[str, int] = {}
    per_family: dict[str, list[int]] = {}
    no_bucket = 0
    for record in records:
        fired = {rule.kind for rule in rules if _rule_fires(record, rule)}
        for kind in fired:
            per_kind[kind] = per_kind.get(kind, 0) + 1
        slot = per_family.setdefault(record.family, [0, 0])
        slot[0] += 1
        if not fired:
            no_bucket += 1
            slot[1] += 1
    ranked = sorted(
        ((family, int(pair[0]), int(pair[1])) for family, pair in per_family.items()),
        key=lambda item: (-(item[2] / item[1]) if item[1] else 0.0, -item[2]),
    )
    families = [
        {"family": family, "n": n, "n_no_bucket": missed, "fraction": (missed / n) if n else 0.0}
        for family, n, missed in ranked
    ]
    return {
        "rules": [asdict(rule) for rule in rules],
        "n_recordings": len(records),
        "n_no_bucket": no_bucket,
        "fraction_no_bucket": no_bucket / len(records) if records else None,
        "n_fired_per_kind": dict(sorted(per_kind.items())),
        "families": families,
    }


def bucket_augmentation(
    records: Sequence[RecordingFeatures],
    base: Sequence[RoutingRule] = BASELINE_RULES,
) -> dict[str, Any]:
    """What each candidate detector would do to the fall-through if added to a rule set.

    Args:
        records: The recordings.
        base: The rule set being augmented.

    Returns:
        One entry per detector not already in ``base``, listing every threshold in its grid with
        the fall-through that adding it there would leave, how many recordings it rescues and how
        many it fires on in total. No threshold is chosen.
    """
    base_names = {rule.detector for rule in base}
    uncovered = [not any(_rule_fires(record, rule) for rule in base) for record in records]
    n_uncovered = sum(uncovered)
    entries: list[dict[str, Any]] = []
    for detector in DETECTORS:
        if detector.name in base_names:
            continue
        values = [detector_value(record, detector) for record in records]
        rows: list[dict[str, Any]] = []
        for threshold in sweep_points(detector):
            fires = 0
            rescued = 0
            for index, value in enumerate(values):
                if value is None:
                    continue
                fired = value <= threshold if detector.polarity == "below" else value >= threshold
                if not fired:
                    continue
                fires += 1
                if uncovered[index]:
                    rescued += 1
            rows.append(
                {
                    "threshold": threshold,
                    "n_fires_corpus": fires,
                    "fraction_fires_corpus": fires / len(records) if records else None,
                    "n_rescued": rescued,
                    "n_no_bucket_after": n_uncovered - rescued,
                    "fraction_no_bucket_after": (n_uncovered - rescued) / len(records) if records else None,
                }
            )
        entries.append(
            {
                "detector": detector.name,
                "kind": detector.kind,
                "unit": detector.unit,
                "polarity": detector.polarity,
                "rows": rows,
            }
        )
    return {
        "base": [asdict(rule) for rule in base],
        "n_recordings": len(records),
        "n_no_bucket_base": n_uncovered,
        "candidates": entries,
    }


def _best_row(scored: dict[str, Any]) -> dict[str, Any] | None:
    """The threshold in one sweep with the largest Youden's J.

    Args:
        scored: What :func:`score_detector` returned.

    Returns:
        That row, or None when no threshold has both rates defined.
    """
    rows = [row for row in scored["rows"] if row.get("youden") is not None]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["youden"]))


def write_report(records: Sequence[RecordingFeatures], out_dir: Path) -> dict[str, Any]:
    """Score every detector, enumerate the disagreements and write both forms of the output.

    Args:
        records: The recordings.
        out_dir: Where ``sweeps.json``, ``sweeps_by_family.json``, ``label_prevalence.json``,
            ``prevalence.json``, ``disagreements.json``, ``index.json`` and ``summary.md`` are
            written. Overwritten on every run.

    Returns:
        The index that is also written as ``index.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sweeps: list[dict[str, Any]] = []
    for detector in DETECTORS:
        for reference in REFERENCE_STANDARDS:
            if reference.kind != detector.kind:
                continue
            sweeps.append(score_detector(records, detector, reference))
    (out_dir / "sweeps.json").write_text(json.dumps(sweeps, indent=1, sort_keys=True))
    families_seen = sorted({record.family for record in records})
    by_family = [
        score_detector(records, detector, reference, family=family)
        for detector in DETECTORS
        if detector.name in PRIMARY_DETECTORS
        for reference in REFERENCE_STANDARDS
        if reference.kind == detector.kind
        for family in families_seen
    ]
    (out_dir / "sweeps_by_family.json").write_text(json.dumps(by_family, indent=1, sort_keys=True))
    (out_dir / "label_prevalence.json").write_text(json.dumps(label_prevalence(records), indent=1, sort_keys=True))
    prevalence_report = prevalence(records)
    (out_dir / "prevalence.json").write_text(json.dumps(prevalence_report, indent=1, sort_keys=True))
    disagreement_report = disagreements(records)
    (out_dir / "disagreements.json").write_text(json.dumps(disagreement_report, indent=1, sort_keys=True))
    coverage = bucket_coverage(records, BASELINE_RULES)
    (out_dir / "buckets.json").write_text(json.dumps(coverage, indent=1, sort_keys=True))
    augmentation = bucket_augmentation(records, BASELINE_RULES)
    (out_dir / "bucket_augmentation.json").write_text(json.dumps(augmentation, indent=1, sort_keys=True))
    baseline = taxonomy_as_run(records)
    index: dict[str, Any] = {
        "n_recordings": len(records),
        "n_families": len(prevalence_report["families"]),
        "taxonomy_as_run": baseline,
        "bucket_coverage": {key: value for key, value in coverage.items() if key != "families"},
        "detectors": [
            {
                "detector": scored["detector"],
                "reference": scored["reference"],
                "reference_is_proxy": scored["reference_is_proxy"],
                "n_scored": scored["n_scored"],
                "n_unavailable": scored["n_unavailable"],
                "n_reference_positive": scored["n_reference_positive"],
                "best_youden_row": _best_row(scored),
                "recall_at_budget": scored["recall_at_budget"],
            }
            for scored in sweeps
        ],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    (out_dir / "summary.md").write_text(_markdown(index, prevalence_report, disagreement_report, sweeps, coverage))
    return index


def _rate(value: Any) -> str:  # noqa: ANN401
    """One rate formatted for the readable summary.

    Args:
        value: A rate or None.

    Returns:
        Four decimal places, or ``n/a``.
    """
    return "n/a" if value is None else f"{float(value):.4f}"


def _budgets_reported(index: dict[str, Any]) -> tuple[float, ...]:
    """The budgets the scored detectors were reported at.

    Args:
        index: What :func:`write_report` assembled.

    Returns:
        The budgets of the first detector's curve, which every detector in one report shares.
    """
    for entry in index["detectors"]:
        return tuple(float(point["budget"]) for point in entry["recall_at_budget"]["points"])
    return ()


def _markdown(
    index: dict[str, Any],
    prevalence_report: dict[str, Any],
    disagreement_report: dict[str, Any],
    sweeps: Sequence[dict[str, Any]],
    coverage: dict[str, Any],
) -> str:
    """The readable summary.

    Args:
        index: What :func:`write_report` assembled.
        prevalence_report: What :func:`prevalence` returned.
        disagreement_report: What :func:`disagreements` returned.
        sweeps: Every scored detector.
        coverage: What :func:`bucket_coverage` returned for :data:`BASELINE_RULES`.

    Returns:
        The Markdown document.
    """
    lines: list[str] = ["# TAXONOMY routing evidence", ""]
    lines.append(f"- recordings: {index['n_recordings']}")
    lines.append(f"- task families: {index['n_families']}")
    lines.append("")
    lines.append("## TAXONOMY as it ran")
    lines.append("")
    lines.append("| kind | states |")
    lines.append("| --- | --- |")
    baseline = index["taxonomy_as_run"]
    for kind, states in sorted(baseline["kind_states"].items()):
        lines.append(f"| {kind} | {states} |")
    lines.append("")
    lines.append("| reference | tp | fp | tn | fn | sens | spec |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, table in sorted(baseline["routing_would_run"].items()):
        lines.append(
            f"| {name} | {table['tp']} | {table['fp']} | {table['tn']} | {table['fn']} | "
            f"{_rate(table['sensitivity'])} | {_rate(table['specificity'])} |"
        )
    lines.append("")
    lines.append("## Branch coverage under the baseline rule set")
    lines.append("")
    for rule in coverage["rules"]:
        lines.append(f"- `{rule['detector']}` >= {rule['threshold']:g} -> {rule['kind']}")
    lines.append("")
    lines.append(
        f"- no branch at all: {coverage['n_no_bucket']} of {coverage['n_recordings']} "
        f"({_rate(coverage['fraction_no_bucket'])})"
    )
    lines.append("")
    lines.append("| family | n | no bucket | fraction |")
    lines.append("| --- | --- | --- | --- |")
    for row in coverage["families"][:AUGMENTATION_TOP_FAMILIES]:
        lines.append(f"| {row['family']} | {row['n']} | {row['n_no_bucket']} | {_rate(row['fraction'])} |")
    lines.append("")
    lines.append("## Prevalence per family")
    lines.append("")
    lines.append("| family | n | agreed_asr | declared_speech | declared_airway | declared_voice |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    families = prevalence_report["families"]
    for family, row in sorted(families.items(), key=lambda item: -int(item[1]["n"])):
        lines.append(
            f"| {family} | {row['n']} | {row.get('agreed_asr', 0)} | {row.get('declared_speech', 0)} | "
            f"{row.get('declared_airway', 0)} | {row.get('declared_voice', 0)} |"
        )
    lines.append("")
    lines.append("## Recall at over-routing budget")
    lines.append("")
    lines.append(
        "The loosest threshold whose false-positive rate over the negatives stays inside the budget, "
        "and the recall it reaches. `limit` says whether the widest budget or the detector's own "
        "availability is what stops the recall rising."
    )
    lines.append("")
    budgets = _budgets_reported(index)
    header = " | ".join(f"R@{budget:.0%}" for budget in budgets)
    lines.append(f"| detector | reference | avail | ceiling | {header} | limit |")
    lines.append("| --- | --- | --- | --- | " + " | ".join(["---"] * len(budgets)) + " | --- |")
    for entry in index["detectors"]:
        curve = entry["recall_at_budget"]
        recalls = " | ".join(_rate(point["recall"]) for point in curve["points"])
        limit = curve["points"][-1]["limit"] if curve["points"] else None
        lines.append(
            f"| {entry['detector']} | {entry['reference']} | {_rate(curve['availability'])} | "
            f"{_rate(curve['recall_ceiling'])} | {recalls} | {limit or 'n/a'} |"
        )
    lines.append("")
    lines.append("## Detector sweeps")
    for scored in sweeps:
        lines.append("")
        proxy = " (proxy standard)" if scored["reference_is_proxy"] else ""
        lines.append(f"### {scored['detector']} vs {scored['reference']}{proxy}")
        lines.append("")
        lines.append(
            f"unit {scored['unit']}, fires {scored['polarity']} threshold; over {scored['population']}; "
            f"scored {scored['n_scored']}, evidence absent {scored['n_unavailable']}, "
            f"reference positive {scored['n_reference_positive']}"
        )
        lines.append("")
        excluded = scored["excluded_by_construction"]
        held_out = f" | spec excl. {excluded}" if excluded else ""
        lines.append(f"| threshold | tp | fp | tn | fn | sens | spec{held_out} | note |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |" + (" --- |" if excluded else "") + " --- |")
        for row in scored["rows"]:
            corrected = f" {_rate(row['excluding_construction']['specificity'])} |" if excluded else ""
            lines.append(
                f"| {row['threshold']:g} | {row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
                f"{_rate(row['sensitivity'])} | {_rate(row['specificity'])} |{corrected} {row.get('marker', '')} |"
            )
    lines.append("")
    lines.append("## Disagreements between agreed ASR and the declaration")
    for direction in ("spoke_in_non_speech_task", "silent_in_speech_task"):
        bucket = disagreement_report[direction]
        lines.append("")
        lines.append(f"### {direction}")
        lines.append("")
        lines.append("| family | n |")
        lines.append("| --- | --- |")
        for family, entry in sorted(bucket.items(), key=lambda item: -int(item[1]["n"])):
            lines.append(f"| {family} | {entry['n']} |")
    lines.append("")
    return "\n".join(lines)


def load_features(path: Path) -> list[RecordingFeatures]:
    """Read a features shard back.

    Args:
        path: A JSONL file of :meth:`RecordingFeatures.as_json` lines.

    Returns:
        The records.
    """
    records: list[RecordingFeatures] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(RecordingFeatures(**json.loads(line)))
    return records


def dump_features(records: Iterable[RecordingFeatures], path: Path) -> int:
    """Write records as one JSONL shard.

    Args:
        records: The records.
        path: Where to write.

    Returns:
        How many lines were written.
    """
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")
            written += 1
    return written
