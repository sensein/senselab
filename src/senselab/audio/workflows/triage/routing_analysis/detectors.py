"""The candidate routing detectors and the corpus-derived grid each is swept over.

A detector reads one number out of a :class:`~senselab.audio.workflows.triage.routing_analysis.
features.RecordingFeatures` and fires when that number is at or above a threshold. No threshold is
preferred here: every one in a detector's grid is scored, and
``taxonomy.consolidation_floor`` (0.2) is marked where it falls rather than adopted.

No grid is written down. Each is derived at catalogue construction from the detector's own
distribution over the corpus, read from the dated profile under ``data/detector_profile/``. A
detector the profile does not cover, or covers as a constant, raises rather than taking a default.
A candidate declared ahead of the sweep that would profile it is staged in
:data:`UNPROFILED_DETECTORS`, outside the catalogue, and carries no grid at all.

``specs/20260912-detector-grids/design.md`` says which quantiles, why the extremes are in, how a
count and a gate are handled, and which cut points are pinned regardless of the corpus.
``specs/20260912-parquet-tables/design.md`` says why the profile is a parquet table and why a
derived grid stops at what its feature attains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.triage.routing_analysis.features import SPAN_CLASSIFIERS, RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.labels import FAMILIES, LABEL_SETS, peak_key
from senselab.audio.workflows.triage.routing_analysis.tables import read_header

GATE_CLOSED = -1.0
"""The value a gated detector reads when its corroborator did not fire: below every threshold."""

GATE_CLOSED_BELOW = math.inf
"""The same, for a ``below``-polarity detector, where a small value is what fires."""

CONSOLIDATION_FLOOR = 0.2
"""``taxonomy.consolidation_floor`` in ``data/config/default.yaml``, marked in every score grid."""

CONSENSUS_CLASSIFIERS: frozenset[str] = frozenset(SPAN_CLASSIFIERS.values())
"""The classifiers the consensus taxonomy consolidates, and so the only ones its stream can carry."""

PROFILE_VERSION = "1"
"""Schema version of the detector profile. A reader refuses any other value."""

PROFILE_DIR = Path(__file__).resolve().parent.parent / "data" / "detector_profile"
"""Where the bundled profiles live, one dated parquet per corpus sweep."""

PROFILE_SUFFIX = ".parquet"
"""What a bundled profile is named with."""

PROFILE_NAME_COLUMN = "detector"
"""The profile table's key column."""

PROFILE_ENTRY_KEYS: tuple[str, ...] = ("n", "availability", "polarity", "state", "min", "max", "distinct", "value")
"""Every non-quantile column one detector's row can carry. A row leaves out what its state has no value for."""

PROFILE_QUANTILE_PREFIX = "q"
"""What a quantile of the ladder is columned as, so ``0.001`` becomes ``q0.001``."""

QUANTILE_LADDER: tuple[str, ...] = (
    "0.001",
    "0.01",
    "0.05",
    "0.1",
    "0.2",
    "0.3",
    "0.4",
    "0.5",
    "0.6",
    "0.7",
    "0.8",
    "0.9",
    "0.95",
    "0.99",
    "0.999",
)
"""The quantiles every profiled detector carries, ascending. A profile missing one is refused."""

COUNT_UNITS: frozenset[str] = frozenset({"words", "spans", "tokens", "phonemes", "segments"})
"""The units whose feature is a count, so its own integers are the operating points, not quantiles."""

DENSE_INTEGER_SPAN = 32
"""How far up a count's integers are enumerated before the quantile ladder takes over."""

PINNED_THRESHOLDS: Mapping[str, tuple[float, ...]] = {
    "cough.yamnet_cough_minus_breath.plain": (0.0,),
    "cough.hear_cough_minus_breath.plain": (0.0,),
    "airway.residual_correlation_residual": (0.0,),
}
"""Cut points a corpus cannot move, unioned into the derived grid. Each is a sign test."""


@dataclass(frozen=True)
class Detector:
    """One candidate rule TAXONOMY could route on.

    Attributes:
        name: The detector's id, unique across the catalogue.
        kind: The branch it would route to. ``speech``, ``airway`` and ``voice`` are the branches
            TAXONOMY carries a state for; ``cough``, ``glide`` and ``ddk`` are analysis-only.
        reader: What to read, as ``(source, *arguments)``; see :func:`detector_value`. A
            ``gated`` reader nests two of these.
        unit: The unit of the number it reads.
        thresholds: Every threshold it is scored at.
        polarity: ``above`` fires at or over the threshold, ``below`` fires at or under it.
    """

    name: str
    kind: str
    reader: tuple[Any, ...]
    unit: str
    thresholds: tuple[float, ...]
    polarity: str = "above"


def _candidate(
    name: str,
    kind: str,
    reader: tuple[Any, ...],
    unit: str,
    polarity: str = "above",
) -> Detector:
    """One catalogue entry, declared without a grid for :func:`build_catalogue` to fill in.

    Args:
        name: The detector's id, unique across the catalogue.
        kind: The branch it would route to.
        reader: What to read, as ``(source, *arguments)``.
        unit: The unit of the number it reads.
        polarity: ``above`` fires at or over the threshold, ``below`` fires at or under it.

    Returns:
        The detector, with an empty threshold grid.
    """
    return Detector(name=name, kind=kind, reader=reader, unit=unit, thresholds=(), polarity=polarity)


def _bundled_profile_path() -> Path:
    """The newest bundled detector profile.

    Returns:
        The last dated profile in the bundled directory.

    Raises:
        FileNotFoundError: If the package ships no profile, which leaves every grid underivable
            rather than silently defaulted.
    """
    bundled = sorted(PROFILE_DIR.glob(f"*{PROFILE_SUFFIX}"))
    if not bundled:
        raise FileNotFoundError(f"no detector profile in {PROFILE_DIR}")
    return bundled[-1]


def _validate(profile: Mapping[str, Any], source: str) -> None:
    """Reject a profile whose schema, corpus size or quantile ladder is not what a grid needs.

    Args:
        profile: The parsed profile.
        source: Where it was read from, for the message.

    Raises:
        ValueError: If the schema version is unknown, if the corpus is empty, if the profile
            carries no detector, or if an entry's state, polarity or quantile ladder is malformed.
    """
    version = str(profile.get("profile_version"))
    if version != PROFILE_VERSION:
        raise ValueError(f"{source}: profile_version {version!r}, expected {PROFILE_VERSION!r}")
    if not int(profile.get("n_recordings") or 0) > 0:
        raise ValueError(f"{source}: carries no recording count")
    detectors = profile.get("detectors") or {}
    if not detectors:
        raise ValueError(f"{source}: carries no detector")
    for name, entry in detectors.items():
        if entry.get("polarity") not in ("above", "below"):
            raise ValueError(f"{source}: {name} has polarity {entry.get('polarity')!r}")
        state = entry.get("state")
        if state == "constant":
            continue
        if state != "varies":
            raise ValueError(f"{source}: {name} has state {state!r}")
        quantiles = entry.get("quantiles") or {}
        missing = [q for q in QUANTILE_LADDER if q not in quantiles]
        if missing:
            raise ValueError(f"{source}: {name} is missing quantiles {missing}")
        ladder = [float(entry["min"]), *(float(quantiles[q]) for q in QUANTILE_LADDER), float(entry["max"])]
        if not all(math.isfinite(value) for value in ladder):
            raise ValueError(f"{source}: {name} carries a non-finite quantile")
        if any(lower > upper for lower, upper in zip(ladder, ladder[1:])):
            raise ValueError(f"{source}: {name} has a quantile ladder that decreases")


@lru_cache(maxsize=None)
def load_detector_profile(path: str | None = None) -> dict[str, Any]:
    """Load and validate a detector profile.

    Args:
        path: Profile path, or ``None`` for the newest bundled one.

    Returns:
        The validated profile.

    Raises:
        FileNotFoundError: If ``path`` names a file that does not exist. A named-but-absent profile
            is an operator error, not a reason to fall back to the bundled one.
        ValueError: If the profile fails validation.
    """
    resolved = _bundled_profile_path() if path is None else Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"detector profile not found: {resolved}")
    profile = profile_from_table(pq.read_table(resolved))
    _validate(profile, str(resolved))
    return profile


def profile_as_table(profile: Mapping[str, Any]) -> tuple[pa.Table, dict[str, Any]]:
    """One profile as its detector table and the header that describes the whole sweep.

    Args:
        profile: A profile in the shape :func:`load_detector_profile` returns.

    Returns:
        One row per detector, and the scalars that belong to no row.
    """
    rows = []
    for name, entry in sorted(profile["detectors"].items()):
        quantiles = entry.get("quantiles") or {}
        row: dict[str, Any] = {key: entry.get(key) for key in PROFILE_ENTRY_KEYS}
        row[PROFILE_NAME_COLUMN] = name
        row.update({f"{PROFILE_QUANTILE_PREFIX}{step}": quantiles.get(step) for step in QUANTILE_LADDER})
        rows.append(row)
    header = {key: value for key, value in profile.items() if key != "detectors"}
    return pa.Table.from_pylist(rows), header


def profile_from_table(table: pa.Table) -> dict[str, Any]:
    """Rebuild a profile from the table :func:`profile_as_table` produced.

    Args:
        table: The detector table, carrying the header in its file metadata.

    Returns:
        The profile, in the shape every reader of it expects. A detector's entry carries only the
        keys its row has a value for, so a ``constant`` entry has no quantile ladder rather than a
        ladder of zeros.
    """
    profile: dict[str, Any] = dict(read_header(table))
    detectors: dict[str, Any] = {}
    for row in table.to_pylist():
        entry: dict[str, Any] = {key: row[key] for key in PROFILE_ENTRY_KEYS if row.get(key) is not None}
        quantiles = {
            step: row[f"{PROFILE_QUANTILE_PREFIX}{step}"]
            for step in QUANTILE_LADDER
            if row.get(f"{PROFILE_QUANTILE_PREFIX}{step}") is not None
        }
        if quantiles:
            entry["quantiles"] = quantiles
        detectors[str(row[PROFILE_NAME_COLUMN])] = entry
    profile["detectors"] = detectors
    return profile


def _ladder(entry: Mapping[str, Any], drop_sentinel: bool) -> set[float]:
    """One detector's measured values: its extremes and every quantile of the ladder.

    Args:
        entry: The detector's profile entry, in the ``varies`` state.
        drop_sentinel: Whether to discard :data:`GATE_CLOSED`, which a gate wrote rather than the
            feature.

    Returns:
        The distinct values, unordered.
    """
    quantiles = entry["quantiles"]
    values = [float(entry["min"]), *(float(quantiles[q]) for q in QUANTILE_LADDER), float(entry["max"])]
    return {value for value in values if math.isfinite(value) and not (drop_sentinel and value == GATE_CLOSED)}


def _profile_entry(name: str, profile: Mapping[str, Any], source: str) -> Mapping[str, Any]:
    """One detector's profile entry, refusing anything a grid cannot be derived from.

    Args:
        name: The detector's id.
        profile: A loaded profile.
        source: Where the profile came from, for the message.

    Returns:
        The entry.

    Raises:
        ValueError: If the detector is absent from the profile, or if its feature was constant
            over the whole corpus.
    """
    entry = profile["detectors"].get(name)
    if entry is None:
        raise ValueError(f"{name} is absent from the detector profile {source}; re-sweep the corpus or drop it")
    if entry.get("state") != "varies":
        value = entry.get("value")
        raise ValueError(
            f"{name} is constant at {value} over all {profile['n_recordings']} recordings in {source}; "
            "no threshold separates it, so it does not belong in the catalogue"
        )
    return entry


def derive_thresholds(
    candidate: Detector,
    siblings: Sequence[str],
    profile: Mapping[str, Any],
    source: str,
) -> tuple[float, ...]:
    """One detector's grid, derived from the corpus distribution of the feature it reads.

    Args:
        candidate: The detector, declared without a grid.
        siblings: Ungated detector ids reading the same primary feature; their ladders are
            unioned into the grid.
        profile: A loaded profile.
        source: Where the profile came from, for the message.

    Returns:
        The thresholds, ascending and distinct, none of them outside what the contributing ladders
        attain apart from the declared cut points.

    Raises:
        ValueError: If the grid cannot be derived, or if it fails to reach the profiled extreme on
            the polarity's firing side.
    """
    entry = _profile_entry(candidate.name, profile, source)
    gated = candidate.reader[0] == "gated"
    points = _ladder(entry, drop_sentinel=gated)
    for sibling in siblings:
        points |= _ladder(_profile_entry(sibling, profile, source), drop_sentinel=False)
    if candidate.unit in COUNT_UNITS:
        points = {float(round(value)) for value in points}
    attained = (min(points), max(points))
    if candidate.unit in COUNT_UNITS:
        floor = max(0, int(math.ceil(attained[0])))
        ceiling = min(DENSE_INTEGER_SPAN, int(math.floor(attained[1])))
        points |= {float(count) for count in range(floor, ceiling + 1)}
    points = {value for value in points if attained[0] <= value <= attained[1]}
    if candidate.unit == "score":
        points.add(CONSOLIDATION_FLOOR)
    points |= {float(pinned) for pinned in PINNED_THRESHOLDS.get(candidate.name, ())}
    grid = tuple(sorted(points))
    if not grid:
        raise ValueError(f"{candidate.name}: the profile yielded no threshold")
    reach = grid[-1] >= float(entry["max"]) if candidate.polarity == "above" else grid[0] <= float(entry["min"])
    if not reach:
        raise ValueError(
            f"{candidate.name}: the derived grid [{grid[0]}, {grid[-1]}] does not reach the profiled "
            f"[{entry['min']}, {entry['max']}] on the {candidate.polarity} side"
        )
    return grid


def build_catalogue(candidates: Iterable[Detector], path: str | None = None) -> tuple[Detector, ...]:
    """Fill every declared candidate's grid in from the profile.

    Args:
        candidates: The detectors, declared without grids.
        path: Profile path, or ``None`` for the bundled one.

    Returns:
        The catalogue, each entry carrying its derived grid.

    Raises:
        ValueError: If a name repeats, or if any one detector's grid cannot be derived.
    """
    declared = list(candidates)
    names = [candidate.name for candidate in declared]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"detector ids are not unique: {duplicated}")
    profile = load_detector_profile(path)
    source = str(_bundled_profile_path() if path is None else Path(path))
    ungated: dict[tuple[Any, ...], list[str]] = {}
    for candidate in declared:
        if candidate.reader[0] != "gated":
            ungated.setdefault(candidate.reader, []).append(candidate.name)
    return tuple(
        replace(
            candidate,
            thresholds=derive_thresholds(
                candidate,
                ungated.get(candidate.reader[1], ()) if candidate.reader[0] == "gated" else (),
                profile,
                source,
            ),
        )
        for candidate in declared
    )


def _check_pins(candidates: Iterable[Detector]) -> None:
    """Refuse a pin that names no declared detector.

    Args:
        candidates: Every detector the catalogue declares.

    Raises:
        ValueError: If :data:`PINNED_THRESHOLDS` names a detector that is not among them.
    """
    unpinnable = sorted(set(PINNED_THRESHOLDS) - {candidate.name for candidate in candidates})
    if unpinnable:
        raise ValueError(f"PINNED_THRESHOLDS names detectors the catalogue does not declare: {unpinnable}")


def _check_unprofiled(candidates: Iterable[Detector], catalogue: Iterable[Detector], path: str | None = None) -> None:
    """Refuse a staged candidate that repeats a catalogue name, or that the profile now covers.

    Args:
        candidates: The candidates staged for the next sweep, declared without a profile entry.
        catalogue: Every candidate the catalogue builds a grid for.
        path: Profile path, or ``None`` for the bundled one.

    Raises:
        ValueError: If a staged name is already in the catalogue, or if the newest profile carries
            it — a profiled detector has a derivable grid and belongs in the catalogue.
    """
    staged = {candidate.name for candidate in candidates}
    repeated = sorted(staged & {candidate.name for candidate in catalogue})
    if repeated:
        raise ValueError(f"staged detectors are already in the catalogue: {repeated}")
    profiled = sorted(name for name in staged if name in load_detector_profile(path)["detectors"])
    if profiled:
        raise ValueError(
            f"staged detectors are in the detector profile: {profiled}; move them into the catalogue, "
            "where their grids are derived, rather than leaving them unscored"
        )


def sweep_points(detector: Detector) -> tuple[float, ...]:
    """The thresholds a detector is scored at.

    Args:
        detector: The detector.

    Returns:
        Its threshold grid, ascending.
    """
    return tuple(sorted(detector.thresholds))


def detector_value(features: RecordingFeatures, detector: Detector) -> float | None:
    """The number a detector reads out of one recording.

    Args:
        features: The recording's extracted evidence.
        detector: The detector.

    Returns:
        The number, or None when the evidence the detector reads is not in the store, which
        excludes the recording from that detector's scoring rather than counting as a zero.

    Raises:
        ValueError: When the detector names a source this function does not implement.
    """
    source, *arguments = detector.reader
    if source == "words":
        return float(features.words.get(arguments[0], 0))
    if source == "bracketed_set":
        if not features.consensus_present:
            return None
        return float(sum(features.bracketed_types.get(str(name), 0) for name in arguments))
    if source == "onomatopoeic":
        if not features.consensus_present:
            return None
        return float(sum(features.onomatopoeic_types.values()))
    if source == "stream_peak_max":
        stream, classifier = arguments
        if f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        prefix = peak_key(stream, classifier, "")
        return max((score for key, score in features.peaks.items() if key.startswith(prefix)), default=0.0)
    if source == "residual":
        if not features.residual:
            return None
        value = features.residual.get(arguments[0])
        return None if value is None else float(value)
    if source == "span_longest":
        return float(features.span_longest_s.get(arguments[0], 0.0))
    if source == "span_total":
        return float(features.span_total_s.get(arguments[0], 0.0))
    if source == "span_count":
        return float(features.span_count.get(arguments[0], 0))
    if source == "peak":
        stream, classifier, kind = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        labels = FAMILIES[kind][classifier]
        if not labels:
            return None
        return max((features.peaks.get(peak_key(stream, classifier, label), 0.0) for label in labels), default=0.0)
    if source == "gated":
        primary, gate, gate_threshold, polarity = arguments
        gate_value = detector_value(features, Detector(detector.name, detector.kind, gate, "score", ()))
        if gate_value is None:
            return None
        passes = gate_value >= float(gate_threshold) if polarity == "above" else gate_value < float(gate_threshold)
        if not passes:
            return GATE_CLOSED if detector.polarity == "above" else GATE_CLOSED_BELOW
        return detector_value(
            features, Detector(detector.name, detector.kind, primary, detector.unit, (), detector.polarity)
        )
    if source == "peak_label":
        stream, classifier, label = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        return float(features.peaks.get(peak_key(stream, classifier, label), 0.0))
    if source == "peak_set":
        stream, classifier, set_name = arguments
        if stream in ("plain", "enhanced", "residual") and f"{stream}|{classifier}" not in features.classifier_streams:
            return None
        labels = LABEL_SETS[set_name][classifier]
        if not labels:
            return None
        return max((features.peaks.get(peak_key(stream, classifier, label), 0.0) for label in labels), default=0.0)
    if source == "span_stat":
        return _optional(features.span_stats, arguments[0])
    if source == "span_label_stat":
        return _optional(features.span_label_stats, arguments[0])
    if source == "span_label_set_stat":
        return _optional(features.span_label_set_stats, arguments[0])
    if source == "squim":
        return _optional(features.squim, arguments[0])
    if source == "level":
        return _optional(features.level, arguments[0])
    if source == "disruptions":
        return _optional(features.disruptions, arguments[0])
    if source == "silence":
        return _optional(features.silence, arguments[0])
    if source == "praat":
        return _optional(features.praat, arguments[0])
    if source == "ppg":
        return _optional(features.ppg, arguments[0])
    if source == "phonation":
        return _optional(features.phonation, arguments[0])
    if source == "ratio":
        numerator = detector_value(features, Detector(detector.name, detector.kind, arguments[0], detector.unit, ()))
        denominator = detector_value(features, Detector(detector.name, detector.kind, arguments[1], detector.unit, ()))
        if numerator is None or denominator is None or denominator == 0.0:
            return None
        return numerator / denominator
    if source == "difference":
        left = detector_value(features, Detector(detector.name, detector.kind, arguments[0], detector.unit, ()))
        right = detector_value(features, Detector(detector.name, detector.kind, arguments[1], detector.unit, ()))
        return None if left is None or right is None else left - right
    raise ValueError(f"unknown detector source {source!r}")


def _optional(table: dict[str, float], key: str) -> float | None:
    """One value out of a feature table, absent rather than zero when the derivative is missing.

    Args:
        table: The feature table.
        key: The key.

    Returns:
        The value, or None when the key is absent or not finite.
    """
    value = table.get(key)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _peak_detectors() -> list[Detector]:
    """One detector per kind, classifier and stream, reading that family's best score.

    The consensus stream is built only for the classifiers in :data:`CONSENSUS_CLASSIFIERS`, which
    is what the consensus taxonomy consolidates.

    Returns:
        The classifier-score detectors.
    """
    built: list[Detector] = []
    for kind, per_classifier in FAMILIES.items():
        for classifier, labels in per_classifier.items():
            if not labels:
                continue
            for stream in ("plain", "enhanced", "residual"):
                built.append(
                    _candidate(
                        name=f"{kind}.{classifier}_peak.{stream}",
                        kind=kind,
                        reader=("peak", stream, classifier, kind),
                        unit="score",
                    )
                )
            if classifier == "hear":
                built.append(
                    _candidate(
                        name=f"{kind}.hear_peak.span",
                        kind=kind,
                        reader=("peak", "span", "hear", kind),
                        unit="score",
                    )
                )
            if classifier in CONSENSUS_CLASSIFIERS:
                built.append(
                    _candidate(
                        name=f"{kind}.{classifier}_peak.consensus",
                        kind=kind,
                        reader=("peak", "consensus", classifier, kind),
                        unit="score",
                    )
                )
    return built


def _singing_detectors() -> list[Detector]:
    """The AudioSet singing-subtree union, on each stream and both AudioSet classifiers.

    Returns:
        The union detectors, for the ``voice`` and ``glide`` kinds.
    """
    built: list[Detector] = []
    for kind in ("voice", "glide"):
        for classifier in ("yamnet", "ast"):
            for stream in ("plain", "enhanced"):
                built.append(
                    _candidate(
                        name=f"{kind}.{classifier}_singing_union.{stream}",
                        kind=kind,
                        reader=("peak_set", stream, classifier, "singing"),
                        unit="score",
                    )
                )
    return built


_COUGH_VS_BREATH: tuple[Detector, ...] = (
    _candidate("cough.longest_amplitude_span", "cough", ("span_longest", "amplitude"), "seconds", "below"),
    _candidate(
        "cough.amplitude_duration_median",
        "cough",
        ("span_stat", "amplitude.duration_median"),
        "seconds",
        "below",
    ),
    _candidate(
        "cough.amplitude_duration_max",
        "cough",
        ("span_stat", "amplitude.duration_max"),
        "seconds",
    ),
    _candidate(
        "cough.amplitude_span_count",
        "cough",
        ("span_count", "amplitude"),
        "spans",
    ),
    _candidate(
        "cough.amplitude_rate_per_s",
        "cough",
        ("span_stat", "amplitude.rate_per_s"),
        "spans/s",
    ),
    _candidate(
        "cough.amplitude_duty_fraction",
        "cough",
        ("span_stat", "amplitude.duty_fraction"),
        "fraction",
        "below",
    ),
    _candidate(
        "cough.amplitude_peak_over_floor_db_max",
        "cough",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
    ),
    _candidate(
        "cough.amplitude_peak_over_floor_db_median",
        "cough",
        ("span_stat", "amplitude.peak_over_floor_db_median"),
        "dB",
    ),
    _candidate("cough.squim_stoi_median", "cough", ("squim", "all.stoi.median"), "stoi", "below"),
    _candidate(
        "cough.squim_stoi_max",
        "cough",
        ("squim", "all.stoi.max"),
        "stoi",
    ),
    _candidate(
        "cough.squim_stoi_iqr",
        "cough",
        ("squim", "all.stoi.iqr"),
        "stoi",
    ),
    _candidate(
        "cough.squim_pesq_max",
        "cough",
        ("squim", "all.pesq.max"),
        "pesq",
    ),
    _candidate(
        "cough.squim_pesq_iqr",
        "cough",
        ("squim", "all.pesq.iqr"),
        "pesq",
    ),
    _candidate("cough.squim_si_sdr_median", "cough", ("squim", "all.si_sdr.median"), "dB", "below"),
    _candidate(
        "cough.squim_si_sdr_max",
        "cough",
        ("squim", "all.si_sdr.max"),
        "dB",
    ),
    _candidate(
        "cough.squim_si_sdr_iqr",
        "cough",
        ("squim", "all.si_sdr.iqr"),
        "dB",
    ),
    _candidate("cough.squim_amplitude_si_sdr_max", "cough", ("squim", "amplitude.si_sdr.max"), "dB", "below"),
    _candidate(
        "cough.yamnet_cough_labels.plain",
        "cough",
        ("peak_set", "plain", "yamnet", "cough_labels"),
        "score",
    ),
    _candidate(
        "cough.hear_cough_labels.plain",
        "cough",
        ("peak_set", "plain", "hear", "cough_labels"),
        "score",
    ),
    _candidate(
        "cough.yamnet_cough_minus_breath.plain",
        "cough",
        (
            "difference",
            ("peak_set", "plain", "yamnet", "cough_labels"),
            ("peak_set", "plain", "yamnet", "breath_labels"),
        ),
        "score",
    ),
    _candidate(
        "cough.hear_cough_minus_breath.plain",
        "cough",
        ("difference", ("peak_set", "plain", "hear", "cough_labels"), ("peak_set", "plain", "hear", "breath_labels")),
        "score",
    ),
    _candidate(
        "cough.yamnet_cough_label.plain",
        "cough",
        ("peak_label", "plain", "yamnet", "Cough"),
        "score",
    ),
    _candidate(
        "cough.yamnet_breathing_label.plain",
        "cough",
        ("peak_label", "plain", "yamnet", "Breathing"),
        "score",
        "below",
    ),
    _candidate(
        "cough.zero_crossing_rate",
        "cough",
        ("disruptions", "zero_crossing_rate"),
        "crossings/s",
    ),
    _candidate(
        "cough.level_peak_dbfs",
        "cough",
        ("level", "peak_dbfs"),
        "dBFS",
    ),
    _candidate(
        "cough.level_crest_db",
        "cough",
        ("difference", ("level", "peak_dbfs"), ("level", "rms_dbfs")),
        "dB",
    ),
    _candidate(
        "cough.silence_fraction",
        "cough",
        ("silence", "fraction"),
        "fraction",
    ),
    _candidate("cough.residual_energy_fraction", "cough", ("residual", "energy_fraction"), "fraction", "below"),
    _candidate("cough.residual_band_0_200", "cough", ("residual", "band_0_200"), "fraction", "below"),
    _candidate(
        "cough.residual_band_1000_4000",
        "cough",
        ("residual", "band_1000_4000"),
        "fraction",
    ),
    _candidate(
        "cough.yamnet_cough_labels.plain+short_span",
        "cough",
        ("gated", ("peak_set", "plain", "yamnet", "cough_labels"), ("span_longest", "amplitude"), 1.0, "below"),
        "score",
    ),
    _candidate(
        "cough.level_crest_db+hear_cough>=0.5",
        "cough",
        (
            "gated",
            ("difference", ("level", "peak_dbfs"), ("level", "rms_dbfs")),
            ("peak_set", "plain", "hear", "cough_labels"),
            0.5,
            "above",
        ),
        "dB",
    ),
)
"""Candidate discriminators between a declared cough family and a declared breath family."""

_GLIDES: tuple[Detector, ...] = (
    _candidate(
        "glide.yamnet_whistle.plain",
        "glide",
        ("peak_set", "plain", "yamnet", "whistle"),
        "score",
    ),
    _candidate(
        "glide.yamnet_humming_label.plain",
        "glide",
        ("peak_label", "plain", "yamnet", "Humming"),
        "score",
    ),
    _candidate(
        "glide.yamnet_chant_label.plain",
        "glide",
        ("peak_label", "plain", "yamnet", "Chant"),
        "score",
    ),
    _candidate(
        "glide.yamnet_peak.plain",
        "glide",
        ("peak", "plain", "yamnet", "voice"),
        "score",
    ),
    _candidate(
        "glide.longest_amplitude_span",
        "glide",
        ("span_longest", "amplitude"),
        "seconds",
    ),
    _candidate(
        "glide.amplitude_duration_iqr",
        "glide",
        ("span_stat", "amplitude.duration_iqr"),
        "seconds",
    ),
    _candidate(
        "glide.amplitude_duty_fraction",
        "glide",
        ("span_stat", "amplitude.duty_fraction"),
        "fraction",
    ),
    _candidate("glide.amplitude_span_count", "glide", ("span_count", "amplitude"), "spans", "below"),
    _candidate(
        "glide.amplitude_peak_over_floor_db_max",
        "glide",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
    ),
    _candidate(
        "glide.squim_stoi_max",
        "glide",
        ("squim", "all.stoi.max"),
        "stoi",
    ),
    _candidate(
        "glide.squim_pesq_max",
        "glide",
        ("squim", "all.pesq.max"),
        "pesq",
    ),
    _candidate(
        "glide.squim_si_sdr_max",
        "glide",
        ("squim", "all.si_sdr.max"),
        "dB",
    ),
    _candidate(
        "glide.squim_si_sdr_iqr",
        "glide",
        ("squim", "all.si_sdr.iqr"),
        "dB",
    ),
    _candidate(
        "glide.squim_amplitude_stoi_median",
        "glide",
        ("squim", "amplitude.stoi.median"),
        "stoi",
    ),
    _candidate("glide.silence_fraction", "glide", ("silence", "fraction"), "fraction", "below"),
    _candidate(
        "glide.level_lufs",
        "glide",
        ("level", "lufs"),
        "dBFS",
    ),
    _candidate(
        "glide.zero_crossing_rate",
        "glide",
        ("disruptions", "zero_crossing_rate"),
        "crossings/s",
    ),
    _candidate("glide.residual_energy_fraction", "glide", ("residual", "energy_fraction"), "fraction", "below"),
    _candidate("glide.words_lexical", "glide", ("words", "lexical"), "words", "below"),
    _candidate(
        "glide.yamnet_singing_union.plain+no_agreed_word",
        "glide",
        ("gated", ("peak_set", "plain", "yamnet", "singing"), ("words", "agreement"), 1, "below"),
        "score",
    ),
    _candidate(
        "glide.yamnet_singing_union.plain+amplitude>=2s",
        "glide",
        ("gated", ("peak_set", "plain", "yamnet", "singing"), ("span_longest", "amplitude"), 2.0, "above"),
        "score",
    ),
    _candidate(
        "glide.longest_amplitude_span+singing>=0.05",
        "glide",
        ("gated", ("span_longest", "amplitude"), ("peak_set", "plain", "yamnet", "singing"), 0.05, "above"),
        "seconds",
    ),
    _candidate(
        "glide.squim_stoi_max+no_agreed_word",
        "glide",
        ("gated", ("squim", "all.stoi.max"), ("words", "agreement"), 1, "below"),
        "stoi",
    ),
)
"""Candidate detectors for the glide families, the worst-served of the voice families."""

_NEW_DERIVATIVES: tuple[Detector, ...] = (
    _candidate(
        "airway.hear_cough_labels.plain",
        "airway",
        ("peak_set", "plain", "hear", "cough_labels"),
        "score",
    ),
    _candidate(
        "airway.amplitude_peak_over_floor_db_max",
        "airway",
        ("span_stat", "amplitude.peak_over_floor_db_max"),
        "dB",
    ),
    _candidate(
        "airway.amplitude_rate_per_s",
        "airway",
        ("span_stat", "amplitude.rate_per_s"),
        "spans/s",
    ),
    _candidate(
        "airway.zero_crossing_rate",
        "airway",
        ("disruptions", "zero_crossing_rate"),
        "crossings/s",
    ),
    _candidate("airway.squim_stoi_median", "airway", ("squim", "all.stoi.median"), "stoi", "below"),
    _candidate(
        "airway.squim_si_sdr_iqr",
        "airway",
        ("squim", "all.si_sdr.iqr"),
        "dB",
    ),
    _candidate(
        "airway.silence_fraction",
        "airway",
        ("silence", "fraction"),
        "fraction",
    ),
    _candidate(
        "airway.residual_band_0_200",
        "airway",
        ("residual", "band_0_200"),
        "fraction",
    ),
    _candidate(
        "airway.residual_band_4000_8000",
        "airway",
        ("residual", "band_4000_8000"),
        "fraction",
    ),
    _candidate(
        "airway.residual_correlation_residual",
        "airway",
        ("residual", "correlation_residual"),
        "r",
    ),
    _candidate(
        "speech.squim_stoi_median",
        "speech",
        ("squim", "all.stoi.median"),
        "stoi",
    ),
    _candidate(
        "speech.squim_pesq_median",
        "speech",
        ("squim", "all.pesq.median"),
        "pesq",
    ),
    _candidate(
        "speech.squim_si_sdr_median",
        "speech",
        ("squim", "all.si_sdr.median"),
        "dB",
    ),
    _candidate("speech.silence_fraction", "speech", ("silence", "fraction"), "fraction", "below"),
    _candidate(
        "speech.amplitude_rate_per_s",
        "speech",
        ("span_stat", "amplitude.rate_per_s"),
        "spans/s",
    ),
    _candidate(
        "speech.level_lufs",
        "speech",
        ("level", "lufs"),
        "dBFS",
    ),
    _candidate(
        "voice.squim_stoi_max",
        "voice",
        ("squim", "all.stoi.max"),
        "stoi",
    ),
    _candidate(
        "voice.squim_amplitude_pesq_max",
        "voice",
        ("squim", "amplitude.pesq.max"),
        "pesq",
    ),
    _candidate(
        "voice.amplitude_duty_fraction",
        "voice",
        ("span_stat", "amplitude.duty_fraction"),
        "fraction",
    ),
    _candidate("voice.silence_fraction", "voice", ("silence", "fraction"), "fraction", "below"),
    _candidate(
        "voice.level_lufs",
        "voice",
        ("level", "lufs"),
        "dBFS",
    ),
)
"""Detectors reading a derivative the first sweep ignored, on the three kinds it already covered."""

_PITCH_SWEEP: tuple[Detector, ...] = (
    _candidate(
        "glide.praat_std_f0_hertz",
        "glide",
        ("praat", "std_f0_hertz"),
        "Hz",
    ),
    _candidate(
        "glide.praat_f0_relative_spread",
        "glide",
        ("ratio", ("praat", "std_f0_hertz"), ("praat", "mean_f0_hertz")),
        "ratio",
    ),
    _candidate(
        "glide.praat_std_f0_hertz+no_agreed_word",
        "glide",
        ("gated", ("praat", "std_f0_hertz"), ("words", "agreement"), 1, "below"),
        "Hz",
    ),
    _candidate(
        "glide.praat_phonation_ratio",
        "glide",
        ("praat", "phonation_ratio"),
        "fraction",
    ),
    _candidate(
        "voice.praat_phonation_ratio",
        "voice",
        ("praat", "phonation_ratio"),
        "fraction",
    ),
    _candidate(
        "voice.praat_mean_hnr_db",
        "voice",
        ("praat", "mean_hnr_db"),
        "dB",
    ),
    _candidate(
        "voice.praat_cepstral_peak_prominence_mean",
        "voice",
        ("praat", "cepstral_peak_prominence_mean"),
        "dB",
    ),
)
"""Pitch spread and phonation read off Praat, neither of which needs a span to reach 3 s."""

_SYLLABLE_REPETITION: tuple[Detector, ...] = (
    _candidate(
        "ddk.praat_articulation_rate",
        "ddk",
        ("praat", "articulation_rate"),
        "syllables/s",
    ),
    _candidate(
        "ddk.praat_speaking_rate",
        "ddk",
        ("praat", "speaking_rate"),
        "syllables/s",
    ),
    _candidate(
        "ddk.ppg_segment_rate_per_s",
        "ddk",
        ("ppg", "segment_rate_per_s"),
        "segments/s",
    ),
    _candidate(
        "ddk.ppg_repetition_peak",
        "ddk",
        ("ppg", "repetition_peak"),
        "fraction",
    ),
    _candidate(
        "ddk.ppg_repetition_prominence",
        "ddk",
        ("ppg", "repetition_prominence"),
        "fraction",
    ),
    _candidate(
        "ddk.ppg_repetition_lag_segments",
        "ddk",
        ("ppg", "repetition_lag_segments"),
        "segments",
    ),
    _candidate(
        "ddk.ppg_segment_duration_median",
        "ddk",
        ("ppg", "segment_duration_median"),
        "seconds",
        "below",
    ),
    _candidate("ddk.ppg_distinct_phonemes", "ddk", ("ppg", "distinct_phonemes"), "phonemes", "below"),
)
"""Candidate detectors for diadochokinesis, which is a rate and a repetition rather than a word."""

_UNVOICED_AIRWAY: tuple[Detector, ...] = (
    _candidate(
        "airway.praat_phonation_ratio",
        "airway",
        ("praat", "phonation_ratio"),
        "fraction",
        "below",
    ),
    _candidate(
        "airway.praat_pause_rate",
        "airway",
        ("praat", "pause_rate"),
        "pauses/s",
    ),
    _candidate(
        "airway.praat_mean_pause_duration",
        "airway",
        ("praat", "mean_pause_duration"),
        "seconds",
    ),
    _candidate("airway.praat_mean_hnr_db", "airway", ("praat", "mean_hnr_db"), "dB", "below"),
    _candidate(
        "airway.ppg_silent_fraction",
        "airway",
        ("ppg", "silent_fraction"),
        "fraction",
    ),
)
"""Candidate detectors for a breath, which is unvoiced and need not accumulate energy to be read."""

_BRACKETED_TOKENS: tuple[Detector, ...] = (
    _candidate(
        "airway.bracketed_breath",
        "airway",
        ("bracketed_set", "breath"),
        "tokens",
    ),
    _candidate(
        "airway.bracketed_cough",
        "airway",
        ("bracketed_set", "cough"),
        "tokens",
    ),
    _candidate(
        "airway.bracketed_throatclearing",
        "airway",
        ("bracketed_set", "throatclearing"),
        "tokens",
    ),
    _candidate(
        "airway.bracketed_sniff",
        "airway",
        ("bracketed_set", "sniff"),
        "tokens",
    ),
    _candidate(
        "airway.bracketed_laughter",
        "airway",
        ("bracketed_set", "laughter"),
        "tokens",
    ),
    _candidate(
        "airway.bracketed_airway_union",
        "airway",
        ("bracketed_set", "breath", "cough", "throatclearing", "sniff"),
        "tokens",
    ),
    _candidate(
        "speech.bracketed_uh",
        "speech",
        ("bracketed_set", "uh"),
        "tokens",
    ),
    _candidate(
        "speech.bracketed_um",
        "speech",
        ("bracketed_set", "um"),
        "tokens",
    ),
    _candidate(
        "speech.bracketed_filler_union",
        "speech",
        ("bracketed_set", "uh", "um"),
        "tokens",
    ),
)
"""Typed bracketed consensus tokens, so the AIRWAY bracket gate can be swept rather than assumed."""

_STREAM_PEAKS: tuple[Detector, ...] = (
    _candidate(
        "speech.enhanced_yamnet_peak_max",
        "speech",
        ("stream_peak_max", "enhanced", "yamnet"),
        "score",
    ),
    _candidate(
        "speech.residual_yamnet_peak_max",
        "speech",
        ("stream_peak_max", "residual", "yamnet"),
        "score",
    ),
)
"""The highest tracked-label score a whole stream carries, which is what an empty recording lacks."""

_LABEL_CONDITIONED_SPANS: tuple[Detector, ...] = (
    _candidate(
        "cough.yamnet_cough_span_peak_over_floor_db_max",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_max"),
        "dB",
    ),
    _candidate(
        "cough.yamnet_cough_span_peak_over_floor_db_p75",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_p75"),
        "dB",
    ),
    _candidate(
        "cough.yamnet_cough_span_peak_over_floor_db_p90",
        "cough",
        ("span_label_stat", "yamnet.Cough.peak_over_floor_db_p90"),
        "dB",
    ),
)
"""Span amplitude read only over the spans YAMNet itself labelled ``Cough``."""

_COUGH_SET = "cough_labels"
"""The :data:`~senselab.audio.workflows.triage.routing_analysis.labels.LABEL_SETS` union read below."""

_LABEL_SET_CONDITIONED_SPANS: tuple[Detector, ...] = tuple(
    _candidate(
        f"cough.yamnet_cough_set_span_peak_over_floor_db_{statistic}",
        "cough",
        ("span_label_set_stat", f"yamnet.{_COUGH_SET}.peak_over_floor_db_{statistic}"),
        "dB",
    )
    for statistic in ("max", "p75", "p90")
)
"""Span amplitude over the spans carrying any cough-set label, not the single ``Cough`` string."""


_CANDIDATES: tuple[Detector, ...] = tuple(
    [
        _candidate(
            "speech.words_agreement",
            "speech",
            ("words", "agreement"),
            "words",
        ),
        _candidate(
            "speech.words_agreement_lexical",
            "speech",
            ("words", "agreement_lexical"),
            "words",
        ),
        _candidate(
            "speech.words_lexical",
            "speech",
            ("words", "lexical"),
            "words",
        ),
        _candidate(
            "speech.words_total",
            "speech",
            ("words", "total"),
            "words",
        ),
        _candidate(
            "speech.residual_speech_coverage",
            "speech",
            ("residual", "speech_coverage_fraction"),
            "fraction",
        ),
        _candidate(
            "airway.residual_energy_fraction",
            "airway",
            ("residual", "energy_fraction"),
            "fraction",
        ),
        _candidate(
            "airway.residual_enhanced_energy_fraction",
            "airway",
            ("residual", "enhanced_energy_fraction"),
            "fraction",
        ),
        _candidate(
            "voice.longest_amplitude_span",
            "voice",
            ("span_longest", "amplitude"),
            "seconds",
        ),
        _candidate(
            "voice.longest_continuity_span",
            "voice",
            ("span_longest", "continuity"),
            "seconds",
        ),
        _candidate(
            "voice.total_amplitude_span",
            "voice",
            ("span_total", "amplitude"),
            "seconds",
        ),
        _candidate(
            "airway.residual_energy_fraction+hear>=0.2",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("peak", "plain", "hear", "airway"), 0.2, "above"),
            "fraction",
        ),
        _candidate(
            "airway.residual_energy_fraction+hear>=0.5",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("peak", "plain", "hear", "airway"), 0.5, "above"),
            "fraction",
        ),
        _candidate(
            "airway.yamnet_peak.plain+hear>=0.2",
            "airway",
            ("gated", ("peak", "plain", "yamnet", "airway"), ("peak", "plain", "hear", "airway"), 0.2, "above"),
            "score",
        ),
        _candidate(
            "airway.residual_energy_fraction+no_agreed_word",
            "airway",
            ("gated", ("residual", "energy_fraction"), ("words", "agreement"), 1, "below"),
            "fraction",
        ),
        _candidate(
            "voice.longest_amplitude_span+no_agreed_word",
            "voice",
            ("gated", ("span_longest", "amplitude"), ("words", "agreement"), 1, "below"),
            "seconds",
        ),
        _candidate(
            "voice.yamnet_peak.plain+no_agreed_word",
            "voice",
            ("gated", ("peak", "plain", "yamnet", "voice"), ("words", "agreement"), 1, "below"),
            "score",
        ),
        _candidate(
            "voice.yamnet_peak.plain+amplitude>=3s",
            "voice",
            ("gated", ("peak", "plain", "yamnet", "voice"), ("span_longest", "amplitude"), 3.0, "above"),
            "score",
        ),
        _candidate(
            "speech.words_agreement+ast>=0.5",
            "speech",
            ("gated", ("words", "agreement"), ("peak", "plain", "ast", "speech"), 0.5, "above"),
            "words",
        ),
        _candidate(
            "voice.yamnet_chant_peak.plain",
            "voice",
            ("peak_label", "plain", "yamnet", "Chant"),
            "score",
        ),
        _candidate(
            "voice.yamnet_mantra_peak.plain",
            "voice",
            ("peak_label", "plain", "yamnet", "Mantra"),
            "score",
        ),
    ]
    + _peak_detectors()
    + _singing_detectors()
    + list(_COUGH_VS_BREATH)
    + list(_GLIDES)
    + list(_NEW_DERIVATIVES)
    + list(_LABEL_CONDITIONED_SPANS)
    + list(_LABEL_SET_CONDITIONED_SPANS)
    + list(_PITCH_SWEEP)
    + list(_SYLLABLE_REPETITION)
    + list(_UNVOICED_AIRWAY)
    + list(_BRACKETED_TOKENS)
    + list(_STREAM_PEAKS)
)
"""Every candidate detector, declared without a grid."""

_check_pins(_CANDIDATES)

DETECTORS: tuple[Detector, ...] = build_catalogue(_CANDIDATES)
"""Every candidate detector, scored at every threshold of its corpus-derived grid."""

_PITCH_TRAJECTORY: tuple[Detector, ...] = (
    _candidate("glide.phonation_monotonicity", "glide", ("phonation", "monotonicity"), "correlation"),
    _candidate("glide.phonation_monotone_fraction", "glide", ("phonation", "monotone_fraction"), "fraction"),
    _candidate("glide.phonation_semitone_range", "glide", ("phonation", "semitone_range"), "semitones"),
    _candidate("glide.phonation_semitone_iqr", "glide", ("phonation", "semitone_iqr"), "semitones"),
    _candidate("glide.phonation_sweep_semitones_abs", "glide", ("phonation", "sweep_semitones_abs"), "semitones"),
    _candidate("glide.phonation_sweep_seconds", "glide", ("phonation", "sweep_seconds"), "seconds"),
    _candidate("glide.phonation_sweep_fraction", "glide", ("phonation", "sweep_fraction"), "fraction"),
    _candidate(
        "glide.phonation_sweep_rate_abs_semitones_per_s",
        "glide",
        ("phonation", "sweep_rate_abs_semitones_per_s"),
        "semitones/s",
    ),
    _candidate(
        "glide.phonation_sweep_over_range",
        "glide",
        ("ratio", ("phonation", "sweep_semitones_abs"), ("phonation", "semitone_range")),
        "ratio",
    ),
    _candidate(
        "glide.phonation_net_over_variation_rising",
        "glide",
        ("phonation", "net_over_variation"),
        "ratio",
    ),
    _candidate(
        "glide.phonation_net_over_variation_falling",
        "glide",
        ("phonation", "net_over_variation"),
        "ratio",
        "below",
    ),
    _candidate(
        "glide.phonation_monotonicity+no_agreed_word",
        "glide",
        ("gated", ("phonation", "monotonicity"), ("words", "agreement"), 1, "below"),
        "correlation",
    ),
    _candidate(
        "glide.phonation_sweep_semitones_abs+no_agreed_word",
        "glide",
        ("gated", ("phonation", "sweep_semitones_abs"), ("words", "agreement"), 1, "below"),
        "semitones",
    ),
    _candidate("glide.phonation_rank_correlation_rising", "glide", ("phonation", "rank_correlation"), "correlation"),
    _candidate(
        "glide.phonation_rank_correlation_falling",
        "glide",
        ("phonation", "rank_correlation"),
        "correlation",
        "below",
    ),
    _candidate("glide.phonation_sweep_semitones_rising", "glide", ("phonation", "sweep_semitones"), "semitones"),
    _candidate(
        "glide.phonation_sweep_semitones_falling",
        "glide",
        ("phonation", "sweep_semitones"),
        "semitones",
        "below",
    ),
    _candidate("glide.phonation_direction_bias_rising", "glide", ("phonation", "direction_bias"), "fraction"),
    _candidate(
        "glide.phonation_direction_bias_falling",
        "glide",
        ("phonation", "direction_bias"),
        "fraction",
        "below",
    ),
    _candidate("voice.phonation_voiced_fraction", "voice", ("phonation", "voiced_fraction"), "fraction"),
    _candidate("voice.phonation_voiced_seconds", "voice", ("phonation", "voiced_seconds"), "seconds"),
    _candidate("voice.phonation_strength_median", "voice", ("phonation", "strength_median"), "strength"),
)
"""Candidates reading ``RecordingFeatures.phonation``, which no shipped profile has measured. Each
``_rising``/``_falling`` pair reads one signed key at both polarities.
``specs/20260817-triage-workflow-dag/family-taxonomy-ruleset.md`` says what each key measures and
what the pairs are there to settle."""

UNPROFILED_DETECTORS: tuple[Detector, ...] = (
    _candidate("cough.words_onomatopoeic", "cough", ("onomatopoeic",), "tokens"),
    *_PITCH_TRAJECTORY,
)
"""Candidates whose feature no shipped profile has measured. Each carries an empty ``thresholds``
and is absent from :data:`DETECTORS`; :func:`detector_value` reads one like any other, which is what
a corpus sweep needs to profile it. Once a profile carries one, :func:`_check_unprofiled` raises
until it is moved into the catalogue."""

_check_unprofiled(UNPROFILED_DETECTORS, _CANDIDATES)
