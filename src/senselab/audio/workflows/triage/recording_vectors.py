"""One compact row per recording, for the corpus and recording views.

Reads finished triage stores and emits a single parquet with one row per recording: three identity
columns, every measurement the graph writes, the decision record, and the drawable content of the
summary figure as fixed-layout binary blobs.

The byte layout of every binary column and the nullability of every scalar are specified in
``specs/20260922-compact-recording-vectors/schema.md``, which is the interface a browser decodes
against. No layout here may change without that file changing with it.

The parquet carries transcript text and marked PII extents and is therefore a sensitive artefact:
mode 600, never committed, never copied to a shared location.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa

SCHEMA_VERSION = 3
"""Bumped whenever a column is added, removed or retyped, or a binary layout changes."""

STORE_NAME = "store.jsonl"
RUN_SUBDIR = "run"

TRACE_POINTS = 256
"""Every decimated trace is exactly this many points."""

TIME_SCALE = 65535
"""Time is ``round(t / duration_s * TIME_SCALE)`` as uint16."""

ENVELOPE_DBFS_RANGE = (-100.0, 0.0)
CONTINUITY_RANGE = (0.0, 1.05)
SCORE_RANGE = (0.0, 1.0)
SQUIM_RANGES = (("stoi", (0.0, 1.0)), ("pesq", (1.0, 4.5)), ("si_sdr", (-10.0, 30.0)))

SPAN_ROWS = ("E", "C", "A", "S", "G")
MEASURE_ROW = {"amplitude": "E", "continuity": "C", "asr": "A", "gap": "G"}
CLASSIFIERS = ("yamnet", "hear", "ast")
WORD_OUTCOMES = ("agreement", "variant", "insertion")
LANES = ("AIRWAY", "SPEECH", "VOICE", "REDACT")
LANE_OF_FAMILY = {"airway": 0, "speech": 1, "voice": 2}
UNKNOWN_CODE = 255
"""What a byte enum carries when the store's value is outside the declared vocabulary."""

REDACTION_NAME = "redaction"
_ROLE_INDEX = re.compile(r"_\d+$")
_TIMESTAMP = re.compile(r"_\d{8}-\d{6}$")
_PARTICIPANT = re.compile(r"(sub-[^_]+)")
_SESSION = re.compile(r"(ses-[^_]+)")
_TASK = re.compile(r"task-(.+)$")

SCALAR_MEASUREMENTS = (
    "breath_coverage_fraction",
    "breath_peak_over_floor_db",
    "cough_peak_over_floor_db",
    "ddk_ppg_period_dispersion",
    "ddk_repetition_count_from_ppg_decode",
    "ddk_syllable_rate_from_envelope_modulation_hz",
    "ddk_syllable_rate_from_ppg_decode_hz",
    "expected_sequence_repeat_fraction",
    "extent_dominant_speaker_share",
    "extent_secondary_source_s",
    "extent_source_active_s",
    "extent_speaker_count",
    "glide_extent_semitones",
    "interruptions",
    "pause_fraction_of_response",
    "phonation_onset_to_offset_s",
    "source_content_coverage",
    "speech_rate_from_consensus_words_per_s",
    "train_fraction_of_recording",
    "verbatim_overlap_fraction",
    "voiced_duration_s",
)
VECTOR_MEASUREMENTS = ("ddk_position_realised_mass",)
MATRIX_MEASUREMENTS = ("ddk_cv_instrument_reading",)
CATEGORICAL_MEASUREMENTS = (
    "carrier_rejected",
    "category_membership",
    "defines_its_cue",
    "effort_absolute",
    "measured_route",
    "phonation_extent",
    "repetition_rule",
    "route",
    "source_overlap",
    "sweep_extent",
)
MEASUREMENTS = SCALAR_MEASUREMENTS + VECTOR_MEASUREMENTS + MATRIX_MEASUREMENTS + CATEGORICAL_MEASUREMENTS

BRANCH_NODES = ("AIRWAY", "SPEECH", "VOICE", "QUALITY")
ROUTED_BRANCHES = ("AIRWAY", "SPEECH", "VOICE")
FLAG_OUTCOMES = frozenset({"flag", "fail", "discard"})

GATE_NAMES = (
    "production_min_s",
    "voiced_fraction_min",
    "f0_spread_max_semitones",
    "continuity_min",
    "dominant_segment_min_fraction",
    "monotone_tolerance_semitones",
    "expected_tokens_matched_min",
    "omissions_max",
    "response_min_s",
    "coverage_min",
    "dominant_speaker_share_min",
    "items_min",
    "events_min",
    "repetitions_min",
    "repeat_overlap_min",
    "echo_overlap_max",
    "verbatim_overlap_max",
    "gap_off_task_min_s",
    "interval_max_s",
    "score_min",
    "train_min_s",
    "rate_prominence_min",
)
"""The gate column order, pinned against ``nodes.gates.GATE_KEYS`` by ``recording_vectors_test``.

Spelled here rather than imported so that reading a store costs no model-framework import, and so
that a gate added to the registry fails a test until this schema is bumped with it.
"""

GATE_UNDETERMINED = "undetermined"
"""What a gate's ``_passed`` column carries when either the reading or the bound was absent."""

RESIDUAL_MEASUREMENT = "residual"
"""The PREPROCESS measurement carrying the enhanced/residual decomposition's energies."""

SECONDARY_ROLE = "secondary_source_extent"
SOLO_ROLE = "solo_extent"
SEPARATED_PREFIX = "separated_"
DOMINANT_SHARE = "extent_dominant_speaker_share"
"""The reading ``dominant_speaker_share_min`` gates. Folded by ``min``, as the gate folds it."""

SPANS_LAYOUT = "BHH"
SPAN_LABELS_LAYOUT = "HBB"
SPAN_SQUIM_LAYOUT = "HBBB"
ASR_WORDS_LAYOUT = "HHB"
PII_MARKS_LAYOUT = "HH"
BRANCH_LANES_LAYOUT = "BHH"


# --------------------------------------------------------------------------------------- encoding


def quantise_time(seconds: float, duration_s: float) -> int:
    """One time, as the uint16 the binary blocks carry.

    Args:
        seconds: A time on the recording's own axis.
        duration_s: The recording's duration, which is the quantiser's full scale.

    Returns:
        ``round(seconds / duration_s * TIME_SCALE)``, clamped to ``0..TIME_SCALE``.
    """
    if not duration_s > 0 or not math.isfinite(seconds):
        return 0
    return int(min(TIME_SCALE, max(0, round(seconds / duration_s * TIME_SCALE))))


def quantise_value(value: float, low: float, high: float) -> int:
    """One value, as the uint8 the binary blocks carry.

    Args:
        value: The reading.
        low: The bottom of the declared range.
        high: The top of the declared range.

    Returns:
        ``round((value - low) / (high - low) * 255)``, clamped to ``0..255``.
    """
    if not math.isfinite(value) or high <= low:
        return 0
    return int(min(255, max(0, round((value - low) / (high - low) * 255))))


def dequantise_time(code: int, duration_s: float) -> float:
    """The inverse of :func:`quantise_time`.

    Args:
        code: The stored uint16.
        duration_s: The recording's duration.

    Returns:
        The time in seconds.
    """
    return code / TIME_SCALE * duration_s


def dequantise_value(code: int, low: float, high: float) -> float:
    """The inverse of :func:`quantise_value`.

    Args:
        code: The stored uint8.
        low: The bottom of the declared range.
        high: The top of the declared range.

    Returns:
        The value.
    """
    return low + code / 255.0 * (high - low)


def decimate(values: Sequence[float] | np.ndarray, points: int, how: str) -> list[float] | None:
    """Reduce a per-sample trace to a fixed number of points.

    Args:
        values: The trace.
        points: How many points to emit.
        how: ``"max"`` or ``"mean"``, the reduction within each bucket.

    Returns:
        The reduced trace, or None when nothing in it is finite.
    """
    array = np.asarray(values, dtype="float64").reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    edges = np.linspace(0, array.size, points + 1).astype(int)
    out: list[float] = []
    previous = float(array[0])
    for start, end in zip(edges[:-1], edges[1:]):
        bucket = array[start:end]
        if bucket.size:
            previous = float(bucket.max() if how == "max" else bucket.mean())
        out.append(previous)
    return out


def encode_trace(values: Sequence[float] | np.ndarray | None, low: float, high: float, how: str) -> bytes | None:
    """One decimated trace, as ``TRACE_POINTS`` uint8 bytes.

    Args:
        values: The per-sample trace, or None when the derivative is absent.
        low: The bottom of the declared range.
        high: The top of the declared range.
        how: The bucket reduction, ``"max"`` or ``"mean"``.

    Returns:
        The bytes, or None when the trace is absent.
    """
    if values is None:
        return None
    reduced = decimate(values, TRACE_POINTS, how)
    if reduced is None:
        return None
    return bytes(quantise_value(v, low, high) for v in reduced)


def encode_waveform(samples: np.ndarray | None) -> tuple[bytes | None, float | None]:
    """A waveform as ``TRACE_POINTS`` (min, max) uint8 pairs over ``[-peak, +peak]``.

    Args:
        samples: The conditioned waveform, or None when no stream decoded.

    Returns:
        The bytes and the peak they are scaled by, or ``(None, None)``.
    """
    if samples is None:
        return None, None
    finite = np.asarray(samples, dtype="float64").reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    peak = float(np.abs(finite).max()) or 1.0
    edges = np.linspace(0, finite.size, TRACE_POINTS + 1).astype(int)
    out = bytearray()
    for start, end in zip(edges[:-1], edges[1:]):
        bucket = finite[start:end] if end > start else finite[start : start + 1]
        out.append(quantise_value(float(bucket.min()), -peak, peak))
        out.append(quantise_value(float(bucket.max()), -peak, peak))
    return bytes(out), peak


def encode_records(records: Iterable[Sequence[int]], layout: str) -> bytes:
    """Pack fixed-width little-endian records.

    Args:
        records: One tuple of field values per record.
        layout: A :mod:`struct` format without a byte-order prefix.

    Returns:
        The concatenated records.
    """
    packer = struct.Struct("<" + layout)
    return b"".join(packer.pack(*record) for record in records)


def decode_records(blob: bytes | None, layout: str) -> list[tuple[int, ...]]:
    """Unpack what :func:`encode_records` packed.

    Args:
        blob: The bytes, or None.
        layout: The same :mod:`struct` format.

    Returns:
        One tuple per record; empty for None and for empty bytes.

    Raises:
        ValueError: When the blob is not a whole number of records.
    """
    if not blob:
        return []
    unpacker = struct.Struct("<" + layout)
    if len(blob) % unpacker.size:
        raise ValueError(f"{len(blob)} bytes is not a whole number of {unpacker.size}-byte records")
    return [unpacker.unpack_from(blob, offset) for offset in range(0, len(blob), unpacker.size)]


# ------------------------------------------------------------------------------------ store view


@dataclass
class Entity:
    """One store entity, as this scan needs it.

    Attributes:
        id: The entity's id.
        prov_type: Its PROV type.
        extent: Its ``[start, end]`` in seconds, or None.
        attributes: Its attribute mapping.
    """

    id: str
    prov_type: str
    extent: tuple[float, float] | None
    attributes: dict[str, Any]


@dataclass
class StoreView:
    """A store, read once, in the shapes this scan reads it in.

    Attributes:
        entities: Every entity, in file order.
        invalidated: The ids of entities some activity invalidated.
        derived: Entity id to the ids it ``wasDerivedFrom``.
        malformed_lines: How many lines did not parse.
    """

    entities: list[Entity] = field(default_factory=list)
    invalidated: set[str] = field(default_factory=set)
    derived: dict[str, list[str]] = field(default_factory=dict)
    malformed_lines: int = 0

    def live(self, prov_type: str) -> list[Entity]:
        """Every live entity of one type, in file order.

        Args:
            prov_type: The PROV type to select.

        Returns:
            The entities that type and that no activity invalidated.
        """
        return [e for e in self.entities if e.prov_type == prov_type and e.id not in self.invalidated]

    def last(self, prov_type: str, **match: Any) -> Entity | None:  # noqa: ANN401 -- attribute values are any type
        """The last live entity of one type whose attributes match.

        Args:
            prov_type: The PROV type to select.
            **match: Attribute equality constraints.

        Returns:
            The last match in file order, or None.
        """
        found = [e for e in self.live(prov_type) if all(e.attributes.get(k) == v for k, v in match.items())]
        return found[-1] if found else None


def read_store(path: Path) -> StoreView:
    """Read one ``store.jsonl``.

    Args:
        path: The store.

    Returns:
        The view. A line that does not parse is counted and skipped, so a store still being
        appended to yields what it holds rather than raising.
    """
    view = StoreView()
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                view.malformed_lines += 1
                continue
            kind = record.get("record")
            if kind == "entity":
                extent = record.get("extent")
                view.entities.append(
                    Entity(
                        id=str(record.get("id")),
                        prov_type=str(record.get("prov_type")),
                        extent=(float(extent[0]), float(extent[1])) if extent else None,
                        attributes=record.get("attributes") or {},
                    )
                )
            elif kind == "relation":
                relation, source, target = record.get("relation"), record.get("source"), record.get("target")
                if relation == "wasInvalidatedBy":
                    view.invalidated.add(str(source))
                elif relation == "wasDerivedFrom":
                    view.derived.setdefault(str(source), []).append(str(target))
    return view


# --------------------------------------------------------------------------------------- the row


def stem_of(run_root: Path) -> str:
    """The recording's stem, from its timestamped run directory's name.

    Args:
        run_root: The directory holding ``run/`` and ``released/``.

    Returns:
        The BIDS stem, with the launch timestamp removed.
    """
    return _TIMESTAMP.sub("", run_root.name)


def identity(stem: str) -> tuple[str | None, str | None, str | None]:
    """Participant, session and task, from a stem.

    Args:
        stem: ``sub-…_ses-…_task-…``.

    Returns:
        The three, each None when the stem does not carry it.
    """
    participant = _PARTICIPANT.search(stem)
    session = _SESSION.search(stem)
    task = _TASK.search(stem)
    return (
        participant.group(1) if participant else None,
        session.group(1) if session else None,
        task.group(1) if task else None,
    )


def role_kind(role: Any) -> str | None:  # noqa: ANN401 -- a store attribute is any type
    """A branch role with its instance index removed.

    Args:
        role: The span's ``role`` attribute.

    Returns:
        ``speech_run_12`` as ``speech_run``, or None when there is no role.
    """
    return _ROLE_INDEX.sub("", str(role)) if role else None


def _npz_array(run_dir: Path, view: StoreView, name: str, key: str) -> np.ndarray | None:
    """One array from a derivative the store names.

    Args:
        run_dir: The ``run/`` directory every stored path is relative to.
        view: The store.
        name: The measurement's ``name``.
        key: The key inside the ``.npz``.

    Returns:
        The array, or None when the measurement, the file or the key is absent.
    """
    measurement = view.last("measurement", name=name)
    if measurement is None or not measurement.attributes.get("path"):
        return None
    path = run_dir / str(measurement.attributes["path"])
    try:
        with np.load(path) as archive:
            if key not in archive.files:
                return None
            return np.asarray(archive[key], dtype="float64")
    except (OSError, ValueError, EOFError):
        return None


def _waveform(run_dir: Path, view: StoreView) -> np.ndarray | None:
    """The conditioned waveform the figure draws, mono.

    Args:
        run_dir: The ``run/`` directory.
        view: The store.

    Returns:
        The samples of the first stream that decodes, in the figure's own preference order, or
        None when none does.
    """
    import soundfile

    for name in ("preemphasised", "plain"):
        stream = view.last("stream", name=name)
        if stream is None or not stream.attributes.get("path"):
            continue
        path = run_dir / str(stream.attributes["path"])
        try:
            samples, _ = soundfile.read(path, dtype="float32", always_2d=True)
        except (OSError, RuntimeError):
            continue
        return np.asarray(samples, dtype="float64").mean(axis=1)
    return None


def _measurement_readings(view: StoreView) -> dict[str, list[Any]]:
    """Every measurement reading the graph wrote, grouped by name.

    Args:
        view: The store.

    Returns:
        Measurement name to its values, in file order. Only measurements carrying a ``value`` are
        here, which is the same selection ``measure_stats`` makes.
    """
    out: dict[str, list[Any]] = {}
    for entity in view.live("measurement"):
        name, value = entity.attributes.get("name"), entity.attributes.get("value")
        if name and value is not None:
            out.setdefault(str(name), []).append(value)
    return out


def _spans(view: StoreView) -> tuple[list[Entity], int]:
    """The general spans the five-row lane draws.

    Args:
        view: The store.

    Returns:
        Live spans carrying no ``family``, an extent and a row code, in file order; and how many
        otherwise-general spans carried no row code and were left out.
    """
    general = [e for e in view.live("span") if e.attributes.get("family") is None and e.extent is not None]
    rowed = [e for e in general if _row_code(e) in SPAN_ROWS]
    return rowed, len(general) - len(rowed)


def _span_top_labels(view: StoreView, index_of: dict[str, int]) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Each span's strongest label per classifier.

    Args:
        view: The store.
        index_of: Span id to its position in the spans block.

    Returns:
        The ``(span_index, classifier, score)`` records and the label name per record.
    """
    peaks: dict[tuple[str, str], dict[str, float]] = {}
    for entity in view.live("measurement"):
        classifier = entity.attributes.get("classifier")
        span_id = entity.attributes.get("span_id")
        scores = entity.attributes.get("raw_scores")
        if not classifier or not span_id or not isinstance(scores, dict):
            continue
        slot = peaks.setdefault((str(span_id), str(classifier)), {})
        for label, score in scores.items():
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            slot[str(label)] = max(slot.get(str(label), 0.0), value)
    records: list[tuple[int, int, int]] = []
    names: list[str] = []
    for (span_id, classifier), slot in sorted(peaks.items()):
        if span_id not in index_of or not slot:
            continue
        label, score = max(slot.items(), key=lambda kv: (kv[1], kv[0]))
        code = CLASSIFIERS.index(classifier) if classifier in CLASSIFIERS else UNKNOWN_CODE
        records.append((index_of[span_id], code, quantise_value(score, *SCORE_RANGE)))
        names.append(label)
    return records, names


def _span_squim(view: StoreView, index_of: dict[str, int]) -> list[tuple[int, int, int, int]]:
    """SQUIM's three readings per span.

    Args:
        view: The store.
        index_of: Span id to its position in the spans block.

    Returns:
        One ``(span_index, stoi, pesq, si_sdr)`` record per measured span.
    """
    records: list[tuple[int, int, int, int]] = []
    for entity in view.live("assertion"):
        if entity.attributes.get("name") != "squim" or "stoi" not in entity.attributes:
            continue
        for span_id in view.derived.get(entity.id, []):
            if span_id not in index_of:
                continue
            codes = []
            for key, (low, high) in SQUIM_RANGES:
                try:
                    codes.append(quantise_value(float(entity.attributes[key]), low, high))
                except (KeyError, TypeError, ValueError):
                    codes.append(0)
            records.append((index_of[span_id], codes[0], codes[1], codes[2]))
    return records


def _branch_lanes(view: StoreView, duration_s: float) -> tuple[list[tuple[int, int, int]], list[str]]:
    """Every branch-proposed span, as a lane rectangle.

    Args:
        view: The store.
        duration_s: The recording's duration.

    Returns:
        The ``(lane, t0, t1)`` records and the role name per record.
    """
    records: list[tuple[int, int, int]] = []
    roles: list[str] = []
    for entity in view.live("span"):
        if entity.extent is None:
            continue
        family = entity.attributes.get("family")
        if entity.attributes.get("name") == REDACTION_NAME:
            lane, role = LANES.index("REDACT"), str(entity.attributes.get("category") or REDACTION_NAME)
        elif family in LANE_OF_FAMILY:
            lane, role = LANE_OF_FAMILY[str(family)], role_kind(entity.attributes.get("role")) or "unlabelled"
        else:
            continue
        records.append((lane, quantise_time(entity.extent[0], duration_s), quantise_time(entity.extent[1], duration_s)))
        roles.append(role)
    return records, roles


def _conformance(value: Any) -> str | None:  # noqa: ANN401 -- the stored value is bool or str
    """One branch's conformance, as a three-valued string.

    Args:
        value: What the branch report or the fold stored.

    Returns:
        ``"true"``, ``"false"``, ``"undetermined"``, or None when nothing was stored.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).lower()


def _number(value: Any) -> float | None:  # noqa: ANN401 -- a stored reading is any type
    """One stored reading as a finite float.

    Args:
        value: The stored value.

    Returns:
        The float, or None when the value is absent, not a number or not finite.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _gate_columns(decision: dict[str, Any]) -> dict[str, Any]:
    """Every gate VERDICT resolved, as one column group per gate plus the fold's own summary.

    Args:
        decision: The VERDICT fold's attributes.

    Returns:
        ``gate_<name>``, ``gate_<name>_bound`` and ``gate_<name>_passed`` for every gate the fold
        applied, and the summary columns. A gate the fold did not apply is null in all three.
    """
    gates = decision.get("gates")
    gates = gates if isinstance(gates, dict) else {}
    applied = [g for g in (gates.get("applied") or []) if isinstance(g, dict)]
    flagging = [g for g in (gates.get("flagging") or []) if isinstance(g, dict)]

    row: dict[str, Any] = {
        "gate_node": gates.get("node"),
        "gate_group": gates.get("group"),
        "gate_family": gates.get("family"),
    }
    outcomes: dict[str, str | None] = {}
    for record in (*applied, *flagging):
        name = str(record.get("gate"))
        if name not in GATE_NAMES:
            continue
        row[f"gate_{name}"] = _number(record.get("value"))
        row[f"gate_{name}_bound"] = _number(record.get("bound"))
        outcomes[name] = _conformance(record.get("passed"))
        row[f"gate_{name}_passed"] = outcomes[name]

    row["gate_applied_n"] = len(applied)
    row["gate_flagging_n"] = len(flagging)
    row["gate_failed_n"] = sum(1 for v in outcomes.values() if v == "false")
    row["gate_undetermined_n"] = sum(1 for v in outcomes.values() if v == GATE_UNDETERMINED)
    row["gate_failed_names"] = sorted(name for name, v in outcomes.items() if v == "false")
    row["gate_flagged_names"] = sorted(str(g.get("gate")) for g in flagging if _conformance(g.get("passed")) == "false")
    return row


def _residual_levels(view: StoreView) -> dict[str, float | None]:
    """The enhanced and residual stream levels, and how far the first sits above the second.

    Args:
        view: The store.

    Returns:
        The residual's own stored levels, the enhanced stream's RMS, and the two RMS differences.
        All null when PREPROCESS wrote no residual measurement.

    The enhanced RMS and both differences are reconstructed from the two energy fractions the
    measurement already carries, which share one denominator and one aligned length; the
    derivation is in ``specs/20260923-enhanced-over-residual/design.md``.
    """
    keys = (
        "residual_peak_dbfs",
        "residual_rms_dbfs",
        "enhanced_rms_dbfs",
        "enhanced_over_residual_rms_db",
        "enhanced_over_residual_rms_fitted_db",
    )
    out: dict[str, float | None] = dict.fromkeys(keys)
    measurement = view.last("measurement", name=RESIDUAL_MEASUREMENT)
    if measurement is None:
        return out
    attributes = measurement.attributes
    out["residual_peak_dbfs"] = _number(attributes.get("peak_dbfs"))
    residual_rms = _number(attributes.get("rms_dbfs"))
    out["residual_rms_dbfs"] = residual_rms
    enhanced_fraction = _number(attributes.get("enhanced_energy_fraction"))
    residual_fraction = _number(attributes.get("energy_fraction"))
    if not enhanced_fraction or not residual_fraction or enhanced_fraction <= 0 or residual_fraction <= 0:
        return out
    difference = 10.0 * math.log10(enhanced_fraction / residual_fraction)
    out["enhanced_over_residual_rms_db"] = difference
    if residual_rms is not None:
        out["enhanced_rms_dbfs"] = residual_rms + difference
    gain_db = _number(attributes.get("gain_db"))
    if gain_db is not None:
        out["enhanced_over_residual_rms_fitted_db"] = difference + gain_db
    return out


def _speaker_columns(view: StoreView, readings: dict[str, list[Any]]) -> dict[str, Any]:
    """What the multi-speaker instrument found: separation, the secondary runs and the solo run.

    Args:
        view: The store.
        readings: Every measurement reading, grouped by name.

    Returns:
        How many sources were separated, the secondary runs and their total seconds, the solo
        run's seconds, and the worst dominant-speaker share over the recording's task extents.

    The share is folded by ``min`` because that is the fold VERDICT's flag gate applies; the
    ``m_extent_dominant_speaker_share`` column beside it is the mean, as every measurement column
    is, and the two disagree whenever a recording carries more than one task extent.
    """
    separated = [e for e in view.live("stream") if str(e.attributes.get("name") or "").startswith(SEPARATED_PREFIX)]
    secondary = [e for e in view.live("span") if e.attributes.get("role") == SECONDARY_ROLE and e.extent]
    solo = [e for e in view.live("span") if e.attributes.get("role") == SOLO_ROLE and e.extent]
    shares = [v for v in (_number(value) for value in readings.get(DOMINANT_SHARE, [])) if v is not None]
    return {
        "separated_n": len(separated),
        "secondary_extent_n": len(secondary),
        "secondary_extent_s": sum(e.extent[1] - e.extent[0] for e in secondary if e.extent) if secondary else None,
        "solo_extent_s": sum(e.extent[1] - e.extent[0] for e in solo if e.extent) if solo else None,
        "extent_dominant_speaker_share_min": min(shares) if shares else None,
    }


def extract(run_root: Path, root: Path, anomalies: dict[str, int] | None = None) -> dict[str, Any] | None:
    """One recording's row.

    Args:
        run_root: The directory holding ``run/store.jsonl``.
        root: The scan root, which ``run_dir`` is reported relative to.
        anomalies: Counted into, per measurement name the schema has no column for.

    Returns:
        The row, or None when the store is unreadable or the fold has not been written yet.
    """
    run_dir = run_root / RUN_SUBDIR
    try:
        view = read_store(run_dir / STORE_NAME)
    except OSError:
        return None
    fold = view.last("verdict", node="VERDICT")
    if fold is None:
        return None
    decision = fold.attributes
    stem = stem_of(run_root)
    participant, session, task = identity(stem)

    recording = view.last("stream", name="recording")
    conditioned = view.last("stream", name="preemphasised") or view.last("stream", name="plain")
    duration_s = recording.extent[1] if recording and recording.extent else None
    conditioned_s = conditioned.extent[1] if conditioned and conditioned.extent else None
    scale_s = conditioned_s or duration_s or 0.0

    row: dict[str, Any] = {
        "participant": participant,
        "task": task,
        "verdict": decision.get("triage"),
        "session": session,
        "stem": stem,
        "run_dir": str(run_root.relative_to(root)) if run_root.is_relative_to(root) else str(run_root),
        "declared_family": decision.get("declared_family"),
        "release": decision.get("release"),
        "release_ground": decision.get("release_ground"),
        "grounds": decision.get("discard_ground"),
        "route_state": decision.get("route_state"),
        "duration_s": duration_s,
        "duration_conditioned_s": conditioned_s,
        "time_scale_s": scale_s or None,
        "sampling_rate": int(conditioned.attributes["sampling_rate"])
        if conditioned and conditioned.attributes.get("sampling_rate")
        else None,
        "schema_version": SCHEMA_VERSION,
        "malformed_store_lines": view.malformed_lines,
    }

    reasons = decision.get("reasons") or []
    flagged = [str(r.get("node")) for r in reasons if isinstance(r, dict) and r.get("outcome") in FLAG_OUTCOMES]
    row["flags_n"] = len(flagged)
    row["flag_nodes"] = flagged
    conformance = decision.get("conformance") or {}
    routes = decision.get("routes") or {}
    for node in BRANCH_NODES:
        row[f"conformance_{node.lower()}"] = _conformance(conformance.get(node))
    for branch in ROUTED_BRANCHES:
        row[f"route_{branch.lower()}"] = routes.get(branch)
    row.update(_gate_columns(decision))
    row.update(_residual_levels(view))

    speech_report = view.last("branch_report", node="SPEECH")
    scanned = bool(speech_report and isinstance(speech_report.attributes.get("pii"), dict))
    pii_entities = [e for e in view.live("pii") if e.extent is not None]
    row["pii_findings_n"] = len(pii_entities) if scanned else None

    readings = _measurement_readings(view)
    row.update(_speaker_columns(view, readings))
    if anomalies is not None:
        for name in readings:
            if name not in MEASUREMENTS:
                anomalies[name] = anomalies.get(name, 0) + 1
    for name in MEASUREMENTS:
        values = readings.get(name, [])
        row[f"m_{name}_n"] = len(values)
        if name in SCALAR_MEASUREMENTS:
            numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
            numbers = [v for v in numbers if math.isfinite(v)]
            row[f"m_{name}"] = sum(numbers) / len(numbers) if numbers else None
        elif name in VECTOR_MEASUREMENTS:
            flat = [None if v is None else float(v) for value in values for v in _as_list(value)]
            row[f"m_{name}"] = flat if values else None
        elif name in MATRIX_MEASUREMENTS:
            rows = [_as_list(inner) for value in values for inner in _as_list(value)]
            width = max((len(r) for r in rows), default=0)
            row[f"m_{name}"] = [None if v is None else float(v) for r in rows for v in r] if values else None
            row[f"m_{name}_width"] = width if values else None
        else:
            row[f"m_{name}"] = sorted({str(v) for v in values}) if values else None

    envelope = _npz_array(run_dir, view, "energy_envelope", "envelope_dbfs")
    floor = _npz_array(run_dir, view, "energy_envelope", "floor_dbfs")
    continuity = _npz_array(run_dir, view, "continuity_trace", "continuity")
    samples = _waveform(run_dir, view)
    waveform, peak = encode_waveform(samples)
    row["wave_minmax"] = waveform
    row["wave_peak"] = peak
    row["env_dbfs"] = encode_trace(envelope, *ENVELOPE_DBFS_RANGE, how="max")
    row["floor_dbfs"] = float(floor[0]) if floor is not None and floor.size and math.isfinite(floor[0]) else None
    row["continuity"] = encode_trace(continuity, *CONTINUITY_RANGE, how="mean")

    preprocess = view.last("verdict", node="PREPROCESS")
    spans, unrowed = _spans(view)
    index_of = {entity.id: index for index, entity in enumerate(spans)}
    row["spans_unrowed_n"] = unrowed if preprocess is not None else None
    if preprocess is None:
        row["spans"] = row["span_labels"] = row["span_label_name"] = row["span_squim"] = None
    else:
        row["spans"] = encode_records(
            (
                (
                    SPAN_ROWS.index(_row_code(e)),
                    quantise_time(e.extent[0], scale_s),
                    quantise_time(e.extent[1], scale_s),
                )
                for e in spans
                if e.extent is not None
            ),
            SPANS_LAYOUT,
        )
        labels, names = _span_top_labels(view, index_of)
        row["span_labels"] = encode_records(labels, SPAN_LABELS_LAYOUT)
        row["span_label_name"] = names
        row["span_squim"] = encode_records(_span_squim(view, index_of), SPAN_SQUIM_LAYOUT)

    transcript = view.last("measurement", name="consensus_transcript")
    if transcript is None:
        row["asr_words"] = row["asr_word_text"] = None
    else:
        words = sorted(
            (e for e in view.live("word") if e.extent is not None),
            key=lambda e: int(e.attributes.get("index", 0)),
        )
        row["asr_words"] = encode_records(
            (
                (
                    quantise_time(e.extent[0], scale_s),
                    quantise_time(e.extent[1], scale_s),
                    WORD_OUTCOMES.index(str(e.attributes.get("outcome")))
                    if str(e.attributes.get("outcome")) in WORD_OUTCOMES
                    else UNKNOWN_CODE,
                )
                for e in words
                if e.extent is not None
            ),
            ASR_WORDS_LAYOUT,
        )
        row["asr_word_text"] = [str(e.attributes.get("text") or "") for e in words]

    if not scanned:
        row["pii_marks"] = row["pii_category"] = None
    else:
        row["pii_marks"] = encode_records(
            (
                (quantise_time(e.extent[0], scale_s), quantise_time(e.extent[1], scale_s))
                for e in pii_entities
                if e.extent is not None
            ),
            PII_MARKS_LAYOUT,
        )
        row["pii_category"] = [str(e.attributes.get("category") or "") for e in pii_entities]

    if not view.live("branch_decision"):
        row["branch_lanes"] = row["branch_lane_role"] = None
    else:
        lanes, roles = _branch_lanes(view, scale_s)
        row["branch_lanes"] = encode_records(lanes, BRANCH_LANES_LAYOUT)
        row["branch_lane_role"] = roles
    return row


def _as_list(value: Any) -> list[Any]:  # noqa: ANN401 -- a measurement's value is its own type
    """A stored value as a list.

    Args:
        value: The value.

    Returns:
        The value itself when it is a list, otherwise a one-element list.
    """
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _row_code(entity: Entity) -> str:
    """The five-row lane code for one span.

    Args:
        entity: The span.

    Returns:
        ``E``, ``C``, ``A``, ``S``, ``G`` or ``?``.
    """
    measure = str(entity.attributes.get("measure"))
    if measure == "amplitude" and entity.attributes.get("signal") == "normalized":
        return "S"
    return MEASURE_ROW.get(measure, "?")


# ---------------------------------------------------------------------------------------- schema


def schema() -> pa.Schema:
    """The parquet schema, in column order.

    Returns:
        The schema every shard is written with, so shards concatenate without a cast.
    """
    fields = [
        pa.field("participant", pa.string()),
        pa.field("task", pa.string()),
        pa.field("verdict", pa.string()),
        pa.field("session", pa.string()),
        pa.field("stem", pa.string()),
        pa.field("run_dir", pa.string()),
        pa.field("declared_family", pa.string()),
        pa.field("release", pa.string()),
        pa.field("release_ground", pa.string()),
        pa.field("grounds", pa.string()),
        pa.field("route_state", pa.string()),
        pa.field("duration_s", pa.float64()),
        pa.field("duration_conditioned_s", pa.float64()),
        pa.field("time_scale_s", pa.float64()),
        pa.field("sampling_rate", pa.int32()),
        pa.field("schema_version", pa.int32()),
        pa.field("malformed_store_lines", pa.int32()),
        pa.field("flags_n", pa.int32()),
        pa.field("flag_nodes", pa.list_(pa.string())),
        *[pa.field(f"conformance_{node.lower()}", pa.string()) for node in BRANCH_NODES],
        *[pa.field(f"route_{branch.lower()}", pa.string()) for branch in ROUTED_BRANCHES],
        pa.field("pii_findings_n", pa.int32()),
        pa.field("gate_node", pa.string()),
        pa.field("gate_group", pa.string()),
        pa.field("gate_family", pa.string()),
        pa.field("gate_applied_n", pa.int32()),
        pa.field("gate_flagging_n", pa.int32()),
        pa.field("gate_failed_n", pa.int32()),
        pa.field("gate_undetermined_n", pa.int32()),
        pa.field("gate_failed_names", pa.list_(pa.string())),
        pa.field("gate_flagged_names", pa.list_(pa.string())),
        *[
            field
            for name in GATE_NAMES
            for field in (
                pa.field(f"gate_{name}", pa.float64()),
                pa.field(f"gate_{name}_bound", pa.float64()),
                pa.field(f"gate_{name}_passed", pa.string()),
            )
        ],
        pa.field("separated_n", pa.int32()),
        pa.field("secondary_extent_n", pa.int32()),
        pa.field("secondary_extent_s", pa.float64()),
        pa.field("solo_extent_s", pa.float64()),
        pa.field("extent_dominant_speaker_share_min", pa.float64()),
        pa.field("residual_peak_dbfs", pa.float64()),
        pa.field("residual_rms_dbfs", pa.float64()),
        pa.field("enhanced_rms_dbfs", pa.float64()),
        pa.field("enhanced_over_residual_rms_db", pa.float64()),
        pa.field("enhanced_over_residual_rms_fitted_db", pa.float64()),
    ]
    for name in MEASUREMENTS:
        if name in SCALAR_MEASUREMENTS:
            fields.append(pa.field(f"m_{name}", pa.float64()))
        elif name in VECTOR_MEASUREMENTS or name in MATRIX_MEASUREMENTS:
            fields.append(pa.field(f"m_{name}", pa.list_(pa.float64())))
        else:
            fields.append(pa.field(f"m_{name}", pa.list_(pa.string())))
        fields.append(pa.field(f"m_{name}_n", pa.int32()))
        if name in MATRIX_MEASUREMENTS:
            fields.append(pa.field(f"m_{name}_width", pa.int32()))
    fields += [
        pa.field("wave_minmax", pa.binary()),
        pa.field("wave_peak", pa.float64()),
        pa.field("env_dbfs", pa.binary()),
        pa.field("floor_dbfs", pa.float64()),
        pa.field("continuity", pa.binary()),
        pa.field("spans", pa.binary()),
        pa.field("spans_unrowed_n", pa.int32()),
        pa.field("span_labels", pa.binary()),
        pa.field("span_label_name", pa.list_(pa.string())),
        pa.field("span_squim", pa.binary()),
        pa.field("asr_words", pa.binary()),
        pa.field("asr_word_text", pa.list_(pa.string())),
        pa.field("pii_marks", pa.binary()),
        pa.field("pii_category", pa.list_(pa.string())),
        pa.field("branch_lanes", pa.binary()),
        pa.field("branch_lane_role", pa.list_(pa.string())),
    ]
    return pa.schema(fields, metadata={b"senselab.recording_vectors.schema_version": str(SCHEMA_VERSION).encode()})


def to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """Rows as a table on :func:`schema`.

    Args:
        rows: What :func:`extract` returned, one per recording.

    Returns:
        The table. A key a row omits is null, never a default.
    """
    target = schema()
    columns = [pa.array([row.get(f.name) for row in rows], type=f.type) for f in target]
    return pa.Table.from_arrays(columns, schema=target)


# ------------------------------------------------------------------------------------- the scan


def shard_of(stem: str, slices: int) -> int:
    """Which shard owns one recording.

    Args:
        stem: The recording's stem.
        slices: How many shards share the tree.

    Returns:
        The shard index. Content-addressed, so it does not move when the tree grows.
    """
    return int(hashlib.sha1(stem.encode(), usedforsecurity=False).hexdigest()[:8], 16) % slices


def recording_dirs(root: Path) -> Iterator[Path]:
    """Every run directory under a tree, tolerating one that is still being written.

    Args:
        root: The tree of ``sub-*/ses-*/<stem>_<timestamp>/`` directories.

    Yields:
        Each directory holding a ``run/store.jsonl``.
    """
    for participant in _listdir(root):
        for session in _listdir(participant):
            for recording in _listdir(session):
                if (recording / RUN_SUBDIR / STORE_NAME).exists():
                    yield recording


def _listdir(path: Path) -> list[Path]:
    """The subdirectories of one directory, or nothing when it is gone.

    Args:
        path: The directory.

    Returns:
        Its subdirectories, sorted; empty when it cannot be read.
    """
    try:
        return sorted(p for p in path.iterdir() if p.is_dir())
    except OSError:
        return []


@dataclass
class ScanReport:
    """What one shard met.

    Attributes:
        considered: Run directories the shard owned.
        written: Rows written.
        incomplete: Directories whose store carries no VERDICT fold yet.
        unreadable: Directories whose store could not be read.
        superseded: Re-runs of a stem this shard skipped in favour of a later one.
        anomalies: Measurement names the store carried that the schema does not.
    """

    considered: int = 0
    written: int = 0
    incomplete: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    anomalies: dict[str, int] = field(default_factory=dict)


def scan(root: Path, slice_index: int = 0, slices: int = 1) -> tuple[list[dict[str, Any]], ScanReport]:
    """Extract every row this shard owns.

    Args:
        root: The tree of finished run directories.
        slice_index: This shard's index.
        slices: How many shards share the tree.

    Returns:
        The rows and what the shard met.
    """
    report = ScanReport()
    latest: dict[str, Path] = {}
    for recording in recording_dirs(root):
        stem = stem_of(recording)
        if shard_of(stem, slices) != slice_index:
            continue
        report.considered += 1
        previous = latest.get(stem)
        if previous is None or recording.name > previous.name:
            if previous is not None:
                report.superseded.append(previous.name)
            latest[stem] = recording
        else:
            report.superseded.append(recording.name)
    rows: list[dict[str, Any]] = []
    for _, recording in sorted(latest.items()):
        try:
            row = extract(recording, root, report.anomalies)
        except (OSError, ValueError, KeyError, TypeError):
            report.unreadable.append(recording.name)
            continue
        if row is None:
            report.incomplete.append(recording.name)
            continue
        rows.append(row)
        report.written += 1
    return rows, report
