"""The shapes a triage node returns, and its store conventions.

There are two, because there are two kinds of node. A node that **decides** returns a
:class:`NodeResult` and writes a ``verdict`` entity. A node that **reports** — the three branches
and QUALITY — returns a :class:`BranchResult` and writes a ``branch_report`` entity carrying its
task conformance and its deviations and no outcome. ``vocabulary.fold_file_verdict`` is where every
decision about the recording is made. See
``specs/20260817-triage-workflow-dag/branch-conventions.md``.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import CppsSettings
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.vocabulary import (
    CONFORMANCE_REFERENTS,
    BranchReport,
    Conformance,
    NodeVerdict,
    Outcome,
    Triage,
)
from senselab.utils.portable_audio_io import NORMALIZE, AudioWriteReport
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore, file_attributes
from senselab.utils.subprocess_venv import venv_environment

RESERVED_REPORT_KEYS = frozenset(
    {"node", "kind", "conformance", "conformance_of", "deviations", "unmeasured", "in_family", "outcome"}
)
"""Attribute names a ``branch_report``'s detail may not carry, ``outcome`` among them."""

MESSAGE_CAP = 200
"""How many characters of an exception's message are recorded."""

STREAM_SUFFIX = ".flac"
"""Container a persisted stream gets. See ``specs/20260907-triage-stream-compression/design.md``."""

HOST_ENV_PACKAGES = ("torch", "torchaudio", "torchcodec", "transformers", "numpy", "scipy", "librosa")
"""The packages whose versions are recorded for the host environment.

See ``specs/20260908-triage-prov-bep028/design.md``."""


def describe_exception(error: BaseException) -> str:
    """An exception as ``"Class: first line"``, bounded, for a record a reader can act on.

    Args:
        error: The exception.

    Returns:
        The class name alone when the exception carries no message, else the class name and the
        message's first line truncated to :data:`MESSAGE_CAP` characters with an ellipsis.
    """
    name = type(error).__name__
    message = str(error).strip().splitlines()
    if not message or not message[0]:
        return name
    first = message[0]
    if len(first) > MESSAGE_CAP:
        first = first[: MESSAGE_CAP - 3] + "..."
    return f"{name}: {first}"


@dataclass(frozen=True)
class NodeResult:
    """What a node that decides returns.

    Attributes:
        verdict: The node's conclusion, in the graph's shared vocabulary.
        view: Ids of the store entities this node wrote or asserted over.
        verdict_entity_id: The verdict entity this node wrote to the store.
    """

    verdict: NodeVerdict
    view: tuple[str, ...]
    verdict_entity_id: str


@dataclass(frozen=True)
class BranchResult:
    """What a node that reports returns. It carries no outcome.

    Attributes:
        report: What the node observed: its task conformance and its deviations.
        view: Ids of the store entities this node wrote or asserted over, its spans included.
        report_entity_id: The ``branch_report`` entity this node wrote to the store.
    """

    report: BranchReport
    view: tuple[str, ...]
    report_entity_id: str


def software_agent(store: ProvStore) -> str:
    """The agent for work senselab itself performed, at the installed version.

    Args:
        store: The provenance store.

    Returns:
        The agent's id.
    """
    return store.agent(agent_type="software", version=f"senselab {version('senselab')}")


def host_environment(store: ProvStore) -> str:
    """Add the host interpreter's environment: python version, platform, and senselab's own version.

    Args:
        store: The provenance store.

    Returns:
        The environment's id.
    """
    dependencies: dict[str, str] = {}
    for name in HOST_ENV_PACKAGES:
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            continue
    return store.environment(
        kind="host",
        label="host",
        python_version=platform.python_version(),
        operating_system=platform.platform(),
        dependencies=dependencies,
        senselab_version=version("senselab"),
    )


def capture_environments(store: ProvStore, used_venvs: dict[str, Path]) -> list[str]:
    """Add one environment entity for the host, plus one for each subprocess venv actually used.

    Args:
        store: The provenance store.
        used_venvs: Venv backend name to resolved directory, collected by wrapping the run's node
            execution in :func:`~senselab.utils.subprocess_venv.record_venv_use`. Empty when the run
            reached no subprocess venv.

    Returns:
        The ids added, host first, then one per venv in name order.
    """
    ids = [host_environment(store)]
    for name, venv_dir in sorted(used_venvs.items()):
        ids.append(store.environment(kind="venv", **venv_environment(name, venv_dir)))
    return ids


def write_verdict(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    node: str,
    outcome: Outcome | Triage,
    kind: str | None,
    why: str,
    detail: dict[str, Any],
) -> tuple[str, NodeVerdict]:
    """Write one node's verdict entity.

    Args:
        store: The provenance store.
        activity_id: The activity that concluded.
        agent_id: The agent answerable for the verdict.
        node: The node's name.
        outcome: What it concluded — an ``Outcome`` for every node, a ``Triage`` for the file fold.
        kind: The kind the node screens, or None.
        why: The reason, in controlled vocabulary — never transcript text.
        detail: The node's verdict fields.

    Returns:
        The verdict entity's id and the vocabulary verdict.

    Raises:
        ValueError: If ``detail`` carries any of the reserved keys ``node``, ``outcome``, ``kind``
            or ``why``.
    """
    shadowed = detail.keys() & {"node", "outcome", "kind", "why"}
    if shadowed:
        raise ValueError(f"detail must not shadow the reserved verdict keys: {sorted(shadowed)}")
    entity_id = store.entity(
        prov_type="verdict",
        extent=None,
        attributes={"node": node, "outcome": outcome.value, "kind": kind, "why": why, **detail},
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    return entity_id, NodeVerdict(node=node, outcome=outcome, kind=kind, why=why)


def write_report(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    node: str,
    kind: str | None,
    conformance: Conformance,
    conformance_of: str,
    deviations: tuple[str, ...],
    unmeasured: tuple[str, ...] = (),
    in_family: bool = False,
    detail: dict[str, Any],
) -> tuple[str, BranchReport]:
    """Write one reporting node's ``branch_report`` entity.

    Args:
        store: The provenance store.
        activity_id: The activity that observed.
        agent_id: The agent answerable for the report.
        node: The node's name.
        kind: The kind it reports on, or None.
        conformance: Whether what was asked for happened.
        conformance_of: What that conformance is about, one of
            :data:`~senselab.audio.workflows.triage.vocabulary.CONFORMANCE_REFERENTS`.
        deviations: The deviation type names found, sorted and deduplicated.
        unmeasured: The config paths this node asked for and nobody has measured, in read order.
        in_family: Whether the node evaluated a declared task of its own kind.
        detail: The node's observation fields.

    Returns:
        The report entity's id and the vocabulary report.

    Raises:
        ValueError: If ``conformance_of`` is not a known referent, or if ``detail`` carries any of
            :data:`RESERVED_REPORT_KEYS`.
    """
    if conformance_of not in CONFORMANCE_REFERENTS:
        raise ValueError(f"conformance_of must be one of {list(CONFORMANCE_REFERENTS)}; got {conformance_of!r}")
    shadowed = detail.keys() & RESERVED_REPORT_KEYS
    if shadowed:
        raise ValueError(f"detail must not shadow the reserved report keys: {sorted(shadowed)}")
    entity_id = store.entity(
        prov_type="branch_report",
        extent=None,
        attributes={
            "node": node,
            "kind": kind,
            "conformance": conformance,
            "conformance_of": conformance_of,
            "deviations": list(deviations),
            "unmeasured": list(unmeasured),
            "in_family": in_family,
            **detail,
        },
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    return entity_id, BranchReport(
        node=node,
        kind=kind,
        conformance=conformance,
        conformance_of=conformance_of,
        deviations=tuple(deviations),
        unmeasured=tuple(unmeasured),
        in_family=in_family,
    )


def write_measurement(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    name: str,
    signal: str,
    attributes: dict[str, Any],
    derived_from: tuple[str, ...] = (),
    extent: tuple[float, float] | None = None,
) -> str:
    """Write one derivative measurement entity with its provenance.

    Args:
        store: The provenance store.
        activity_id: The activity that measured.
        agent_id: The agent answerable for the measurement.
        name: The measurement's name, which is how readers find it.
        signal: The stream it was measured over.
        attributes: The measurement's own fields.
        derived_from: Ids this measurement was derived from.
        extent: The ``(start, end)`` it covers, in seconds, or None for a whole-file measurement.

    Returns:
        The measurement entity's id.
    """
    entity_id = store.entity(
        prov_type="measurement", extent=extent, attributes={"name": name, "signal": signal, **attributes}
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    for source_id in derived_from:
        store.was_derived_from(entity_id, source_id)
    return entity_id


def clamp_extent(extent: tuple[float, float], audio: Audio) -> tuple[float, float]:
    """Bound an extent's end by the decoded audio, when the overshoot is under one sample period.

    Args:
        extent: The ``(start, end)`` about to be sliced, in seconds.
        audio: The audio being sliced; the length it decoded to is the bound.

    Returns:
        The extent, with ``end`` replaced by the audio's duration when it overshot within tolerance.

    Raises:
        ValueError: If ``end`` exceeds the duration by more than one sample period.
    """
    start, end = float(extent[0]), float(extent[1])
    sampling_rate = int(audio.sampling_rate)
    duration = audio.waveform.shape[-1] / sampling_rate
    if end <= duration:
        return start, end
    if (end - duration) * sampling_rate > 1.0:
        raise ValueError(
            f"extent ends at {end}s, past the {duration}s this audio decoded to by "
            f"{(end - duration) * sampling_rate:.3f} samples; more than one sample period outside "
            "the recording is an inconsistency, not rounding"
        )
    return start, duration


def bound_reading(extent: tuple[float, float], audio: Audio) -> tuple[float, float] | None:
    """Bound another instrument's reading by the audio this node slices, however far it reaches.

    Any overshoot is truncated rather than refused; :func:`clamp_extent` is the stricter form, for
    an extent this node composed itself.

    Args:
        extent: The ``(start, end)`` the instrument reported, in seconds.
        audio: The audio being sliced; the length it decoded to is the bound.

    Returns:
        The reading bounded by the audio's duration, or None when it names no part of the audio.
    """
    start, end = float(extent[0]), float(extent[1])
    duration = audio.waveform.shape[-1] / int(audio.sampling_rate)
    if start >= duration:
        return None
    end = min(end, duration)
    return (start, end) if end > start else None


def find_measurement(store: ProvStore, name: str) -> Entity | None:
    """The latest non-invalidated measurement entity carrying this name, or None.

    Args:
        store: The provenance store.
        name: The measurement's ``name`` attribute.

    Returns:
        The entity, or None when nothing live carries the name.
    """
    found = [
        e for e in store.entities("measurement") if e.attributes.get("name") == name and not store.is_invalidated(e.id)
    ]
    return found[-1] if found else None


def find_verdict(store: ProvStore, node: str) -> Entity | None:
    """The latest non-invalidated verdict entity one node wrote, or None.

    Args:
        store: The provenance store.
        node: The node's name, as the verdict's ``node`` attribute carries it.

    Returns:
        The entity, or None when that node concluded nothing that is still live.
    """
    found = [
        e for e in store.entities("verdict") if e.attributes.get("node") == node and not store.is_invalidated(e.id)
    ]
    return found[-1] if found else None


def find_branch_report(store: ProvStore, node: str) -> Entity | None:
    """The latest non-invalidated ``branch_report`` entity one node wrote, or None.

    Args:
        store: The provenance store.
        node: The node's name, as the report's ``node`` attribute carries it.

    Returns:
        The entity, or None when that node reported nothing that is still live.
    """
    found = [
        e
        for e in store.entities("branch_report")
        if e.attributes.get("node") == node and not store.is_invalidated(e.id)
    ]
    return found[-1] if found else None


def find_measurements(store: ProvStore, name: str) -> list[Entity]:
    """Every live measurement entity carrying this name, in write order.

    The plural of :func:`find_measurement`, for a name one node writes many of.

    Args:
        store: The provenance store.
        name: The measurement's ``name`` attribute.

    Returns:
        The entities, oldest first. Empty when nothing live carries the name.
    """
    return [
        e for e in store.entities("measurement") if e.attributes.get("name") == name and not store.is_invalidated(e.id)
    ]


def live_entities(store: ProvStore, prov_type: PROV_TYPE) -> list[Entity]:
    """Every non-invalidated entity of one type, in write order.

    Args:
        store: The provenance store.
        prov_type: The entity type to read.

    Returns:
        The live entities, oldest first.
    """
    return [e for e in store.entities(prov_type) if not store.is_invalidated(e.id)]


def consensus_words(store: ProvStore) -> list[Entity]:
    """The consensus stream: every live ``word`` entity, in ``index`` order.

    ``index`` is the position PREPROCESS's consensus emitted the word at, and is the only order a
    reader may use.

    Args:
        store: The provenance store.

    Returns:
        The live ``word`` entities, sorted by their ``index`` attribute.
    """
    return sorted(live_entities(store, "word"), key=lambda word: int(word.attributes["index"]))


def lexical_words(store: ProvStore) -> list[Entity]:
    """The consensus words that are not bracketed, in ``index`` order.

    Args:
        store: The provenance store.

    Returns:
        The subset of :func:`consensus_words` whose ``bracketed`` attribute is False.
    """
    return [word for word in consensus_words(store) if not word.attributes["bracketed"]]


def word_hull(word: Entity) -> tuple[float, float]:
    """The hull of a word's per-source timings — every recognizer's placement of it.

    The union of the derived extent and every source's own reading.

    Args:
        word: A consensus ``word`` entity.

    Returns:
        ``(min member start, max member end)``, or the derived extent when no source timed it.
    """
    spans = [tuple(span) for span in (word.attributes.get("timings") or {}).values()]
    if word.extent is not None:
        spans.append(word.extent)
    if not spans:
        return (0.0, 0.0)
    return min(float(span[0]) for span in spans), max(float(span[1]) for span in spans)


PITCH_NARROWING_KEYS = (
    "pitch_floor_divisor",
    "pitch_ceiling_quartile_multiplier",
    "pitch_pinned_percentile",
    "pitch_excursion_multiplier",
    "pitch_pinned_octave_ratio",
)
"""The five coefficients of the per-recording F0 narrowing, as ``praat_features`` config keys."""

F0_RANGE_PARAMETER_NAMES = ("search_floor_hz", "search_ceiling_hz", *PITCH_NARROWING_KEYS)
"""Every parameter ``derive_f0_range`` takes besides the audio, in the order it declares them."""


def f0_range_parameters(config: TriageConfig) -> dict[str, float]:
    """Read the wide search range and the five narrowing coefficients.

    Args:
        config: The triage configuration.

    Returns:
        The wide search bounds and the five coefficients, keyed by the parameter names
        ``derive_f0_range`` and ``extract_pitch_values`` take.

    Raises:
        ValueError: If the search range or any coefficient is unmeasured.
    """
    search = config.require("voice.f0_search_range_hz")
    parameters = {"search_floor_hz": float(search[0]), "search_ceiling_hz": float(search[1])}
    parameters.update({name: float(config.require(f"praat_features.{name}")) for name in PITCH_NARROWING_KEYS})
    return parameters


CPPS_SCALAR_KEYS = (
    "pitch_floor_hz",
    "time_step_s",
    "max_frequency_hz",
    "preemphasis_from_hz",
    "time_averaging_s",
    "quefrency_averaging_s",
    "robust_tolerance",
)
"""The ``praat_features.cpps`` keys that are a bare number, in the order the settings declare them."""


def cpps_settings(config: TriageConfig) -> CppsSettings:
    """Read every setting the smoothed cepstral peak prominence is computed under.

    Args:
        config: The triage configuration.

    Returns:
        The settings, built from ``praat_features.cpps``.

    Raises:
        ValueError: If any key is unmeasured.
    """
    peak_floor, peak_ceiling = config.require("praat_features.cpps.peak_search_range_hz")
    trend_start, trend_end = config.require("praat_features.cpps.trend_range_s")
    scalars = {name: float(config.require(f"praat_features.cpps.{name}")) for name in CPPS_SCALAR_KEYS}
    return CppsSettings(
        peak_search_floor_hz=float(peak_floor),
        peak_search_ceiling_hz=float(peak_ceiling),
        trend_start_s=float(trend_start),
        trend_end_s=float(trend_end),
        subtract_tilt_before_smoothing=bool(config.require("praat_features.cpps.subtract_tilt_before_smoothing")),
        tilt_line_type=str(config.require("praat_features.cpps.tilt_line_type")),
        peak_interpolation=str(config.require("praat_features.cpps.peak_interpolation")),
        **scalars,
    )


def resolve_stream(store: ProvStore, run_dir: Path, name: str) -> tuple[str, Audio]:
    """Load a stream the graph wrote earlier, by its name.

    Args:
        store: The provenance store.
        run_dir: The run directory sidecar paths are relative to.
        name: The stream entity's ``name`` attribute.

    Returns:
        The stream entity's id and its audio, loaded lazily from the sidecar.

    Raises:
        LookupError: If no live stream entity carries that name.
    """
    found = [e for e in store.entities("stream") if e.attributes.get("name") == name and not store.is_invalidated(e.id)]
    if not found:
        raise LookupError(f"no stream named {name!r} in the store; the node that writes it has not run")
    entity = found[-1]
    path = Path(entity.attributes["path"])
    if not path.is_absolute():
        path = run_dir / path
    return entity.id, Audio(filepath=str(path))


def write_stream(audio: Audio, run_dir: Path, stem: str) -> tuple[str, AudioWriteReport]:
    """Persist a stream under ``run_dir/streams/<stem><STREAM_SUFFIX>``.

    Written with ``out_of_range="normalize"``, which never truncates: only a peak the write would
    otherwise clip is scaled down, and the gain comes back in the report. See
    ``specs/20260907-triage-stream-compression/design.md``.

    Any existing entry at the destination is removed first, so the write replaces the name rather
    than following whatever it points at. See
    ``specs/20260922-replay-decisions-over-a-finished-corpus/design.md``.

    Args:
        audio: The stream's audio.
        run_dir: The run directory streams live under.
        stem: The stream's file stem, e.g. ``"plain"`` or ``"separated_0"``.

    Returns:
        ``(relative_path, report)`` — the path to record as the stream entity's ``"path"``
        attribute (relative to ``run_dir``), and the write report (``report.gain`` is 1.0 unless
        the write scaled the samples down to fit).
    """
    relative = f"streams/{stem}{STREAM_SUFFIX}"
    destination = run_dir / relative
    destination.unlink(missing_ok=True)
    report = audio.save_to_file(str(destination), out_of_range=NORMALIZE)
    return relative, report


def path_attributes(relative: str, run_dir: Path) -> dict[str, Any]:
    """The ``path``, digest, size and mtime of a file just written under ``run_dir``.

    Args:
        relative: The file's path relative to ``run_dir``, as the entity records it.
        run_dir: The run directory the path is relative to.

    Returns:
        ``path`` and either ``checksum_sha256`` or ``checksum_unresolved_reason``, plus
        ``size_bytes`` and ``mtime_ns`` when the file can be stat'd.
    """
    return {"path": relative, **file_attributes(run_dir / relative)}


#: The report-detail keys each branch carries, in the order a reader prints them.
BRANCH_MEASURES: dict[str, tuple[str, ...]] = {
    "AIRWAY": ("labelled_n", "contested_n", "merged_n"),
    "SPEECH": (
        "speaker_count",
        "words_n",
        "speech_s",
        "nontarget_speech_s",
        "trains_n",
        "train_s",
        "train_fraction",
        "modulation_peak_hz",
        "modulation_unit",
        "ppg_syllable_rate_hz",
        "ppg_cycle_rate_hz",
        "ppg_repetitions",
        "ppg_typical_repetitions",
        "ppg_period_s",
        "ppg_period_cv",
        "ppg_period_trend_s_per_step",
        "ppg_positions",
        "ppg_realised_mass",
        "ppg_occupancy_s",
        "ppg_filler_fraction",
        "ppg_score_per_frame",
        "ppg_contradicted_words_n",
    ),
    "VOICE": ("spans_n", "phonation_s", "longest_span_s"),
}

UNLABELLED = "unlabelled"
"""What a span reading falls back to when the producer stamped none."""

BRANCH_QUALIFIERS: tuple[str, ...] = ("label", "production", "attributed_to")
"""The attributes a proposal carries to distinguish itself inside its role, in the order preferred."""

UNROLED = "unroled"
"""What :func:`span_role_kind` falls back to for a span carrying no role."""

ENVELOPE_SPAN_KIND = "envelope"
"""PREPROCESS's name for an envelope span."""

_ROLE_INDEX = re.compile(r"_\d+$")


def report_entities(store: ProvStore) -> dict[str, Entity]:
    """The latest live ``branch_report`` entity per node, keyed by node name.

    Args:
        store: The provenance store.

    Returns:
        ``{node: entity}`` over every node that reported.
    """
    latest: dict[str, Entity] = {}
    for entity in store.entities("branch_report"):
        if not store.is_invalidated(entity.id):
            latest[str(entity.attributes.get("node"))] = entity
    return latest


def span_sources(store: ProvStore) -> dict[str, list[Entity]]:
    """Every live span, indexed by the live span it names in ``wasDerivedFrom``.

    Built once per renderer, in one pass over the store's relations.

    Args:
        store: The provenance store.

    Returns:
        ``{span id: [span it was derived from, ...]}``, in write order. A derivation naming
        anything that is not a live span with an extent contributes nothing; a span whose whole
        derivation is such a name is absent from the mapping.
    """
    spans = {span.id: span for span in live_entities(store, "span") if span.extent is not None}
    index: dict[str, list[Entity]] = {}
    for relation, source, target in store.relations():
        if relation != "wasDerivedFrom" or source not in spans:
            continue
        parent = spans.get(target)
        if parent is not None:
            index.setdefault(source, []).append(parent)
    return index


def envelope_span_label(span: Entity) -> str:
    """One envelope span's own reading, which is the level PREPROCESS measured over it.

    Args:
        span: An envelope span.

    Returns:
        The level, rounded, or :data:`UNLABELLED` when the span carries none.
    """
    level = span.attributes.get("peak_over_floor_db")
    return UNLABELLED if level is None else f"{float(level):.0f} dB"


def initial_span_label(span: Entity) -> str:
    """One upstream span's own reading, as the row that shows what came in states it.

    Args:
        span: A span another span was derived from.

    Returns:
        The producer's own reading of it: the level PREPROCESS measured, the family and role a
        proposer stamped, or :data:`UNLABELLED`.
    """
    if "peak_over_floor_db" in span.attributes:
        reading = envelope_span_label(span)
        return reading if reading == UNLABELLED else f"{ENVELOPE_SPAN_KIND} {reading}"
    family, role = span.attributes.get("family"), span.attributes.get("role")
    if family and role:
        return f"{family}/{role}"
    return str(family or span.attributes.get("name") or "") or UNLABELLED


def proposed_span_label(span: Entity) -> tuple[str, str]:
    """One proposed span's caption and the shorter one a narrow bar falls back to.

    Args:
        span: A span a branch proposed.

    Returns:
        ``(label, short)``. The label is the role every proposer stamps, qualified by whichever of
        :data:`BRANCH_QUALIFIERS` the proposal carries a value for and marked ``nontarget`` when it
        says so; the short form is the qualifier alone.
    """
    role = str(span.attributes.get("role") or "")
    qualifier = next(
        (str(span.attributes[key]) for key in BRANCH_QUALIFIERS if span.attributes.get(key) is not None), ""
    )
    label = f"{role}/{qualifier}" if role and qualifier else (role or qualifier or UNLABELLED)
    if span.attributes.get("nontarget"):
        return f"{label} nontarget", f"{qualifier or role} nontarget"
    return label, qualifier or role or UNLABELLED


def span_role_kind(span: Entity) -> str:
    """What kind of thing one proposed span is, with any per-instance index dropped.

    Args:
        span: A span a branch proposed.

    Returns:
        The role with a trailing ``_<number>`` removed, or :data:`UNROLED` when the span carries no
        role.
    """
    role = str(span.attributes.get("role") or "")
    return _ROLE_INDEX.sub("", role) or UNROLED
