"""The shape every triage node shares: its result type and its store conventions."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.vocabulary import NodeVerdict, Outcome, Triage
from senselab.utils.portable_audio_io import NORMALIZE, AudioWriteReport
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore, file_attributes
from senselab.utils.subprocess_venv import venv_environment

MESSAGE_CAP = 200
"""How much of an exception's message is recorded. The bound on what a message can leak."""

STREAM_SUFFIX = ".flac"
"""Container a persisted stream gets. See ``specs/20260907-triage-stream-compression/design.md``."""

HOST_ENV_PACKAGES = ("torch", "torchaudio", "torchcodec", "transformers", "numpy", "scipy", "librosa")
"""The packages named for the host environment: the ones that decide numerical results.

See ``specs/20260908-triage-prov-bep028/design.md``."""


def describe_exception(error: BaseException) -> str:
    """An exception as ``"Class: first line"``, bounded, for a record a reader can act on.

    Args:
        error: The exception.

    Returns:
        The class name alone when the exception carries no message, else the class name and the
        message's first line truncated to :data:`MESSAGE_CAP` characters with an ellipsis. Only the
        first line is kept: a multi-line message is a traceback or a dump, and neither belongs in a
        one-line record.
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
    """What every node returns.

    Attributes:
        verdict: The node's conclusion, in the graph's shared vocabulary.
        view: Ids of the store entities this node wrote or asserted over.
        verdict_entity_id: The verdict entity this node wrote to the store.
    """

    verdict: NodeVerdict
    view: tuple[str, ...]
    verdict_entity_id: str


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
            never reached a subprocess venv (e.g. it failed at ADMIT).

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
        detail: The node's design-named verdict fields.

    Returns:
        The verdict entity's id and the vocabulary verdict.

    Raises:
        ValueError: If ``detail`` carries any of the reserved keys ``node``, ``outcome``, ``kind``
            or ``why``, which would let the stored attributes diverge from the returned verdict.
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

    Shared across nodes: PREPROCESS writes most measurements, but a node further downstream (e.g.
    TAXONOMY proposing phonation spans from PREPROCESS's own track measurement) writes in the same
    shape, so both call this rather than each keeping its own copy.
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

    The tolerance is one sample period of ``audio``, which is a numerical identity rather than a
    tunable: an end within one sample of the last sample names that same sample boundary.

    Args:
        extent: The ``(start, end)`` about to be sliced, in seconds.
        audio: The audio being sliced; the length it decoded to is the bound.

    Returns:
        The extent, with ``end`` replaced by the audio's duration when it overshot within tolerance.

    Raises:
        ValueError: If ``end`` exceeds the duration by more than one sample period. The message
            carries bounds only, never any text the extent covers.
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


def find_measurement(store: ProvStore, name: str) -> Entity | None:
    """The latest non-invalidated measurement entity carrying this name, or None.

    Reads by the store's shared rule: invalidated entities are never returned, and of the survivors
    the latest write wins — the same rule ``resolve_stream`` applies to streams.

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


def find_measurements(store: ProvStore, name: str) -> list[Entity]:
    """Every live measurement entity carrying this name, in write order.

    The plural of :func:`find_measurement`, for a name one node writes many of — the per-window
    classifications, the per-span formant tracks. Reads by the store's shared rule: an invalidated
    entity is never returned.

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

    The store's shared read rule in its simplest form, so no node re-derives the filter and forgets
    the invalidation check.

    Args:
        store: The provenance store.
        prov_type: The entity type to read.

    Returns:
        The live entities, oldest first.
    """
    return [e for e in store.entities(prov_type) if not store.is_invalidated(e.id)]


def consensus_words(store: ProvStore) -> list[Entity]:
    """The consensus stream: every live ``word`` entity, in ``index`` order.

    ``index`` is the position PREPROCESS's consensus emitted the word at, and it is the only order a
    reader may use; a word's extent is metadata on the position, never a sort key.

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

    The union of the derived extent and every source's own reading. The fit can place a word
    outside all of its sources, and pooling can place it outside its own sources' hull, so a
    consumer that must not miss the word takes the whole of both.

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


def resolve_stream(store: ProvStore, run_dir: Path, name: str) -> tuple[str, Audio]:
    """Load a stream the graph wrote earlier, by its name.

    Reads by the store's shared rule: invalidated entities are never returned, and of the survivors
    the latest write wins — the same rule ``find_measurement`` applies to measurements.

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

    The one place every node that writes a persisted stream (as opposed to a transient hand-off
    file a subprocess worker reads and a caller deletes) calls ``Audio.save_to_file``, so the
    container and the out-of-range policy are decided once. ``out_of_range="normalize"`` never
    truncates: a peak already at or below +-1 (a genuinely clipped recording, which is content)
    passes through unchanged, and only a peak the write would otherwise clip is scaled down, with
    the gain in the returned report -- a caller records it rather than letting it go unaccounted.

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
    report = audio.save_to_file(str(run_dir / relative), out_of_range=NORMALIZE)
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
