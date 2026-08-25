"""The shape every triage node shares: its result type and its store conventions."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.vocabulary import NodeVerdict, Outcome
from senselab.utils.prov_store import PROV_TYPE, Entity, ProvStore


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


def write_verdict(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    node: str,
    outcome: Outcome,
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
        outcome: What it concluded.
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
