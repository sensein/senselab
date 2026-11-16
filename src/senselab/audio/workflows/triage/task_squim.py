"""SQUIM over the task extent: fixed windows of the plain stream, read by the objective head.

:func:`task_windows` places the windows; :func:`measure_task_squim` reads them over one finished
store and records one ``task_squim`` measurement on the ``plain`` stream. The window lengths, the
model's pinned weights and the branches that get no value are ``data/task_squim.yaml``.

See ``specs/20261027-task-extent-squim/design.md``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as dist_version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml  # type: ignore[import-untyped]

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_verdict,
    live_entities,
    software_agent,
    write_measurement,
)
from senselab.audio.workflows.triage.nodes.verdict import NODE as VERDICT_NODE
from senselab.audio.workflows.triage.recording_vectors import stem_of
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
from senselab.audio.workflows.triage.task_audio import MEASUREMENT as TASK_AUDIO_MEASUREMENT
from senselab.audio.workflows.triage.task_audio import PLAIN, sample_bounds, task_extent
from senselab.utils.prov_store import Entity, ProvStore

NODE = "TASK_SQUIM"
"""The node name this measurement's activities carry."""

MEASUREMENT = "task_squim"
"""The measurement's ``name``."""

PROCESS_VERSION = 1
"""The measurement's behaviour version. Bump when what it writes for the same inputs changes."""

PARAMETERS_PATH = Path(__file__).parent / "data" / "task_squim.yaml"

HEADS = ("pesq", "stoi", "si_sdr")
"""The objective head's three readings."""

EXTENT_FROM_CUT = "task_audio"
"""``extent_source``: the live ``task_audio`` measurement's bounds."""

EXTENT_FROM_SPANS = "task_extent"
"""``extent_source``: :func:`~senselab.audio.workflows.triage.task_audio.task_extent` over the live spans."""

OK = "ok"
PRESENT = "present"
ABSENT = "absent"

Scorer = Callable[[Sequence[Audio]], list[dict[str, float]]]
"""Reads a batch of mono 16 kHz windows; returns ``pesq``, ``stoi`` and ``si_sdr`` per window."""

ScorerFactory = Callable[[], Scorer]
"""Builds a :data:`Scorer`; called only once a window is to be read."""


@functools.cache
def task_squim_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_squim.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(PARAMETERS_PATH.read_text(encoding="utf-8")) or {})


@dataclass(frozen=True)
class WindowConfig:
    """Where windows go across an extent.

    Attributes:
        window_s: The fixed window length.
        min_window_s: The shortest trailing window kept, and the length under which an extent is one window.
        floor_s: The shortest extent given any window.
    """

    window_s: float
    min_window_s: float
    floor_s: float

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any] | None = None) -> WindowConfig:
        """The configuration ``data/task_squim.yaml`` declares.

        Args:
            parameters: The parsed file; the packaged one when None.

        Returns:
            The window configuration.

        Raises:
            ValueError: If the lengths are not ``0 < floor_s <= min_window_s <= window_s``.
        """
        windows = (parameters or task_squim_parameters())["windows"]
        config = cls(float(windows["window_s"]), float(windows["min_window_s"]), float(windows["floor_s"]))
        if not 0.0 < config.floor_s <= config.min_window_s <= config.window_s:
            raise ValueError(f"task_squim windows need 0 < floor_s <= min_window_s <= window_s, got {config}")
        return config

    def as_dict(self) -> dict[str, float]:
        """The configuration as the measurement records it."""
        return {"window_s": self.window_s, "min_window_s": self.min_window_s, "floor_s": self.floor_s}


def task_windows(start_s: float, end_s: float, config: WindowConfig) -> list[tuple[float, float]]:
    """Non-overlapping fixed windows across ``[start_s, end_s]``.

    :func:`~senselab.audio.tasks.features_extraction.torchaudio_squim.plan_windows` with a trailing
    remainder under ``min_window_s`` dropped: full windows of ``window_s`` from the start, the remainder
    one more window where it is at least ``min_window_s``. An extent shorter than ``min_window_s`` is one
    window over the whole extent, and one shorter than ``floor_s`` gets none.

    Args:
        start_s: The extent's start, in seconds.
        end_s: The extent's end, in seconds.
        config: The window configuration.

    Returns:
        ``(start, end)`` per window, in order.
    """
    from senselab.audio.tasks.features_extraction.torchaudio_squim import plan_windows

    return plan_windows(
        start_s,
        end_s,
        window_s=config.window_s,
        min_window_s=config.min_window_s,
        floor_s=config.floor_s,
        short_tail="drop",
    )


def summarise(readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The medians over the windows SQUIM read, and how many it read.

    Args:
        readings: One record per window, carrying the three heads where it was read.

    Returns:
        ``pesq``, ``stoi`` and ``si_sdr`` medians (None where no window was read), ``windows_n``
        (windows read) and ``covered_s`` (their total length).
    """
    read = [r for r in readings if all(_finite(r.get(head)) for head in HEADS)]
    out: dict[str, Any] = {head: float(np.median([float(r[head]) for r in read])) if read else None for head in HEADS}
    out["windows_n"] = len(read)
    out["covered_s"] = float(sum(float(r["end_s"]) - float(r["start_s"]) for r in read))
    return out


def _finite(value: Any) -> bool:  # noqa: ANN401
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def branch_of_family(family: str | None) -> str | None:
    """The branch that owns a declared family, or None.

    Args:
        family: The declared family.

    Returns:
        ``AIRWAY``, ``SPEECH`` or ``VOICE``, or None for a family no branch declares.
    """
    owners = [branch for branch, families in EXPECTATIONS.items() if family and family in families]
    return owners[0] if owners else None


def declared_family(store: ProvStore, stem: str | None) -> str | None:
    """The recording's declared family: the fold's, else the stem's task id collapsed.

    Args:
        store: The run's store.
        stem: The recording's BIDS stem, with or without the run's launch timestamp, or None.

    Returns:
        The family, or None where neither is known.
    """
    folded = find_verdict(store, VERDICT_NODE)
    family = None if folded is None else folded.attributes.get("declared_family")
    if family:
        return str(family)
    return task_family(task_id_of(stem_of(Path(stem)))) if stem else None


def weights_path(asset: str) -> Path:
    """Where torchaudio keeps one hosted asset.

    Args:
        asset: The asset key, such as ``models/squim_objective_dns2020.pth``.

    Returns:
        ``<torch hub dir>/torchaudio/<asset>``, the path ``SQUIM_OBJECTIVE.get_model()`` loads.
    """
    import torch

    return Path(torch.hub.get_dir()) / "torchaudio" / asset


def verify_weights(asset: str, sha256: str) -> str:
    """Fetch the objective head's weights if absent and check them against the pinned digest.

    Args:
        asset: The asset key.
        sha256: The pinned SHA-256.

    Returns:
        The digest, equal to ``sha256``.

    Raises:
        ValueError: If the file's digest is not the pinned one.
    """
    from torchaudio.utils import _download_asset

    path = Path(_download_asset(asset, progress=False))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path} has sha256 {digest}, not the pinned {sha256}; refusing to read SQUIM with it")
    return digest


@functools.cache
def squim_scorer() -> Scorer:
    """The objective-SQUIM reader senselab ships, after the pinned weights are verified.

    Returns:
        A scorer reading each window in one pass of
        ``torchaudio_squim.extract_objective_quality_features_from_audios``.
    """
    from senselab.audio.tasks.features_extraction.torchaudio_squim import (
        extract_objective_quality_features_from_audios,
    )
    from senselab.utils.data_structures import DeviceType

    model = task_squim_parameters()["model"]
    verify_weights(str(model["asset"]), str(model["weights_sha256"]))

    def score(audios: Sequence[Audio]) -> list[dict[str, float]]:
        results = extract_objective_quality_features_from_audios(list(audios), window_s=None, device=DeviceType.CPU)
        return [{head: float(result[head]) for head in HEADS} for result in results]

    return score


@dataclass(frozen=True)
class TaskSquimOutcome:
    """What one pass did to one run.

    Attributes:
        status: ``ok`` (written), ``present`` (the live measurement already matches) or ``absent``.
        reason: Why nothing was measured, for ``absent``.
        summary: The medians, ``windows_n``, ``covered_s``, ``extent_s`` and ``extent_source``, where measured.
        changed: Whether the store was written to.
    """

    status: str
    reason: str | None
    summary: dict[str, Any] | None
    changed: bool


def _live_stream(store: ProvStore, name: str) -> Entity | None:
    found = [e for e in live_entities(store, "stream") if e.attributes.get("name") == name]
    return found[-1] if found else None


def extent_bounds(store: ProvStore, config: TriageConfig) -> tuple[float, float, str, str | None] | None:
    """The task extent SQUIM reads: the live cut's bounds, else the same definition over the live spans.

    Args:
        store: The run's store.
        config: The triage configuration, for ``task_audio.padding_s``.

    Returns:
        ``(start_s, end_s, extent_source, task_audio measurement id or None)``, or None where the
        recording has no task extent.
    """
    cut = find_measurement(store, TASK_AUDIO_MEASUREMENT)
    if cut is not None and cut.attributes.get("start_s") is not None and cut.attributes.get("end_s") is not None:
        return float(cut.attributes["start_s"]), float(cut.attributes["end_s"]), EXTENT_FROM_CUT, cut.id
    extent = task_extent(store, padding_s=float(config.require("task_audio.padding_s")))
    if extent is None:
        return None
    return extent.start_s, extent.end_s, EXTENT_FROM_SPANS, None


def measure_key(plain_sha256: str | None, start_s: float, end_s: float, windows: WindowConfig, weights: str) -> str:
    """The key a measurement is matched on: its input bytes, its extent, its windows, the weights and version.

    Args:
        plain_sha256: The plain stream's recorded digest.
        start_s: The extent's start.
        end_s: The extent's end.
        windows: The window configuration.
        weights: The weights' SHA-256.

    Returns:
        A 64-character hex digest.
    """
    payload = {
        "process": NODE,
        "process_version": PROCESS_VERSION,
        "plain": plain_sha256,
        "extent": [start_s, end_s],
        "windows": windows.as_dict(),
        "weights_sha256": weights,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _retire(store: ProvStore, entity: Entity, *, software: str, reason: str) -> None:
    activity = store.activity(node=NODE, step="superseded", parameters={"superseded": entity.id, "reason": reason})
    store.was_associated_with(activity, software)
    store.used(activity, entity.id)
    store.was_invalidated_by(entity.id, activity)


def measure_task_squim(
    store: ProvStore,
    config: TriageConfig,
    *,
    run_dir: Path,
    stem: str | None = None,
    scorer: ScorerFactory | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> TaskSquimOutcome:
    """Read SQUIM over fixed windows of the task extent on the plain stream and record it.

    A store whose live ``task_squim`` measurement carries the same key is left unwritten. Otherwise the
    old one is retired and a new one written. A family of a branch ``no_value_branches`` names (airway,
    voice), a recording with no task extent or no
    plain stream, and an extent under the floor get no value, and any earlier measurement is retired.

    Args:
        store: The run's store.
        config: The triage configuration, for ``task_audio.padding_s``.
        run_dir: The run directory the plain stream's path is relative to.
        stem: The recording's BIDS stem, for its family where the fold records none.
        scorer: Builds the SQUIM reader, called only once a window is to be read; :func:`squim_scorer`
            when None.
        parameters: ``data/task_squim.yaml``'s mapping; the packaged one when None.

    Returns:
        The outcome.
    """
    p = dict(parameters or task_squim_parameters())
    windows_config = WindowConfig.from_parameters(p)
    model = p["model"]
    weights = str(model["weights_sha256"])
    rate = int(model["sampling_rate"])
    held = find_measurement(store, MEASUREMENT)

    def absent(reason: str) -> TaskSquimOutcome:
        if held is None:
            return TaskSquimOutcome(ABSENT, reason, None, changed=False)
        _retire(store, held, software=software_agent(store), reason=reason)
        return TaskSquimOutcome(ABSENT, reason, None, changed=True)

    family = declared_family(store, stem)
    branch = branch_of_family(family)
    if branch in set(p.get("no_value_branches") or ()):
        return absent(f"{str(branch).lower()} family")
    bounds = extent_bounds(store, config)
    if bounds is None:
        return absent("no task extent")
    start_s, end_s, source, cut_id = bounds
    plain = _live_stream(store, PLAIN)
    if plain is None:
        return absent("no plain stream")
    windows = task_windows(start_s, end_s, windows_config)
    if not windows:
        return absent(f"task extent {end_s - start_s:.3f} s is under floor_s {windows_config.floor_s} s")
    key = measure_key(plain.attributes.get("checksum_sha256"), start_s, end_s, windows_config, weights)
    if held is not None and held.attributes.get("measure_key") == key:
        return TaskSquimOutcome(PRESENT, None, _summary_of(held), changed=False)

    path = Path(str(plain.attributes["path"]))
    audio = Audio(filepath=str(path if path.is_absolute() else run_dir / path))
    if audio.sampling_rate != rate:
        from senselab.audio.tasks.preprocessing.preprocessing import resample_audios

        [audio] = resample_audios([audio], rate)
    waveform = audio.waveform
    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    n = int(waveform.shape[-1])
    pieces: list[Audio] = []
    for a, b in windows:
        lo, hi = sample_bounds(a, b, rate, n)
        pieces.append(Audio(waveform=waveform[:, lo:hi].contiguous(), sampling_rate=rate))

    read = (scorer or squim_scorer)()
    readings: list[dict[str, Any]] = []
    for (a, b), piece in zip(windows, pieces):
        record: dict[str, Any] = {"start_s": a, "end_s": b}
        if int(piece.waveform.shape[-1]) == 0:
            record["unmeasured"] = "empty window"
        else:
            try:
                [scores] = read([piece])
                record.update({head: float(scores[head]) for head in HEADS})
            except Exception as error:  # noqa: BLE001 — a window SQUIM refuses is unmeasured
                record["unmeasured"] = type(error).__name__
        readings.append(record)
    summary = summarise(readings)

    software = software_agent(store)
    agent = store.agent(
        agent_type="model",
        model_id=str(model["model_id"]),
        unresolved_reason=(
            f"torchaudio-hosted asset {model['asset']}, sha256 {weights}, torchaudio {_version('torchaudio')}"
        ),
    )
    if held is not None:
        _retire(store, held, software=software, reason="task extent, plain stream, windows or weights changed")
    activity = store.activity(
        node=NODE,
        step="measure",
        parameters={
            "process_version": PROCESS_VERSION,
            "parameters_version": p.get("version"),
            **windows_config.as_dict(),
            "weights_sha256": weights,
            "sampling_rate": rate,
        },
    )
    store.was_associated_with(activity, software)
    store.was_associated_with(activity, agent)
    store.used(activity, plain.id)
    derived = (plain.id,) if cut_id is None else (plain.id, cut_id)
    if cut_id is not None:
        store.used(activity, cut_id)
    write_measurement(
        store,
        activity,
        agent,
        name=MEASUREMENT,
        signal=PLAIN,
        attributes={
            **summary,
            "extent_s": end_s - start_s,
            "extent_source": source,
            "windows": readings,
            "window_config": windows_config.as_dict(),
            "parameters_version": p.get("version"),
            "model_id": str(model["model_id"]),
            "weights_sha256": weights,
            "plain_sha256": plain.attributes.get("checksum_sha256"),
            "measure_key": key,
            "process_version": PROCESS_VERSION,
        },
        derived_from=derived,
        extent=(start_s, end_s),
    )
    return TaskSquimOutcome(OK, None, {**summary, "extent_s": end_s - start_s, "extent_source": source}, changed=True)


def _summary_of(entity: Entity) -> dict[str, Any]:
    return {k: entity.attributes.get(k) for k in (*HEADS, "windows_n", "covered_s", "extent_s", "extent_source")}


def _version(name: str) -> str:
    try:
        return dist_version(name)
    except PackageNotFoundError:
        return "unknown"
