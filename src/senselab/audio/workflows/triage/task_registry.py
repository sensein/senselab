"""What a Bridge2AI-Voice task asks the participant to do and to say, from the study's own registry.

The BIDS sidecars carry each recording's ``instructions`` and ``stimulus_text``; they were built from
b2aiprep's flat ``audio_task_descriptions.json``, whose instructions its curated file corrects for a
few tasks. :func:`task_text` returns the text to use for one recording and names where each part came
from: the sidecar, or the curated correction where the registry says the harvested instructions were
wrong, and the flat descriptions' prompts where the sidecar carries no stimulus. The files are vendored
under ``data/b2ai_task_registry`` with their provenance; see
``specs/20260927-mask-placement-and-second-speaker/design.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

REGISTRY_DIR = Path(__file__).parent / "data" / "b2ai_task_registry"

SIDECAR = "sidecar"
"""A part read from the recording's own BIDS sidecar."""

CURATED = "curated"
"""Instructions from the registry's curated correction, which says the harvested text was wrong."""

DESCRIPTIONS = "descriptions"
"""A recall task's stimulus from the flat task descriptions' prompts, where the sidecar carries none."""

RECALL = "recall"
"""The sidecar ``speech_type`` of a task that asks the participant to retell what they were given."""


def description_key(name: str) -> str:
    """A task name as the flat descriptions and the BIDS task entities are compared.

    Args:
        name: A task name, as a sidecar, a path or the descriptions file writes it.

    Returns:
        Lower-cased, parentheses dropped: ``free-speech-(v2)-1`` and ``free-speech-v2-1`` agree.
    """
    return str(name or "").strip().lower().replace("(", "").replace(")", "")


@lru_cache(maxsize=1)
def _descriptions() -> dict[str, dict[str, Any]]:
    raw = json.loads((REGISTRY_DIR / "audio_task_descriptions.json").read_text())
    return {description_key(name): dict(entry) for name, entry in raw.items()}


@lru_cache(maxsize=1)
def _registry() -> dict[str, Any]:
    return dict(json.loads((REGISTRY_DIR / "registry.json").read_text()))


@lru_cache(maxsize=1)
def _curated() -> dict[str, dict[str, Any]]:
    return dict(json.loads((REGISTRY_DIR / "task_instructions_curated.json").read_text()).get("tasks") or {})


def _version_of(name: str) -> str:
    return "v2" if "-v2" in description_key(name) else "v1"


def registry_task(name: str, population: str = "adult") -> tuple[str | None, str | None]:
    """The registry task a recording's task name belongs to, and its recording key within it.

    An alias several task versions share (``story-recall`` is v1's and v2's) is resolved by the name
    itself: a name carrying ``-v2`` is the v2 task, any other the unversioned or v1 one. The
    registry's own ``alias_index`` maps such an alias to the current version, which is not what a
    recording named for the retired one was.

    Args:
        name: The recording's task name.
        population: ``adult`` or ``pediatric``.

    Returns:
        ``(task id, recording key)``; the key is the part of the name after the matched alias, empty
        where the whole name is the alias; ``(None, None)`` where no alias matches.
    """
    key = description_key(name)
    wanted = _version_of(name)
    candidates: list[tuple[int, str, str]] = []
    for task_id, task in (_registry().get("tasks") or {}).items():
        if not str(task_id).startswith(f"{population}."):
            continue
        for alias in task.get("aliases") or ():
            alias_key = description_key(str(alias))
            if key == alias_key:
                candidates.append((len(alias_key), str(task_id), ""))
            elif key.startswith(alias_key + "-"):
                candidates.append((len(alias_key), str(task_id), key[len(alias_key) + 1 :]))
    if not candidates:
        return None, None
    longest = max(length for length, _, _ in candidates)
    best = [(task_id, rest) for length, task_id, rest in candidates if length == longest]
    if len(best) > 1:
        versioned = [(task_id, rest) for task_id, rest in best if task_id.endswith(f".{wanted}")]
        unversioned = [(task_id, rest) for task_id, rest in best if not task_id.rsplit(".", 1)[-1].startswith("v")]
        best = versioned or unversioned or best
    return best[0]


@dataclass(frozen=True)
class TaskText:
    """What one recording's task asked for, and where each part came from.

    Attributes:
        instructions: What the participant was told to do, or None.
        stimulus: What the participant was given to say or recall, or None.
        speech_type: The sidecar's ``speech_type`` (``read``, ``recall``, ``free`` ...), or None.
        instructions_source: :data:`SIDECAR` or :data:`CURATED`; empty where there are none.
        stimulus_source: :data:`SIDECAR` or :data:`DESCRIPTIONS`; empty where there is none.
        registry_task: The registry task id the name resolved to, or None.
    """

    instructions: str | None
    stimulus: str | None
    speech_type: str | None
    instructions_source: str
    stimulus_source: str
    registry_task: str | None


def task_text(name: str, sidecar: Mapping[str, Any], population: str = "adult") -> TaskText:
    """The instructions and stimulus to give the pipeline for one recording.

    Args:
        name: The recording's task name (the sidecar's ``task_name``, else the path's task entity).
        sidecar: The recording's merged sidecar fields.
        population: ``adult`` or ``pediatric``.

    Returns:
        The sidecar's instructions unless the registry's curated file corrects that task (per
        recording where it keys its corrections so); the sidecar's ``stimulus_text`` where it is not
        empty, else, for a recall task only, the flat descriptions' prompts for the name. A sidecar
        stimulus left empty by design -- a breath, a picture description -- stays empty.
    """
    task_id, recording = registry_task(name, population)
    instructions = str(sidecar.get("instructions") or "").strip() or None
    instructions_source = SIDECAR if instructions else ""
    curated = _curated().get(task_id or "") if task_id else None
    if curated:
        corrected = str(curated.get("instructions") or "").strip()
        per_recording = curated.get("recordings") or {}
        if recording and recording in per_recording:
            corrected = str(per_recording[recording] or "").strip()
        if corrected:
            instructions, instructions_source = corrected, CURATED
    stimulus = str(sidecar.get("stimulus_text") or "").strip() or None
    stimulus_source = SIDECAR if stimulus else ""
    speech_type = str(sidecar.get("speech_type") or "").strip() or None
    if stimulus is None and speech_type == RECALL:
        prompts = [str(prompt) for prompt in (_descriptions().get(description_key(name)) or {}).get("prompts") or ()]
        joined = " ".join(prompt.strip() for prompt in prompts if prompt.strip())
        if joined:
            stimulus, stimulus_source = joined, DESCRIPTIONS
    return TaskText(instructions, stimulus, speech_type, instructions_source, stimulus_source, task_id)


def described(name: str) -> dict[str, Any] | None:
    """The flat descriptions' entry for a task name, for cross-checking a sidecar against it.

    Args:
        name: The task name.

    Returns:
        ``{instructions, prompts}``, or None where the file has no such name.
    """
    return _descriptions().get(description_key(name))
