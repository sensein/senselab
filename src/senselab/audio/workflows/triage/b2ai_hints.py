"""The hint for one Bridge2AI-Voice recording, read from its BIDS sidecars and the study's registry.

``<stem>_recording-metadata.json`` and ``<stem>_acoustictask-metadata.json`` carry the recording's
declared task. :func:`build_hint` maps them onto :class:`~senselab.audio.data_structures.AudioHints`:
the task's ``instructions`` and ``speech_type`` as typed fields, its stimulus as one
``expected_speech`` entry, and ``language``, ``task_name`` and where each text came from in
``metadata``. Instructions the registry's curated file corrects, and a recall task's stimulus the
sidecar leaves empty, come from :mod:`~senselab.audio.workflows.triage.task_registry`. A campaign's
``hints.py`` is ``from senselab.audio.workflows.triage.b2ai_hints import build_hint``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from senselab.audio.data_structures import AudioHints
from senselab.audio.data_structures.audio_hints import ExpectedSpeech
from senselab.audio.workflows.triage.nodes.branches import ITEM_CATEGORY_FROM_KEY, ITEM_CATEGORY_KEY
from senselab.audio.workflows.triage.task_registry import prompt_ref, questionnaire_field, task_text

SIDECAR_SUFFIXES = ("_recording-metadata.json", "_acoustictask-metadata.json")
CARRIED = (
    "instructions",
    "stimulus_text",
    "speech_type",
    "stimulus_source",
    "task_name",
    "acoustic_task_name",
    "language",
    "acoustic_task_id",
)
_TASK = re.compile(r"_task-([^_]+)$")
DATASET_DESCRIPTION = "dataset_description.json"
"""The file that marks a BIDS dataset's root."""
CATEGORY_FIELD_SUFFIX = "_category"
"""A questionnaire field ending so declares the category an item-list task asked for."""


def bids_root_of(wav: Path) -> Path | None:
    """The BIDS dataset root a recording lies under.

    Args:
        wav: The recording.

    Returns:
        The nearest ancestor holding :data:`DATASET_DESCRIPTION`, or None.
    """
    for parent in wav.resolve().parents:
        if (parent / DATASET_DESCRIPTION).is_file():
            return parent
    return None


def task_id_of(stem: str) -> str:
    """The BIDS task entity of a recording's stem.

    Args:
        stem: The recording's file stem.

    Returns:
        The lower-cased task entity, or ``unknown``.
    """
    match = _TASK.search(stem)
    return match.group(1).lower() if match else "unknown"


def read_sidecars(wav: Path) -> dict[str, Any]:
    """Merge the recording's sidecars; the first non-empty value for a key wins.

    Args:
        wav: The recording.

    Returns:
        The carried fields present across both sidecars. Empty when neither exists.
    """
    merged: dict[str, Any] = {}
    base = str(wav.with_suffix(""))
    for suffix in SIDECAR_SUFFIXES:
        path = Path(base + suffix)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in CARRIED:
            value = payload.get(key)
            if value not in (None, "") and merged.get(key) in (None, ""):
                merged[key] = value
    return merged


def build_hint(wav: Path) -> tuple[AudioHints | None, dict[str, Any]]:
    """The hint for one recording, and a record of what was declared and where it came from.

    Args:
        wav: The recording.

    Returns:
        The hint, or None when the recording has no sidecar at all, and a small dict for the run
        record naming the instructions' and the stimulus's source.
    """
    fields = read_sidecars(wav)
    if not fields:
        return None, {"sidecar": False}
    name = str(fields.get("task_name") or fields.get("acoustic_task_name") or task_id_of(wav.stem))
    text = task_text(name, fields)
    prompts = [ExpectedSpeech(text=text.stimulus, prompt_id=name)] if text.stimulus else []
    metadata: dict[str, Any] = {"task_name": name}
    for key in ("language", "stimulus_source"):
        if fields.get(key) not in (None, ""):
            metadata[key] = fields[key]
    metadata["instructions_from"] = text.instructions_source
    metadata["stimulus_from"] = text.stimulus_source
    if text.registry_task:
        metadata["registry_task"] = text.registry_task
    ref = prompt_ref(text.registry_task)
    joined = questionnaire_field(ref, bids_root_of(wav), fields.get("acoustic_task_id"))
    if joined is not None and str((ref or {}).get("field") or "").endswith(CATEGORY_FIELD_SUFFIX):
        metadata[ITEM_CATEGORY_KEY], metadata[ITEM_CATEGORY_FROM_KEY] = joined[0], f"questionnaire:{joined[1]}"
    hint = AudioHints(
        expected_speech=prompts,
        instructions=text.instructions,
        speech_type=text.speech_type,
        metadata=metadata,
    )
    record = {
        "sidecar": True,
        "expected_speech_n": len(prompts),
        "stimulus_text_len": len(text.stimulus) if text.stimulus else None,
        "speech_type": text.speech_type,
        "instructions_from": text.instructions_source,
        "stimulus_from": text.stimulus_source,
        "registry_task": text.registry_task,
        "item_category_from": metadata.get(ITEM_CATEGORY_FROM_KEY),
    }
    return hint, record
