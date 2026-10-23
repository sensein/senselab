"""The hint a run was given, as ADMIT records it in the store: the hint itself and a digest of each field.

ADMIT writes :func:`hint_attributes` onto the ``recording`` stream, and onto its own activity when it refuses
the recording. The stored hint is canonical JSON of the fields that differ from their defaults; ``metadata``
is digested key by key; ``target_speaker`` is stored as its dimension, provenance and vector digest only.
:func:`recorded_hint`, :func:`recorded_hint_digest` and :func:`changed_fields` read the record back.

A store written before ADMIT recorded hints has none: :func:`recorded_hint_digest` is then None ("unknown"),
which is not :data:`NO_HINT`. See ``specs/20261022-triage-dataset-profile/plan.md``, step 3.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError

from senselab.audio.data_structures import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.identity import admit_record
from senselab.utils.prov_store import ProvStore

HINT = "hint"
HINT_DIGEST = "hint_digest"
HINT_FIELD_DIGESTS = "hint_field_digests"
"""The attributes ADMIT records the hint under."""

NO_HINT = "none"
""":data:`HINT_DIGEST` for a run given no hint at all."""

TARGET_SPEAKER = "target_speaker"
METADATA = "metadata"


def _digest(value: Any) -> str:  # noqa: ANN401 -- any JSON value
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _dumped(model: BaseModel) -> dict[str, Any]:
    """``model`` as JSON, fields at their defaults left out; a value JSON cannot hold is written as a string."""
    try:
        return model.model_dump(mode="json", exclude_defaults=True)
    except PydanticSerializationError:
        return json.loads(json.dumps(model.model_dump(exclude_defaults=True), default=str))


def _reduced_target_speaker(value: Mapping[str, Any]) -> dict[str, Any]:
    vector = value.get("vector") or []
    return {
        "vector_dim": len(vector),
        "vector_digest": _digest(vector),
        "provenance": value.get("provenance"),
        "distribution": value.get("distribution"),
    }


def field_digests(hint: AudioHints) -> dict[str, str]:
    """A digest of each field of ``hint`` that differs from its default, ``metadata`` as ``metadata.<key>``.

    Args:
        hint: The hint.

    Returns:
        Field name to the digest of the field's full JSON value.
    """
    dumped = _dumped(hint)
    digests = {name: _digest(value) for name, value in dumped.items() if name != METADATA}
    for key, value in (dumped.get(METADATA) or {}).items():
        digests[f"{METADATA}.{key}"] = _digest(value)
    return digests


def expected_speech_digest(prompts: Sequence[ExpectedSpeech]) -> str:
    """The :func:`field_digests` entry for ``expected_speech``, from the prompts alone.

    Args:
        prompts: The hint's ``expected_speech``.

    Returns:
        The digest.
    """
    return _digest([_dumped(prompt) for prompt in prompts])


def hint_digest(hint: AudioHints | None) -> str:
    """A digest of the whole hint, or :data:`NO_HINT`.

    Args:
        hint: The hint, or None.

    Returns:
        The digest of its canonical JSON.
    """
    return NO_HINT if hint is None else _digest(_dumped(hint))


def hint_attributes(hint: AudioHints | None) -> dict[str, Any]:
    """The attributes ADMIT records for ``hint``.

    Args:
        hint: What the recording was declared to contain, or None.

    Returns:
        :data:`HINT` (canonical JSON with ``target_speaker`` reduced; None for no hint), :data:`HINT_DIGEST`
        and :data:`HINT_FIELD_DIGESTS` (empty for no hint).
    """
    if hint is None:
        return {HINT: None, HINT_DIGEST: NO_HINT, HINT_FIELD_DIGESTS: {}}
    stored = _dumped(hint)
    if stored.get(TARGET_SPEAKER) is not None:
        stored[TARGET_SPEAKER] = _reduced_target_speaker(stored[TARGET_SPEAKER])
    return {HINT: stored, HINT_DIGEST: hint_digest(hint), HINT_FIELD_DIGESTS: field_digests(hint)}


def _record(store: ProvStore) -> Mapping[str, Any] | None:
    found = admit_record(store)
    return None if found is None else found[0]


def recorded_hint_digest(store: ProvStore) -> str | None:
    """The whole-hint digest ADMIT recorded, :data:`NO_HINT`, or None where the store records no hint.

    Args:
        store: The provenance store.

    Returns:
        The digest; None for a store written before ADMIT recorded hints, or holding no ADMIT record.
    """
    record = _record(store)
    if record is None or HINT_DIGEST not in record:
        return None
    return str(record[HINT_DIGEST])


def recorded_hint(store: ProvStore) -> AudioHints | None:
    """The hint ADMIT recorded, rebuilt, or None where it recorded none or the store predates the record.

    ``target_speaker`` is not rebuilt, since only its digest is stored: for a hint that carried one, the
    rebuilt hint differs from the original in that field.

    Args:
        store: The provenance store.

    Returns:
        The hint, or None.
    """
    record = _record(store)
    stored = None if record is None else record.get(HINT)
    if not isinstance(stored, Mapping):
        return None
    return AudioHints.model_validate({key: value for key, value in stored.items() if key != TARGET_SPEAKER})


def changed_fields(store: ProvStore, hint: AudioHints | None) -> list[str] | None:
    """The fields whose digest differs between the recorded hint and ``hint``, sorted.

    Args:
        store: The provenance store.
        hint: The hint now held.

    Returns:
        The changed field names (``metadata.<key>`` for metadata), ``[]`` when nothing changed, or None
        when the store records no hint to compare with.
    """
    record = _record(store)
    if record is None or HINT_DIGEST not in record:
        return None
    if hint_digest(hint) == record[HINT_DIGEST]:
        return []
    recorded = dict(record.get(HINT_FIELD_DIGESTS) or {})
    now = field_digests(hint) if hint is not None else {}
    return sorted(name for name in recorded.keys() | now.keys() if recorded.get(name) != now.get(name))
