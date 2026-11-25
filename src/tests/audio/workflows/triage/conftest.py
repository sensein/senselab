"""Every fold any triage test runs emits only scalar evidence rows with a comparison of their type.

Also :func:`declare_hint_identity` and :func:`record_task`, which record a hint's identity and the task it names on
a seeded store the way ADMIT does, and :func:`record_admitted_hint`, which records both.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage import vocabulary
from senselab.audio.workflows.triage.b2ai_hints import b2ai_task
from senselab.audio.workflows.triage.decision import EvidenceItem, row_problems
from senselab.audio.workflows.triage.identity import (
    PARTICIPANT_KEY,
    RECORDING_STREAM,
    SESSION_KEY,
    TASK_TOKEN_KEY,
    identity_of,
)
from senselab.audio.workflows.triage.task_catalogue import task_attributes
from senselab.utils.prov_store import ProvStore
from tests.conftest import activate_b2ai_profile_for_collection  # noqa: E402

activate_b2ai_profile_for_collection()

_ADMITTED_PATH = "/admitted/recording.wav"
"""The path a store seeded without a ``recording`` stream is given one at: a stem that names no task."""


def _recording_stream(store: ProvStore) -> Any:  # noqa: ANN401 -- the store's entity
    """The seeded store's latest ``recording`` stream with a ``path``, written at :data:`_ADMITTED_PATH` if absent."""
    found = [
        e for e in store.entities("stream") if e.attributes.get("name") == RECORDING_STREAM and e.attributes.get("path")
    ]
    if found:
        return found[-1]
    store.entity(prov_type="stream", extent=None, attributes={"name": RECORDING_STREAM, "path": _ADMITTED_PATH})
    return [e for e in store.entities("stream") if e.attributes.get("name") == RECORDING_STREAM][-1]


def record_task(
    store: ProvStore,
    task: str | None,
    *,
    variant: dict[str, str] | None = None,
    take: int | None = None,
    item_category: str | None = None,
    description: dict[str, Any] | None = None,
) -> str:
    """Record the task a hint names on a seeded store's recording, as ADMIT does on a real run.

    Every stage reads the task from ADMIT's record (``task_catalogue.recorded_task``), never from the file name,
    so a test that seeds a store by hand records the task it means here. A new ``recording`` stream entity,
    carrying the latest one's attributes plus :func:`~senselab.audio.workflows.triage.task_catalogue.task_attributes`
    of ``AudioHints(task=task, ...)``, becomes the one every reader takes. A store seeded without a ``recording``
    stream gets one, at a path whose stem names no task.

    Args:
        store: The seeded store.
        task: The task's name in the profile's task list (``respiration-and-cough-fivebreaths``), or None to record
            that the hint named none.
        variant: The task's declared variants (``{"route": "mouth"}``).
        take: Which take of the task.
        item_category: The word-generation category declared for the recording.
        description: An inline task description (``AudioHints.task_description``), its branch under ``branch``.

    Returns:
        The new stream entity's id.
    """
    hint = AudioHints(
        task=task, task_variant=variant or {}, task_take=take, item_category=item_category, task_description=description
    )
    latest = _recording_stream(store)
    return store.entity(
        prov_type="stream", extent=latest.extent, attributes={**latest.attributes, **task_attributes(hint)}
    )


def declare_hint_identity(store: ProvStore, hint: AudioHints) -> str:
    """Record what ``hint`` declares on a seeded store's recording, as ADMIT does on a real run.

    Stages read the task from the store, never from the live hint, so a test that seeds a store by hand
    and wants a hint's declaration seen must record it first. A new ``recording`` stream entity, carrying
    the latest one's attributes plus the identity, becomes the one every reader takes.

    Args:
        store: A store already holding a ``recording`` stream with a ``path``.
        hint: The hint whose declaration is recorded.

    Returns:
        The new stream entity's id.
    """
    latest = [e for e in store.entities("stream") if e.attributes.get("name") == RECORDING_STREAM][-1]
    return store.entity(
        prov_type="stream",
        extent=latest.extent,
        attributes={**latest.attributes, **identity_of(hint).as_attributes()},
    )


def record_admitted_hint(store: ProvStore, hint: AudioHints | None) -> None:
    """Record ``hint``'s identity and task on a seeded store, as ADMIT has by the time any later node runs.

    For node tests that seed a store by hand and then call a node with a hint. Only a hint that declares an
    identity key or a task is recorded; any other store is left exactly as seeded. The task is the hint's
    ``task`` (or inline ``task_description``); a hint naming none but carrying a B2AI ``task_token`` records the
    task, variant and take B2AI's hints derive from that id (``b2ai_hints.b2ai_task``). A store seeded without a
    ``recording`` stream gets one, at a path whose stem names no task, as ADMIT would have written it.

    Args:
        store: The seeded store.
        hint: The hint the node is about to be called with.
    """
    if hint is None:
        return
    token = str((hint.metadata or {}).get(TASK_TOKEN_KEY) or "").strip().lower()
    declares_identity = any((hint.metadata or {}).get(key) for key in (PARTICIPANT_KEY, SESSION_KEY, TASK_TOKEN_KEY))
    if not declares_identity and hint.task is None and hint.task_description is None:
        return
    _recording_stream(store)
    if declares_identity:
        declare_hint_identity(store, hint)
    if hint.task is not None or hint.task_description is not None:
        record_task(
            store,
            hint.task,
            variant=dict(hint.task_variant),
            take=hint.task_take,
            item_category=hint.item_category,
            description=hint.task_description,
        )
    elif token:
        task, variant, take = b2ai_task(token)
        record_task(store, task, variant=variant, take=take, item_category=hint.item_category)


@pytest.fixture(autouse=True)
def _scalar_evidence_rows(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail any test whose fold emits a dict, a list or a comparison of the wrong type.

    Yields:
        Nothing; the fold is checked for the duration of the test.
    """
    fold_evidence = vocabulary._decision_evidence

    def checked(**kwargs: Any) -> list[EvidenceItem]:  # noqa: ANN401 -- the fold's own keywords
        items = fold_evidence(**kwargs)
        problems = [problem for entry in items for problem in row_problems(entry)]
        assert not problems, problems
        return items

    monkeypatch.setattr(vocabulary, "_decision_evidence", checked)
    yield


@pytest.fixture(autouse=True)
def _triage_reads_the_b2ai_profile(b2ai_triage_profile: Any) -> None:  # noqa: ANN401 -- the fixture's profile
    """Every triage test reads its dataset data from the Bridge2AI-Voice adult profile (src/tests/conftest.py)."""


def b2ai_identity_attributes(path_or_stem: str) -> dict[str, Any]:
    """The identity attributes ADMIT records for a B2AI-named recording, as B2AI's hint builder declares them.

    For a test that seeds a ``recording`` stream by hand: senselab reads nothing off the file name (plan step 6a),
    so the identity the name carries is recorded beside it, as on a real run.

    Args:
        path_or_stem: The recording's path or stem, B2AI-named; a run stamp after it is ignored.

    Returns:
        ``identity_of(...).as_attributes()`` for the participant, session and task the name carries.
    """
    from senselab.audio.workflows.triage.b2ai_hints import identity_from_name  # noqa: PLC0415

    stem = re.sub(
        r"_\d{8}-\d{6}$", "", Path(str(path_or_stem)).stem if "." in Path(str(path_or_stem)).name else str(path_or_stem)
    )
    named = identity_from_name(stem)
    metadata = {
        key: value
        for key, value in (
            (PARTICIPANT_KEY, named.participant),
            (SESSION_KEY, named.session),
            (TASK_TOKEN_KEY, named.task_id),
        )
        if value and value != "unknown"
    }
    return identity_of(AudioHints(metadata=metadata)).as_attributes()


def seed_recorded_identity(store: ProvStore, stem: str) -> str:
    """Record on a hand-seeded store the identity ADMIT records on every run, as B2AI's hints declare it.

    Corpus passes refuse a run that records no identity (plan step 6a: replay it first), so a test that seeds a
    finished run by hand records one. The fields are what B2AI's hint builder reads off ``stem``, declared.

    Args:
        store: The seeded store.
        stem: The recording's file stem, B2AI-named, or its run directory's name (the stem and a stamp).

    Returns:
        The ``recording`` stream entity's id.
    """
    return store.entity(
        prov_type="stream",
        extent=None,
        attributes={
            "name": RECORDING_STREAM,
            "path": f"{stem}.wav",
            **b2ai_identity_attributes(stem),
        },
    )
