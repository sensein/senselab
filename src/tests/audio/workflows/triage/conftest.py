"""Every fold any triage test runs emits only scalar evidence rows with a comparison of their type.

Also the helpers a test declares a recording's identity with (plan step 8): :func:`hint_for` builds the hint a
dataset's hint builder would write, :func:`declared_identity_attributes` the identity ADMIT records from it, and
:func:`seed_recorded_identity` records it on a hand-seeded store. Test recordings carry plain names (``rec-0007``);
nothing is read off a file name. :func:`declare_hint_identity` and :func:`record_task` record a hint's identity and
task on a seeded store the way ADMIT does, and :func:`record_admitted_hint` records both.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage import vocabulary
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
from tests.conftest import activate_fixture_profile_for_collection  # noqa: E402

activate_fixture_profile_for_collection()

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
    ``task`` (or inline ``task_description``); a ``task_token`` alone records no task, as on a real run. A store
    seeded without a ``recording`` stream gets one, at a path whose stem names no task, as ADMIT would have written
    it.

    Args:
        store: The seeded store.
        hint: The hint the node is about to be called with.
    """
    if hint is None:
        return
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
def _triage_reads_the_fixture_profile(triage_profile: Any) -> None:  # noqa: ANN401 -- the fixture's profile
    """Every triage test reads its dataset data from the fixture profile (src/tests/conftest.py)."""


def hint_for(
    participant: str | None = None,
    session: str | None = None,
    task: str | None = None,
    *,
    token: str | None = None,
    variant: dict[str, str] | None = None,
    take: int | None = None,
    item_category: str | None = None,
    **fields: Any,  # noqa: ANN401 -- any other AudioHints field
) -> AudioHints:
    """The hint a dataset's hint builder writes for one recording: its identity declared, its task named.

    Args:
        participant: The participant, e.g. ``p07``.
        session: The session, e.g. ``s2``.
        task: The task's name in the profile's task list, e.g. ``rainbow-passage``.
        token: A dataset's own task id (``task_token``), with any trailing index (``harvard-sentences-list-3``); None
            declares the task in the typed fields alone, so the identity's task id is the task's name and take.
        variant: The task's declared variants.
        take: Which take of the task.
        item_category: The word-generation category declared for the recording.
        **fields: Any other :class:`AudioHints` field.

    Returns:
        The hint.
    """
    metadata = {
        key: value
        for key, value in ((PARTICIPANT_KEY, participant), (SESSION_KEY, session), (TASK_TOKEN_KEY, token))
        if value
    }
    return AudioHints(
        metadata={**metadata, **dict(fields.pop("metadata", None) or {})},
        task=task,
        task_variant=variant or {},
        task_take=take,
        item_category=item_category,
        **fields,
    )


def declared_identity_attributes(
    participant: str | None = None, session: str | None = None, task: str | None = None, *, token: str | None = None
) -> dict[str, Any]:
    """The identity and task attributes ADMIT records for a recording whose hint declares these (:func:`hint_for`).

    For a test that seeds a ``recording`` stream by hand: senselab reads nothing off the file name (plan step 6a),
    so the identity, and the task resolved from the profile's task list, are recorded beside it, as on a real run.

    Args:
        participant: The participant.
        session: The session.
        task: The task's name.
        token: The task id with its trailing index, where it differs from the name.

    Returns:
        ``identity_of(...).as_attributes()`` and ``task_attributes(...)`` of the hint.
    """
    hint = hint_for(participant, session, task, token=token)
    return {**identity_of(hint).as_attributes(), **task_attributes(hint)}


def seed_recorded_identity(
    store: ProvStore,
    stem: str,
    participant: str | None = None,
    session: str | None = None,
    task: str | None = None,
    *,
    token: str | None = None,
) -> str:
    """Record on a hand-seeded store the identity ADMIT records on every run, as the recording's hint declares it.

    Corpus passes refuse a run that records no identity (plan step 6a: replay it first), so a test that seeds a
    finished run by hand records one.

    Args:
        store: The seeded store.
        stem: The recording's file stem, a plain name such as ``rec-0007``.
        participant: The declared participant.
        session: The declared session.
        task: The declared task.
        token: The task id with its trailing index, where it differs from the task's name.

    Returns:
        The ``recording`` stream entity's id.
    """
    return store.entity(
        prov_type="stream",
        extent=None,
        attributes={
            "name": RECORDING_STREAM,
            "path": f"{stem}.wav",
            **declared_identity_attributes(participant, session, task, token=token),
        },
    )
