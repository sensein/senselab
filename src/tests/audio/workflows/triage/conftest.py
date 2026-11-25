"""Every fold any triage test runs emits only scalar evidence rows with a comparison of their type.

Also :func:`declare_hint_identity`, which records a hint's identity on a seeded store the way ADMIT does.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
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
from senselab.utils.prov_store import ProvStore


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
    stem = Path(str(latest.attributes.get("path") or "")).stem
    return store.entity(
        prov_type="stream",
        extent=latest.extent,
        attributes={**latest.attributes, **identity_of(stem, hint).as_attributes()},
    )


def record_admitted_hint(store: ProvStore, hint: AudioHints | None) -> None:
    """Record ``hint``'s identity on a seeded store, as ADMIT has by the time any later node runs.

    For node tests that seed a store by hand and then call a node with a hint. Only a hint that declares an
    identity key is recorded; any other store is left exactly as seeded. A store seeded without a
    ``recording`` stream gets one, at a path whose stem names no task, as ADMIT would have written it.

    Args:
        store: The seeded store.
        hint: The hint the node is about to be called with.
    """
    if hint is None or not any(
        (hint.metadata or {}).get(key) for key in (PARTICIPANT_KEY, SESSION_KEY, TASK_TOKEN_KEY)
    ):
        return
    if not any(
        e.attributes.get("name") == RECORDING_STREAM and e.attributes.get("path") for e in store.entities("stream")
    ):
        store.entity(
            prov_type="stream", extent=None, attributes={"name": RECORDING_STREAM, "path": "/admitted/recording.wav"}
        )
    declare_hint_identity(store, hint)


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
