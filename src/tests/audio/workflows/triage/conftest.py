"""Every fold any triage test runs emits only scalar evidence rows with a comparison of their type."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from senselab.audio.workflows.triage import vocabulary
from senselab.audio.workflows.triage.decision import EvidenceItem, row_problems


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
