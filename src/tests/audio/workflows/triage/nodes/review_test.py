"""REVIEW: the reviewer reads every transcript that could be released, however it got there.

The population the reviewer exists for is the one the detectors never saw. 13,810 of the r3 corpus
carry lexical speech and were never scanned, because SPEECH declined the scan where every lexical
word sits in the task's own stimulus; a further 11,526 were scanned and marked nothing. A reviewer
reached only through REDACT can second-guess neither. These pin that REVIEW is reached from the
transcript rather than from the scan, and that what it concludes stays a reading.

``specs/20260924-reviewer-over-every-transcript/design.md`` holds the reasoning.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import review as review_module
from senselab.audio.workflows.triage.nodes.redact import transcript_texts
from senselab.audio.workflows.triage.nodes.review import NODE, review
from senselab.audio.workflows.triage.vocabulary import PII_SCAN, REDACTION_LLM_ANNOTATION, SCANNED, Outcome
from senselab.text.tasks.pii_detection.redaction_review import ReviewProposal, ReviewResult
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import word_attributes

LLM_ON = "redaction:\n  llm_check:\n    enabled: true\n"


def _config(tmp_path: Path, text: str = LLM_ON) -> TriageConfig:
    """The packaged config with a partial YAML deep-merged over it."""
    path = tmp_path / "override.yaml"
    path.write_text(text, encoding="utf-8")
    return load_triage_config(path)


def _extent(index: int) -> tuple[float, float]:
    """One word's bounds, a second apart."""
    return (float(index), float(index) + 0.5)


def _seed(
    store: ProvStore,
    *,
    words: Sequence[str] = ("hello", "world"),
    scan: str | None = "ran",
    redacted: Sequence[tuple[float, float]] = (),
    redact_outcome: Outcome | None = None,
    findings_n: int = 0,
) -> None:
    """The store SPEECH and REDACT leave behind, in the four shapes REVIEW must tell apart.

    ``scan`` is ``"ran"``, ``"declined"`` or None for a store carrying no scan measurement at all.
    ``redacted`` are the planned extents REDACT left as ``redaction`` spans.
    """
    software = store.agent(agent_type="software", version="senselab test-seed")
    consensus = store.activity(node="PREPROCESS", step="consensus", parameters={})
    store.was_associated_with(consensus, software)
    word_ids: list[str] = []
    for index, text in enumerate(words):
        word_id = store.entity(
            prov_type="word", extent=_extent(index), attributes=word_attributes(text, _extent(index), index=index)
        )
        store.was_generated_by(word_id, consensus)
        word_ids.append(word_id)
    transcript = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "role": "consensus",
            "n_words": len(words),
            "word_ids": word_ids,
            "text": " ".join(words),
        },
    )
    store.was_generated_by(transcript, consensus)
    if scan is not None:
        attributes: dict[str, Any] = {"name": PII_SCAN, "signal": "consensus_transcript", "findings_n": findings_n}
        if scan == "declined":
            attributes[SCANNED] = False
        scan_id = store.entity(prov_type="measurement", extent=None, attributes=attributes)
        store.was_generated_by(scan_id, consensus)
    for index in range(findings_n):
        finding = store.entity(
            prov_type="pii", extent=_extent(index), attributes={"category": "PERSON", "detector": "stub"}
        )
        store.was_generated_by(finding, consensus)
    redact_act = store.activity(node="REDACT", step="plan", parameters={})
    store.was_associated_with(redact_act, software)
    for bounds in redacted:
        span = store.entity(prov_type="span", extent=bounds, attributes={"name": "redaction", "category": "PERSON"})
        store.was_generated_by(span, redact_act)
    if redact_outcome is not None:
        from senselab.audio.workflows.triage.nodes.common import write_verdict

        write_verdict(
            store, redact_act, software, node="REDACT", outcome=redact_outcome, kind=None, why="seeded", detail={}
        )


def _stub(monkeypatch: pytest.MonkeyPatch, rounds: Sequence[ReviewResult]) -> list[str]:
    """Replace the node's reviewer with one answer per round, recording every text it was handed."""
    seen: list[str] = []
    remaining = list(rounds)

    def _fake(original: str, *, redacted: str | None = None, **kw: Any) -> ReviewResult:  # noqa: ANN401
        seen.append(redacted if redacted is not None else original)
        assert remaining, "the node reviewed more times than the test declared answers for"
        return remaining.pop(0)

    monkeypatch.setattr(review_module, "review_transcript", _fake)
    return seen


def _clean(redaction: str = "not_applicable") -> ReviewResult:
    """A round that read both texts and would change nothing."""
    return ReviewResult(
        available=True,
        reasoning="Nothing here identifies the speaker.",
        redaction=redaction,
        original="clean",
        speakers="one",
        model_id="s/m",
        revision="a" * 40,
    )


def _flags(*proposal: ReviewProposal, original: str = "carries_pii") -> ReviewResult:
    """A round that read both texts and would remove something."""
    return ReviewResult(
        available=True,
        reasoning="A name survives.",
        redaction="incomplete",
        original=original,
        speakers="one",
        proposal=list(proposal),
        model_id="s/m",
        revision="a" * 40,
    )


def _annotation(store: ProvStore) -> dict[str, Any]:
    """The reading's summary, as VERDICT reads it."""
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == REDACTION_LLM_ANNOTATION]
    assert len(found) == 1, f"expected exactly one annotation, found {len(found)}"
    return dict(found[0].attributes)


class TestTheReviewerIsReachedFromTheTranscriptNotFromTheScan:
    """The 13,810 were never scanned; a reviewer behind the scan gate cannot check the gate."""

    def test_a_declined_scan_is_still_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEECH declined because every word is in the stimulus; near-match widened that twice."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["buttercup", "meadow"], scan="declined")
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == ["buttercup meadow"]
        annotation = _annotation(store)
        assert annotation["status"] == "clean"
        assert annotation["detector_state"] == "declined"

    def test_a_transcript_no_detector_ever_saw_is_still_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No scan measurement at all is the third detector state, and not a reason to skip."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan=None)
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == ["hello world"]
        assert _annotation(store)["detector_state"] == "unscanned"

    def test_a_scanned_clean_transcript_is_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The miss direction: the detectors marked nothing, which is what wants a second reader."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan="ran", findings_n=0)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        annotation = _annotation(store)
        assert annotation["detector_state"] == "scanned"
        assert annotation["detector_findings_n"] == 0

    def test_a_redacted_transcript_is_read_as_redacted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Where REDACT ran, the reviewer reads what would be released, never the findings in the clear."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "alice"], scan="ran", findings_n=1, redacted=[(1.0, 1.5)])
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == ["hello [PERSON]"]


class TestTheFourSilencesStayApart:
    """Switched off, nothing to read, tried and could not load, and ran: each its own state."""

    def test_switched_off_contacts_nothing_and_says_so(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off means no subprocess and an annotation recording the absence as a choice."""
        store = ProvStore(run_id="review-test")
        _seed(store)
        seen = _stub(monkeypatch, [])
        review(store, _config(tmp_path, "redaction:\n  llm_check:\n    enabled: false\n"))
        assert seen == []
        assert _annotation(store)["status"] == "disabled"

    def test_an_empty_transcript_is_nothing_to_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 20,518 with no lexical word; there is no text, so there is no reading."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=[], scan=None)
        seen = _stub(monkeypatch, [])
        review(store, _config(tmp_path))
        assert seen == []
        assert _annotation(store)["status"] == "nothing_to_read"

    def test_a_model_that_would_not_load_is_absent_not_clean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GPU queue must not read as a clean transcript."""
        store = ProvStore(run_id="review-test")
        _seed(store)
        _stub(monkeypatch, [ReviewResult(available=False, failure="OSError: no such model", model_id="s/m")])
        review(store, _config(tmp_path))
        annotation = _annotation(store)
        assert annotation["status"] == "absent"
        assert annotation["failure"] == "OSError: no such model"

    def test_the_retired_state_is_gone(self) -> None:
        """``not_run`` meant "the detectors marked nothing", which is now a reviewed population."""
        assert "not_run" not in review_module.REVIEW_STATES
        assert "not_run" not in review_module.DETECTOR_STATES, "and nothing else may reuse the name"
        assert "not_run" not in inspect.getsource(review_module)


class TestTheReadingIsARecordAndNeverAnAct:
    """It annotates. VERDICT may act on it; it acts on nothing itself."""

    def test_the_node_writes_no_verdict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A node with no verdict cannot reach the release axis, which is the whole guarantee."""
        store = ProvStore(run_id="review-test")
        _seed(store)
        _stub(
            monkeypatch,
            [_flags(ReviewProposal(text="world", action="redact", category="LOCATION", why="a place")), _clean()],
        )
        review(store, _config(tmp_path))
        assert [e for e in store.entities("verdict") if e.attributes["node"] == NODE] == []

    def test_a_flag_on_a_never_scanned_transcript_is_legible_as_exactly_that(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate let something through: flagged beside declined is the one reading that says so."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "alicia"], scan="declined")
        _stub(
            monkeypatch,
            [_flags(ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")), _clean()],
        )
        review(store, _config(tmp_path))
        annotation = _annotation(store)
        assert annotation["status"] == "flagged"
        assert annotation["detector_state"] == "declined"
        assert annotation["detector_findings_n"] == 0

    def test_the_annotation_names_redacts_outcome_where_there_was_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean reading beside a REDACT fail is a disagreement; beside no REDACT it is neither."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan="ran", findings_n=1, redact_outcome=Outcome.FAIL)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert _annotation(store)["detector_outcome"] == "fail"

    def test_no_redact_verdict_leaves_the_outcome_empty_rather_than_guessed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REDACT not having run is not a pass, and the annotation must not report one."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan="declined")
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert _annotation(store)["detector_outcome"] == ""


class TestTheTextIsWhatWouldBeReleased:
    """One renderer, so the reviewer and the release path cannot drift apart."""

    def test_an_unredacted_transcript_renders_as_its_words(self) -> None:
        """No redaction span means the released text is the transcript itself."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["one", "two", "three"], scan="declined")
        assert transcript_texts(store) == ("one two three", None)

    def test_a_redaction_span_renders_as_its_placeholder(self) -> None:
        """The same rendering REDACT writes into the released pair, recovered from the store alone."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["one", "two"], scan="ran", findings_n=1, redacted=[(1.0, 1.5)])
        assert transcript_texts(store) == ("one two", "one [PERSON]")

    def test_the_prompt_does_not_assert_a_redaction_happened(self) -> None:
        """Most of the population was never scanned; priming the model otherwise is a false premise."""
        from senselab.text.tasks.pii_detection import redaction_review

        assert "has already been automatically redacted" not in redaction_review._PROMPT
        assert "where an automatic redaction has already been applied" in redaction_review._PROMPT
