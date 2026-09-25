"""REVIEW: the reviewer reads the lexical residue the detectors read, and what it concludes stays a reading.

``specs/20260925-lexical-only-pii-pathway/design.md`` holds the reasoning.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Sequence

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import review as review_module
from senselab.audio.workflows.triage.nodes.redact import transcript_texts
from senselab.audio.workflows.triage.nodes.review import NODE, refine_plan, review
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
    residue: Sequence[int] | None = None,
) -> None:
    """The store SPEECH and REDACT leave behind, in the four shapes REVIEW must tell apart.

    ``scan`` is ``"ran"``, ``"declined"`` or None for a store carrying no scan measurement at all.
    ``redacted`` are the planned extents REDACT left as ``redaction`` spans. ``residue`` are the
    positions of the words SPEECH's scan read; every word by default where the scan ran, none where
    it was declined.
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
        read = (range(len(words)) if scan == "ran" else ()) if residue is None else residue
        attributes: dict[str, Any] = {
            "name": PII_SCAN,
            "signal": "consensus_transcript",
            "findings_n": findings_n,
            "residue_word_ids": [word_ids[position] for position in read],
        }
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


def _annotate(store: ProvStore, proposal: list[dict[str, str]]) -> None:
    """Write the annotation a completed REVIEW would have left, carrying this proposal."""
    software = store.agent(agent_type="software", version="senselab test-seed")
    activity = store.activity(node=NODE, step="llm_check", parameters={})
    store.was_associated_with(activity, software)
    entity = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": REDACTION_LLM_ANNOTATION, "signal": "consensus_transcript", "proposal": proposal},
    )
    store.was_generated_by(entity, activity)


def _rounds_in(store: ProvStore) -> list[dict[str, Any]]:
    """Every per-round record the reading left, in order.

    Returns:
        One mapping per round.
    """
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == "redaction_llm_review"]
    return [dict(entity.attributes) for entity in found]


def _annotation(store: ProvStore) -> dict[str, Any]:
    """The reading's summary, as VERDICT reads it."""
    found = [e for e in store.entities("measurement") if e.attributes.get("name") == REDACTION_LLM_ANNOTATION]
    assert len(found) == 1, f"expected exactly one annotation, found {len(found)}"
    return dict(found[0].attributes)


class TestTheReviewerReadsTheResidue:
    """The reviewer reads exactly the string SPEECH's scan read, and nothing where it read nothing."""

    def test_a_declined_scan_is_nothing_to_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEECH declined because nothing lexical lay outside the task: no model is contacted."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["buttercup", "meadow"], scan="declined")
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == []
        assert _annotation(store)["status"] == "nothing_to_read"

    def test_a_store_with_no_scan_is_nothing_to_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """SPEECH never ran, so no residue was computed and nothing reaches the pathway."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan=None)
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == []
        assert _annotation(store)["detector_state"] == "unscanned"

    def test_only_the_residue_is_read(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A syllable train around a disclosure: the reviewer reads the disclosure alone."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["pa", "pa", "my", "name", "is", "alice", "pa"], scan="ran", residue=[2, 3, 4, 5])
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert seen == ["my name is alice"]

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
        _seed(store, words=["one", "two", "three"], scan="ran")
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


class TestTheProposalBecomesTheAppliedRedaction:
    """One artefact, cut from the span set that was reasoned about. The owner, 2026-09-24."""

    def test_a_proposed_removal_is_added_to_the_applied_set(self) -> None:
        """The reviewer found what the detectors missed; the audio must lose it."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "alicia"], scan="ran")
        _annotate(store, [{"text": "alicia", "action": "redact", "category": "PERSON", "why": "a name"}])
        applied = refine_plan(store, padding_ms=0)
        assert [(round(e.start, 3), round(e.end, 3), e.category) for e in applied.extents] == [(1.0, 1.5, "PERSON")]
        assert applied.unplaced == ()

    def test_a_proposed_release_drops_a_detector_span(self) -> None:
        """96.5% of DATE_TIME marks carry no calendar anchor; this is how one stops being cut."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "world"], scan="ran", findings_n=1, redacted=[(1.0, 1.5)])
        _annotate(store, [{"text": "world", "action": "release", "category": "DATE_TIME", "why": "no anchor"}])
        applied = refine_plan(store, padding_ms=0)
        assert applied.extents == []
        assert applied.released_n == 1

    def test_an_unlocatable_removal_keeps_every_detector_span_and_is_recorded(self) -> None:
        """An unplaceable finding once widened to the whole transcript; it may only fail closed."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "world"], scan="ran", findings_n=1, redacted=[(1.0, 1.5)])
        _annotate(store, [{"text": "nowhere in this text", "action": "redact", "category": "ID", "why": "an id"}])
        applied = refine_plan(store, padding_ms=0)
        assert [(round(e.start, 3), round(e.end, 3)) for e in applied.extents] == [(1.0, 1.5)]
        assert applied.unplaced == ("nowhere in this text",)

    def test_an_unlocatable_release_is_refused(self) -> None:
        """A release it cannot place must never widen what is handed on."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "world"], scan="ran", findings_n=1, redacted=[(1.0, 1.5)])
        _annotate(store, [{"text": "not here", "action": "release", "category": "DATE_TIME", "why": "no anchor"}])
        applied = refine_plan(store, padding_ms=0)
        assert [(round(e.start, 3), round(e.end, 3)) for e in applied.extents] == [(1.0, 1.5)]
        assert applied.released_n == 0
        assert applied.unplaced == ("not here",)

    def test_every_applied_span_carries_the_reason_it_is_there(self) -> None:
        """The audit trail the discarded second artefact would otherwise have been."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["hello", "alicia"], scan="ran")
        _annotate(store, [{"text": "alicia", "action": "redact", "category": "PERSON", "why": "a name"}])
        applied = refine_plan(store, padding_ms=0)
        assert applied.reasons == {"PERSON": "a name"}


class TestTheLoopIsBoundedAndReReadsTheMaskedText:
    """Carried over from the REDACT step the reviewer used to be. The loop moved; the bound did not.

    These are the guards the move to REVIEW dropped along with ``redact_test.py``'s LLM block. Each
    one is about behaviour that survived the move intact, so losing it lost only the check.
    """

    def test_the_loop_is_bounded_by_the_config_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reviewer that flags forever stops at ``max_iterations`` and says so."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        proposal = ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")
        seen = _stub(monkeypatch, [_flags(proposal), _flags(proposal)])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 2\n"))
        annotation = _annotation(store)
        assert len(seen) == 2
        assert annotation["iterations"] == 2
        assert annotation["status"] == "flagged"

    def test_a_flagged_round_reviews_again_on_the_masked_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The second round must not be handed the text the first round objected to."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        proposal = ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")
        seen = _stub(monkeypatch, [_flags(proposal), _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert "alicia" in seen[0]
        assert "alicia" not in seen[1]

    def test_a_clean_first_round_stops_there(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bound is a ceiling, not a count."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "world"])
        seen = _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 1
        assert _annotation(store)["iterations"] == 1


class TestTheWeightsAreReleasedUnlessTheRunSaysOtherwise:
    """A resident worker amortises the load; a run that did not ask for one must not leak it."""

    def test_the_check_releases_the_worker_when_it_ends(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default. One recording's review must not hold a card after it concludes."""
        released: list[bool] = []
        monkeypatch.setattr(
            review_module, "shutdown_review_worker", lambda **kw: released.append(kw.get("forget_failure"))
        )
        store = ProvStore(run_id="r")
        _seed(store)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        assert released == [False]

    def test_a_run_that_asks_for_residency_keeps_them(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """What makes a corpus pass affordable: the weights survive one recording."""
        released: list[bool] = []
        monkeypatch.setattr(
            review_module, "shutdown_review_worker", lambda **kw: released.append(kw.get("forget_failure"))
        )
        store = ProvStore(run_id="r")
        _seed(store)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path, LLM_ON + "    keep_worker_resident: true\n"))
        assert released == []

    def test_a_review_that_raised_still_releases(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raise must not be the path that leaks the card."""
        released: list[bool] = []
        monkeypatch.setattr(
            review_module, "shutdown_review_worker", lambda **kw: released.append(kw.get("forget_failure"))
        )

        def _boom(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("the worker died mid-read")

        monkeypatch.setattr(review_module, "review_transcript", _boom)
        store = ProvStore(run_id="r")
        _seed(store)
        with pytest.raises(RuntimeError):
            review(store, _config(tmp_path))
        assert released == [False]


class TestTheLoopStopsWhenAnotherRoundCannotDiffer:
    """The loop's whole action is the mask. Masking nothing means the next round reads the same text.

    Measured over the first 9,670 reviewed recordings: 3,131 ran to the ceiling, 86% of them saying
    the redaction was incomplete while proposing nothing to remove, and 99.2% of every multi-round
    recording gained nothing after round one. Those re-reads were 39% of the pass's GPU time.
    """

    def test_a_flag_with_no_removal_stops_at_one_round(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 86% case: flagged on the redaction judgment alone, nothing proposed, nothing to mask."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        bare = ReviewResult(
            available=True,
            reasoning="Something is still there.",
            redaction="incomplete",
            original="clean",
            speakers="one",
            model_id="s/m",
            revision="a" * 40,
        )
        seen = _stub(monkeypatch, [bare])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 1, "a second round would have read the same string"
        annotation = _annotation(store)
        assert annotation["status"] == "flagged"
        assert annotation["iterations"] == 1

    def test_a_flag_that_proposes_a_removal_still_iterates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case the loop exists for: the mask changes the text, so another round can differ."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        proposal = ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")
        seen = _stub(monkeypatch, [_flags(proposal), _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 2
        assert "alicia" in seen[0] and "alicia" not in seen[1]

    def test_a_release_only_proposal_stops_too(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A proposal that only asks for less to be removed masks nothing either."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        release_only = ReviewResult(
            available=True,
            reasoning="Too much was taken.",
            redaction="incomplete",
            original="clean",
            speakers="one",
            proposal=[ReviewProposal(text="alicia", action="release", category="PERSON", why="not a name")],
            model_id="s/m",
            revision="a" * 40,
        )
        seen = _stub(monkeypatch, [release_only])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 1


class TestTheRoundsReachTheStoreAndTheSummaryStaysClean:
    """The chain of thought and the cost are per-round records; the annotation is the summary."""

    def test_every_round_reaches_the_store_with_its_reasoning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What the reader said, verbatim, once per round, which is what the step exists for."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        proposal = ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")
        _stub(monkeypatch, [_flags(proposal), _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        rounds = _rounds_in(store)
        assert [entry["iteration"] for entry in rounds] == [1, 2]
        assert [entry["reasoning"] for entry in rounds] == ["A name survives.", "Nothing here identifies the speaker."]

    def test_each_round_records_its_clock_and_how_much_of_it_was_the_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A store answers what the pass cost without a stopwatch outside the graph."""
        store = ProvStore(run_id="r")
        _seed(store)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        recorded = _rounds_in(store)[0]
        assert {"elapsed_s", "load_s"} <= set(recorded)

    def test_no_timing_reaches_the_annotation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """VERDICT reads the summary, and how long a card was busy decides nothing about a recording."""
        store = ProvStore(run_id="r")
        _seed(store)
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        annotation = _annotation(store)
        assert {"elapsed_s", "load_s"} <= set(_rounds_in(store)[0]), "the rounds must still carry them"
        assert not [key for key in annotation if key.endswith(("_s", "_mib"))]


class TestTheReviewerDegradesHonestly:
    """A model that will not load is a gap in the record, never a clean reading."""

    def test_a_model_lost_after_it_already_flagged_keeps_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reading that already happened is not undone by the round that could not."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        proposal = ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name")
        lost = ReviewResult(available=False, model_id="s/m", failure="the worker would not start")
        _stub(monkeypatch, [_flags(proposal), lost])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        annotation = _annotation(store)
        assert annotation["status"] == "flagged"
        assert annotation["flagged"] == ["PERSON"]
        assert annotation["failure"] == "the worker would not start"


class TestTheBracketedTokensAreNotSpeech:
    """The reviewer reads the string the detectors read, which is the one without the markers.

    Measured on the r4 corpus: 85.2% of the non-lexical tasks' transcripts are nothing but
    ``[breath]``, ``[UH]``, ``[cough]``, ``[laughter]``. A probe over 20 of them read all 20 clean,
    so this is not about the model being fooled -- it is about the reviewer and the detectors
    reading one string, and about not spending a card to be told that ``[breath]`` is not a name.
    """

    def test_a_transcript_of_only_markers_is_nothing_to_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It renders empty, so the node records the silence rather than contacting a model."""
        store = ProvStore(run_id="r")
        _seed(store, words=["[breath]", "[cough]"])
        seen = _stub(monkeypatch, [])
        review(store, _config(tmp_path))
        assert seen == [], "no model may be contacted for a transcript of markers"
        assert _annotation(store)["status"] == "nothing_to_read"

    def test_a_marker_between_two_words_leaves_the_words_joined(self, tmp_path: Path) -> None:
        """What the reviewer is shown, and therefore what its quotes are taken from."""
        store = ProvStore(run_id="r")
        _seed(store, words=["my", "[UH]", "name"])
        original, redacted = transcript_texts(store)
        assert original == "my name"
        assert redacted is None

    def test_a_quote_spanning_a_dropped_marker_still_locates_its_words(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The locator must search the string the reviewer read, or the redaction is lost silently.

        ``my [UH] name`` reads as ``my name``, so a proposal quoting ``my name`` has to resolve to
        the two real words. Searching the unfiltered join would not find it, and an unplaceable
        removal keeps everything -- a dropped redaction that reports as success.
        """
        store = ProvStore(run_id="r")
        _seed(store, words=["my", "[UH]", "name"])
        _annotate(store, [{"text": "my name", "action": "redact", "category": "PERSON", "why": "a name"}])
        plan = refine_plan(store, padding_ms=0)
        assert tuple(plan.unplaced) == ()
        assert plan.added_n == 1
        # The extent is the time hull of the two real words, so it spans the marker that sits
        # between them. That is right: the marker is dropped from the *text*, and redacting the
        # audio under a breath between two redacted words takes nothing away.
        assert plan.extents[0].start == pytest.approx(0.0)
        assert plan.extents[0].end == pytest.approx(2.5)
