"""REVIEW: the reviewer reads the lexical residue the detectors read, and what it concludes stays a reading.

``specs/20260925-lexical-only-pii-pathway/design.md`` holds the reasoning.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    REVIEW_CARRIED,
    REVIEW_NEEDS_REREAD,
    REVIEW_NONE,
    REVIEW_READ,
    carry_reading_forward,
    held_reading,
    live_decisions,
    retire_decisions,
)
from senselab.audio.workflows.triage.nodes import review as review_module
from senselab.audio.workflows.triage.nodes.common import consensus_words, find_measurement, software_agent
from senselab.audio.workflows.triage.nodes.redact import transcript_texts
from senselab.audio.workflows.triage.nodes.review import (
    BACKFILL_HELD,
    BACKFILL_SKIPPED,
    BACKFILL_STORED,
    LLM_REVIEW_MEASUREMENT,
    NODE,
    backfill_from_store,
    refine_plan,
    review,
    review_cache_key,
)
from senselab.audio.workflows.triage.vocabulary import PII_SCAN, REDACTION_LLM_ANNOTATION, SCANNED, Outcome
from senselab.text.tasks.pii_detection.redaction_review import (
    PROMPT_VERSION,
    ReviewProposal,
    ReviewResult,
    parse_completion,
)
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
            prov_type="pii",
            extent=_extent(index),
            attributes={"category": "PERSON", "detector": "stub", "word_ids": []},
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


def _clean(redaction: str = "not_applicable", proposal: Sequence[ReviewProposal] = ()) -> ReviewResult:
    """A round that read both texts and would change nothing, or would only release ``proposal``."""
    return ReviewResult(
        available=True,
        reasoning="Nothing here identifies the speaker.",
        redaction=redaction,
        original="clean",
        speakers="one",
        proposal=list(proposal),
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
        release = ReviewProposal(text="alice", action="release", category="PERSON", why="a common word here")
        seen = _stub(monkeypatch, [_clean(proposal=[release])])
        review(store, _config(tmp_path))
        assert seen == ["hello [PERSON]"]


class TestTheReviewerSeesEveryRecogniserSReading:
    """Owner, 2026-10-09: the reviewer reads the consensus column by column, and its reading records that it did."""

    def test_the_whole_consensus_reaches_the_reviewer_with_each_recogniser_s_word(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every consensus word, not only the residue, with its kind, its PII findings and both readings."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["pa", "um", "my", "name", "is", "alice"], scan="ran", residue=[2, 3, 4, 5])
        alice = consensus_words(store)[5]
        finding = store.entity(
            prov_type="pii",
            extent=_extent(5),
            attributes={"category": "PERSON", "source": "gliner", "haystack": "asr_qwen", "word_ids": [alice.id]},
        )
        store.was_generated_by(finding, store.activity(node="SPEECH", step="pii", parameters={}))
        contexts: list[dict[str, Any]] = []

        def _fake(original: str, *, context: dict[str, Any] | None = None, **kw: Any) -> ReviewResult:  # noqa: ANN401
            contexts.append(dict(context or {}))
            return _clean()

        monkeypatch.setattr(review_module, "review_transcript", _fake)
        review(store, _config(tmp_path))
        (context,) = contexts
        transcript = context["consensus_transcript"]
        assert transcript["recognisers"] == ["asr_crisperwhisper", "asr_qwen"]
        assert transcript["findings_n"] == 1
        assert [word[4] for word in transcript["words"]] == ["task", "non_lexical", "", "", "", ""]
        assert transcript["words"][0][:4] == [0, "pa", "agreement", ["pa", "pa"]]
        assert transcript["words"][5][5] == [["PERSON", "gliner", "asr_qwen"]]
        assert len(transcript["words"]) == 6

    def test_the_reading_and_each_round_record_their_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No task declared: the consensus and the readings were given, the task was not."""
        store = ProvStore(run_id="review-test")
        _seed(store, scan="ran")
        _stub(monkeypatch, [_clean()])
        review(store, _config(tmp_path))
        inputs = _annotation(store)["review_inputs"]
        assert (inputs["task"], inputs["full_transcript"], inputs["pii_annotations"], inputs["asr_readings"]) == (
            False,
            True,
            True,
            True,
        )
        assert inputs["prompt_version"] == PROMPT_VERSION and inputs["version"] == 3
        assert [round_["review_inputs"] for round_ in _rounds_in(store)] == [inputs]

    def test_the_task_reading_reaches_the_annotation_and_the_context_names_the_nature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v12: the five task-reading fields are stored, and the task's nature, name and speech type are sent."""
        from senselab.audio.data_structures import AudioHints

        store = ProvStore(run_id="review-test")
        _seed(store, words=("hello", "world"), scan="ran")
        answer = _clean()
        answer.off_task_speech = "some"
        answer.off_task_quotes = ["hello"]
        answer.other_speaker = "assistant"
        answer.other_speaker_quotes = ["world"]
        answer.task_content_quotes = ["hello world"]
        _stub(monkeypatch, [answer])
        hint = AudioHints(
            instructions="Say it.",
            speech_type="elicited",
            metadata={"task_token": "animal-fluency", "task_name": "Animal fluency"},
        )
        review(store, _config(tmp_path), hint)
        annotation = _annotation(store)
        assert {key: annotation[key] for key in ("off_task_speech", "other_speaker")} == {
            "off_task_speech": "some",
            "other_speaker": "assistant",
        }
        assert annotation["off_task_quotes"] == ["hello"] and annotation["other_speaker_quotes"] == ["world"]
        assert annotation["task_content_quotes"] == ["hello world"]
        context = annotation["task_context"]
        assert context["task_nature"] == "item_generation" and context["speech_type"] == "elicited"
        assert context["task_name"] == "Animal fluency"
        assert annotation["review_inputs"]["instructions"] is True


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

    def test_a_quote_across_a_task_word_lands_on_the_residue_words_only(self) -> None:
        """Prompt 11 shows the task word "pa"; a quote running across it covers the residue words around it."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["alicia", "pa", "smith"], scan="ran", residue=[0, 2])
        _annotate(store, [{"text": "alicia pa smith", "action": "redact", "category": "PERSON", "why": "a name"}])
        applied = refine_plan(store, padding_ms=0)
        assert applied.unplaced == ()
        assert [(round(e.start, 3), round(e.end, 3)) for e in applied.extents] == [(0.0, 2.5)]

    def test_a_quote_of_a_task_word_alone_is_unplaced(self) -> None:
        """The task word is shown for context; the redaction never reaches it."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["alicia", "pa", "smith"], scan="ran", residue=[0, 2])
        _annotate(store, [{"text": "pa", "action": "redact", "category": "OTHER", "why": "?"}])
        applied = refine_plan(store, padding_ms=0)
        assert applied.extents == [] and applied.unplaced == ("pa",)

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

    def test_a_judgment_with_no_words_is_fed_back_until_the_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner, 2026-09-28: "incomplete" with no entries is not an answer; each round names the problem."""
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
        seen = _stub(monkeypatch, [bare, bare, bare])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 3, "the same text, asked again with feedback"
        annotation = _annotation(store)
        assert (annotation["status"], annotation["iterations"], annotation["converged"]) == ("flagged", 3, False)
        assert "listed no words" in annotation["problem"]
        rounds = _rounds_in(store)
        assert rounds[0]["feedback"] is None
        assert all("listed no words" in entry["feedback"] for entry in rounds[1:])

    def test_a_judgment_corrected_by_feedback_converges(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second round names the word, and the reading records that it converged."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        bare = _flags()
        named = _flags(ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name"))
        seen = _stub(monkeypatch, [bare, named, _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        assert len(seen) == 3
        annotation = _annotation(store)
        assert annotation["converged"] is True and annotation["problem"] is None

    def test_an_answer_without_its_conditions_part_is_read_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reading that leaves out CONDITIONS is fed back; the next round's condition is recorded."""
        store = ProvStore(run_id="r")
        _seed(store, words=["i", "have", "essential", "tremors"])
        silent = ReviewResult(
            available=True,
            reasoning="Nothing identifies the speaker.",
            redaction="not_applicable",
            original="clean",
            speakers="one",
            conditions_answered=False,
            model_id="s/m",
            revision="a" * 40,
        )
        listed = ReviewResult(
            available=True,
            reasoning="A diagnosis is named.",
            redaction="not_applicable",
            original="clean",
            speakers="one",
            proposal=[
                ReviewProposal(text="essential tremors", action="redact", category="CONDITION", why="a diagnosis")
            ],
            model_id="s/m",
            revision="a" * 40,
        )
        seen = _stub(monkeypatch, [silent, listed, _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        rounds = _rounds_in(store)
        assert len(seen) >= 2
        assert "no CONDITIONS part" in rounds[1]["feedback"]
        assert any(
            entry["category"] == "CONDITION" and entry["text"] == "essential tremors"
            for round_ in rounds
            for entry in round_["proposal"]
        )

    def test_a_quote_not_in_the_text_is_fed_back(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A proposal quoting words that are not there cannot be applied, so the reviewer is told which."""
        store = ProvStore(run_id="r")
        _seed(store, words=["hello", "alicia"])
        wrong = _flags(ReviewProposal(text="[PERSON]", action="redact", category="PERSON", why="a name"))
        right = _flags(ReviewProposal(text="alicia", action="redact", category="PERSON", why="a name"))
        _stub(monkeypatch, [wrong, right, _clean()])
        review(store, _config(tmp_path, LLM_ON + "    max_iterations: 3\n"))
        rounds = _rounds_in(store)
        assert "do not occur in the ORIGINAL" in rounds[1]["feedback"] and '"[PERSON]"' in rounds[1]["feedback"]
        assert _annotation(store)["converged"] is True

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


def _answered(completion: str) -> ReviewResult:
    """A fake backend's round: the completion parsed exactly as the shipped reviewer parses it."""
    parsed = parse_completion(completion)
    return ReviewResult(
        available=True,
        reasoning=parsed.reasoning,
        redaction=parsed.redaction,
        original=parsed.original,
        speakers=parsed.speakers,
        proposal=parsed.proposal,
        other_speakers=parsed.other_speakers,
        conditions_answered=parsed.conditions_answered,
        model_id="s/m",
        revision="a" * 40,
    )


def _speakers_answer(speakers: str, others: list[dict[str, Any]]) -> str:
    return (
        "REASONING: read it.\nREDACTION: not_applicable\nORIGINAL: clean\n"
        f"SPEAKERS: {speakers}\nOTHER_SPEAKERS: {json.dumps(others)}\nCONDITIONS: []\nPROPOSAL: []"
    )


class TestTheSpeakerJudgmentWeighsTheInstructions:
    """Owner, 2026-09-29: a second voice is quoted and judged against what the task expects."""

    def test_a_participant_asking_the_examiner_is_one_speaker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Animal fluency: "Is that enough?" is the participant talking, and nothing is quoted."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["cat", "dog", "horse", "is", "that", "enough"], scan="ran")
        _stub(monkeypatch, [_answered(_speakers_answer("one", []))])
        review(store, _config(tmp_path))
        annotation = _annotation(store)
        assert (annotation["speakers"], annotation["other_speakers"]) == ("one", [])
        assert annotation["prompt_version"] == PROMPT_VERSION

    def test_an_expected_examiner_is_recorded_as_expected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Story recall: the examiner's instruction is quoted and marked as a voice the task expects."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["tell", "me", "the", "story", "once", "a", "fox"], scan="ran")
        examiner = {"text": "tell me the story", "expected": True, "why": "the examiner's instruction"}
        _stub(monkeypatch, [_answered(_speakers_answer("more_than_one", [examiner]))])
        review(store, _config(tmp_path))
        assert _annotation(store)["other_speakers"] == [examiner]

    def test_more_than_one_without_a_quote_is_asked_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second voice named without its words is fed back; the next round quotes the interjection."""
        store = ProvStore(run_id="review-test")
        _seed(store, words=["pa", "pa", "who", "is", "that", "pa"], scan="ran")
        intruder = {"text": "who is that", "expected": False, "why": "nobody the task provides for"}
        _stub(
            monkeypatch,
            [
                _answered(_speakers_answer("more_than_one", [])),
                _answered(_speakers_answer("more_than_one", [intruder])),
            ],
        )
        review(store, _config(tmp_path))
        rounds = _rounds_in(store)
        assert [bool(r["problem"]) for r in rounds] == [True, False]
        assert "OTHER_SPEAKERS quoted no words" in str(rounds[1]["feedback"])
        annotation = _annotation(store)
        assert annotation["iterations"] == 2 and annotation["other_speakers"] == [intruder]


class TestAReadingIsKeptInTheResultCache:
    """E: the bounded loop's final reading is a result-cache entry keyed on exactly what it read."""

    @pytest.fixture(autouse=True)
    def _resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(review_module, "_resolved", lambda settings: "a" * 40)

    def _reviewed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: Sequence[ReviewResult]) -> ProvStore:
        store = ProvStore(run_id="review-cache-test")
        _seed(store, words=["my", "name", "is", "alice"], scan="ran")
        _stub(monkeypatch, rounds)
        review(store, _config(tmp_path))
        return store

    def test_the_same_text_is_read_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second recording with the same words, context and settings takes the first's reading."""
        first = self._reviewed(tmp_path, monkeypatch, [_clean()])
        miss = _annotation(first)["result_cache"]
        assert miss["hit"] is False and miss["stored"] is True
        second = self._reviewed(tmp_path, monkeypatch, [])
        hit = _annotation(second)
        assert hit["result_cache"] == {"key": miss["key"], "hit": True}
        assert (hit["status"], hit["speakers"], hit["iterations"]) == ("clean", "one", 1)
        assert len(_rounds_in(second)) == len(_rounds_in(first)) == 1

    def test_a_reading_equal_to_a_retired_one_comes_back_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache hit identical to a reading a replay retired is minted live, not born retired."""
        self._reviewed(tmp_path, monkeypatch, [_clean()])
        store = self._reviewed(tmp_path, monkeypatch, [])
        held = find_measurement(store, REDACTION_LLM_ANNOTATION)
        assert held is not None
        retire = store.activity(node="REPLAY", step="decision_superseded", parameters={"superseded": held.id})
        store.was_invalidated_by(held.id, retire)
        _stub(monkeypatch, [])
        review(store, _config(tmp_path))
        again = find_measurement(store, REDACTION_LLM_ANNOTATION)
        assert again is not None and again.id != held.id
        assert again.attributes["result_cache"] == held.attributes["result_cache"]

    def test_a_prompt_version_bump_misses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prompt and its parse are in the key, so a new prompt reads the text again."""
        first = self._reviewed(tmp_path, monkeypatch, [_clean()])
        monkeypatch.setattr(review_module, "PROMPT_VERSION", PROMPT_VERSION + 1)
        second = self._reviewed(tmp_path, monkeypatch, [_clean()])
        assert _annotation(second)["result_cache"]["hit"] is False
        assert _annotation(second)["result_cache"]["key"] != _annotation(first)["result_cache"]["key"]

    def test_the_settings_and_the_context_are_in_the_key(self) -> None:
        """A different iteration bound, generation ceiling, commit or task context is a different reading."""
        settings = {"model_id": "s/m", "max_new_tokens": 1024, "max_iterations": 3}
        base = review_cache_key("a b", None, {"task": "x"}, settings, "a" * 40)
        assert base == review_cache_key("a b", None, {"task": "x"}, dict(settings), "a" * 40)
        assert base != review_cache_key("a b", None, {"task": "x"}, {**settings, "max_iterations": 2}, "a" * 40)
        assert base != review_cache_key("a b", None, {"task": "x"}, {**settings, "max_new_tokens": 512}, "a" * 40)
        assert base != review_cache_key("a b", None, {"task": "y"}, settings, "a" * 40)
        assert base != review_cache_key("a b", "[PERSON] b", {"task": "x"}, settings, "a" * 40)
        assert base != review_cache_key("a b", None, {"task": "x"}, settings, "b" * 40)

    def test_an_absent_reading_is_not_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reviewer that did not load is recorded, and asked again next time."""
        failed = ReviewResult(available=False, failure="no GPU", model_id="s/m")
        first = self._reviewed(tmp_path, monkeypatch, [failed])
        assert _annotation(first)["result_cache"]["stored"] is False
        second = self._reviewed(tmp_path, monkeypatch, [_clean()])
        assert _annotation(second)["result_cache"]["hit"] is False

    def test_a_finished_store_backfills_the_key_review_would_ask_for(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seeding from a store under an empty cache, then reviewing the same text, is a hit."""
        first = self._reviewed(tmp_path, monkeypatch, [_clean()])
        monkeypatch.setenv("SENSELAB_RESULT_CACHE", str(tmp_path / "fresh-cache"))
        state, key = backfill_from_store(first, _config(tmp_path))
        assert (state, key) == (BACKFILL_STORED, _annotation(first)["result_cache"]["key"])
        assert backfill_from_store(first, _config(tmp_path))[0] == BACKFILL_HELD
        second = self._reviewed(tmp_path, monkeypatch, [])
        assert _annotation(second)["result_cache"]["hit"] is True

    def test_a_reading_from_another_prompt_version_is_not_backfilled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """r8's readings carry no prompt version, so no key they could fill will ever be asked for."""
        store = ProvStore(run_id="review-cache-test")
        _seed(store, scan="ran")
        _annotate(store, [])
        state, why = backfill_from_store(store, _config(tmp_path))
        assert state == BACKFILL_SKIPPED and "prompt version" in why


class TestAReplayKeepsTheReadingItDidNotReadAgain:
    """A replay under a configuration that does not read keeps an answered reading whose inputs stand."""

    @pytest.fixture(autouse=True)
    def _resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(review_module, "_resolved", lambda settings: "a" * 40)

    def _read_then_retire(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProvStore, Any, str]:
        """Read with the LLM on, re-open the store under a replay's own run id, and retire as a replay does."""
        finished = ProvStore(run_id="replay-review-test")
        _seed(finished, words=["my", "name", "is", "alice"], scan="ran")
        _stub(monkeypatch, [_clean()])
        review(finished, _config(tmp_path))
        finished.write_jsonl(tmp_path / "store.jsonl")
        store = ProvStore.read_jsonl(tmp_path / "store.jsonl", run_id="replay-review-test+replay")
        held = held_reading(store)
        assert held is not None and len(held.rounds) == 1
        software = software_agent(store)
        retire_decisions(store, live_decisions(store), software=software)
        return store, held, software

    def test_unchanged_inputs_keep_an_equivalent_live_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live annotation is the earlier reading, derived from it, and the replay's own is retired."""
        store, held, software = self._read_then_retire(tmp_path, monkeypatch)
        off = load_triage_config()
        review(store, off)
        assert carry_reading_forward(store, off, None, held, software=software) == (REVIEW_CARRIED, None)
        live = find_measurement(store, REDACTION_LLM_ANNOTATION)
        assert live is not None
        assert (live.attributes["status"], live.attributes["speakers"]) == ("clean", "one")
        assert store.derived_from(live.id) == [held.annotation.id]
        unread = [
            e
            for e in store.entities("measurement")
            if e.attributes.get("name") == REDACTION_LLM_ANNOTATION and e.attributes.get("status") == "disabled"
        ]
        assert len(unread) == 1 and store.is_invalidated(unread[0].id)
        rounds = [e for e in store.entities("measurement") if e.attributes.get("name") == LLM_REVIEW_MEASUREMENT]
        assert sum(1 for e in rounds if not store.is_invalidated(e.id)) == 1

    def test_changed_inputs_are_reported_and_nothing_is_carried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A redaction the earlier reading never saw leaves the replay's own annotation and says why."""
        store, held, software = self._read_then_retire(tmp_path, monkeypatch)
        plan = store.activity(node="REDACT", step="plan-again", parameters={})
        span = store.entity(prov_type="span", extent=_extent(3), attributes={"name": "redaction", "category": "PERSON"})
        store.was_generated_by(span, plan)
        off = load_triage_config()
        review(store, off)
        state, why = carry_reading_forward(store, off, None, held, software=software)
        assert state == REVIEW_NEEDS_REREAD and why is not None and "changed" in why
        live = find_measurement(store, REDACTION_LLM_ANNOTATION)
        assert live is not None and live.attributes["status"] == "disabled"

    def test_a_replay_that_reads_again_keeps_its_own_reading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the replay's REVIEW answered, the earlier reading is not re-attached."""
        store, held, software = self._read_then_retire(tmp_path, monkeypatch)
        _stub(monkeypatch, [_clean()])
        monkeypatch.setenv("SENSELAB_RESULT_CACHE", str(tmp_path / "fresh-cache"))
        review(store, _config(tmp_path))
        assert carry_reading_forward(store, _config(tmp_path), None, held, software=software) == (REVIEW_READ, None)

    def test_no_answered_reading_is_nothing_to_keep(self) -> None:
        """A store whose REVIEW never answered holds nothing a replay could lose."""
        store = ProvStore(run_id="replay-review-test")
        _seed(store, scan="ran")
        review(store, load_triage_config())
        assert held_reading(store) is None
        assert carry_reading_forward(store, load_triage_config(), None, None, software="s") == (REVIEW_NONE, None)
