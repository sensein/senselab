"""The review worker's lifetime: one load per process, and every failure a recorded absence.

The subprocess is real and the line protocol is the shipped one; only the *payload* is fake. The
worker script is replaced with a few lines of pure Python that speak the same protocol and load no
weights, so spawn counts, restarts, timeouts and the refusal record are all observable on a laptop.
A stub of ``review_transcript`` — which is what ``redact_test.py`` uses — could not see any of
them, because they all live underneath it.

See ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from senselab.text.tasks.pii_detection import redaction_review
from senselab.text.tasks.pii_detection.redaction_review import (
    review_transcript,
    shutdown_review_worker,
)

SHA = "b" * 40

# A worker that speaks the shipped protocol and loads nothing. It appends one line to `log` per
# spawn, so a test can count loads; `script` is a JSON queue of directives, and each load and each
# request pops the head: "ok" answers, "raise" raises, "hang" never answers, "exit" dies silently,
# "fail_load" makes the load itself raise, which is the no-GPU-reachable shape. The queue is on disk
# and popped destructively so that a *replacement* worker takes the next directive rather than
# repeating the one that killed its predecessor.
_FAKE_WORKER = r"""
import json, sys, time

MARKER = "%(marker)s"
LOG = "%(log)s"
SCRIPT = "%(script)s"
_replies = sys.stdout
sys.stdout = sys.stderr


def emit(payload):
    _replies.write(MARKER + json.dumps(payload) + "\n")
    _replies.flush()


def head():
    with open(SCRIPT) as handle:
        return (json.loads(handle.read()) or [None])[0]


def pop():
    with open(SCRIPT) as handle:
        steps = json.loads(handle.read())
    directive = steps.pop(0) if steps else "ok"
    with open(SCRIPT, "w") as handle:
        handle.write(json.dumps(steps))
    return directive


def main():
    init = json.loads(sys.stdin.readline())
    with open(LOG, "a") as handle:
        handle.write(json.dumps(init) + "\n")
    if head() == "fail_load":
        pop()
        raise RuntimeError("no device found")
    print("loading shards 1/9", file=sys.stderr)
    emit({"ready": True, "revision": init.get("revision"), "load_s": 0.01})
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        request = json.loads(line)
        if request.get("stop"):
            return
        directive = pop()
        if directive == "raise":
            raise RuntimeError("CUDA out of memory")
        if directive == "hang":
            time.sleep(600)
        if directive == "exit":
            sys.exit(3)
        print(json.dumps({"completion": "LEAKED", "output_tokens": 999}), file=_replies)
        print("chatter on stdout that is not a reply", file=_replies)
        _replies.flush()
        emit(
            {
                "completion": (
                    "REASONING: nothing identifying remains.\nREDACTION: not_applicable\n"
                    "ORIGINAL: clean\nSPEAKERS: one\nPROPOSAL: []"
                ),
                "generate_s": 0.01,
                "output_tokens": 11,
                "peak_reserved_mib": 71000,
                "resident_mib": 23000,
            }
        )


try:
    main()
except Exception as exc:
    emit({"error": {"type": type(exc).__name__, "message": str(exc)}})
    sys.exit(1)
"""


class Harness:
    """The fake worker's control surface.

    Attributes:
        log: The file each spawn appends its load payload to.
        script: The file holding the per-request directives.
    """

    def __init__(self, log: Path, script: Path) -> None:
        """Bind the two files the fake worker reads and writes.

        Args:
            log: Where spawns are recorded.
            script: Where the directives live.
        """
        self.log = log
        self.script = script

    @property
    def loads(self) -> list[dict[str, Any]]:
        """One load payload per spawn, in order."""
        if not self.log.is_file():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def directives(self, *steps: str) -> None:
        """Set what the worker does to each request in turn.

        Args:
            steps: The directives.
        """
        self.script.write_text(json.dumps(list(steps)))


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """A review backend whose worker is the fake script, with no worker left running afterwards."""
    shutdown_review_worker()
    log, script = tmp_path / "spawns.jsonl", tmp_path / "script.json"
    script.write_text("[]")
    monkeypatch.setattr(
        redaction_review,
        "_REVIEW_WORKER_SCRIPT",
        _FAKE_WORKER % {"marker": redaction_review._WORKER_MARKER, "log": log, "script": script},
    )
    monkeypatch.setattr(redaction_review, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(redaction_review, "venv_python", lambda *a, **k: sys.executable)
    monkeypatch.setattr(redaction_review, "_staged_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(redaction_review, "hf_subprocess_env", lambda *a, **k: {"PATH": "/usr/bin:/bin"})
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: SHA)
    try:
        yield Harness(log, script)
    finally:
        shutdown_review_worker()


class TestTheWeightsAreLoadedOncePerProcess:
    """The measured shipped cost was ~90% model load, paid again for every round of every file."""

    def test_three_reviews_pay_one_load(self, harness: Harness) -> None:
        """The whole point: a second review reuses the first one's worker."""
        results = [review_transcript(f"text {n}", model_id="stub/model") for n in range(3)]
        assert all(result.available for result in results)
        assert len(harness.loads) == 1, f"one load expected, the worker was spawned {len(harness.loads)} times"

    def test_only_the_round_that_loaded_says_it_loaded(self, harness: Harness) -> None:
        """``load_s`` is how a store separates the amortised rounds from the one that paid."""
        first = review_transcript("one", model_id="stub/model")
        second = review_transcript("two", model_id="stub/model")
        assert first.load_s > 0.0, "the first review paid the load and must say so"
        assert second.load_s == 0.0, "a reused worker loaded nothing"
        assert first.elapsed_s >= first.load_s, "the round's clock has to contain the load it paid"
        assert second.elapsed_s < first.elapsed_s, "the amortised round is the cheaper one"

    def test_a_different_commit_is_a_different_worker(self, harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weights already resident are the wrong weights when the resolved commit changes."""
        review_transcript("one", model_id="stub/model")
        monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "c" * 40)
        review_transcript("two", model_id="stub/model")
        assert [load["revision"] for load in harness.loads] == [SHA, "c" * 40]

    def test_shutdown_hands_the_weights_back(self, harness: Harness) -> None:
        """A caller that wants the memory before the process ends can have it."""
        review_transcript("one", model_id="stub/model")
        shutdown_review_worker()
        review_transcript("two", model_id="stub/model")
        assert len(harness.loads) == 2

    def test_the_answer_carries_what_it_generated(self, harness: Harness) -> None:
        """Output tokens are the thing generation cost tracks, so the store carries them."""
        result = review_transcript("one", model_id="stub/model")
        assert result.output_tokens == 11
        assert result.revision == SHA

    def test_the_answer_carries_what_it_holds_on_the_card(self, harness: Harness) -> None:
        """A worker kept alive occupies a GPU; how much decides whether anything else fits beside it."""
        result = review_transcript("one", model_id="stub/model")
        assert result.peak_reserved_mib == 71000
        assert result.resident_mib == 23000

    def test_a_worker_that_did_not_answer_claims_no_memory(self, harness: Harness) -> None:
        """Zero is the honest reading when nothing ran; a stale number would size the next run."""
        harness.directives("fail_load")
        result = review_transcript("one", model_id="stub/model")
        assert result.peak_reserved_mib == 0 and result.resident_mib == 0


class TestAWorkerThatCannotStartCostsOneAttempt:
    """A corpus of 15,000 recordings on a host with no reachable GPU must not pay 15,000 loads."""

    def test_the_first_failure_is_recorded_and_not_retried(self, harness: Harness) -> None:
        """The second call reports the same absence without spawning anything."""
        harness.directives("fail_load")
        first = review_transcript("one", model_id="stub/model")
        second = review_transcript("two", model_id="stub/model")
        assert not first.available and not second.available
        assert first.failure is not None and "no device found" in first.failure
        assert second.failure == first.failure
        assert len(harness.loads) == 1, f"a refused load was retried: {len(harness.loads)} spawns"

    def test_a_refusal_is_an_absence_and_not_a_raise(self, harness: Harness) -> None:
        """The absent path's contract: a recorded absence, a named failure, no proposal."""
        harness.directives("fail_load")
        result = review_transcript("one", model_id="stub/model")
        assert result.proposal == []
        assert result.model_id == "stub/model"
        assert result.elapsed_s > 0.0

    def test_shutdown_clears_the_refusal(self, harness: Harness) -> None:
        """A deliberate retry is the caller's to ask for; a silent one is not."""
        harness.directives("fail_load")
        review_transcript("one", model_id="stub/model")
        shutdown_review_worker()
        assert review_transcript("two", model_id="stub/model").available
        assert len(harness.loads) == 2

    def test_releasing_the_weights_between_recordings_does_not_clear_it(self, harness: Harness) -> None:
        """The caller that hands the memory back after every recording is not asking for a retry.

        This is the interaction, not either half of it: a node that releases the worker when its
        check ends would, if release also forgot the refusal, re-attempt a 23 GB load once per
        recording on a host that has no GPU — which is the exact stall the record exists to stop.
        """
        harness.directives("fail_load")
        for _ in range(3):
            review_transcript("a recording", model_id="stub/model")
            shutdown_review_worker(forget_failure=False)
        assert len(harness.loads) == 1, f"the refusal was forgotten: {len(harness.loads)} load attempts"


class TestAFailedReviewDoesNotPoisonTheProcess:
    """A worker that died mid-generation is gone; the *next* recording must still get a verdict."""

    def test_a_worker_that_raised_is_replaced(self, harness: Harness) -> None:
        """An out-of-memory on one transcript costs one reload, not every later review."""
        harness.directives("raise")
        failed = review_transcript("one", model_id="stub/model")
        recovered = review_transcript("two", model_id="stub/model")
        assert not failed.available and failed.failure is not None
        assert "CUDA out of memory" in failed.failure
        assert recovered.available, "a dead worker must be replaced, not reported dead forever"
        assert len(harness.loads) == 2

    def test_a_worker_that_vanished_is_replaced(self, harness: Harness) -> None:
        """A killed worker answers nothing at all; that is an absence, then a fresh load."""
        harness.directives("exit")
        failed = review_transcript("one", model_id="stub/model")
        assert not failed.available
        assert review_transcript("two", model_id="stub/model").available
        assert len(harness.loads) == 2

    def test_a_worker_that_never_answers_is_an_absence_not_a_hang(self, harness: Harness) -> None:
        """``timeout_s`` still bounds a review; the worker is killed rather than waited on."""
        harness.directives("hang")
        result = review_transcript("one", model_id="stub/model", timeout_s=2)
        assert not result.available
        assert result.failure is not None and "did not answer in 2s" in result.failure
        assert review_transcript("two", model_id="stub/model").available


class TestTheProtocolIsNotConfusedByTheLoaderSOwnOutput:
    """transformers writes progress to both streams; a reply is what carries the marker."""

    def test_unmarked_lines_on_either_stream_are_not_replies(self, harness: Harness) -> None:
        """The fake worker writes to stderr during load, and to stdout a line shaped like a reply.

        Prose on stdout would be rejected by the JSON parse whether the marker were checked or not,
        so it cannot show that the marker does any work. A well-formed object that is not a reply
        can: without the marker it is read as one, and the review returns the wrong completion.
        """
        result = review_transcript("one", model_id="stub/model")
        assert result.available
        assert "LEAKED" not in result.reasoning and "LEAKED" not in result.raw
        assert result.reasoning == "nothing identifying remains."
        assert result.output_tokens == 11, "a line the loader wrote was read as the model's answer"
        assert result.proposal == []

    def test_a_failure_message_carries_what_the_worker_was_last_doing(self, harness: Harness) -> None:
        """A load failure with no context is a bug report nobody can act on."""
        harness.directives("hang")
        result = review_transcript("one", model_id="stub/model", timeout_s=2)
        assert result.failure is not None
        assert "chatter on stdout that is not a reply" in result.failure or "loading shards" in result.failure


def test_the_request_names_the_task_its_instructions_and_its_stimulus() -> None:
    """Owner, 2026-09-27: the reviewer is told what the participant was asked to do and to say."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose

    body = _compose(
        "he is ninety-three",
        None,
        {
            "task": "story-recall",
            "speech_type": "recall",
            "instructions": "Recall the story in your own words.",
            "asked_to_say": "he is nearly ninety-three years old",
        },
    )
    assert "TASK: story-recall" in body
    assert "SPEECH TYPE: recall" in body
    assert "INSTRUCTIONS GIVEN TO THE PARTICIPANT: Recall the story in your own words." in body
    assert "STIMULUS THE PARTICIPANT WAS GIVEN TO SAY OR RECALL: he is nearly ninety-three years old" in body
    assert "not identifying for being said" in body
    assert body.index("TASK:") < body.index("ORIGINAL:")


def test_a_request_with_no_task_facts_carries_no_task_lines() -> None:
    """Nothing declared, nothing claimed."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose

    body = _compose("hello", None, {})
    assert "INSTRUCTIONS" not in body and "not identifying for being said" not in body


def test_the_prompt_states_every_safe_harbor_identifier_and_the_residual_clause() -> None:
    """Owner, 2026-09-28: the reviewer applies Safe Harbor, rendered from the packaged data."""
    from senselab.text.tasks.pii_detection import redaction_review

    standard = redaction_review.safe_harbor()
    codes = [identifier["code"] for identifier in standard["identifiers"]]
    assert codes == [chr(ord("A") + i) for i in range(18)]
    assert standard["citation"] in redaction_review._PROMPT
    assert all(f"({code})" in redaction_review._PROMPT for code in codes)
    assert "Actual knowledge" in redaction_review._PROMPT and "St. Petersburg" in redaction_review._PROMPT
    assert redaction_review.safe_harbor_codes("location") == ("B",)
    assert redaction_review.safe_harbor_codes("CONDITION") == ()


def test_a_place_released_without_a_reason_is_fed_back() -> None:
    """A place may be released under Safe Harbor only with a reason weighed against the transcript."""
    from senselab.text.tasks.pii_detection.redaction_review import (
        ReviewProposal,
        ReviewResult,
        answer_problem,
    )

    def result(*proposal: ReviewProposal) -> ReviewResult:
        return ReviewResult(available=True, redaction="complete", original="clean", proposal=list(proposal))

    bare = ReviewProposal(text="Florida", action="release", category="LOCATION", why="")
    reasoned = ReviewProposal(text="Florida", action="release", category="LOCATION", why="a state, nothing else")
    assert "give no reason" in (answer_problem(result(bare), "i grew up in Florida", "i grew up in [LOCATION]") or "")
    assert answer_problem(result(reasoned), "i grew up in Florida", "i grew up in [LOCATION]") is None
    empty = ReviewResult(available=True, redaction="incomplete", original="clean")
    assert "listed no words" in (answer_problem(empty, "hello alice", "hello [PERSON]") or "")
    agreeing = ReviewResult(available=True, redaction="complete", original="carries_pii")
    assert answer_problem(agreeing, "hello alice", "hello [PERSON]") is None


def test_the_parser_keeps_the_safe_harbor_letter() -> None:
    """Each entry carries the identifier the model named, one letter, upper-cased."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    parsed = parse_completion(
        "REASONING: x\nREDACTION: complete\nORIGINAL: clean\nSPEAKERS: one\nPROPOSAL: "
        '[{"text": "Florida", "action": "release", "category": "LOCATION", "safe_harbor": "b", "why": "a state"}]'
    )
    assert [(entry.safe_harbor, entry.why) for entry in parsed.proposal] == [("B", "a state")]


def test_the_conditions_part_is_parsed_into_condition_entries() -> None:
    """Every listed health condition becomes a ``redact`` entry of category CONDITION, beside the proposal."""
    from senselab.text.tasks.pii_detection.redaction_review import CONDITION, REDACT, parse_completion

    parsed = parse_completion(
        "REASONING: the speaker names two conditions and a city\nREDACTION: incomplete\nORIGINAL: carries_pii\n"
        'SPEAKERS: one\nCONDITIONS: [{"text": "essential tremors", "why": "a diagnosis"}, '
        '{"text": "synovial joint cyst", "why": "a diagnosis"}]\n'
        'PROPOSAL: [{"text": "St. Petersburg", "action": "redact", "category": "LOCATION", "safe_harbor": "B", '
        '"why": "a city"}]'
    )
    assert parsed.conditions_answered
    assert [(entry.text, entry.action, entry.category) for entry in parsed.proposal] == [
        ("St. Petersburg", REDACT, "LOCATION"),
        ("essential tremors", REDACT, CONDITION),
        ("synovial joint cyst", REDACT, CONDITION),
    ]
    assert parsed.reasoning == "the speaker names two conditions and a city"


def test_a_condition_listed_twice_is_one_entry_and_order_of_parts_does_not_matter() -> None:
    """A condition in both PROPOSAL and CONDITIONS is kept once; CONDITIONS may follow PROPOSAL."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    parsed = parse_completion(
        "REASONING: x\nREDACTION: complete\nORIGINAL: carries_pii\nSPEAKERS: one\n"
        'PROPOSAL: [{"text": "Parkinson\'s", "action": "redact", "category": "CONDITION", "why": "x"}]\n'
        'CONDITIONS: [{"text": "parkinson\'s", "why": "a diagnosis"}, {"text": "tremor", "why": "a symptom"}]'
    )
    assert [entry.text for entry in parsed.proposal] == ["Parkinson's", "tremor"]
    assert parsed.conditions_answered


def test_an_answer_without_its_conditions_part_is_fed_back() -> None:
    """A missing CONDITIONS part is an unusable answer; an empty list is an answer."""
    from senselab.text.tasks.pii_detection.redaction_review import (
        ParsedCompletion,
        ReviewResult,
        answer_problem,
        parse_completion,
    )

    missing = parse_completion("REASONING: x\nREDACTION: complete\nORIGINAL: clean\nSPEAKERS: one\nPROPOSAL: []")
    assert not missing.conditions_answered and missing.proposal == []
    none = parse_completion(
        "REASONING: x\nREDACTION: complete\nORIGINAL: clean\nSPEAKERS: one\nCONDITIONS: []\nPROPOSAL: []"
    )
    assert none.conditions_answered

    def result(parsed: ParsedCompletion) -> ReviewResult:
        return ReviewResult(
            available=True, redaction="complete", original="clean", conditions_answered=parsed.conditions_answered
        )

    assert "no CONDITIONS part" in (answer_problem(result(missing), "hello", None) or "")
    assert answer_problem(result(none), "hello", None) is None


def test_the_prompt_asks_for_conditions_apart_from_safe_harbor() -> None:
    """Conditions are a required part of their own, and no longer a PROPOSAL category."""
    assert "CONDITIONS: a JSON array" in redaction_review._PROMPT
    assert "exactly seven parts" in redaction_review._PROMPT
    assert "CONTACT, CONDITION, OTHER" not in redaction_review._PROMPT


def test_the_other_speakers_part_is_parsed_and_does_not_shadow_the_speakers_label() -> None:
    """``OTHER_SPEAKERS:`` ends in ``SPEAKERS:``; the one-word label is still the SPEAKERS line's."""
    parsed = redaction_review.parse_completion(
        "REASONING: two voices.\nREDACTION: not_applicable\nORIGINAL: clean\nSPEAKERS: more_than_one\n"
        'OTHER_SPEAKERS: [{"text": "who is that", "expected": false, "why": "an interjection"}]\n'
        "CONDITIONS: []\nPROPOSAL: []"
    )
    assert parsed.speakers == "more_than_one"
    assert [(o.text, o.expected) for o in parsed.other_speakers] == [("who is that", False)]
    assert parsed.conditions_answered and parsed.proposal == []
    assert "OTHER_SPEAKERS" not in parsed.reasoning


def test_more_than_one_with_no_quote_is_a_problem_to_feed_back() -> None:
    """The fold can only weigh a second voice against the task if the reading quotes it."""
    unquoted = redaction_review.ReviewResult(available=True, speakers="more_than_one", original="clean")
    assert "OTHER_SPEAKERS quoted no words" in str(redaction_review.answer_problem(unquoted, "who is that", None))
    quoted = redaction_review.ReviewResult(
        available=True,
        speakers="more_than_one",
        original="clean",
        other_speakers=[redaction_review.OtherSpeaker("who is that", False)],
    )
    assert redaction_review.answer_problem(quoted, "pa pa who is that", None) is None
    elsewhere = redaction_review.ReviewResult(
        available=True,
        speakers="more_than_one",
        original="clean",
        other_speakers=[redaction_review.OtherSpeaker("never said", False)],
    )
    assert "do not occur in the ORIGINAL" in str(redaction_review.answer_problem(elsewhere, "pa pa", None))


def test_the_prompt_counts_an_expected_voice_as_another_voice() -> None:
    """Owner, 2026-09-30: an examiner or a model speaker is still more_than_one; the participant asking is not."""
    prompt = redaction_review._PROMPT
    assert "an expected voice is still another voice" in prompt
    assert '"Is that enough?"' in prompt and "is still one speaker" in prompt
    assert "You were given the text" in prompt and "repeat after them" in prompt
    assert redaction_review.PROMPT_VERSION == 3
