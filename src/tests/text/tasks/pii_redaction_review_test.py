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


def test_the_prompt_states_the_policy_and_the_remaining_safe_harbor_identifiers() -> None:
    """Policy v7 replaces Safe Harbor (A)-(C); the identifiers (D)-(R) are listed as written."""
    from senselab.text.tasks.pii_detection import redaction_review

    standard = redaction_review.safe_harbor()
    codes = [identifier["code"] for identifier in standard["identifiers"]]
    assert codes == [chr(ord("A") + i) for i in range(18)]
    assert standard["citation"] in redaction_review._PROMPT
    assert all(f"({code})" in redaction_review._PROMPT for code in codes[3:])
    assert not any(f"  ({code})" in redaction_review._PROMPT for code in codes[:3])
    for rule in ("Ray Bradbury", "I had COVID in 2021", "Halloween", "USF voice center", "never hold a recording back"):
        assert rule in redaction_review._PROMPT
    assert redaction_review.safe_harbor_codes("location") == ("B",)
    assert redaction_review.safe_harbor_codes("CONDITION") == ()


def test_a_release_the_policy_never_allows_is_fed_back() -> None:
    """A year, a season, an age or a state is never released; a name or a country needs its reason."""
    from senselab.text.tasks.pii_detection.redaction_review import (
        ReviewProposal,
        ReviewResult,
        answer_problem,
    )

    def result(*proposal: ReviewProposal) -> ReviewResult:
        return ReviewResult(available=True, redaction="complete", original="clean", proposal=list(proposal))

    for text in ("Florida", "2021", "October", "73 years old", "Halloween"):
        forbidden = ReviewProposal(text=text, action="release", category="LOCATION", why="looks fine")
        assert "always removes" in (answer_problem(result(forbidden), f"i said {text}", None) or ""), text
    bare = ReviewProposal(text="Mexico", action="release", category="LOCATION", why="")
    reasoned = ReviewProposal(text="Mexico", action="release", category="LOCATION", why="a country, nothing else")
    assert "give no reason" in (answer_problem(result(bare), "we moved from Mexico", "we moved from [LOCATION]") or "")
    assert answer_problem(result(reasoned), "we moved from Mexico", "we moved from [LOCATION]") is None
    unnamed = ReviewProposal(text="Ray Bradbury", action="release", category="PERSON", why="")
    assert "public figure" in (answer_problem(result(unnamed), "i read Ray Bradbury", None) or "")
    weekday = ReviewProposal(text="on Monday", action="release", category="DATE_TIME", why="a weekday")
    assert answer_problem(result(weekday), "i went on Monday", "i went [DATE_TIME]") is None
    empty = ReviewResult(available=True, redaction="incomplete", original="clean")
    assert "listed no words" in (answer_problem(empty, "hello alice", "hello [PERSON]") or "")
    agreeing = ReviewResult(available=True, redaction="complete", original="carries_pii")
    assert answer_problem(agreeing, "hello alice", "hello [PERSON]") is None


def test_v8_places_need_a_reason_and_relabels_come_from_the_set() -> None:
    """A place smaller than a country goes only with a place reason; a state only as historical; relabels are closed."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewProposal, ReviewResult, answer_problem

    def result(*proposal: ReviewProposal) -> ReviewResult:
        return ReviewResult(available=True, redaction="complete", original="clean", proposal=list(proposal))

    original = "a fighter in the Roman Empire, and the Civil War in Virginia, and Star Wars"
    bare = ReviewProposal(text="Roman Empire", action="release", category="LOCATION", why="history")
    assert "without a place_reason" in (answer_problem(result(bare), original, None) or "")
    reasoned = ReviewProposal(
        text="Roman Empire", action="release", category="LOCATION", why="history", place_reason="historical"
    )
    assert answer_problem(result(reasoned), original, None) is None
    state = ReviewProposal(text="Virginia", action="release", category="LOCATION", why="war", place_reason="fictional")
    assert "always removes" in (answer_problem(result(state), original, None) or "")
    historical = ReviewProposal(
        text="Virginia", action="release", category="LOCATION", why="war", place_reason="historical"
    )
    assert answer_problem(result(historical), original, None) is None
    unknown = ReviewProposal(text="Star Wars", action="release", category="PERSON", why="a film", relabel="movie")
    assert "outside the allowed values" in (answer_problem(result(unknown), original, None) or "")
    titled = ReviewProposal(text="Star Wars", action="release", category="PERSON", why="a film", relabel="work_title")
    assert answer_problem(result(titled), original, None) is None
    as_place = ReviewProposal(text="Star Wars", action="release", category="PERSON", why="a planet", relabel="place")
    assert "without a place_reason" in (answer_problem(result(as_place), original, None) or "")


def test_v8_the_parser_and_the_payload_keep_the_relabel_and_the_place_reason() -> None:
    """Both keys survive the parse and the stored payload, lower-cased."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewResult, parse_completion, review_payload

    completion = (
        "REASONING: a film and an empire.\nREDACTION: complete\nORIGINAL: clean\nSPEAKERS: one\n"
        "OTHER_SPEAKERS: []\nCONDITIONS: []\nINSTRUCTIONS_SPOKEN: []\nPROPOSAL: "
        '[{"text": "Star Wars", "action": "release", "category": "PERSON", "why": "a film", "relabel": "Work_Title"},'
        ' {"text": "Roman Empire", "action": "release", "category": "LOCATION", "why": "history",'
        ' "place_reason": "historical"}]'
    )
    parsed = parse_completion(completion)
    assert [(entry.relabel, entry.place_reason) for entry in parsed.proposal] == [
        ("work_title", ""),
        ("", "historical"),
    ]
    payload = review_payload(ReviewResult(available=True, proposal=parsed.proposal))
    assert payload["proposal"][0]["relabel"] == "work_title" and payload["proposal"][1]["place_reason"] == "historical"


def test_v8_each_task_family_gets_its_task_content_guidance() -> None:
    """Productive vocabulary is told its definitions are task content; a family with no entry gets none."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose, task_guidance, task_guidance_digest

    assert "definition task" in task_guidance("productive-vocabulary")
    assert "picture" in task_guidance("picture-description-option1")
    assert "reads a given text" in task_guidance("harvard-sentences-list")
    assert task_guidance("free-speech") == "" and task_guidance(None) == ""
    body = _compose("Roman fighter", None, {"task": "productive-vocabulary", "asked_to_say": "gladiator"})
    assert "WHAT IS TASK CONTENT IN THIS TASK:" in body and "Roman Empire" in body
    assert len(task_guidance_digest()) == 64


def test_the_parser_keeps_the_safe_harbor_letter() -> None:
    """Each entry carries the identifier the model named, one letter, upper-cased."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    parsed = parse_completion(
        "REASONING: x\nREDACTION: complete\nORIGINAL: clean\nSPEAKERS: one\nPROPOSAL: "
        '[{"text": "Florida", "action": "release", "category": "LOCATION", "safe_harbor": "b", "why": "a state"}]'
    )
    assert [(entry.safe_harbor, entry.why) for entry in parsed.proposal] == [("B", "a state")]


def test_the_conditions_part_is_parsed_apart_from_the_proposal() -> None:
    """Every listed health condition is a condition for the record, never a ``redact`` entry."""
    from senselab.text.tasks.pii_detection.redaction_review import REDACT, parse_completion

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
    ]
    assert [(entry.text, entry.why) for entry in parsed.conditions] == [
        ("essential tremors", "a diagnosis"),
        ("synovial joint cyst", "a diagnosis"),
    ]
    assert parsed.reasoning == "the speaker names two conditions and a city"


def test_a_condition_listed_twice_is_one_entry_and_order_of_parts_does_not_matter() -> None:
    """A condition listed twice is kept once; CONDITIONS may follow PROPOSAL."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    parsed = parse_completion(
        "REASONING: x\nREDACTION: complete\nORIGINAL: carries_pii\nSPEAKERS: one\n"
        "PROPOSAL: []\n"
        'CONDITIONS: [{"text": "Parkinson\'s", "why": "a diagnosis"}, {"text": "parkinson\'s", "why": "again"}, '
        '{"text": "tremor", "why": "a symptom"}]'
    )
    assert [entry.text for entry in parsed.conditions] == ["Parkinson's", "tremor"]
    assert parsed.proposal == [] and parsed.conditions_answered


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
    assert "exactly thirteen parts" in redaction_review._PROMPT
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
    assert redaction_review.PROMPT_VERSION >= 5


def test_the_prompt_lists_only_specific_diagnoses() -> None:
    """Owner, 2026-09-30: CONDITIONS are named diagnoses, not symptoms, moods, events or bare treatments."""
    prompt = redaction_review._PROMPT
    assert "every specific medical diagnosis the speaker attributes to themselves" in prompt
    assert "Do not list symptoms or sensations" in prompt and "feelings or moods" in prompt
    assert "surgery, a fall, an accident" in prompt
    assert "lists Parkinson's disease" in prompt
    assert "every diagnosis under point 5" in prompt


def test_the_prompt_attributes_instructions_addressed_to_the_participant_to_another_voice() -> None:
    """r9 story recall: 'I said you have up to five minutes' was read as the participant repeating the instructions."""
    prompt = redaction_review._PROMPT
    assert "do not assume the participant is repeating the instructions" in prompt
    assert '"I said you have up to five minutes"' in prompt
    assert "answer unclear and still quote those words in OTHER_SPEAKERS" in prompt


STORY_RECALL_CARD = (
    "Okay. So what about it? You were given the test, read the text, have you familiarized? I said you have up "
    "to five minutes to read it as many times as you want. Okay. So am I supposed to be recalling the story now?"
)


def test_the_instructions_spoken_part_is_parsed_into_quotes() -> None:
    """Owner, 2026-10-01: the reviewer quotes where the task's instructions are spoken; the story-recall card."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    quote = "I said you have up to five minutes to read it as many times as you want"
    parsed = parse_completion(
        "REASONING: the examiner restates the instructions\nREDACTION: not_applicable\nORIGINAL: clean\n"
        "SPEAKERS: one\nOTHER_SPEAKERS: []\nCONDITIONS: []\n"
        f'INSTRUCTIONS_SPOKEN: [{{"text": "{quote}", "why": "the five-minute reading instruction"}}]\nPROPOSAL: []'
    )
    assert parsed.instructions_answered
    assert parsed.instructions_spoken == [quote]
    assert parsed.proposal == [] and "INSTRUCTIONS_SPOKEN" not in parsed.reasoning


def test_an_answer_without_its_instructions_spoken_part_is_fed_back() -> None:
    """A missing INSTRUCTIONS_SPOKEN part is unusable; [] is an answer; a quote must occur in the ORIGINAL."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewResult, answer_problem, parse_completion

    head = "REASONING: x\nREDACTION: not_applicable\nORIGINAL: clean\nSPEAKERS: one\nCONDITIONS: []\n"
    missing = parse_completion(head + "PROPOSAL: []")
    empty = parse_completion(head + "INSTRUCTIONS_SPOKEN: []\nPROPOSAL: []")
    invented = parse_completion(head + 'INSTRUCTIONS_SPOKEN: [{"text": "read it ten times", "why": "x"}]\nPROPOSAL: []')

    def result(parsed: object) -> ReviewResult:
        return ReviewResult(
            available=True,
            redaction="not_applicable",
            original="clean",
            instructions_spoken=list(parsed.instructions_spoken),  # type: ignore[attr-defined]
            instructions_answered=parsed.instructions_answered,  # type: ignore[attr-defined]
        )

    assert not missing.instructions_answered
    assert "no INSTRUCTIONS_SPOKEN part" in (answer_problem(result(missing), STORY_RECALL_CARD, None) or "")
    assert answer_problem(result(empty), STORY_RECALL_CARD, None) is None
    assert "do not occur in the ORIGINAL" in (answer_problem(result(invented), STORY_RECALL_CARD, None) or "")


def test_the_prompt_asks_whether_the_instructions_are_spoken() -> None:
    """Point 6 counts paraphrase, names the story-recall example and keeps the stimulus out."""
    from senselab.text.tasks.pii_detection import redaction_review

    prompt = redaction_review._PROMPT
    assert "INSTRUCTIONS_SPOKEN: a JSON array" in prompt
    assert "paraphrase" in prompt and "have you familiarized" in prompt
    assert "stimulus" in prompt and "is never instructions" in prompt


def test_quotes_match_across_case_punctuation_and_a_transcription_spelling() -> None:
    """A quote that differs from the ORIGINAL only in case, punctuation, apostrophes or one ASR spelling places."""
    from senselab.text.tasks.pii_detection.redaction_review import quote_occurs

    original = "Well, I've been to Sandals in Grenada for a week -- last two years, maybe."
    assert quote_occurs("sandals in grenada", original)
    assert quote_occurs("Ive been", original)
    assert quote_occurs("last two years", original)
    assert quote_occurs("Grenadda", original)
    assert not quote_occurs("Sandals in Jamaica", original)
    assert not quote_occurs("two years ago", original)


def test_a_venue_the_reasoning_names_without_a_proposal_entry_is_fed_back() -> None:
    """Owner card sub-005bd146: the reviewer called "Sandals" a hotel chain and proposed nothing for it."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewProposal, ReviewResult, answer_problem

    original = "we spent a week at Sandals in Grenada"
    reasoning = 'The speaker mentions "Sandals", a hotel chain (Organization), and "Grenada", a country.'
    bare = ReviewResult(available=True, redaction="complete", original="clean", reasoning=reasoning)
    assert "names" in (answer_problem(bare, original, None) or "")
    with_entry = ReviewResult(
        available=True,
        redaction="complete",
        original="clean",
        reasoning=reasoning,
        proposal=[ReviewProposal(text="Sandals", action="redact", category="ORGANIZATION", why="a resort")],
    )
    assert answer_problem(with_entry, original, None) is None
    ordinary = ReviewResult(
        available=True,
        redaction="complete",
        original="clean",
        reasoning='The word "summer" is a season; no school or employer is named.',
    )
    assert answer_problem(ordinary, "i love the summer", None) is None


def test_the_prompt_states_the_time_expression_rule_and_the_cue_word_rule() -> None:
    """v7: weekdays and relative times are released, absolute dates are not; a cue word's definition is task content."""
    from senselab.text.tasks.pii_detection import redaction_review as r

    assert r.PROMPT_VERSION == 13
    assert "2-3 weeks ago" in r._PROMPT and "this morning" in r._PROMPT and '"Monday"' in r._PROMPT
    assert "gladiator" in r._PROMPT and "hotel" in r._PROMPT


_TRANSCRIPT: dict[str, Any] = {
    "recognisers": ["asr_crisperwhisper", "asr_qwen"],
    "words": [
        [0, "What", "variant", ["What", "Barakat"], "", [["PERSON", "gliner", "asr_qwen"]]],
        [1, "the", "agreement", ["the", "the"], "", []],
        [2, "time", "insertion", [None, "time"], "", []],
        [3, "um", "agreement", ["um", "um"], "non_lexical", []],
        [4, "Cinderella", "agreement", ["Cinderella", "Cinderella"], "task", [["PERSON", "presidio", "consensus"]]],
        [5, "danced.", "agreement", ["danced.", "danced"], "task", []],
        [
            6,
            "Maria",
            "agreement",
            ["Maria", "maria"],
            "",
            [
                ["PERSON", "gliner", "consensus"],
                ["PERSON", "presidio", "asr_crisperwhisper"],
                ["LOCATION", "rules", "consensus"],
            ],
        ],
    ],
    "findings_n": 4,
}


def test_the_request_carries_the_whole_transcript_with_its_pii_and_variation() -> None:
    """Owner, 2026-10-09 (C): every consensus word, its PII annotation and each recogniser's reading, in one text."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose

    body = _compose("What the time Maria", None, {"task": "ddk", "consensus_transcript": _TRANSCRIPT})
    assert "ORIGINAL:\nTRANSCRIPT (recognisers: A = asr_crisperwhisper, B = asr_qwen; C = the consensus):\n" in body
    line = body.split("C = the consensus):\n", 1)[1].split("\n", 1)[0]
    assert line == (
        "What{pii PERSON by gliner@B | variant A=What B=Barakat} the time{insertion A=- B=time} "
        "<nonlex>um</nonlex> <task>Cinderella{pii PERSON by presidio@C} danced.</task> "
        "Maria{pii PERSON by gliner@C, presidio@A ; LOCATION by rules@C}"
    )
    assert "What the time Maria" not in body
    assert body.count("ORIGINAL:") == 1


def test_without_a_transcript_the_original_is_the_residue() -> None:
    """A caller that gives no transcript is sent the text it passed."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose

    assert "ORIGINAL:\nWhat the time\n" in _compose("What the time", None, {"task": "ddk"})


def test_a_reading_differing_in_more_than_case_is_shown_and_quoted_when_it_needs_to_be() -> None:
    """An agreement spelled differently keeps its readings; a reading with a space or a brace is JSON-quoted."""
    from senselab.text.tasks.pii_detection.redaction_review import transcript_block

    block = transcript_block(
        {
            "recognisers": ["a", "b", "c"],
            "words": [
                [0, "okay", "agreement", ["okay", "OK", "ok ay"], "", []],
                [1, "x", "variant", ["x", "y{", "x"], "", []],
            ],
        }
    )
    assert block.splitlines()[0] == "TRANSCRIPT (recognisers: A = a, B = b, D = c; C = the consensus):"
    assert block.splitlines()[1] == 'okay{agreement A=okay B=OK D="ok ay"} x{variant A=x B="y{" D=x}'


def test_quotes_are_checked_against_the_whole_transcript_and_the_residue() -> None:
    """A quote across a task word occurs in the transcript; one that skips a filler occurs in the residue."""
    from senselab.text.tasks.pii_detection.redaction_review import quotable_texts, quote_occurs

    texts = quotable_texts("What the time Maria", {"consensus_transcript": _TRANSCRIPT})
    assert texts == ("What the time um Cinderella danced. Maria", "What the time Maria")
    assert quote_occurs("time um Cinderella", texts)
    assert quote_occurs("time Maria", texts)
    assert not quote_occurs("Barakat", texts)


def test_the_reading_records_which_inputs_it_had() -> None:
    """The task with its instruction or stimulus, the whole transcript, its PII and two readings: complete."""
    from senselab.text.tasks.pii_detection.redaction_review import review_inputs, review_inputs_complete

    full = review_inputs(
        {"task": "cinderella-story", "instructions": "Tell the story.", "consensus_transcript": _TRANSCRIPT}
    )
    assert {key: full[key] for key in ("version", "prompt_version", "task", "full_transcript", "pii_annotations")} == {
        "version": 3,
        "prompt_version": 13,
        "task": True,
        "full_transcript": True,
        "pii_annotations": True,
    }
    assert (full["asr_readings"], full["words_n"], full["pii_words_n"], full["variation_words_n"]) == (True, 7, 3, 2)
    assert review_inputs_complete(full)
    assert not review_inputs_complete(review_inputs({"task": "cinderella-story", "consensus_transcript": _TRANSCRIPT}))
    assert not review_inputs_complete(review_inputs({"task": "x", "asked_to_say": "y"}))
    one = {**_TRANSCRIPT, "recognisers": ["asr_qwen"]}
    assert not review_inputs_complete(review_inputs({"task": "x", "asked_to_say": "y", "consensus_transcript": one}))
    unannotated = {key: value for key, value in _TRANSCRIPT.items() if key != "findings_n"}
    assert not review_inputs_complete(
        review_inputs({"task": "x", "asked_to_say": "y", "consensus_transcript": unannotated})
    )
    columns = {"recognisers": _TRANSCRIPT["recognisers"], "words": [word[:4] for word in _TRANSCRIPT["words"]]}
    assert not review_inputs_complete(
        review_inputs({"task": "x", "asked_to_say": "y", "consensus_transcript": {**columns, "findings_n": 0}})
    )
    assert not review_inputs_complete(None)
    assert not review_inputs_complete({**full, "version": 1})


def test_v12_the_request_carries_the_task_nature_name_speech_type_and_instructions() -> None:
    """The family's nature, the declared task name, the speech type and the instructions are all sent."""
    from senselab.text.tasks.pii_detection.redaction_review import _compose, task_nature, task_nature_description

    assert task_nature("animal-fluency") == "item_generation"
    assert task_nature("story-recall-v2") == "read_or_recall"
    assert task_nature("respiration-and-cough-v2-breath") == "non_lexical"
    assert task_nature("open-response-questions") == "open_response"
    assert task_nature("unheard-of") == "" and task_nature(None) == ""
    body = _compose(
        "dog cat",
        None,
        {
            "task": "animal-fluency",
            "task_name": "Animal fluency",
            "speech_type": "elicited",
            "instructions": "Name as many animals as you can.",
        },
    )
    assert f"NATURE OF THIS TASK: {task_nature_description('animal-fluency')}" in body
    assert "TASK NAME THE RECORDING DECLARES: Animal fluency" in body
    assert "SPEECH TYPE: elicited" in body
    assert "INSTRUCTIONS GIVEN TO THE PARTICIPANT: Name as many animals as you can." in body


def test_v12_the_prompt_asks_about_task_content_off_task_speech_and_other_speakers() -> None:
    """The prompt asks the three task questions and names the five new parts."""
    from senselab.text.tasks.pii_detection import redaction_review as r

    for heading in (
        "OTHER_SPEAKER_ROLE:",
        "OTHER_SPEAKER_QUOTES:",
        "OFF_TASK_SPEECH:",
        "OFF_TASK_QUOTES:",
        "TASK_CONTENT_QUOTES:",
    ):
        assert heading in r._PROMPT
    assert "NATURE OF THIS TASK" in r._PROMPT and "non-lexical task" in r._PROMPT
    assert r.redaction_policy_text() in r._PROMPT


_V12_ANSWER = (
    "REASONING: plain.\nREDACTION: not_applicable\nORIGINAL: clean\nSPEAKERS: more_than_one\n"
    'OTHER_SPEAKERS: [{"text": "go ahead", "expected": true, "why": "prompt"}]\n'
    "OTHER_SPEAKER_ROLE: assistant\n"
    'OTHER_SPEAKER_QUOTES: ["go ahead"]\n'
    "OFF_TASK_SPEECH: some\n"
    'OFF_TASK_QUOTES: ["is that enough"]\n'
    'TASK_CONTENT_QUOTES: ["lion", {"text": "zebra"}]\n'
    "CONDITIONS: []\nINSTRUCTIONS_SPOKEN: []\nPROPOSAL: []\n"
)


def test_v12_the_five_task_reading_fields_are_parsed_and_recorded() -> None:
    """Each new part parses into its field and reaches the payload under its exact key."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewResult, parse_completion, review_payload

    parsed = parse_completion(_V12_ANSWER)
    assert parsed.speakers == "more_than_one" and len(parsed.other_speakers) == 1
    assert parsed.other_speaker == "assistant" and parsed.other_speaker_quotes == ["go ahead"]
    assert parsed.off_task_speech == "some" and parsed.off_task_quotes == ["is that enough"]
    assert parsed.task_content_quotes == ["lion", "zebra"]
    assert parsed.proposal == [] and parsed.conditions_answered and parsed.instructions_answered
    assert parsed.reasoning == "plain."
    payload = review_payload(
        ReviewResult(
            available=True,
            off_task_speech=parsed.off_task_speech,
            off_task_quotes=parsed.off_task_quotes,
            other_speaker=parsed.other_speaker,
            other_speaker_quotes=parsed.other_speaker_quotes,
            task_content_quotes=parsed.task_content_quotes,
        )
    )
    assert (payload["off_task_speech"], payload["other_speaker"]) == ("some", "assistant")
    assert payload["off_task_quotes"] == ["is that enough"] and payload["other_speaker_quotes"] == ["go ahead"]
    assert payload["task_content_quotes"] == ["lion", "zebra"]


def test_v12_malformed_task_reading_parts_read_as_unanswered() -> None:
    """An unknown word reads as None, a missing or broken array as []."""
    from senselab.text.tasks.pii_detection.redaction_review import parse_completion

    broken = (
        "REASONING: x\nSPEAKERS: one\nOTHER_SPEAKER_ROLE: narrator\nOTHER_SPEAKER_QUOTES: [not json\n"
        'OFF_TASK_SPEECH: lots\nOFF_TASK_QUOTES: {"a": 1}\nTASK_CONTENT_QUOTES: [1, "", null]\nPROPOSAL: []\n'
    )
    parsed = parse_completion(broken)
    assert parsed.speakers == "one"
    assert parsed.other_speaker is None and parsed.off_task_speech is None
    assert parsed.other_speaker_quotes == [] and parsed.off_task_quotes == [] and parsed.task_content_quotes == []
    silent = parse_completion("REASONING: x\nPROPOSAL: []\n")
    assert silent.other_speaker is None and silent.off_task_speech is None and silent.off_task_quotes == []


def test_v13_an_item_set_task_is_asked_about_phrases_instead_of_items() -> None:
    """Owner, 2026-10-10: an item-set task's point 8 asks about phrases, not off-task speech; the rest is shared."""
    from senselab.text.tasks.pii_detection import redaction_review as r

    items = r.review_prompt(True)
    assert r.review_prompt(False) == r._PROMPT
    assert "PHRASES_INSTEAD_OF_ITEMS:" in items and "PHRASE_QUOTES:" in items
    assert "OFF_TASK_SPEECH:" not in items and "OFF_TASK_QUOTES:" not in items
    assert "let me think" in items and "my dog\'s name is" in items
    assert items.replace(r._POINT_8_PHRASES, "").replace(r._PHRASE_PARTS, "") == r._PROMPT.replace(
        r._POINT_8_OFF_TASK, ""
    ).replace(r._OFF_TASK_PARTS, "")


def test_v13_the_phrase_reading_is_parsed_checked_and_recorded() -> None:
    """The phrases part parses into its field, needs a quote when not none, and reaches the payload."""
    from senselab.text.tasks.pii_detection.redaction_review import (
        ReviewResult,
        answer_problem,
        parse_completion,
        review_payload,
    )

    answer = _V12_ANSWER.replace("OFF_TASK_SPEECH: some\n", "PHRASES_INSTEAD_OF_ITEMS: predominant\n").replace(
        'OFF_TASK_QUOTES: ["is that enough"]', 'PHRASE_QUOTES: ["my dog is old"]'
    )
    parsed = parse_completion(answer)
    assert parsed.phrases_instead_of_items == "predominant" and parsed.phrase_quotes == ["my dog is old"]
    assert parsed.off_task_speech is None and parsed.task_content_quotes == ["lion", "zebra"]
    assert parse_completion(answer.replace("predominant", "lots")).phrases_instead_of_items is None
    unquoted = ReviewResult(available=True, phrases_instead_of_items="occasional")
    assert "PHRASE_QUOTES" in str(answer_problem(unquoted, "lion zebra", None))
    unheard = ReviewResult(available=True, phrases_instead_of_items="predominant", phrase_quotes=["purple"])
    assert "do not occur" in str(answer_problem(unheard, "lion zebra", None))
    payload = review_payload(ReviewResult(available=True, phrases_instead_of_items="none"))
    assert payload["phrases_instead_of_items"] == "none" and payload["phrase_quotes"] == []


def test_v12_task_reading_quotes_are_checked_and_a_judgment_needs_one() -> None:
    """A task-reading quote not in the ORIGINAL, or an off-task judgment without one, is fed back."""
    from senselab.text.tasks.pii_detection.redaction_review import ReviewResult, answer_problem

    base: dict[str, Any] = {"available": True, "off_task_speech": "none", "other_speaker": "none"}
    assert answer_problem(ReviewResult(**base), "lion zebra", None) is None
    unquoted = ReviewResult(**{**base, "off_task_speech": "extensive"})
    assert "OFF_TASK_QUOTES" in str(answer_problem(unquoted, "lion zebra", None))
    unheard = ReviewResult(**{**base, "other_speaker": "background", "other_speaker_quotes": ["purple"]})
    assert "do not occur" in str(answer_problem(unheard, "lion zebra", None))
    unheard_task = ReviewResult(**{**base, "task_content_quotes": ["purple"]})
    assert "do not occur" in str(answer_problem(unheard_task, "lion zebra", None))


def test_v12_review_inputs_v3_require_the_instructions() -> None:
    """Version 3 records whether instructions were given, and a reading without them is incomplete."""
    from senselab.text.tasks.pii_detection.redaction_review import review_inputs, review_inputs_complete

    stimulus_only = review_inputs(
        {"task": "cinderella-story", "asked_to_say": "y", "consensus_transcript": _TRANSCRIPT}
    )
    assert stimulus_only["version"] == 3 and stimulus_only["task"] and not stimulus_only["instructions"]
    assert not review_inputs_complete(stimulus_only)
    full = review_inputs({"task": "cinderella-story", "instructions": "Tell it.", "consensus_transcript": _TRANSCRIPT})
    assert full["instructions"] and review_inputs_complete(full)
    assert not review_inputs_complete({**full, "version": 2})
    assert not review_inputs_complete({key: value for key, value in full.items() if key != "instructions"})
