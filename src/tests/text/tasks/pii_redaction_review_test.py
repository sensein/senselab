"""The review worker's lifetime: one load per process, and every failure a recorded absence.

The subprocess is real and the line protocol is the shipped one; only the *payload* is fake. The
worker script is replaced with a few lines of pure Python that speak the same protocol and load no
weights, so spawn counts, restarts, timeouts and the refusal record are all observable on a laptop.
A stub of ``review_redacted_text`` — which is what ``redact_test.py`` uses — could not see any of
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
    review_redacted_text,
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
                "completion": "REASONING: nothing identifying remains.\nFINDINGS: []",
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
        results = [review_redacted_text(f"text {n}", model_id="stub/model") for n in range(3)]
        assert all(result.available for result in results)
        assert len(harness.loads) == 1, f"one load expected, the worker was spawned {len(harness.loads)} times"

    def test_only_the_round_that_loaded_says_it_loaded(self, harness: Harness) -> None:
        """``load_s`` is how a store separates the amortised rounds from the one that paid."""
        first = review_redacted_text("one", model_id="stub/model")
        second = review_redacted_text("two", model_id="stub/model")
        assert first.load_s > 0.0, "the first review paid the load and must say so"
        assert second.load_s == 0.0, "a reused worker loaded nothing"
        assert first.elapsed_s >= first.load_s, "the round's clock has to contain the load it paid"
        assert second.elapsed_s < first.elapsed_s, "the amortised round is the cheaper one"

    def test_a_different_commit_is_a_different_worker(self, harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        """Weights already resident are the wrong weights when the resolved commit changes."""
        review_redacted_text("one", model_id="stub/model")
        monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "c" * 40)
        review_redacted_text("two", model_id="stub/model")
        assert [load["revision"] for load in harness.loads] == [SHA, "c" * 40]

    def test_shutdown_hands_the_weights_back(self, harness: Harness) -> None:
        """A caller that wants the memory before the process ends can have it."""
        review_redacted_text("one", model_id="stub/model")
        shutdown_review_worker()
        review_redacted_text("two", model_id="stub/model")
        assert len(harness.loads) == 2

    def test_the_answer_carries_what_it_generated(self, harness: Harness) -> None:
        """Output tokens are the thing generation cost tracks, so the store carries them."""
        result = review_redacted_text("one", model_id="stub/model")
        assert result.output_tokens == 11
        assert result.revision == SHA

    def test_the_answer_carries_what_it_holds_on_the_card(self, harness: Harness) -> None:
        """A worker kept alive occupies a GPU; how much decides whether anything else fits beside it."""
        result = review_redacted_text("one", model_id="stub/model")
        assert result.peak_reserved_mib == 71000
        assert result.resident_mib == 23000

    def test_a_worker_that_did_not_answer_claims_no_memory(self, harness: Harness) -> None:
        """Zero is the honest reading when nothing ran; a stale number would size the next run."""
        harness.directives("fail_load")
        result = review_redacted_text("one", model_id="stub/model")
        assert result.peak_reserved_mib == 0 and result.resident_mib == 0


class TestAWorkerThatCannotStartCostsOneAttempt:
    """A corpus of 15,000 recordings on a host with no reachable GPU must not pay 15,000 loads."""

    def test_the_first_failure_is_recorded_and_not_retried(self, harness: Harness) -> None:
        """The second call reports the same absence without spawning anything."""
        harness.directives("fail_load")
        first = review_redacted_text("one", model_id="stub/model")
        second = review_redacted_text("two", model_id="stub/model")
        assert not first.available and not second.available
        assert first.failure is not None and "no device found" in first.failure
        assert second.failure == first.failure
        assert len(harness.loads) == 1, f"a refused load was retried: {len(harness.loads)} spawns"

    def test_a_refusal_is_an_absence_and_not_a_raise(self, harness: Harness) -> None:
        """The absent path's contract: a recorded absence, a named failure, no findings."""
        harness.directives("fail_load")
        result = review_redacted_text("one", model_id="stub/model")
        assert result.findings == []
        assert result.model_id == "stub/model"
        assert result.elapsed_s > 0.0

    def test_shutdown_clears_the_refusal(self, harness: Harness) -> None:
        """A deliberate retry is the caller's to ask for; a silent one is not."""
        harness.directives("fail_load")
        review_redacted_text("one", model_id="stub/model")
        shutdown_review_worker()
        assert review_redacted_text("two", model_id="stub/model").available
        assert len(harness.loads) == 2

    def test_releasing_the_weights_between_recordings_does_not_clear_it(self, harness: Harness) -> None:
        """The caller that hands the memory back after every recording is not asking for a retry.

        This is the interaction, not either half of it: a node that releases the worker when its
        check ends would, if release also forgot the refusal, re-attempt a 23 GB load once per
        recording on a host that has no GPU — which is the exact stall the record exists to stop.
        """
        harness.directives("fail_load")
        for _ in range(3):
            review_redacted_text("a recording", model_id="stub/model")
            shutdown_review_worker(forget_failure=False)
        assert len(harness.loads) == 1, f"the refusal was forgotten: {len(harness.loads)} load attempts"


class TestAFailedReviewDoesNotPoisonTheProcess:
    """A worker that died mid-generation is gone; the *next* recording must still get a verdict."""

    def test_a_worker_that_raised_is_replaced(self, harness: Harness) -> None:
        """An out-of-memory on one transcript costs one reload, not every later review."""
        harness.directives("raise")
        failed = review_redacted_text("one", model_id="stub/model")
        recovered = review_redacted_text("two", model_id="stub/model")
        assert not failed.available and failed.failure is not None
        assert "CUDA out of memory" in failed.failure
        assert recovered.available, "a dead worker must be replaced, not reported dead forever"
        assert len(harness.loads) == 2

    def test_a_worker_that_vanished_is_replaced(self, harness: Harness) -> None:
        """A killed worker answers nothing at all; that is an absence, then a fresh load."""
        harness.directives("exit")
        failed = review_redacted_text("one", model_id="stub/model")
        assert not failed.available
        assert review_redacted_text("two", model_id="stub/model").available
        assert len(harness.loads) == 2

    def test_a_worker_that_never_answers_is_an_absence_not_a_hang(self, harness: Harness) -> None:
        """``timeout_s`` still bounds a review; the worker is killed rather than waited on."""
        harness.directives("hang")
        result = review_redacted_text("one", model_id="stub/model", timeout_s=2)
        assert not result.available
        assert result.failure is not None and "did not answer in 2s" in result.failure
        assert review_redacted_text("two", model_id="stub/model").available


class TestTheProtocolIsNotConfusedByTheLoaderSOwnOutput:
    """transformers writes progress to both streams; a reply is what carries the marker."""

    def test_unmarked_lines_on_either_stream_are_not_replies(self, harness: Harness) -> None:
        """The fake worker writes to stderr during load, and to stdout a line shaped like a reply.

        Prose on stdout would be rejected by the JSON parse whether the marker were checked or not,
        so it cannot show that the marker does any work. A well-formed object that is not a reply
        can: without the marker it is read as one, and the review returns the wrong completion.
        """
        result = review_redacted_text("one", model_id="stub/model")
        assert result.available
        assert "LEAKED" not in result.reasoning and "LEAKED" not in result.raw
        assert result.reasoning == "nothing identifying remains."
        assert result.output_tokens == 11, "a line the loader wrote was read as the model's answer"
        assert result.findings == []

    def test_a_failure_message_carries_what_the_worker_was_last_doing(self, harness: Harness) -> None:
        """A load failure with no context is a bug report nobody can act on."""
        harness.directives("hang")
        result = review_redacted_text("one", model_id="stub/model", timeout_s=2)
        assert result.failure is not None
        assert "chatter on stdout that is not a reply" in result.failure or "loading shards" in result.failure
