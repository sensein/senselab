"""The optional local-LLM PII engine.

Never contacted in these tests: an engine that reaches the network during a unit
test is an engine that will reach the network during a compliance scan. The one
test that exercises the request path points at a closed port on purpose.
"""

import pytest

from senselab.text.tasks.pii_detection import api, local_llm
from senselab.text.tasks.pii_detection.subprocess_backend import _KNOWN_DETECTORS


def test_llm_is_off_by_default() -> None:
    """The default detector set must not include a detector that needs a server.

    Default-on would mean a scan silently depends on whether a local server happens to
    be listening — the same corpus would score differently on two machines.
    """
    assert "llm" not in api.default_detectors()


def test_llm_is_a_known_detector_so_it_counts_in_the_agreement_denominator() -> None:
    """Off by default is not the same as unknown.

    When a caller does turn it on it has to be accepted by name and counted among the
    detectors that ran, or cross-detector agreement would be computed against the wrong
    denominator.
    """
    assert "llm" in _KNOWN_DETECTORS


def test_base_url_must_be_loopback() -> None:
    """A remote endpoint is refused at construction.

    Transcript text never leaves the machine. A remote base_url would send clinical
    speech to a third party, which is the one thing this module promises not to do.
    """
    with pytest.raises(ValueError, match="localhost|loopback|127.0.0.1"):
        local_llm.LocalLlmConfig(base_url="https://api.example.com")


def test_loopback_forms_are_accepted() -> None:
    """All three spellings of "this machine" work, and the string is stored verbatim.

    Rewriting it would make the value the caller reads back differ from the one they set.
    """
    for url in ("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434"):
        assert local_llm.LocalLlmConfig(base_url=url).base_url == url


def test_a_hostname_resolving_to_loopback_is_still_rejected() -> None:
    """Only literals count, never a name that merely resolves to loopback.

    DNS is not stable between the check and the request, so a name pointing at 127.0.0.1
    today is not a guarantee about where the transcript actually goes.
    """
    with pytest.raises(ValueError, match="localhost|loopback|127.0.0.1"):
        local_llm.LocalLlmConfig(base_url="http://localhost.evil.example.com:11434")


def test_an_unreachable_server_is_a_recorded_failure_not_a_clean_pass() -> None:
    """A detector that could not run must say so.

    If the LLM cannot be reached the report must record that it did not run. Returning
    no spans would read as "the LLM found no PII".
    """
    result = local_llm.scan_or_fail("some text", local_llm.LocalLlmConfig(base_url="http://127.0.0.1:1"))
    assert result.spans == []
    assert result.failure is not None
    assert "llm" in result.failure.lower() or "connect" in result.failure.lower()


def test_a_fenced_or_chatty_reply_still_parses() -> None:
    """Prose and code fences around the array are tolerated.

    Models wrap the array often enough that treating either as unparsable would fail
    scans that actually succeeded.
    """
    payload = 'Sure! Here you go:\n```json\n[{"text": "Jane Doe", "category": "person", "score": 0.9}]\n```'
    spans = local_llm._parse_response(payload, "llama3.1:8b")
    assert len(spans) == 1
    assert spans[0]["text"] == "Jane Doe"
    assert spans[0]["category"] == "PERSON"
    assert spans[0]["source"] == "llm/llama3.1:8b"


def test_an_out_of_range_score_is_clamped_not_trusted() -> None:
    """A score outside [0, 1] is clamped rather than taken at face value.

    ``_compute_detection_confidence`` reads the score as a probability, so a model
    answering 95 when asked for 0-1 would otherwise dominate every aggregate.
    """
    spans = local_llm._parse_response('[{"text": "x", "category": "OTHER", "score": 95}]', "m")
    assert spans[0]["score"] == 1.0


def test_a_malformed_element_does_not_discard_the_whole_scan() -> None:
    """Bad elements are skipped, not fatal.

    One malformed entry in an otherwise good array is not a reason to throw away the
    findings that did parse.
    """
    spans = local_llm._parse_response('["not an object", {"text": "Jane", "category": "PERSON"}, {"no": "text"}]', "m")
    assert len(spans) == 1
    assert spans[0]["text"] == "Jane"
    assert spans[0]["score"] is None


def test_a_reply_with_no_array_at_all_is_a_failure() -> None:
    """Prose with no array is a failure, not an empty result.

    ``"[]"`` means the model scanned and found nothing; prose means it did not answer
    the question, and the two must not collapse into the same report.
    """
    with pytest.raises(ValueError, match="no JSON array"):
        local_llm._parse_response("I cannot help with that.", "m")
