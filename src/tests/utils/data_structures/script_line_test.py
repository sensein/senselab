"""Tests the ScriptLine data structure.

Focus is on the ``validate_text_and_speaker`` rule: empty ``text`` is a
meaningful "model called, no speech detected" signal and must be
distinguishable from ``text=None`` (model never called / field absent).
Earlier behavior rejected ``ScriptLine(text="")`` outright, which caused
ASR backends honest enough to emit empty transcripts on silent /
unintelligible audio (notably IBM Granite-Speech on heavily-enhanced
short clips) to fail validation while less-honest models silently
hallucinated filler and passed.
"""

import pytest
from pydantic import ValidationError

from senselab.utils.data_structures import ScriptLine

# ── Empty / None / whitespace text handling ────────────────────────


def test_empty_text_without_speaker_is_accepted() -> None:
    """``ScriptLine(text="")`` is a valid "model returned no transcript" signal.

    Regression test for the Granite-Speech failure mode observed on
    enhanced short clips, where the model honestly emits ``""`` instead
    of hallucinating filler. The earlier validator rejected this with
    ``"At least text or speaker must be provided."`` even though the
    intent was clearly "the call happened and produced nothing".
    """
    line = ScriptLine(text="")
    assert line.text == ""
    assert line.speaker is None
    assert line.get_text() == ""


def test_whitespace_only_text_is_accepted_and_normalized_to_empty() -> None:
    """Whitespace-only ``text`` survives validation and is stripped to ``""``.

    Without this, the system would treat ``text=" "`` and ``text=""`` as
    behaving differently — the former passing the truthy check then
    being normalized, the latter being rejected. With the fix, both are
    accepted and converge on ``text=""``.
    """
    line = ScriptLine(text="   ")
    assert line.text == ""


def test_text_none_without_speaker_still_rejected() -> None:
    """``ScriptLine()`` with both fields absent / None must still raise.

    ``None`` means "the field was never set" — the validator's whole job
    is to refuse rows with no signal at all.
    """
    with pytest.raises(ValidationError):
        ScriptLine()
    with pytest.raises(ValidationError):
        ScriptLine(text=None)
    with pytest.raises(ValidationError):
        ScriptLine(text=None, speaker=None)


def test_only_speaker_provided_is_accepted() -> None:
    """Speaker without text is a valid diarization line."""
    line = ScriptLine(speaker="spk1")
    assert line.speaker == "spk1"
    assert line.text is None


def test_empty_text_with_speaker_is_accepted() -> None:
    """Diarization line with explicit empty transcript is valid.

    This is the "speaker turn detected, but the ASR returned nothing for
    that turn" case — surprisingly common at silence boundaries.
    """
    line = ScriptLine(text="", speaker="spk1")
    assert line.text == ""
    assert line.speaker == "spk1"


def test_normal_text_unchanged() -> None:
    """Existing behavior preserved: real transcripts still construct cleanly."""
    line = ScriptLine(text="hello world")
    assert line.text == "hello world"


def test_text_stripped_of_surrounding_whitespace() -> None:
    """Field validator still strips whitespace from non-empty input."""
    line = ScriptLine(text="  hello  ")
    assert line.text == "hello"


# ── from_dict shape coverage ───────────────────────────────────────


def test_from_dict_accepts_empty_text_entry() -> None:
    """``ScriptLine.from_dict({"text": ""})`` no longer raises.

    Matters because some ASR backends serialize their output through a
    dict-shaped IPC payload (e.g. subprocess workers) before reaching
    ``ScriptLine``; an empty transcript shouldn't be lost or remapped
    along the way.
    """
    line = ScriptLine.from_dict({"text": ""})
    assert line.text == ""


def test_from_dict_rejects_dict_with_no_text_or_speaker() -> None:
    """A bare ``from_dict({})`` still raises — nothing to construct from."""
    with pytest.raises(ValidationError):
        ScriptLine.from_dict({})


def test_from_dict_preserves_empty_text_in_nested_chunks() -> None:
    """Nested chunks with ``text=""`` survive ``from_dict``'s child filter.

    The internal ``_is_meaningful`` filter used to drop any chunk whose
    text was falsy, including the empty string. After the validator
    change, empty text is a meaningful "model produced no transcript for
    this subsegment" signal, so the filter must keep it — only chunks
    where BOTH text and speaker are absent / ``None`` should be dropped.

    Without this guarantee the empty-text fix is silently undone for
    payloads that travel through ``from_dict`` (subprocess IPC, saved
    JSON), which is the very path that motivated the validator change.
    """
    line = ScriptLine.from_dict({"text": "parent", "chunks": [{"text": ""}]})
    assert line.chunks is not None
    assert len(line.chunks) == 1
    assert line.chunks[0].text == ""


def test_from_dict_still_drops_chunks_missing_both_text_and_speaker() -> None:
    """Children with neither text nor speaker still get filtered out.

    The original purpose of the ``_is_meaningful`` filter (avoid tripping
    the root validator on wholly-empty placeholder children, e.g. from
    MMS-style aligners) is preserved — only its overly-aggressive
    truthiness check is loosened.
    """
    line = ScriptLine.from_dict({"text": "parent", "chunks": [{"text": ""}, {}, {"start": 0.0, "end": 1.0}]})
    # The empty-text chunk survives; the two text-and-speaker-absent
    # chunks are dropped before they reach the validator.
    assert line.chunks is not None
    assert len(line.chunks) == 1
    assert line.chunks[0].text == ""


# ── Score / timestamp interactions (unchanged behavior, asserted here
#     so future validator tweaks don't quietly regress them) ─────────


def test_empty_text_with_timestamps_is_accepted() -> None:
    """A timestamped empty transcript is valid.

    The segment exists, it just didn't transcribe to anything.
    """
    line = ScriptLine(text="", start=0.0, end=1.5)
    assert line.text == ""
    assert line.start == 0.0
    assert line.end == 1.5


def test_negative_timestamp_still_rejected_independent_of_text() -> None:
    """Timestamp validator is independent of the text/speaker rule."""
    with pytest.raises(ValidationError):
        ScriptLine(text="", start=-1.0)
