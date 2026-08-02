"""Corroboration enters transcript fusion as a weight, never as a deletion.

The ensemble used to receive `purged_spans` and drop those words before voting. Nothing
downstream could tell they had existed — not `transcript.json`, not `final/`. It also made a
word's survival depend on whether an intervention had been *admitted within budget*, so budget
accounting decided what reached the deliverable.

The replacement is one measured quantity per word. These tests pin where it must bite, where it
must not, and that "unmeasured" is not "zero".
"""

from typing import Any

import pytest

from senselab.audio.tasks.speech_to_text_ensemble import MIN_CORROBORATION, fuse_word_streams


def _w(text: str, start: float, end: float, **extra: Any) -> dict[str, Any]:  # noqa: ANN401
    return {"text": text, "start": start, "end": end, **extra}


def test_a_wholly_uncorroborated_word_keeps_its_text_and_its_sources() -> None:
    """Erasing it is how a quiet or overlapped speaker disappears with no trace to appeal to.

    The word must stay in the output carrying the number that condemned it, so a reader can
    disagree with the threshold without re-running a model.
    """
    words = fuse_word_streams(
        {"m1": [_w("whisper", 2.0, 2.3, corroboration=0.0)], "m2": [_w("hello", 0.0, 0.4)]},
    )
    texts = [w["text"] for w in words]
    assert "whisper" in texts
    stray = next(w for w in words if w["text"] == "whisper")
    assert stray["sources"] == ["m1"]
    assert stray["corroboration"] == pytest.approx(MIN_CORROBORATION)


def test_corroboration_reaches_a_single_source_word() -> None:
    """``share`` is identically 1.0 for a one-member slot.

    A factor entering only the vote weight is therefore a provable no-op on exactly the case this
    mechanism exists for — a word one model produced where nothing else did. It has to reach
    ``coverage``.
    """
    streams = {"m1": [_w("stray", 2.0, 2.3)], "m2": [_w("hello", 0.0, 0.4)]}
    plain = next(w for w in fuse_word_streams(streams) if w["text"] == "stray")

    doubted_streams = {"m1": [_w("stray", 2.0, 2.3, corroboration=0.1)], "m2": [_w("hello", 0.0, 0.4)]}
    doubted = next(w for w in fuse_word_streams(doubted_streams) if w["text"] == "stray")

    assert doubted["confidence"] < plain["confidence"]
    assert doubted["coverage"] < plain["coverage"]
    assert doubted["confidence"] > 0.0


def test_member_confidence_is_not_contaminated_by_corroboration() -> None:
    """``member_conf`` reports what the models said about themselves.

    Folding corroboration into it would make one number mean two measurements — the defect being
    removed — and would corrupt the field a calibration profile is fitted against.
    """
    streams = {"m1": [_w("stray", 2.0, 2.3, confidence=0.8, corroboration=0.1)]}
    word = fuse_word_streams(streams)[0]
    # confidence = share(1.0) x member_conf(0.8) x coverage(1.0 x 0.1)
    assert word["confidence"] == pytest.approx(0.8 * 0.1, abs=1e-4)


def test_a_uniform_doubt_does_not_move_the_winner_or_its_share() -> None:
    """Relative preference among candidates must not shift because the whole region is doubtful.

    Attenuation should say "trust this less", not "prefer a different word here".
    """
    plain = fuse_word_streams(
        {
            "m1": [_w("cat", 0.0, 0.4)],
            "m2": [_w("cat", 0.02, 0.42)],
            "m3": [_w("cot", 0.01, 0.41)],
        }
    )[0]
    doubted = fuse_word_streams(
        {
            "m1": [_w("cat", 0.0, 0.4, corroboration=0.3)],
            "m2": [_w("cat", 0.02, 0.42, corroboration=0.3)],
            "m3": [_w("cot", 0.01, 0.41, corroboration=0.3)],
        }
    )[0]
    assert doubted["text"] == plain["text"]
    assert doubted["confidence"] < plain["confidence"]


def test_an_attenuated_loser_is_still_recorded_as_an_alternate() -> None:
    """Alternates are gated on the *uncorroborated* tally.

    Otherwise attenuation quietly decides what is recorded: the loser's share falls below
    ``alternate_min_share`` and the winner's rises above ``winner_margin``, the alternates block is
    suppressed, and the doubted reading vanishes as completely as purging removed it. Attenuation
    may decide who wins; it may never decide who is written down.
    """
    words = fuse_word_streams(
        {
            "m1": [_w("cat", 0.0, 0.4)],
            "m2": [_w("cot", 0.02, 0.42, corroboration=MIN_CORROBORATION)],
        }
    )
    assert len(words) == 1
    alternates = words[0]["alternates"]
    assert [a["text"] for a in alternates] == ["cot"]
    assert alternates[0]["share_uncorroborated"] == pytest.approx(0.5)
    assert alternates[0]["share"] < alternates[0]["share_uncorroborated"]


def test_unmeasured_corroboration_applies_no_discount() -> None:
    """Absent is not zero.

    On a run with no informative presence voter every word is unmeasured, and the mechanism must be
    inert rather than universally condemning — a missing model must not look like a wrong one.
    """
    measured_full = fuse_word_streams({"m1": [_w("hi", 0.0, 0.4, corroboration=1.0)]})[0]
    unmeasured = fuse_word_streams({"m1": [_w("hi", 0.0, 0.4)]})[0]
    explicit_none = fuse_word_streams({"m1": [_w("hi", 0.0, 0.4, corroboration=None)]})[0]

    assert unmeasured["confidence"] == pytest.approx(measured_full["confidence"])
    assert explicit_none["confidence"] == pytest.approx(measured_full["confidence"])
    assert unmeasured["corroboration"] is None, "unmeasured must stay distinguishable from 1.0"
    assert explicit_none["corroboration"] is None


def test_a_zero_floor_is_refused() -> None:
    """A configurable zero floor is purging with extra steps.

    At zero the vote weight and the coverage contribution both vanish, which is deletion reached
    through configuration rather than through code.
    """
    with pytest.raises(ValueError, match="min_corroboration"):
        fuse_word_streams({"m1": [_w("hi", 0.0, 0.4)]}, min_corroboration=0.0)


def test_every_member_of_a_slot_reports_its_weight_including_the_losers() -> None:
    """The losing evidence has to stay visible somewhere in the record.

    ``sources`` lists only the winner's models, so without this the doubted reading's measurement
    would exist nowhere in the artifact.
    """
    word = fuse_word_streams(
        {
            "m1": [_w("cat", 0.0, 0.4, corroboration=0.9)],
            "m2": [_w("cot", 0.02, 0.42, corroboration=0.2)],
            "m3": [_w("cat", 0.01, 0.41)],
        }
    )[0]
    assert word["member_corroboration"] == {"m1": pytest.approx(0.9), "m2": pytest.approx(0.2), "m3": None}


def test_the_low_presence_flag_is_gone() -> None:
    """It thresholded the same quantity this contract now carries continuously, at a bare 0.5.

    Nothing read the flag. Keeping it alongside the number would give two answers to one question,
    one of them un-re-decidable.
    """
    word = fuse_word_streams({"m1": [_w("hi", 0.0, 0.4, corroboration=0.01)]})[0]
    assert "low_speech_presence" not in word["flags"]
    assert word["corroboration"] == pytest.approx(MIN_CORROBORATION)
