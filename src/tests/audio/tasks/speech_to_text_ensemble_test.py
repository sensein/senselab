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


# ── D-27: a word's confidence has two parts ───────────────────────────────────
#
# Measured on `english_conversation_higgs_audio_v2_20260805-034348`: 61 of 62 fused words carried
# `confidence` exactly 1.0, because `member_conf` falls back to 1.0 when a model reports none and
# all three default recognizers report `avg_logprob`/`no_speech_prob`/`token_entropy` as None. The
# product measured agreement and coverage and called it confidence. Meanwhile two of those
# recognizers produced word-identical transcripts whose per-bucket texts still differed, because
# they timed the same words differently — a doubt no field on the word could express.


def test_absent_member_confidence_is_recorded_as_absent_not_as_one() -> None:
    """``member_confidence`` is ``None`` when nobody reported one; 1.0 is a claim nobody made.

    The absent-vs-zero rule in its other direction. A reader cannot otherwise tell a word three
    confident models agreed on from a word three silent models agreed on.
    """
    out = fuse_word_streams(
        {"a": [_w("hello", 0.0, 0.5)], "b": [_w("hello", 0.0, 0.5)]},
        min_corroboration=MIN_CORROBORATION,
    )
    assert out[0]["member_confidence"] is None, out[0]
    assert out[0]["existence_confidence"] == pytest.approx(1.0), "two models agreeing is full agreement"


def test_reported_member_confidence_still_reaches_the_word() -> None:
    """When a model does report one, it is used — the fallback was the defect, not the term."""
    out = fuse_word_streams(
        {"a": [_w("hello", 0.0, 0.5, confidence=0.5)], "b": [_w("hello", 0.0, 0.5, confidence=0.5)]},
        min_corroboration=MIN_CORROBORATION,
    )
    assert out[0]["member_confidence"] == pytest.approx(0.5)
    assert out[0]["existence_confidence"] < 1.0, "a model's own doubt is doubt about the word existing"


def test_a_word_all_models_time_alike_is_temporally_confident() -> None:
    """Agreement on *when* is a separate measurement from agreement on *what*."""
    out = fuse_word_streams(
        {"a": [_w("hello", 1.00, 1.40)], "b": [_w("hello", 1.01, 1.41)]},
        min_corroboration=MIN_CORROBORATION,
    )
    assert out[0]["temporal_confidence"] > 0.9, out[0]


def test_a_word_models_place_differently_is_temporally_doubtful() -> None:
    """The failure the old scheme could not express.

    Both models say the same word; they disagree about where it is by more than its own length.
    Under a time-bucketed WER this read as textual disagreement in whichever buckets they fell in;
    here it is what it is — a word whose text is certain and whose position is not.
    """
    out = fuse_word_streams(
        {"a": [_w("hello", 1.00, 1.40)], "b": [_w("hello", 1.45, 1.85)]},
        min_corroboration=MIN_CORROBORATION,
        slot_mid_tol_s=0.6,
    )
    assert len(out) == 1, f"the two readings must land in one slot for this to be a timing question: {out}"
    assert out[0]["existence_confidence"] == pytest.approx(1.0), "both models said the word"
    assert out[0]["temporal_confidence"] < 0.5, out[0]


def test_a_lone_source_has_no_temporal_agreement_to_measure() -> None:
    """One timing source cannot corroborate itself, so the quantity is absent rather than perfect."""
    out = fuse_word_streams({"a": [_w("hello", 0.0, 0.5)]}, min_corroboration=MIN_CORROBORATION)
    assert out[0]["temporal_confidence"] is None, out[0]


def test_two_transcripts_sharing_an_aligner_are_one_timing_source() -> None:
    """Canary was timed by Qwen's aligner, so their onsets agree by construction.

    Counting them as two agreeing opinions about *when* would manufacture temporal confidence out
    of a shared dependency. Provenance is the only thing that can see it — an aligner is not a
    ``Source``, so closure intersection cannot.

    The labels differ and the aligner does not, which is the whole trap: ``TimestampSource`` is
    ``native | bundled_aligner | external_aligner`` — a *kind*. Qwen3-ASR's timings come from
    ``Qwen/Qwen3-ForcedAligner-0.6B`` shipped with it (``bundled_aligner``) and Canary's come from
    the workflow aligning it with **the same model** (``external_aligner``). Grouping on the kind
    reads one aligner as two independent opinions. Verified on a real run: their word onsets are
    bit-identical across all 62 words, max |Δ| = 0.0000 s.
    """
    shared = fuse_word_streams(
        {
            "canary": [
                _w(
                    "hello",
                    1.0,
                    1.4,
                    timestamp_source="external_aligner",
                    timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
                )
            ],
            "qwen": [
                _w(
                    "hello",
                    1.0,
                    1.4,
                    timestamp_source="bundled_aligner",
                    timestamp_model="Qwen/Qwen3-ForcedAligner-0.6B",
                )
            ],
        },
        min_corroboration=MIN_CORROBORATION,
    )
    assert shared[0]["temporal_confidence"] is None, (
        f"two transcripts sharing one aligner are one timing source: {shared[0]}"
    )
    assert shared[0]["timing_sources"] == 1


def test_the_two_edges_are_reported_separately() -> None:
    """Onset and offset agreement are different findings and must not be pooled into one number.

    A word agreed at its start and disputed at its end localises a boundary; a word disputed at
    both does not localise the word. Reporting only the worse edge — which is what a single
    ``temporal_confidence`` did — told a reader neither which edge nor whether both.
    """
    out = fuse_word_streams(
        {
            "a": [_w("hello", 1.00, 1.40)],
            "b": [_w("hello", 1.01, 1.75)],
        },
        min_corroboration=MIN_CORROBORATION,
    )
    word = out[0]
    assert word["onset_confidence"] > 0.9, f"the starts agree to 10 ms: {word}"
    assert word["offset_confidence"] < word["onset_confidence"], "the ends disagree by a third of a word"
    assert word["temporal_confidence"] == pytest.approx(word["offset_confidence"]), (
        "the pooled figure is the worse edge"
    )


def test_a_lone_timing_source_reports_neither_edge() -> None:
    """Unmeasured stays unmeasured at both edges — a renderer must draw nothing, not a green mark."""
    out = fuse_word_streams({"a": [_w("hello", 0.0, 0.5)]}, min_corroboration=MIN_CORROBORATION)
    assert out[0]["onset_confidence"] is None and out[0]["offset_confidence"] is None
