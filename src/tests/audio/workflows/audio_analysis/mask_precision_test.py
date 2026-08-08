"""Mask temporal precision: the mask may use every axis, at each one's own resolution."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.background_mask import target_spans_from_evidence


def test_word_spans_pin_speech_far_more_precisely_than_a_bucket_grid() -> None:
    """ASR words carry ~10 ms boundaries; the bucket grid the mask used is 0.5 s.

    A bucket containing one short word was wholly target-active, so every gap shorter than the
    bucket was invisible and the mask could only ever be as precise as its coarsest source.
    """
    spans = target_spans_from_evidence(word_spans=[(0.10, 0.32), (2.55, 2.80)], duration_s=4.0)
    assert spans["target"] == [(0.10, 0.32), (2.55, 2.80)]
    # The gaps between words are candidate background, at word resolution.
    assert spans["free"][0] == pytest.approx((0.0, 0.10))
    assert (0.32, 2.55) in [(round(a, 6), round(b, 6)) for a, b in spans["free"]]


def test_evidence_from_several_axes_is_unioned_not_averaged() -> None:
    """Any axis asserting the target was active is sufficient; they are not votes to average.

    A word the ASR heard and a span the diarizer attributed are both positive evidence, and a
    region is target-free only where *nothing* claimed activity.
    """
    spans = target_spans_from_evidence(
        word_spans=[(0.0, 1.0)],
        speaker_spans=[(2.0, 3.0)],
        duration_s=4.0,
    )
    assert (1.0, 2.0) in [(round(a, 6), round(b, 6)) for a, b in spans["free"]]
    assert (3.0, 4.0) in [(round(a, 6), round(b, 6)) for a, b in spans["free"]]


def test_frame_posteriors_contribute_at_their_own_resolution() -> None:
    """16.9 ms frames are the finest evidence in the system; downsampling them discards the point."""
    frames = [0.9] * 10 + [0.0] * 40 + [0.9] * 10
    spans = target_spans_from_evidence(frame_speech=frames, frame_hop_s=0.01, duration_s=0.6)
    free = [(round(a, 3), round(b, 3)) for a, b in spans["free"]]
    assert any(abs(a - 0.1) < 0.02 and abs(b - 0.5) < 0.02 for a, b in free)


def test_a_gap_shorter_than_the_guard_is_not_offered_as_background() -> None:
    """A between-words pause is not a background opportunity — the target is still present.

    Without a floor every inter-word gap becomes a region, and the mask reports hundreds of
    50 ms "backgrounds" that no classifier could characterise.
    """
    spans = target_spans_from_evidence(word_spans=[(0.0, 1.0), (1.05, 2.0)], duration_s=2.0, min_free_s=0.25)
    assert spans["free"] == []


def test_no_evidence_at_all_yields_no_claim_either_way() -> None:
    """Nothing measured is not "the whole clip is free" — that would be the loudest possible guess."""
    assert target_spans_from_evidence(duration_s=5.0) == {"target": [], "free": []}


# ── word evidence reaches the per-bucket confidence ──────────────────────


def test_a_word_raises_target_confidence_in_the_bucket_it_falls_in() -> None:
    """ASR words are direct evidence the target spoke, and the mask was not consulting them.

    `target_confidence_by_bucket` combined diarization and classifier scores by taking the
    maximum, because either is sufficient to establish activity. A recognised word is evidence of
    exactly the same kind and was simply absent from the union.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import apply_span_evidence

    rows = [
        {"start": 0.0, "end": 0.5, "target_confidence": 0.1, "uncertainty": 0.4},
        {"start": 0.5, "end": 1.0, "target_confidence": 0.1, "uncertainty": 0.4},
        {"start": 1.0, "end": 1.5, "target_confidence": 0.1, "uncertainty": 0.4},
    ]
    # 0.2 s of words in the middle bucket (40% of it); the last bucket is filled.
    out = apply_span_evidence(rows, target_spans=[(0.6, 0.8), (1.0, 1.5)], active_at=0.6, free_at=0.2)

    assert out[0]["target_confidence"] == pytest.approx(0.1), "no word here, unchanged"
    # Was ``== 1.0``. That expectation *was* the defect: any overlap, however small, pinned the
    # bucket to absolute confidence, so every bucket of a conversation scored identically and the
    # mask had nothing left to be uncertain about. What the rule actually owes is that a word makes
    # the bucket target-active — above the threshold, not at the ceiling.
    assert out[1]["target_confidence"] >= 0.6, "a word here is direct evidence the target was active"
    assert out[1]["target_confidence"] < out[2]["target_confidence"], "partly covered is weaker than filled"
    assert out[2]["target_confidence"] == pytest.approx(1.0), "a bucket full of words is fully confident"


def test_span_evidence_only_raises_confidence_never_lowers_it() -> None:
    """ASR misses words, so silence in the transcript is not evidence the target was absent.

    Letting an empty span lower confidence would turn a recogniser's miss into a positive claim
    that nobody spoke — which is exactly the region a background characterisation would then
    wrongly treat as usable.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import apply_span_evidence

    rows = [{"start": 0.0, "end": 0.5, "target_confidence": 0.9, "uncertainty": 0.1}]
    assert apply_span_evidence(rows, target_spans=[])[0]["target_confidence"] == pytest.approx(0.9)


def test_a_bucket_raised_by_span_evidence_becomes_more_certain_too() -> None:
    """A direct observation resolves doubt — in proportion to how much of the bucket it observed.

    Previously ``uncertainty`` was set to ``min(u, 0.0)``, so any overlap made the bucket
    *perfectly* certain. The claim being tested survives; the absoluteness does not, and how much
    doubt a word resolves now depends on how much of the bucket it accounts for.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import apply_span_evidence

    rows = [
        {"start": 0.0, "end": 0.5, "target_confidence": 0.1, "uncertainty": 0.8},
        {"start": 0.5, "end": 1.0, "target_confidence": 0.1, "uncertainty": 0.8},
    ]
    out = apply_span_evidence(rows, target_spans=[(0.05, 0.45), (0.9, 1.0)], active_at=0.6, free_at=0.2)

    assert out[0]["uncertainty"] < 0.8, "a bucket mostly filled with words is more certain than before"
    assert out[0]["uncertainty"] > 0.0, "and still not certain, having observed only part of it"
    assert out[1]["uncertainty"] > out[0]["uncertainty"], "a glancing word resolves less doubt"
