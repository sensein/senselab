"""The speaker axis emits attribution voters, and stops emitting change-detection ones.

The regression being fixed, restated as a test: a bucket every diarizer agrees on must read low even
when the previous bucket held a different speaker. The change-detection composition scored exactly
that case high, which is how a clean two-speaker conversation reported 0.666 while its per-speaker
presence doubt averaged 0.168 and its count posterior was 2 at 0.978.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.speaker import harvest_speaker_votes


def _diar(segments: list[tuple[float, float, str]]) -> dict[str, Any]:
    segs = [SimpleNamespace(start=s, end=e, speaker=spk, text="") for s, e, spk in segments]
    return {"status": "ok", "result": [segs], "cache_key": "k"}


def _summary(mask_state: str | None = None, mask_uncertainty: float = 1.0) -> dict[str, Any]:
    """Two diarizers agreeing on one speaker for the first half and another for the second.

    ``mask_state`` attaches a single whole-clip mask region in that state, which is the only mask
    shape these tests need.
    """
    a = [(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")]
    summary: dict[str, Any] = {
        "duration_s": 1.0,
        "diarization": {"by_model": {"pyannote": _diar(a), "sortformer": _diar(a)}},
    }
    if mask_state is not None:
        summary["background_mask"] = {
            "status": "ok",
            "result": {"regions": [{"start": 0.0, "end": 1.0, "state": mask_state, "uncertainty": mask_uncertainty}]},
        }
    return summary


def _votes(
    pass_summary: dict[str, Any] | None = None,
    fused_words: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return harvest_speaker_votes(
        pass_summary=pass_summary if pass_summary is not None else _summary(),
        grid=BucketGrid(),
        per_window_embeddings={},
        fused_words=fused_words,
    )


def test_agreeing_diarizers_read_low_even_across_a_speaker_change() -> None:
    """The regression. Both models change speaker at 0.5 s and agree about it throughout."""
    buckets = _votes()
    assert buckets, "the harvest produced no buckets"
    for bucket in buckets:
        doubt = per_signal_uncertainty(bucket)
        assert doubt.get("per_speaker_presence") == pytest.approx(0.0), (
            f"agreeing models must carry no attribution doubt at {bucket['start']}"
        )


def test_the_change_detection_entries_are_no_longer_scored() -> None:
    """They are the 0.666. Nothing the fold reads may carry them."""
    for bucket in _votes():
        for name, entry in (bucket["votes"] or {}).items():
            if not isinstance(entry, dict):
                continue
            for field in ("same_label_uncertainty", "change_inconsistency_uncertainty"):
                assert field not in entry, f"{name} still carries {field}"
        read = set(per_signal_uncertainty(bucket))
        assert not {n for n in read if "::" in n}, f"a (diar::emb) pair is still scored: {read}"
        assert "__cross_diar_label_disagreement__" not in read
        assert "__overlap_count__" not in read


def test_the_cluster_assignments_survive_for_their_other_readers() -> None:
    """`per_speaker_tracks`, `cluster_active_time` and identity repair all read these."""
    from senselab.audio.workflows.audio_analysis.speaker import cluster_active_time, per_speaker_tracks

    buckets = _votes()
    assert per_speaker_tracks(buckets), "the per-speaker deliverable lost its input"
    assert cluster_active_time(buckets), "cluster ranking lost its input"


def test_word_location_doubt_reaches_the_axis() -> None:
    """A poorly localised word raises attribution doubt: we do not know whose it is."""
    words = [{"start": 0.0, "end": 0.5, "temporal_confidence": 0.2}]
    buckets = _votes(fused_words=words)
    first = per_signal_uncertainty(buckets[0])
    assert first.get("asr_location") == pytest.approx(0.8)


def test_an_indeterminate_mask_raises_attribution_doubt() -> None:
    """Not knowing whether the target was active is not knowing whether anyone is here."""
    buckets = _votes(pass_summary=_summary(mask_state="indeterminate", mask_uncertainty=1.0))
    assert per_signal_uncertainty(buckets[0]).get("target_activity") == pytest.approx(1.0)


def test_a_confidently_target_free_bucket_makes_no_claim() -> None:
    """No one to attribute, so no vote at all — None, never 0.0."""
    for bucket in _votes(pass_summary=_summary(mask_state="target_free", mask_uncertainty=0.02)):
        assert bucket["votes"] == {}, "a target-free bucket must carry no attribution vote"
        assert per_signal_uncertainty(bucket) == {}


def test_a_target_active_mask_adds_nothing() -> None:
    """Where the mask is sure the target is active, the attribution question is simply live."""
    for bucket in _votes(pass_summary=_summary(mask_state="target_active", mask_uncertainty=0.1)):
        assert "target_activity" not in (bucket["votes"] or {})
