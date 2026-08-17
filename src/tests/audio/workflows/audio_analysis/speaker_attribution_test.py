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
        assert doubt.get("speaker_assignment") == pytest.approx(0.0), (
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
        # ``overlap_count`` is an L1 measurement now, not a synthetic block: it lost the ``__``
        # prefix so ``_signal_rows_from_buckets`` records it, and its scored ``value`` so the fold
        # does not read it. Under the old name-and-field it was recorded by nobody.
        # Emitted only where the count is actually ambiguous, so its presence is conditional; what
        # is unconditional is that the fold must not score it.
        assert "overlap_count" not in read, "the overlap distribution must not be scored"


def test_the_cluster_assignments_survive_for_their_other_readers() -> None:
    """`per_speaker_tracks`, `cluster_active_time` and identity repair all read these."""
    from senselab.audio.workflows.audio_analysis.speaker import cluster_active_time, per_speaker_tracks

    buckets = _votes()
    assert per_speaker_tracks(buckets), "the per-speaker deliverable lost its input"
    assert cluster_active_time(buckets), "cluster ranking lost its input"


def test_words_gate_the_axis_and_never_vote_on_it() -> None:
    """Word timing says when there is speech to attribute; it is not evidence about *who*.

    It voted, as ``1 - temporal_confidence``, and contributed ~0.223 of standing doubt in every
    bucket — swamping a per-speaker term that read 0.0 across 86% of a clean recording. Boundary
    jitter of tens of milliseconds cannot tell you which of two speakers said a word.
    """
    words = [{"start": 0.0, "end": 0.5, "temporal_confidence": 0.2}]
    buckets = _votes(fused_words=words)
    for bucket in buckets:
        assert "asr_location" not in (bucket["votes"] or {}), "word timing must not vote on identity"
    # The word covers [0.0, 0.5], so those buckets keep their per-speaker claim...
    assert "speaker_assignment" in (buckets[0]["votes"] or {})
    # ...and the wordless remainder of the clip makes no claim at all.
    assert buckets[-1]["votes"] == {}, "a bucket with no words has no speech to attribute"


def test_a_wordless_bucket_makes_no_claim() -> None:
    """The sharpening, as one property: 22 of 29 flagged buckets on a real clip were inter-turn gaps.

    Four diarizers disagreeing about exactly where a turn ends is not doubt about *who* is speaking —
    in a gap between turns there is no speaker to get wrong.
    """
    for bucket in _votes(fused_words=[{"start": 0.0, "end": 0.2}]):
        if bucket["start"] >= 0.2:
            # Wordless: the whole bucket is cleared, so no measurement rides through either.
            assert bucket["votes"] == {}, f"at {bucket['start']}"
        else:
            # Worded: exactly one scored voter, alongside the unscored measurements.
            assert set(per_signal_uncertainty(bucket)) == {"speaker_assignment"}, f"at {bucket['start']}"


def test_no_asr_at_all_leaves_the_axis_intact() -> None:
    """Unmeasured is not measured-empty: a run without ASR keeps its speaker axis.

    ``fused_words=None`` means the stage did not run. Gating on that would null every bucket and
    delete the axis on any run with ``stages.asr: false``.
    """
    buckets = _votes(fused_words=None)
    assert any("speaker_assignment" in (b["votes"] or {}) for b in buckets)


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


def _wordless_buckets(mask_state: str | None) -> list[dict[str, Any]]:
    """Buckets from 0.2 s on: worded nowhere, with one whole-clip mask region in ``mask_state``.

    The word gate needs at least one word *somewhere* to be a measurement, so the clip carries one
    over ``[0.0, 0.2]`` and every bucket after it is the wordless case under test.
    """
    buckets = _votes(pass_summary=_summary(mask_state=mask_state), fused_words=[{"start": 0.0, "end": 0.2}])
    return [b for b in buckets if b["start"] >= 0.2]


def test_a_wordless_bucket_the_mask_calls_target_active_keeps_the_speaker_voter() -> None:
    """An ASR miss is not silence, and must not null the speaker axis.

    Word absence stands in for speech absence, and the mask is the one source that can contradict
    it: ``target_active`` is a positive report that the target was vocalising here, so a bucket with
    no words is an ASR failure rather than a gap between turns. Zeroing the axis there hides the
    failure behind a confident-looking empty bucket.

    ``target_activity`` is *not* among the survivors: ``target_activity_doubt`` returns ``None`` for
    ``target_active``, since the mask being sure the target is up leaves the attribution question
    simply live. ``speaker_assignment`` alone is what the gate was discarding.
    """
    buckets = _wordless_buckets("target_active")
    assert buckets, "the fixture produced no wordless buckets"
    for bucket in buckets:
        assert set(per_signal_uncertainty(bucket)) == {"speaker_assignment"}, f"at {bucket['start']}"


def test_a_wordless_nontarget_active_bucket_keeps_its_word_independent_voters() -> None:
    """A non-lexical vocalization has no words and is still someone making a sound.

    This is the infant-cry case. ``data/audioset_source_map.json`` maps "Baby cry, infant cry" to
    the ``people`` *background source* category, while the speech task's target vocabulary is
    speech/breath/mouth_noise — so a cry lands in a ``nontarget_active`` region rather than a
    ``target_free`` one, reaches the word gate with no words, and used to be zeroed outright. Both
    voters after the gate are word-independent (diarizer-cluster entropy, mask region state), so
    word absence is the wrong reason to discard either.
    """
    buckets = _wordless_buckets("nontarget_active")
    assert buckets, "the fixture produced no wordless buckets"
    for bucket in buckets:
        read = per_signal_uncertainty(bucket)
        assert set(read) == {"speaker_assignment", "target_activity"}, f"at {bucket['start']}"
        assert read["target_activity"] == pytest.approx(1.0), f"at {bucket['start']}"


def test_a_wordless_indeterminate_bucket_still_makes_no_claim() -> None:
    """The case the gate exists for: the mask reports no vocal activity, so silence is the reading.

    ``indeterminate`` is the mask declining to say, which is not a positive report of anyone
    vocalising — so nothing contradicts the word proxy, and adult inter-turn silence keeps being
    read as inter-turn silence. 22 of the 29 buckets the axis flagged on a clean two-speaker
    conversation were exactly this.
    """
    for bucket in _wordless_buckets("indeterminate"):
        assert bucket["votes"] == {}, f"at {bucket['start']}"


def test_a_wordless_target_free_bucket_still_clears_everything() -> None:
    """``target_free`` is decided before the word gate is consulted, and stays that way.

    The mask positively reporting nobody present is a stronger statement than any word evidence, so
    its branch runs first. Making the gate mask-aware must not reorder the two.
    """
    for bucket in _wordless_buckets("target_free"):
        assert bucket["votes"] == {}, f"at {bucket['start']}"
        assert per_signal_uncertainty(bucket) == {}, f"at {bucket['start']}"


def test_a_wordless_bucket_with_no_mask_region_still_makes_no_claim() -> None:
    """No mask is not evidence of vocal activity, so the word proxy is all there is.

    A ``None`` state covers both "no region covers this bucket" and "the mask stage never ran", and
    neither can license retaining evidence the word gate would otherwise drop.
    """
    for bucket in _wordless_buckets(None):
        assert bucket["votes"] == {}, f"at {bucket['start']}"


def test_no_intervention_recomputes_the_per_speaker_term() -> None:
    """The axis follows the per-speaker presence, so nothing may overwrite that term mid-loop.

    ``I2_recluster`` used to shadow the harvest's ``speaker_assignment`` with a value recomputed
    over its own repaired clusters. On a real run that took the published axis from 0.288 to 0.608,
    because the repair emits 5 clusters against a count posterior of 2 at 0.978 and five clusters
    spread across the sources drop each share to ~0.2 (``H(0.2) = 0.722``).

    That reintroduced the defect this axis exists to remove: ``final/per_speaker_presence.parquet`` is
    built by ``build_speech_presence_tracks(speaker_harvest)`` — from the *harvest*, never from
    ``refined_identity`` — so it still read 0.1196 while the axis read 0.608. A deliverable and the
    axis describing it must not disagree.

    Read off the source because the failure is a *second producer*: the harvest-level tests here
    cannot see a vote another module adds, which is how this shipped once already.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.adaptive import interventions

    source = inspect.getsource(interventions)
    assert 'source="speaker_assignment"' not in source, (
        "an intervention is overwriting the axis's per-speaker term; the axis must follow the "
        "per-speaker presence the run publishes"
    )
