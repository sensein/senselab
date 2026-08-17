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

from senselab.audio.workflows.audio_analysis.background_mask import MASK_STATES, MaskState
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
    buckets = _votes(fused_words=[{"start": 0.0, "end": 0.2}])
    assert buckets, "the harvest produced no buckets"
    for bucket in buckets:
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


def _wordless_buckets(mask_state: str | None, mask_uncertainty: float = 0.4) -> list[dict[str, Any]]:
    """Buckets from 0.2 s on: worded nowhere, with one whole-clip mask region in ``mask_state``.

    The word gate needs at least one word *somewhere* to be a measurement, so the clip carries one
    over ``[0.0, 0.2]`` and every bucket after it is the wordless case under test.

    ``mask_uncertainty`` defaults to a value that is neither of the two the doubt path could invent
    on its own: ``1.0`` is both ``_summary``'s own default and the clamp ceiling in
    ``attribution.target_activity_doubt`` *and* in ``fuse``, so asserting on it cannot tell a
    passthrough from an upward transform.

    The non-empty assertion lives here rather than in each caller so no state case can go vacuously
    green: the filter is over a 1.0 s clip on a 0.1 s grid, and a grid change would empty it.
    """
    buckets = _votes(
        pass_summary=_summary(mask_state=mask_state, mask_uncertainty=mask_uncertainty),
        fused_words=[{"start": 0.0, "end": 0.2}],
    )
    wordless = [b for b in buckets if b["start"] >= 0.2]
    assert wordless, f"the fixture produced no wordless buckets for {mask_state!r}"
    return wordless


def _measurement_keys(bucket: dict[str, Any]) -> set[str]:
    """The unscored entries a bucket carries — everything that is not one of the two voters.

    These are what ``per_speaker_tracks``, ``cluster_active_time`` and identity repair read, and
    what ``per_signal_uncertainty`` cannot see, so a test asserting only through the fold cannot
    tell a surviving bucket from one stripped to its voters.
    """
    return set(bucket["votes"] or {}) - {"speaker_assignment", "target_activity"}


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
    for bucket in _wordless_buckets("target_active"):
        assert set(per_signal_uncertainty(bucket)) == {"speaker_assignment"}, f"at {bucket['start']}"


def test_a_wordless_target_active_bucket_keeps_the_measurements_nobody_scores() -> None:
    """The voters are not what the pre-fix gate cost most; the measurements under them were.

    ``per_signal_uncertainty`` reads scored fields only, so a bucket stripped to its two voters
    reads identically to an intact one — while ``per_speaker_tracks``, ``cluster_active_time`` and
    identity repair, which read the per-diarizer ``cluster_id`` entries and
    ``__cross_diar_label_disagreement__``, lose their whole input. Pinned against a word-covered
    bucket from the same run rather than against a literal key list, so a measurement added later
    (``overlap_count``, emitted only where the count is actually ambiguous, is not in this fixture)
    is covered without this test being edited.
    """
    from senselab.audio.workflows.audio_analysis.speaker import cluster_active_time, per_speaker_tracks

    worded = [b for b in _votes(fused_words=[{"start": 0.0, "end": 0.2}]) if b["start"] < 0.2]
    expected = _measurement_keys(worded[0])
    assert "__cross_diar_label_disagreement__" in expected, "the fixture stopped producing the measurement"

    buckets = _wordless_buckets("target_active")
    for bucket in buckets:
        assert _measurement_keys(bucket) == expected, f"at {bucket['start']}"
        cross = bucket["votes"]["__cross_diar_label_disagreement__"]
        assert set(cross["cluster_ids"]) == {"pyannote", "sortformer"}, f"at {bucket['start']}"
        for model in ("pyannote", "sortformer"):
            assert bucket["votes"][model]["cluster_id"], f"{model} lost its cluster at {bucket['start']}"

    assert per_speaker_tracks(buckets), "the per-speaker deliverable lost its input in wordless buckets"
    assert cluster_active_time(buckets), "cluster ranking lost its input in wordless buckets"


def test_a_wordless_nontarget_active_bucket_keeps_its_word_independent_voters() -> None:
    """A non-lexical vocalization has no words and is still someone making a sound.

    This is the infant-cry case. ``data/audioset_source_map.json`` maps "Baby cry, infant cry" to
    the ``people`` *background source* category, while the speech task's target vocabulary is
    speech/breath/mouth_noise — so a cry lands in a ``nontarget_active`` region rather than a
    ``target_free`` one, reaches the word gate with no words, and used to be zeroed outright. Both
    voters after the gate are word-independent (diarizer-cluster entropy, mask region state), so
    word absence is the wrong reason to discard either.

    The doubt is asserted at the region's own ``0.4``, not at a clamp bound: ``target_activity`` is
    the mask's number passed through, and a test pinned to ``1.0`` cannot see it being raised.
    """
    for bucket in _wordless_buckets("nontarget_active"):
        read = per_signal_uncertainty(bucket)
        assert set(read) == {"speaker_assignment", "target_activity"}, f"at {bucket['start']}"
        assert read["target_activity"] == pytest.approx(0.4), f"at {bucket['start']}"


def test_a_wordless_indeterminate_bucket_still_makes_no_claim() -> None:
    """The case the gate exists for: the mask reports no vocal activity, so silence is the reading.

    ``indeterminate`` is the mask declining to say, which is not a positive report of anyone
    vocalising — so nothing contradicts the word proxy, and adult inter-turn silence keeps being
    read as inter-turn silence.

    The 22-of-29 measurement is *not* a measurement of this state. What was recorded about those
    buckets is that they were wordless inter-turn gaps; no mask state was recorded for them, and the
    mask's regions do not reach this code on a real run at all (see the note at ``mask_regions`` in
    ``speaker``). This pins the reading the fix must not change, not that measurement.
    """
    for bucket in _wordless_buckets("indeterminate"):
        assert bucket["votes"] == {}, f"at {bucket['start']}"


def test_a_word_covered_target_free_bucket_is_still_cleared() -> None:
    """``target_free`` clears on its own authority, not by borrowing the word gate's.

    Word-covered, so the gate cannot fire and the only branch that can empty this bucket is the
    ``target_free`` one — which is what makes the two clearings distinguishable. On a *wordless*
    ``target_free`` bucket both branches produce ``{}``, so a mutant conditioning the ``target_free``
    clear on wordlessness (the word gate winning the precedence) is invisible there and passes.

    The mask positively reporting nobody present is a stronger statement than any word evidence: it
    holds whether or not a recognizer put words here, because a word in a target-free region is the
    recognizer's error, not a person.
    """
    buckets = _votes(
        pass_summary=_summary(mask_state="target_free", mask_uncertainty=0.02),
        fused_words=[{"start": 0.0, "end": 1.0}],
    )
    assert buckets, "the harvest produced no buckets"
    for bucket in buckets:
        assert bucket["votes"] == {}, f"at {bucket['start']}"
        assert per_signal_uncertainty(bucket) == {}, f"at {bucket['start']}"


def test_a_partly_word_covered_bucket_is_not_gated() -> None:
    """The gate fires on *no* word, not on *little* word, and a tenth of a word is not no word.

    Every word in the other fixtures aligns to a bucket boundary, so coverage is only ever 0.0 or
    1.0 there and any threshold in ``(0, 1]`` reads the same — ``coverage <= 0.0`` widened to
    ``< 0.5`` passes all of them. A word ending mid-bucket is the only shape that separates them.

    The word ends 0.01 s into the bucket, for a coverage of 0.1, rather than at the halfway mark:
    ``< 0.5`` is false at exactly 0.5, so half a bucket is the one fraction at which the widened
    threshold still reads correctly. A tenth kills every widening down to it.

    Run under ``indeterminate``, the state where nothing else can rescue the bucket, so the survival
    is the threshold's doing alone.
    """
    buckets = _votes(
        pass_summary=_summary(mask_state="indeterminate", mask_uncertainty=0.4),
        fused_words=[{"start": 0.0, "end": 0.21}],
    )
    partial = [b for b in buckets if b["start"] == pytest.approx(0.2)]
    assert partial, "the fixture produced no partly covered bucket"
    for bucket in partial:
        assert set(per_signal_uncertainty(bucket)) == {"speaker_assignment", "target_activity"}, (
            f"a bucket a word partly occupies has speech to attribute, at {bucket['start']}"
        )


# Every mask state, and the wordless reading each one must produce. Keyed by state rather than
# written as separate cases so the enumeration is checked for completeness below: a fifth state
# added to ``MASK_STATES`` upstream fails here instead of silently inheriting the gate's behaviour.
_WORDLESS_READING: dict[MaskState, set[str]] = {
    "target_active": {"speaker_assignment"},
    "nontarget_active": {"speaker_assignment", "target_activity"},
    "indeterminate": set(),
    "target_free": set(),
}


def test_the_gate_exemption_is_drawn_from_the_masks_own_vocabulary() -> None:
    """``_VOCAL_ACTIVITY`` is a subset of the states the mask can actually emit.

    A typo or a renamed state upstream would otherwise leave a member that matches nothing, and the
    branch it guards would read as present while never firing — the same failure mode as the
    unwired ``regions`` key this whole path already has.

    ``target_free`` is excluded here explicitly because its exclusion is *behaviourally*
    unobservable: its own branch returns before the gate is reached, so adding it to
    ``_VOCAL_ACTIVITY`` changes no output on any input. The constant is the only place that
    statement can be pinned.
    """
    from senselab.audio.workflows.audio_analysis.speaker import _VOCAL_ACTIVITY

    assert set(_VOCAL_ACTIVITY) <= set(MASK_STATES), f"not a mask state: {set(_VOCAL_ACTIVITY) - set(MASK_STATES)}"
    assert "target_free" not in _VOCAL_ACTIVITY, "the mask reporting nobody present cannot exempt anything"


@pytest.mark.parametrize("state", MASK_STATES)
def test_every_mask_state_has_a_pinned_wordless_reading(state: MaskState) -> None:
    """One case per state the mask defines, so a new state cannot arrive unread.

    The named tests above carry the reasoning for each state; this one carries the enumeration, and
    fails on a state nobody has decided about rather than applying the word gate to it by default.
    """
    assert set(_WORDLESS_READING) == set(MASK_STATES), (
        f"a mask state has no pinned wordless reading: {set(MASK_STATES) - set(_WORDLESS_READING)}"
    )
    for bucket in _wordless_buckets(state):
        assert set(per_signal_uncertainty(bucket)) == _WORDLESS_READING[state], f"{state} at {bucket['start']}"


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
