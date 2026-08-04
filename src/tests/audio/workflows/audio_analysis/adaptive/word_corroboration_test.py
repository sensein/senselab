"""Words carry a measured weight into fusion; nothing removes them from the stream.

`collect_word_streams` used to take `purged_spans` and drop every word of a model overlapping an
adjudicated span. Two structural faults beyond the erasure: the drop was gated on an intervention
having *fired and been admitted within budget*, so budget accounting decided what reached the
deliverable; and the trigger measured the ASR against a `p_voice` that folds in the ASR's own
presence vote — the model indicting itself.

These tests pin the replacement: one measurement, taken against voters that are independent of
ASR by construction, applied to every word whether or not any rule fired.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.tasks.speech_to_text_ensemble import MIN_CORROBORATION
from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, Vote, VoteStore, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.corroboration import (
    apply_corroboration,
    independent_presence_pool,
    make_corroboration_lookup,
)
from senselab.audio.workflows.audio_analysis.adaptive.fusion import rollup_segments
from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

STREAM = "raw"


def _presence_vote(source: str, bucket: tuple[float, float], p_speech: float, **extra: Any) -> Vote:  # noqa: ANN401
    """A presence vote linked the way ``speech_presence_link`` links one: directed confidence."""
    speaks = p_speech >= 0.5
    return Vote(
        axis="speech_presence",
        bucket=bucket,
        source=source,
        stream=STREAM,
        scope="file",
        round=1,
        payload={
            "speaks": speaks,
            "native_confidence": p_speech if speaks else 1.0 - p_speech,
            **extra,
        },
    )


def _store(speech_profile: list[float], *, asr_claims_everywhere: bool = True) -> VoteStore:
    """One frame voter tracking ``speech_profile`` per 0.5 s bucket, plus an ASR claiming speech."""
    store = VoteStore()
    for index, p in enumerate(speech_profile):
        bucket = bucket_key(index * 0.5, (index + 1) * 0.5)
        store.add_vote(_presence_vote("frame_brouhaha_vad", bucket, p, frame_mean=p))
        if asr_claims_everywhere:
            store.add_vote(_presence_vote("openai/whisper-large-v3", bucket, 0.9, word_overlap_s=0.4))
    return store


# speech for 1 s, silence for 1 s, speech for 1 s
_PROFILE = [0.95, 0.9, 0.02, 0.03, 0.92, 0.88]


def _lookup(store: VoteStore) -> Any:  # noqa: ANN401
    pool, _rejected = independent_presence_pool(store, STREAM)
    return make_corroboration_lookup(store, STREAM, pool=pool)


def test_the_evidence_pool_never_contains_an_asr_model() -> None:
    """`p_voice` folds the ASR's own presence vote into the number.

    Measuring an ASR word against it lets the model indict — or exonerate — its own words, which is
    the self-confirmation failure `adaptive.provenance.classify_resolution` exists to catch. The
    exclusion is structural (`support.evidence_signal_names`), not a name list that can drift.
    """
    pool, _rejected = independent_presence_pool(_store(_PROFILE), STREAM)
    assert pool == ["frame_brouhaha_vad"]
    assert not any("whisper" in name for name in pool)


def test_an_always_affirmative_proxy_is_rejected_from_the_pool() -> None:
    """Corroboration runs entirely on negative evidence.

    A voter that never reports absence cannot withhold support from anything, and pooled with max
    it pins every word near 1.0 — silently making the whole measure inert rather than visibly
    disabling it.
    """
    store = VoteStore()
    for index in range(8):
        bucket = bucket_key(index * 0.5, (index + 1) * 0.5)
        store.add_vote(_presence_vote("frame_brouhaha_vad", bucket, _PROFILE[index % len(_PROFILE)], frame_mean=0.5))
        store.add_vote(_presence_vote("acoustic_always_yes", bucket, 0.97))
    pool, rejected = independent_presence_pool(store, STREAM)
    assert "acoustic_always_yes" not in pool
    assert rejected["acoustic_always_yes"] == "never_reports_absence"


def test_a_word_in_a_silent_span_is_weighted_down_but_not_removed() -> None:
    """Erasing it is how a quiet or overlapped speaker leaves no trace to appeal to."""
    streams = {"m1": [{"text": "phantom", "start": 1.1, "end": 1.4}, {"text": "real", "start": 0.1, "end": 0.4}]}
    stamped, provenance = apply_corroboration(
        streams, _lookup(_store(_PROFILE)), exponent=1.0, min_corroboration=MIN_CORROBORATION, pool=[], rejected={}
    )
    phantom = next(w for w in stamped["m1"] if w["text"] == "phantom")
    real = next(w for w in stamped["m1"] if w["text"] == "real")
    assert phantom["corroboration"] == pytest.approx(MIN_CORROBORATION)
    assert real["corroboration"] > 0.9
    assert provenance["n_words_measured"] == 2
    # The raw measurement travels, so the weight can be re-derived under a different exponent
    # without re-running a model.
    assert phantom["corroboration_evidence"]["p_independent"] == pytest.approx(0.02)
    assert phantom["corroboration_evidence"]["n_buckets"] == 1


def test_a_word_is_measured_against_the_best_overlapping_bucket() -> None:
    """Presence buckets are 0.5 s; words are shorter and straddle boundaries.

    Max over the overlapping buckets is deliberately permissive: a coarse measurement must not
    confidently indict a finer one. The counts ride along so the coarseness stays auditable.
    """
    streams = {"m1": [{"text": "edge", "start": 0.9, "end": 1.2}]}
    stamped, _ = apply_corroboration(
        streams, _lookup(_store(_PROFILE)), exponent=1.0, min_corroboration=MIN_CORROBORATION, pool=[], rejected={}
    )
    word = stamped["m1"][0]
    assert word["corroboration_evidence"]["n_buckets"] == 2
    assert word["corroboration"] == pytest.approx(0.9)  # the speech bucket, not the silent one


def test_an_empty_pool_leaves_every_word_unmeasured() -> None:
    """Absent is not zero, and inertness must be visible rather than inferred.

    `informative_evidence`'s constants were measured on one 697-bucket recording. On a run where
    no voter is informative the pool is empty and the mechanism does nothing — which is correct,
    but has to be reported, or an inert run looks like a run where nothing was doubtful.
    """
    streams = {"m1": [{"text": "hello", "start": 1.1, "end": 1.4}]}
    lookup = make_corroboration_lookup(_store(_PROFILE), STREAM, pool=[])
    stamped, provenance = apply_corroboration(
        streams, lookup, exponent=1.0, min_corroboration=MIN_CORROBORATION, pool=[], rejected={"x": "why"}
    )
    assert stamped["m1"][0]["corroboration"] is None
    assert provenance["n_words_unmeasured"] == 1
    assert provenance["n_words_measured"] == 0
    assert provenance["evidence_pool"] == []
    assert provenance["evidence_pool_rejected"] == {"x": "why"}


def test_measurement_does_not_depend_on_any_intervention_having_fired() -> None:
    """Whether a word survives must not depend on budget accounting.

    The old drop was gated on P3 having fired *and* having been admitted within budget, so a
    deferred intervention silently changed the transcript. The measurement is now taken for every
    word on every run, and reads nothing from the iteration log.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.adaptive import fusion

    source = inspect.getsource(fusion.collect_word_streams)
    assert "purged" not in source
    assert "iterations" not in source
    assert "purged_spans" not in str(inspect.signature(fusion.collect_word_streams))


# ── rendering: withheld from the readable text, retained in the record ────


def test_withheld_words_stay_in_the_record_with_their_measurement() -> None:
    """The one remaining decision is at the rendering layer, and it is reproducible.

    Keeping an uncorroborated word in the readable transcript would let it *win* — the deliverable
    would assert it and the text consumers downstream would ingest it. Dropping it from `words[]`
    would be the erasure this work removed. Recording the indices makes the rollup a pure function
    of `words[]` plus one number, so the exclusion can be re-decided by re-reading one file.
    """
    words = [
        {"text": "real", "start": 0.1, "end": 0.4, "confidence": 0.9, "corroboration": 0.95, "speaker": "C0"},
        {"text": "phantom", "start": 1.1, "end": 1.4, "confidence": 0.2, "corroboration": 0.05, "speaker": "C0"},
        {"text": "again", "start": 2.1, "end": 2.4, "confidence": 0.9, "corroboration": 0.9, "speaker": "C0"},
    ]
    segments, withheld = rollup_segments(words, min_corroboration=0.2)
    assert withheld == [1]
    assert "phantom" not in " ".join(s["text"] for s in segments)
    assert len(words) == 3, "the rollup must not mutate the record it renders from"


def test_the_rollup_threshold_never_removes_a_word_from_the_record() -> None:
    """`segment_min_corroboration` is the number people will tune.

    Raised far enough it reproduces purging *in effect* on the readable text — so the invariant
    that has to hold at any setting is that `words[]` is untouched.
    """
    words = [
        {"text": "a", "start": 0.0, "end": 0.2, "confidence": 0.9, "corroboration": 0.9},
        {"text": "b", "start": 0.3, "end": 0.5, "confidence": 0.9, "corroboration": 0.5},
    ]
    for threshold in (0.05, 0.2, 0.6, 0.99):
        segments, withheld = rollup_segments(words, min_corroboration=threshold)
        assert len(words) == 2
        assert len(withheld) + sum(len(s["text"].split()) for s in segments) == 2


def test_unmeasured_words_are_never_withheld() -> None:
    """A word nothing was measured about must render. Absent is not below-threshold."""
    words = [{"text": "hi", "start": 0.0, "end": 0.2, "confidence": 0.9, "corroboration": None}]
    segments, withheld = rollup_segments(words, min_corroboration=0.99)
    assert withheld == []
    assert segments[0]["text"] == "hi"


def test_transcript_records_the_pool_it_measured_against(tmp_path: Path) -> None:
    """An artifact that cannot say what pool it measured against is not interpretable."""
    pytest.importorskip("pandas")
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import build_final_outputs

    store = _store(_PROFILE)
    state = BeliefState.from_store(store, aggregator="min", round_index=1)
    build_final_outputs(
        out_dir=tmp_path,
        words=[
            {"text": "real", "start": 0.1, "end": 0.4, "confidence": 0.9, "corroboration": 0.95},
            {"text": "phantom", "start": 1.1, "end": 1.4, "confidence": 0.2, "corroboration": 0.05},
        ],
        store=store,
        state=state,
        stream=STREAM,
        policy=load_policy(),
        generated_from_round=1,
        corroboration_provenance={
            "evidence_pool": ["frame_brouhaha_vad"],
            "evidence_pool_rejected": {},
            "pool_derivation": "support.evidence_signal_names + support.informative_evidence",
            "exponent": 1.0,
            "min_corroboration": MIN_CORROBORATION,
            "n_words_measured": 2,
            "n_words_unmeasured": 0,
        },
    )
    doc = json.loads((tmp_path / "final" / "transcript.json").read_text())
    corroboration = doc["corroboration"]
    assert corroboration["evidence_pool"] == ["frame_brouhaha_vad"]
    assert corroboration["withheld_word_indices"] == [1]
    assert corroboration["n_words_withheld_from_segments"] == 1
    assert len(doc["words"]) == 2, "withheld words stay in the record"
    assert "phantom" not in " ".join(s["text"] for s in doc["segments"])


def test_a_zero_corroboration_floor_is_refused_by_the_policy_loader(tmp_path: Path) -> None:
    """A floor that can be configured to zero is not a floor."""
    override = tmp_path / "p.yaml"
    override.write_text("fusion:\n  corroboration:\n    min_corroboration: 0\n")
    with pytest.raises(ValueError, match="min_corroboration"):
        load_policy(override)
