"""Per-speaker speaker uncertainty (T086-T093, FR-001 to FR-011).

The motivating case is concrete: two diarizers each reported one speaker for a whole clip
while embedding clustering reported five regions aligned to name boundaries. A single
speaker scalar registered 0.67 — correct, but unreadable, because it cannot distinguish
"we disagree about who spoke" from "we disagree about whether this is one person or four".

These tests assert **representation, not accuracy**. The spec deliberately does not require
resolving that clip in a particular direction; it requires that the disagreement be
expressible and attributable.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.speaker_identity import (
    PerSpeakerPresenceTrack,
    SourceCountClaim,
    SourceLabelCorrespondence,
    SpeakerHypothesis,
    speaker_count_posterior,
)

GATES = {"independent": 1.0, "derived": 0.4}


# ── the posterior (T086, FR-002 / FR-006 / FR-008) ────────────────────


def test_probabilities_sum_to_one() -> None:
    """A distribution, not a score."""
    p = speaker_count_posterior([SourceCountClaim("a", 1), SourceCountClaim("b", 2)], gates=GATES)
    assert sum(p.probabilities.values()) == pytest.approx(1.0)


def test_every_supported_count_appears_in_the_probabilities() -> None:
    """Support and probabilities cannot disagree about which counts exist."""
    p = speaker_count_posterior([SourceCountClaim("a", 1), SourceCountClaim("b", 4)], gates=GATES)
    assert set(p.support) <= set(p.probabilities)


def test_disagreement_is_multimodal_not_averaged() -> None:
    """A mean would report "2.5 speakers", which describes nobody's claim.

    The disagreement is the finding; collapsing it is what made the original scalar
    unreadable.
    """
    p = speaker_count_posterior([SourceCountClaim("a", 1), SourceCountClaim("b", 4)], gates=GATES)
    assert p.is_multimodal is True
    assert set(p.probabilities) == {1, 4}


def test_support_names_which_source_backed_each_count() -> None:
    """FR-006: an analyst must be able to attribute a count without opening intermediates."""
    p = speaker_count_posterior(
        [SourceCountClaim("pyannote", 1), SourceCountClaim("sortformer", 1), SourceCountClaim("emb", 5)],
        gates=GATES,
    )
    assert sorted(p.support[1]) == ["pyannote", "sortformer"]
    assert p.support[5] == ["emb"]


# ── the motivating case (T093) ────────────────────────────────────────


def test_two_diarizers_versus_a_less_supported_clusterer_stays_multimodal() -> None:
    """The case that started this work, asserted as representation not accuracy.

    Two independent diarizers say one speaker; a clustering-derived source says five. The
    posterior must keep both and name their backers rather than collapsing to either. The
    spec deliberately does not require deciding which is right — that needs ground truth
    this recording does not have.
    """
    p = speaker_count_posterior(
        [
            SourceCountClaim("pyannote", 1, support=1.0),
            SourceCountClaim("sortformer", 1, support=1.0),
            SourceCountClaim("embedding_silhouette", 5, support=0.5),
        ],
        gates=GATES,
    )
    assert p.is_multimodal is True
    assert set(p.probabilities) == {1, 5}
    assert p.support[5] == ["embedding_silhouette"]


def test_a_less_supported_source_carries_less_weight() -> None:
    """Authority follows measured support, not a declared kind.

    A source whose speakers sit where no voice detector reports speech is attenuated; one
    whose claims the audio backs is not. Its claim survives in the posterior either way,
    without dominating.
    """
    p = speaker_count_posterior(
        [
            SourceCountClaim("pyannote", 1, support=1.0),
            SourceCountClaim("embedding_silhouette", 5, support=0.3),
        ],
        gates=GATES,
    )
    assert p.probabilities[1] > p.probabilities[5]
    assert p.weights["embedding_silhouette"] < p.weights["pyannote"]


def test_an_uncertain_source_carries_less_weight() -> None:
    """A source that does not trust itself moves the posterior less."""
    p = speaker_count_posterior(
        [SourceCountClaim("sure", 1, uncertainty=0.0), SourceCountClaim("unsure", 3, uncertainty=0.9)],
        gates=GATES,
    )
    assert p.probabilities[1] > p.probabilities[3]


# ── agreement and the empty cases (T087, T088, SC-001) ────────────────


def test_unanimous_agreement_concentrates_the_mass() -> None:
    """SC-001: at least 90% on one count when every source agrees."""
    p = speaker_count_posterior([SourceCountClaim(f"d{i}", 1, support=1.0) for i in range(3)], gates=GATES)
    assert p.probabilities[1] == pytest.approx(1.0)
    assert p.is_multimodal is False
    assert p.modal_count == 1


def test_unanimous_agreement_admits_no_second_count() -> None:
    """FR-009: no phantom speaker when everyone agrees."""
    p = speaker_count_posterior([SourceCountClaim("a", 1), SourceCountClaim("b", 1)], gates=GATES)
    assert set(p.probabilities) == {1}


def test_no_claims_means_no_speakers() -> None:
    """The honest reading of "no source reported anybody"."""
    p = speaker_count_posterior([], gates=GATES)
    assert p.probabilities == {0: 1.0}
    assert p.modal_count == 0


def test_a_zero_speaker_claim_is_representable() -> None:
    """Silence is a count, not an absence of one."""
    p = speaker_count_posterior([SourceCountClaim("vad", 0)], gates=GATES)
    assert p.modal_count == 0


def test_all_sources_fully_uncertain_yields_no_confidence() -> None:
    """Total weight zero must not divide by zero or invent a winner."""
    p = speaker_count_posterior(
        [SourceCountClaim("a", 1, uncertainty=1.0), SourceCountClaim("b", 4, uncertainty=1.0)], gates=GATES
    )
    assert sum(p.probabilities.values()) == pytest.approx(1.0)
    assert p.is_multimodal is True


def test_out_of_range_support_rejected() -> None:
    """A support outside [0, 1] would silently invert or inflate a weight."""
    with pytest.raises(ValueError, match="support"):
        speaker_count_posterior([SourceCountClaim("x", 1, support=-0.2)], gates=GATES)


def test_posterior_serializes_with_string_keys() -> None:
    """JSON object keys are strings; counts must survive the round trip."""
    doc = speaker_count_posterior([SourceCountClaim("a", 1), SourceCountClaim("b", 4)], gates=GATES).to_json()
    assert set(doc["probabilities"]) == {"1", "4"}
    assert doc["modal_count"] == 1 or doc["modal_count"] == 4


# ── hypotheses (T090, T092, FR-004 / FR-007) ──────────────────────────


def _hyp(**kw: object) -> SpeakerHypothesis:
    base = {
        "speaker_id": "S0",
        "existence_uncertainty": 0.2,
        "supporting_sources": ["pyannote"],
        "source_support": {"pyannote": 1.0},
    }
    base.update(kw)
    return SpeakerHypothesis(**base)  # type: ignore[arg-type]


def test_existence_uncertainty_is_separate_from_speech_presence_uncertainty() -> None:
    """FR-004: "might not exist" and "unsure where they spoke" call for different follow-up.

    One number cannot say which is meant, so they are distinct fields on distinct objects.
    """
    hyp = _hyp(existence_uncertainty=0.05)
    track = PerSpeakerPresenceTrack(
        speaker_id="S0", start=0.0, end=0.5, speech_presence_confidence=0.4, speech_presence_uncertainty=0.9
    )
    assert hyp.existence_uncertainty < 0.1
    assert track.speech_presence_uncertainty is not None and track.speech_presence_uncertainty > 0.8


def test_hypothesis_reports_whether_any_independent_source_backs_it() -> None:
    """A hypothesis resting only on derived sources is not wrong, but must be visible."""
    assert _hyp().has_supported_evidence is True
    derived_only = _hyp(supporting_sources=["emb"], source_support={"emb": 0.1})
    assert derived_only.has_supported_evidence is False


def test_hypothesis_serializes_every_contract_field() -> None:
    """contracts/speaker-speaker.md - every field a consumer reads."""
    doc = _hyp().to_json()
    for key in (
        "speaker_id",
        "existence_uncertainty",
        "supporting_sources",
        "source_support",
        "has_supported_evidence",
        "converged",
        "revisions",
    ):
        assert key in doc, f"missing {key}"


# ── speech_presence tracks (T089, FR-003) ────────────────────────────────────


def test_overlapping_speakers_are_both_present() -> None:
    """Two people can talk at once; the representation must permit it."""
    a = PerSpeakerPresenceTrack("S0", 1.0, 1.5, 0.9, 0.1, overlap_with=["S1"])
    b = PerSpeakerPresenceTrack("S1", 1.0, 1.5, 0.8, 0.2, overlap_with=["S0"])
    assert a.overlap_with == ["S1"] and b.overlap_with == ["S0"]


def test_a_gap_is_a_null_confidence_row_not_an_absent_row() -> None:
    """SC-003 requires full-duration coverage, so silence is represented explicitly."""
    row = PerSpeakerPresenceTrack("S0", 2.0, 2.5, None, None).to_row()
    assert row["speech_presence_confidence"] is None
    assert row["start"] == pytest.approx(2.0)


def test_speech_presence_row_carries_every_contract_column() -> None:
    """final/per_speaker_presence.parquet columns."""
    row = PerSpeakerPresenceTrack("S0", 0.0, 0.5, 0.9, 0.1).to_row()
    for col in (
        "speaker_id",
        "start",
        "end",
        "speech_presence_confidence",
        "speech_presence_uncertainty",
        "overlap_with",
        "contributing_sources",
        "round",
        "resolution_kind",
    ):
        assert col in row, f"missing {col}"


# ── label correspondence (T091, FR-005) ───────────────────────────────


def test_correspondence_maps_unrelated_naming_conventions_to_one_hypothesis() -> None:
    """`SPEAKER_00` and `speaker_2` can be the same person; the mapping must be auditable."""
    a = SourceLabelCorrespondence("pyannote", "SPEAKER_00", "S0", 1.0, cluster_id="c0")
    b = SourceLabelCorrespondence("sortformer", "speaker_2", "S0", 1.0, cluster_id="c0")
    assert a.speaker_id == b.speaker_id
    assert a.cluster_id == b.cluster_id


def test_correspondence_records_the_measured_support() -> None:
    """So a consumer can weight a mapping by how far the audio backed its source."""
    doc = SourceLabelCorrespondence("emb", "k3", "S1", 0.3).to_json()
    assert doc["source_support"] == pytest.approx(0.3)


# ── source-kind classification is a policy decision, not a constant ────


def _policy(**influence: object) -> dict:
    return {"influence": {"source_kinds": {"embedding_silhouette": "derived"}, **influence}}


def test_source_kind_is_read_from_policy_not_hardcoded() -> None:
    """The classification is a judgement about pipeline wiring, so it must be arguable.

    The same clustering component would be independent in a pipeline that did not also use
    its embeddings to harmonise other sources' labels.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    assert source_kind_for("embedding_silhouette", _policy()) == "derived"
    assert source_kind_for("embedding_silhouette", {"influence": {"source_kinds": {}}}) == "independent"


def test_versioned_source_ids_resolve_to_the_same_kind() -> None:
    """Sources are emitted as ``embedding_silhouette/<model>``, so the prefix must match."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    assert source_kind_for("embedding_silhouette/speechbrain-ecapa", _policy()) == "derived"


def test_undeclared_sources_default_to_independent() -> None:
    """A new diarizer is an independent observer unless something says otherwise."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    assert source_kind_for("pyannote/speaker-diarization-3.1", _policy()) == "independent"


def test_policy_can_reclassify_the_clusterer_as_independent() -> None:
    """The escape hatch that matters.

    On one validation recording the clusterer reported five speakers where two
    "independent" diarizers reported one, and re-examination suggested it was the closer
    answer — so the gate may suppress correct results. An operator must be able to say so
    without editing code.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    pol = {"influence": {"source_kinds": {"embedding_silhouette": "independent"}}}
    assert source_kind_for("embedding_silhouette", pol) == "independent"


def test_reclassifying_changes_the_posterior() -> None:
    """The classification is load-bearing, so its effect is asserted rather than assumed."""
    claims_derived = [
        SourceCountClaim("pyannote", 1, support=1.0),
        SourceCountClaim("embedding_silhouette", 5, support=0.3),
    ]
    claims_independent = [
        SourceCountClaim("pyannote", 1, support=1.0),
        SourceCountClaim("embedding_silhouette", 5, support=1.0),
    ]
    gated = speaker_count_posterior(claims_derived, gates=GATES)
    equal = speaker_count_posterior(claims_independent, gates=GATES)
    assert gated.probabilities[5] < equal.probabilities[5]
    assert equal.probabilities[5] == pytest.approx(0.5), "as peers the two counts tie"


def test_unknown_declared_kind_is_rejected() -> None:
    """A typo in policy must fail loudly rather than silently defaulting."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import source_kind_for

    with pytest.raises(ValueError, match="unknown source kind"):
        source_kind_for("x", {"influence": {"source_kinds": {"x": "peer"}}})


# ── perturbation-derived reliability ──────────────────────────────────
#
# Preferred over hand-assigning a gate. A hand-set constant encodes a judgement about
# pipeline wiring; a perturbation measures what the source actually does. The pipeline
# already generates the evidence -- the raw and enhanced passes are the same recording
# under a transform, and the gain sweep is another axis -- it simply was not being used
# to weight anything.


def _ev(source: str, **answers: object):  # noqa: ANN202 — test helper
    from senselab.audio.workflows.audio_analysis.speaker_identity import PerturbationEvidence

    return PerturbationEvidence(source=source, answers=dict(answers))


def test_a_stable_source_has_zero_measured_uncertainty() -> None:
    """Same answer under every perturbation is the definition of having earned confidence."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import perturbation_uncertainty

    assert perturbation_uncertainty(_ev("s", raw=4, enhanced=4, gain_up=4)) == pytest.approx(0.0)


def test_a_source_that_flips_under_preprocessing_is_maximally_uncertain() -> None:
    """Preprocessing is a perturbation; disagreeing with yourself across it is evidence.

    A recording that is genuinely hard for diarizers is one where off-the-shelf models
    disagree with *themselves* between raw and enhanced audio — that instability is the
    finding, not noise to be smoothed away by a constant.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import perturbation_uncertainty

    assert perturbation_uncertainty(_ev("s", raw=1, enhanced=3)) == pytest.approx(1.0)


def test_partial_instability_lands_between() -> None:
    """Entropy rather than a modal fraction: how the disagreement spreads matters."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import perturbation_uncertainty

    value = perturbation_uncertainty(_ev("s", raw=4, enhanced=4, gain_up=4, band_limited=1))
    assert value is not None and 0.0 < value < 1.0


def test_a_single_observation_yields_no_stability_evidence() -> None:
    """Reporting 0 there would award full confidence for having been asked once."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import perturbation_uncertainty

    assert perturbation_uncertainty(_ev("s", raw=4)) is None


def test_claims_take_the_modal_answer_with_measured_uncertainty() -> None:
    """The claim is the modal answer; the weight is how stable that answer was."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import claims_from_perturbations

    claims = claims_from_perturbations([_ev("d", raw=4, enhanced=4, gain_up=1)])
    assert claims[0].count == 4
    assert 0.0 < claims[0].uncertainty < 1.0


def test_a_single_observation_source_is_neither_trusted_nor_discarded() -> None:
    """One perturbation point is no evidence either way."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import claims_from_perturbations

    claims = claims_from_perturbations([_ev("d", raw=4)], fallback_uncertainty=0.5)
    assert claims[0].uncertainty == pytest.approx(0.5)


def test_measured_stability_can_outweigh_weaker_support() -> None:
    """The point of the change, on the case that motivated it.

    A clusterer that answers identically on raw and enhanced audio, against diarizers that
    each flip between the two, ends up carrying more weight than its ``derived`` label
    alone would give it. Evidence decides, not the label — which matters because the
    recording that provoked this is exactly the kind where off-the-shelf models disagree
    with themselves across preprocessing.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import claims_from_perturbations

    claims = claims_from_perturbations(
        [
            _ev("pyannote", raw=1, enhanced=3),  # flips under preprocessing
            _ev("sortformer", raw=1, enhanced=2),  # also flips
            _ev("embedding_silhouette", raw=5, enhanced=5),  # stable
        ],
        support={"pyannote": 1.0, "sortformer": 1.0, "embedding_silhouette": 0.4},
    )
    posterior = speaker_count_posterior(claims, gates=GATES)
    by_source = {c.source: c for c in claims}
    assert by_source["embedding_silhouette"].uncertainty == pytest.approx(0.0)
    assert by_source["pyannote"].uncertainty == pytest.approx(1.0)
    assert posterior.weights["embedding_silhouette"] > posterior.weights["pyannote"], (
        "a stable but less-supported source must be able to outweigh an unstable well-supported one"
    )
    assert posterior.modal_count == 5


def test_a_better_supported_source_wins_when_stability_is_equal() -> None:
    """Support is the secondary term, not a discarded one: with equal stability it decides."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import claims_from_perturbations

    claims = claims_from_perturbations(
        [_ev("pyannote", raw=1, enhanced=1), _ev("embedding_silhouette", raw=5, enhanced=5)],
        support={"pyannote": 1.0, "embedding_silhouette": 0.4},
    )
    posterior = speaker_count_posterior(claims, gates=GATES)
    assert posterior.probabilities[1] > posterior.probabilities[5]


# ── deriving speaker from a run's passes ─────────────────────────────


def _diar(count: int) -> dict:
    from types import SimpleNamespace

    segs = [SimpleNamespace(start=float(i), end=float(i + 1), speaker=f"SPEAKER_{i:02d}") for i in range(count)]
    return {"status": "ok", "result": [segs]}


def _passes(**per_pass: dict) -> dict:
    return {label: {"diarization": {"by_model": models}} for label, models in per_pass.items()}


def test_evidence_uses_the_two_passes_as_perturbation_points() -> None:
    """The raw and enhanced passes are the same recording under a transform.

    No extra inference is needed to measure stability — the pipeline already ran the
    diarizers twice on transformed versions of the same audio.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import evidence_from_passes

    ev = evidence_from_passes(_passes(raw_16k={"pyannote": _diar(1)}, enhanced_16k={"pyannote": _diar(3)}))
    assert len(ev) == 1
    assert ev[0].answers == {"raw_16k": 1, "enhanced_16k": 3}


def test_a_diarizer_stable_across_enhancement_gets_full_weight() -> None:
    """A stable diarizer's count is taken at full weight."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    posterior, _h, _c = build_speaker_identity(
        _passes(raw_16k={"pyannote": _diar(2)}, enhanced_16k={"pyannote": _diar(2)})
    )
    assert posterior.modal_count == 2
    assert posterior.probabilities[2] == pytest.approx(1.0)


def test_a_diarizer_that_flips_under_enhancement_is_attenuated() -> None:
    """Its answer is not robust on this recording, and the posterior should say so."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    posterior, _h, _c = build_speaker_identity(
        _passes(
            raw_16k={"stable": _diar(1), "flipper": _diar(4)},
            enhanced_16k={"stable": _diar(1), "flipper": _diar(2)},
        )
    )
    assert posterior.weights["stable"] > posterior.weights["flipper"]
    assert posterior.modal_count == 1


def test_hypotheses_inherit_the_doubt_when_sources_are_split() -> None:
    """A split posterior must not produce a confident-looking speaker list.

    The doubt lands on the speakers it is actually about. With sources split between 1 and 3
    speakers, nobody disputes that *someone* is there, so the first hypothesis is certain —
    but the second and third exist only under the 3-speaker reading and carry the whole
    disagreement. Spreading a flat off-modal value over every hypothesis would both
    overstate the doubt about the first speaker and understate which speaker to check.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    posterior, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1), "b": _diar(3)}, enhanced_16k={"a": _diar(1), "b": _diar(3)})
    )
    assert posterior.is_multimodal is True
    assert hyps[0].existence_uncertainty == 0.0
    assert all(h.existence_uncertainty > 0.0 for h in hyps[1:])
    assert all(h.converged is False for h in hyps)


def test_one_hypothesis_per_speaker_in_the_modal_count() -> None:
    """The modal count determines how many hypotheses exist."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(_passes(raw_16k={"a": _diar(3)}, enhanced_16k={"a": _diar(3)}))
    assert [h.speaker_id for h in hyps] == ["S0", "S1", "S2"]


def test_no_diarization_yields_zero_speakers_not_an_invented_one() -> None:
    """No source reported anybody, so nobody is reported."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    posterior, hyps, _c = build_speaker_identity({"raw_16k": {"diarization": {"by_model": {}}}})
    assert posterior.modal_count == 0
    assert hyps == []


def test_a_failed_diarizer_contributes_nothing() -> None:
    """A failed outcome must not be read as a speaker count of zero."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import evidence_from_passes

    passes = {"raw_16k": {"diarization": {"by_model": {"broken": {"status": "failed"}}}}}
    assert evidence_from_passes(passes) == [], "a failed outcome must not be read as a count"


# ── per-speaker structure fed by the harvested speaker evidence (T098) ──


def _votes(*per_bucket: dict[str, str]) -> list[dict]:
    """Harvested speaker buckets: one dict of ``{diar_model: cluster_id}`` per 0.5 s."""
    out = []
    for i, clusters in enumerate(per_bucket):
        votes: dict[str, object] = {
            m: {"speaker_label": f"{m}-{c}", "cluster_id": c, "speaker_changed_from_prev": None}
            for m, c in clusters.items()
        }
        votes["__cross_diar_label_disagreement__"] = {"cluster_ids": dict(clusters)}
        out.append({"start": i * 0.5, "end": (i + 1) * 0.5, "votes": votes})
    return out


def test_the_nth_speaker_inherits_the_doubt_about_there_being_n_speakers() -> None:
    """FR-004: existence uncertainty must differ *between* speakers, not be a shared scalar.

    With mass split between 1 and 3 speakers, the first speaker is near-certain — every
    source agrees someone is there — while the third exists only under the 3-speaker
    reading. A flat off-modal value would report the same doubt for both and give a
    consumer no way to know which speaker to go looking for.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1), "b": _diar(3)}, enhanced_16k={"a": _diar(1), "b": _diar(3)}),
        speaker_votes=_votes({"a": "Sx"}, {"a": "Sx", "b": "Sy"}, {"b": "Sz"}),
    )
    assert hyps[0].existence_uncertainty < hyps[-1].existence_uncertainty


def test_clusters_the_count_posterior_does_not_back_still_get_a_hypothesis() -> None:
    """Observed-but-unbacked speakers are contested, not deleted.

    Truncating to the modal count would drop evidence the run actually gathered, leaving
    no record that a source separated more speakers than the posterior believes.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1)}, enhanced_16k={"a": _diar(1)}),
        speaker_votes=_votes({"a": "Sx"}, {"a": "Sy"}, {"a": "Sz"}),
    )
    assert len(hyps) == 3
    assert hyps[0].existence_uncertainty < hyps[2].existence_uncertainty


def test_a_speaker_carries_when_it_was_active() -> None:
    """Without spans, a hypothesis cannot be checked against the audio it came from."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1)}, enhanced_16k={"a": _diar(1)}),
        speaker_votes=_votes({"a": "Sx"}, {"a": "Sx"}, {"a": "Sx"}),
    )
    assert (hyps[0].first_seen, hyps[0].last_seen, hyps[0].total_active_s) == (0.0, 1.5, 1.5)


def test_correspondence_names_the_real_diarizer_labels_when_evidence_is_available() -> None:
    """FR-005: a placeholder like ``<a:count=1>`` cannot be traced back to any output."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, _h, corr = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1)}, enhanced_16k={"a": _diar(1)}),
        speaker_votes=_votes({"a": "Sx"}),
    )
    assert [(c.source, c.source_label, c.speaker_id, c.cluster_id) for c in corr] == [("a", "a-Sx", "S0", "Sx")]


def test_the_builder_still_works_with_no_speaker_evidence() -> None:
    """The count posterior comes from the passes alone; harvested votes only add detail."""
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    posterior, hyps, _c = build_speaker_identity(_passes(raw_16k={"a": _diar(2)}, enhanced_16k={"a": _diar(2)}))
    assert posterior.modal_count == 2 and len(hyps) == 2


# ── attribution and convergence per hypothesis (found on a real run) ──


def test_a_hypothesis_names_the_sources_whose_labels_landed_in_it() -> None:
    """Attribution must be per speaker, not the modal-count supporters copied everywhere.

    On a real recording the two real diarizers each contributed one label while a derived
    clusterer over-split into five. Copying the modal supporters onto every hypothesis
    credited pyannote and sortformer with four speakers neither of them ever reported.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"real": _diar(1), "derived": _diar(2)}, enhanced_16k={"real": _diar(1), "derived": _diar(2)}),
        speaker_votes=_votes({"real": "Cx", "derived": "Cx"}, {"derived": "Cy"}),
    )
    assert hyps[0].supporting_sources == ["derived", "real"]
    assert hyps[1].supporting_sources == ["derived"]


def test_a_speaker_resting_on_unsupported_claims_says_so() -> None:
    """Unsupported backing is visible per speaker.

    A speaker whose only backer made claims the audio does not carry must read differently
    from one two well-supported diarizers both heard — and the number says so, not a label.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"real": _diar(1)}, enhanced_16k={"real": _diar(1)}),
        speaker_votes=_votes({"real": "Cx"}, {"clusterer": "Cy"}),
        support={"real": 1.0, "clusterer": 0.2},
    )
    assert hyps[1].source_support == {"clusterer": 0.2}
    assert hyps[1].has_supported_evidence is False
    assert hyps[0].has_supported_evidence is True


def test_a_speaker_that_might_not_exist_is_not_reported_as_converged() -> None:
    """Convergence is per speaker, not per run.

    A run can settle on "one speaker" while a surplus hypothesis stays maximally doubtful.
    Reporting that one as converged tells a consumer the question is closed when it is the
    single most open thing in the output.
    """
    from senselab.audio.workflows.audio_analysis.speaker_identity import build_speaker_identity

    _p, hyps, _c = build_speaker_identity(
        _passes(raw_16k={"a": _diar(1)}, enhanced_16k={"a": _diar(1)}),
        speaker_votes=_votes({"a": "Cx"}, {"a": "Cy"}),
    )
    assert hyps[0].converged is True
    assert hyps[1].converged is False


# The packaged-policy gate test lived here. The posterior no longer resolves a source's
# authority from policy: authority is stability x measured support, and no source is named.
# ``source_kind_for`` survives for the adaptive loop's influence gates, which govern how far
# one signal may *revise* another, and is tested above.
