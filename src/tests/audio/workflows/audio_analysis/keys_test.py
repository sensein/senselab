"""The key algebra: what a measurement is, where it came from, and where it lands.

Signals and derivatives share one key space (D-22): a signal key is the degenerate derivative key,
``(Target, Producer, Source)``, where a signal's producer is a model and its source is a route, and a
derivative's producer is an operator and its source is the keys it consumed.

Three things the keying has to make computable rather than assumed, and each has tests below:

- **cross-tool disagreement** — every voter on one target shares a first element, so gathering them
  needs no string matching over ``"::"``-joined names;
- **what a difference means** — same pathway is a fold, different pathway is a compose (D-23);
- **shared evidence** — two inputs to one axis whose source closures intersect are the same evidence
  twice, and the fold must weigh that rather than count both (D-21 rule 6).
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.keys import (
    Arity,
    DerivativeKey,
    EstimateKey,
    Operator,
    Route,
    SignalKey,
)

BROUHAHA_SNR = SignalKey(target="snr", producer="pyannote/brouhaha", route=Route())
BROUHAHA_C50 = SignalKey(target="c50", producer="pyannote/brouhaha", route=Route())
AST_LABELS = SignalKey(target="scene_labels", producer="MIT/ast-finetuned-audioset", route=Route())
YAMNET_LABELS = SignalKey(target="scene_labels", producer="yamnet", route=Route())


# ── routes ─────────────────────────────────────────────────────────────


def test_the_default_route_is_the_direct_pathway_unperturbed() -> None:
    """What used to be spelled ``raw``. Two dimensions, both defaulted, neither special-cased."""
    assert Route() == Route(pathway="direct", perturbation="identity")


def test_the_identity_route_gets_no_special_spelling() -> None:
    """``raw`` as a distinct name is how ``raw_vs_enhanced`` came to masquerade as a third pass.

    Every route spells both dimensions, so a reader cannot mistake a comparison for a member.
    """
    assert Route().path == "direct/identity"
    assert Route(perturbation="enhanced").path == "direct/enhanced"
    assert Route(pathway="background").path == "background/identity"


def test_the_route_is_two_path_levels_so_the_joiner_cannot_be_ambiguous() -> None:
    """Joining with ``__`` — what ``slug`` uses for ``/`` — would let two routes collide.

    ``a_b`` + ``c`` and ``a`` + ``b_c`` join identically under a ``__`` joiner. The filesystem
    separates the dimensions for free, and each becomes globbable on its own.
    """
    assert Route(pathway="a_b", perturbation="c").path != Route(pathway="a", perturbation="b_c").path


def test_a_pathway_change_and_a_perturbation_change_are_distinguishable() -> None:
    """The distinction D-23 turns on: only one of the two is a stability sample."""
    assert Route().same_pathway_as(Route(perturbation="enhanced")) is True
    assert Route().same_pathway_as(Route(pathway="background")) is False


# ── signal keys ────────────────────────────────────────────────────────


def test_tools_measuring_one_target_share_a_first_element() -> None:
    """The load-bearing property of target-first keying (D-20).

    Under the old ``(family, *instances)`` keying these lived under ``frame_*``, ``diarization`` and
    a model name, and only a ``"::"``-style selector could gather them.
    """
    assert AST_LABELS.target == YAMNET_LABELS.target
    assert BROUHAHA_SNR.target != BROUHAHA_C50.target, "one forward pass, two targets, two signals"


def test_a_signal_path_is_derived_from_its_key() -> None:
    """Paths are built from keys, never parsed back out of them."""
    assert BROUHAHA_SNR.relative_path(".parquet") == "L1/signals/snr/pyannote__brouhaha/direct/identity.parquet"


def test_a_producer_id_that_could_collide_after_slugging_is_refused() -> None:
    """Two producers landing in one file would read as one producer measured twice.

    Refused at construction rather than escaped, so the segment stays readable — an escape encoding
    would make every path unreadable to buy safety against an id no real model has.
    """
    ok = SignalKey(target="transcript", producer="nvidia/canary-qwen-2.5b", route=Route())
    assert ok.relative_path(".json") == "L1/signals/transcript/nvidia__canary-qwen-2.5b/direct/identity.json"
    with pytest.raises(ValueError, match="__"):
        SignalKey(target="transcript", producer="nvidia__canary-qwen-2.5b", route=Route())


def test_the_collision_is_refused_where_the_key_is_made_not_where_a_file_is_written() -> None:
    """A key that cannot be stored unambiguously should not exist, even unstored.

    Validating in ``relative_path`` would let a colliding key be built, compared, put in a source
    list and used for an overlap test, failing only at the moment bytes were about to land.
    """
    with pytest.raises(ValueError, match="__"):
        Route(pathway="a__b")
    with pytest.raises(ValueError, match="__"):
        Operator("project__labels")


# ── derivative arity, derived rather than declared ─────────────────────


def test_one_source_is_a_projection() -> None:
    """Tool and route survive into the key, so the result is still that tool's measurement."""
    d = DerivativeKey(target="speech", operator=Operator("project_labels", "speech_v3"), sources=(AST_LABELS,))
    assert d.arity is Arity.PROJECT


def test_sources_sharing_a_target_are_a_fold() -> None:
    """Inputs answering the same question, so a spread across them is meaningful."""
    d = DerivativeKey(target="scene_labels", operator=Operator("agree"), sources=(AST_LABELS, YAMNET_LABELS))
    assert d.arity is Arity.FOLD


def test_sources_with_different_targets_are_a_compose() -> None:
    """Different quantities. A spread across them measures nothing — the ``units: mixed`` defect."""
    d = DerivativeKey(target="target_free", operator=Operator("mask", "speech"), sources=(BROUHAHA_SNR, AST_LABELS))
    assert d.arity is Arity.COMPOSE


def test_a_compose_refuses_to_report_a_spread() -> None:
    """D-21 rule 3, enforced where the arity is known rather than trusted downstream."""
    d = DerivativeKey(target="target_free", operator=Operator("mask"), sources=(BROUHAHA_SNR, AST_LABELS))
    assert d.spread_is_meaningful is False
    fold = DerivativeKey(target="scene_labels", operator=Operator("agree"), sources=(AST_LABELS, YAMNET_LABELS))
    assert fold.spread_is_meaningful is True


def test_a_derivative_with_no_source_is_refused() -> None:
    """A derivative of nothing is a measurement pretending to be a derivation."""
    with pytest.raises(ValueError, match="source"):
        DerivativeKey(target="speech", operator=Operator("invent"), sources=())


# ── the arity decides what a cross-route difference means (D-23) ───────


def test_a_fold_across_perturbations_within_a_pathway_is_a_stability_sample() -> None:
    """One question answered twice, so |Δ| is instability and sets the fusion weight."""
    d = DerivativeKey(
        target="snr",
        operator=Operator("mean_abs_delta"),
        sources=(BROUHAHA_SNR, SignalKey("snr", "pyannote/brouhaha", Route(perturbation="enhanced"))),
    )
    assert d.arity is Arity.FOLD
    assert d.folds_within_one_pathway is True


def test_a_fold_across_pathways_is_not_a_stability_sample() -> None:
    """Different primary target, so the difference is complementary rather than corroborative.

    Reading it as instability down-weights the signal that noticed the background speaker.
    """
    d = DerivativeKey(
        target="snr",
        operator=Operator("mean_abs_delta"),
        sources=(BROUHAHA_SNR, SignalKey("snr", "pyannote/brouhaha", Route(pathway="background"))),
    )
    assert d.arity is Arity.FOLD, "same target, so structurally a fold"
    assert d.folds_within_one_pathway is False, "but not a stability sample"


# ── shared evidence (D-21 rule 6) ──────────────────────────────────────


def test_a_derivative_shares_evidence_with_the_signal_it_came_from() -> None:
    """The hazard the merged input pool creates: a projection and its source both voting."""
    projected = DerivativeKey(target="speech", operator=Operator("project_labels", "v3"), sources=(AST_LABELS,))
    assert projected.shares_evidence_with(AST_LABELS) is True


def test_two_derivatives_of_different_signals_do_not_share_evidence() -> None:
    """Sharing only the recording is not evidence-sharing — every pair in a run shares that."""
    a = DerivativeKey(target="speech", operator=Operator("project_labels", "v3"), sources=(AST_LABELS,))
    b = DerivativeKey(target="speech", operator=Operator("resample", "mean"), sources=(BROUHAHA_SNR,))
    assert a.shares_evidence_with(b) is False


def test_the_closure_is_transitive_through_derivatives() -> None:
    """A consensus built from transcripts is not an independent voter against them."""
    once = DerivativeKey(target="speech", operator=Operator("project_labels", "v3"), sources=(AST_LABELS,))
    twice = DerivativeKey(target="speech", operator=Operator("smooth"), sources=(once,))
    assert twice.shares_evidence_with(AST_LABELS) is True
    assert twice.source_closure() == frozenset({AST_LABELS})


def test_two_targets_from_one_model_are_a_different_overlap_than_shared_evidence() -> None:
    """Brouhaha's SNR and C50 come from one forward pass but are not the same measurement.

    Correlated through a shared trunk, so a fold should know — but not the same evidence twice, and
    collapsing the two notions would either double-discount or miss the correlation entirely. Same
    reason the aligner case needs provenance comparison rather than closure intersection.
    """
    snr = DerivativeKey(target="snr", operator=Operator("resample", "mean"), sources=(BROUHAHA_SNR,))
    c50 = DerivativeKey(target="c50", operator=Operator("resample", "mean"), sources=(BROUHAHA_C50,))
    assert snr.shares_evidence_with(c50) is False
    assert snr.shares_producer_with(c50) is True


# ── estimates in a source: the coupling edge (D-21 rule 5) ─────────────


def test_an_estimate_from_an_earlier_round_may_be_a_source() -> None:
    """The coupling channel, visible in the key rather than inferable from a function name."""
    d = DerivativeKey(
        target="target_free",
        operator=Operator("mask", "speech"),
        sources=(BROUHAHA_SNR, EstimateKey(axis="speech_presence", round=0)),
        round=1,
    )
    assert d.arity is Arity.COMPOSE


def test_an_estimate_from_the_same_round_is_refused() -> None:
    """Strictness is what keeps the round DAG acyclic: ``estimates[<n]``, never ``estimates[n]``."""
    with pytest.raises(ValueError, match="round"):
        DerivativeKey(
            target="target_free",
            operator=Operator("mask"),
            sources=(EstimateKey(axis="speech_presence", round=1),),
            round=1,
        )


def test_an_estimate_contributes_no_signal_to_the_closure() -> None:
    """An axis is a fold over everything; treating it as evidence would make the closure universal.

    Recorded as a limit rather than papered over: a derivative conditioned on an axis shares evidence
    with almost anything, so the closure test does not speak to that edge and the round index does.
    """
    d = DerivativeKey(
        target="target_free",
        operator=Operator("mask"),
        sources=(EstimateKey(axis="speech_presence", round=0),),
        round=1,
    )
    assert d.source_closure() == frozenset()


# ── estimate keys ──────────────────────────────────────────────────────


def test_an_estimate_key_has_no_producer_and_no_route() -> None:
    """An axis aggregates across both, so neither can index its output (D-16)."""
    e = EstimateKey(axis="speaker", round=2)
    assert not hasattr(e, "producer")
    assert not hasattr(e, "route")
    assert e.relative_path(".parquet") == "L2/round/2/estimates/speaker.parquet"


# ── derivative paths ───────────────────────────────────────────────────


def test_a_projection_path_keeps_the_source_producer_and_route() -> None:
    """Dropping either is the reduction D-18 found: a value describing something else."""
    d = DerivativeKey(target="speech", operator=Operator("project_labels", "speech_v3"), sources=(AST_LABELS,), round=0)
    assert d.relative_path(".parquet") == (
        "L2/round/0/derivatives/speech/project_labels__speech_v3/MIT__ast-finetuned-audioset/direct/identity.parquet"
    )


def test_a_fold_collapses_its_sources_to_the_operator_and_materialises_them_as_columns() -> None:
    """An unbounded source list cannot go in a path, so the key rule puts it in the rows."""
    d = DerivativeKey(target="scene_labels", operator=Operator("agree"), sources=(AST_LABELS, YAMNET_LABELS), round=0)
    assert d.relative_path(".parquet") == "L2/round/0/derivatives/scene_labels/agree.parquet"
    assert d.required_columns == ("contributing_producers", "contributing_routes")
