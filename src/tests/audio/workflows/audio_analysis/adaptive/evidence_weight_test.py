"""Attenuation, not erasure: a non-corroborated vote keeps aggregating at reduced weight.

These tests hold the line the belief store crossed when it had a purge path. Purging removed a
source's payload from ``active_votes``, so ``reaggregate_bucket`` never saw it — and it did that on
the weakest signal in the system, non-corroboration between two sources, applied asymmetrically
(presence indicted ASR, never the reverse) though word boundaries are the finer measurement. A
quiet, distant or overlapped speaker produces exactly that signature.

Every attenuation mechanism here is floored on one stated ground: the dissenter may be the only
source that noticed something.
"""

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, Vote, VoteStore, bucket_key
from senselab.audio.workflows.audio_analysis.aggregate import speech_presence_p_voice
from senselab.audio.workflows.audio_analysis.floors import MIN_EVIDENCE_WEIGHT
from senselab.audio.workflows.audio_analysis.support import (
    bucket_corroboration,
    evidence_weight_from_corroboration,
)

BK = bucket_key(0.0, 0.5)
STREAM = "raw"


def _frame_vote(bucket: tuple[float, float], p_speech: float) -> Vote:
    """A frame posterior, linked the way ``speech_presence_link._link_frame`` links it.

    ``native_confidence`` is confidence in the voter's own direction, so a low ``p_speech`` is a
    *high*-confidence "no". Building the fixture any other way tests an inversion, not the rule.
    """
    speaks = p_speech >= 0.5
    return Vote(
        axis="speech_presence",
        bucket=bucket,
        source="frame_brouhaha_vad",
        stream=STREAM,
        scope="file",
        round=1,
        payload={
            "speaks": speaks,
            "native_confidence": p_speech if speaks else 1.0 - p_speech,
            "frame_mean": p_speech,
        },
    )


def _store_with_lone_asr_claim(*, frame_p: float = 0.02) -> VoteStore:
    """One ASR asserting speech where the only independent voter reports near-silence."""
    store = VoteStore()
    store.add_vote(_frame_vote(BK, frame_p))
    store.add_vote(
        Vote(
            axis="speech_presence",
            bucket=BK,
            source="openai/whisper-large-v3",
            stream=STREAM,
            scope="file",
            round=1,
            payload={"speaks": True, "native_confidence": 0.3, "word_overlap_s": 0.4},
        )
    )
    return store


def _attenuate(store: VoteStore, corroboration: float, *, round_idx: int = 2) -> list[dict[str, Any]]:
    return store.attenuate_source_in_bucket(
        STREAM,
        BK,
        "openai/whisper-large-v3",
        corroboration=corroboration,
        evidence_sources=["frame_brouhaha_vad"],
        reason="uncorroborated_speech_claim",
        round_idx=round_idx,
        measured_on=("speech_presence", BK),
    )


def test_uncorroborated_vote_still_aggregates() -> None:
    """A quiet or overlapped speaker produces exactly the signature this rule fires on.

    If the vote leaves aggregation, the only source that heard them is deleted and the bucket
    reports confident silence — with nothing in the aggregate to appeal to.
    """
    store = _store_with_lone_asr_claim()
    before = store.reaggregate_bucket("speech_presence", BK, aggregator="min")
    _attenuate(store, 0.02)
    after = store.reaggregate_bucket("speech_presence", BK, aggregator="min")

    assert "openai/whisper-large-v3" in store.active_votes(STREAM, "speech_presence", BK)
    # The claim is still in the fold — p_voice must sit strictly above what the independent
    # voter alone would report, which is what erasure would have collapsed it to.
    lone_evidence = speech_presence_p_voice({"frame_brouhaha_vad": _frame_vote(BK, 0.02).payload})
    assert after["p_voice"] is not None and lone_evidence is not None
    assert after["p_voice"] > lone_evidence
    assert after["p_voice"] < before["p_voice"]  # attenuated, so it no longer carries equal weight


def test_attenuated_weight_never_reaches_zero() -> None:
    """A maximally-uncorroborated source is attenuated, not zeroed.

    ``_weighted_p_voice`` drops voters at ``weight <= 0``, so a zero weight is erasure by another
    name. The floor is the single shared constant, not a literal repeated here.
    """
    store = _store_with_lone_asr_claim()
    records = _attenuate(store, 0.0)
    assert records
    assert records[0]["evidence_weight"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    weights = store.evidence_weights(STREAM, "speech_presence", BK)
    assert weights["openai/whisper-large-v3"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    assert weights["openai/whisper-large-v3"] > 0.0


def test_repeated_attenuation_still_stops_at_the_floor() -> None:
    """Two rules attenuating the same vote must not multiply their way to zero.

    Weights compose, and a product of floored factors is not itself floored — which is how a
    floor silently stops holding once a second rule is added.
    """
    store = _store_with_lone_asr_claim()
    _attenuate(store, 0.0, round_idx=2)
    _attenuate(store, 0.0, round_idx=3)
    weights = store.evidence_weights(STREAM, "speech_presence", BK)
    assert weights["openai/whisper-large-v3"] == pytest.approx(MIN_EVIDENCE_WEIGHT)


def test_weight_equals_measured_corroboration() -> None:
    """The degree of attenuation is the measurement, continuously.

    A fixed multiplier would be a constant nobody measured; the independent evidence for the same
    event already *is* a probability in [0, 1], and that is how far the assertion carries.
    """
    seen = []
    for corroboration in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        store = _store_with_lone_asr_claim()
        _attenuate(store, corroboration)
        seen.append(store.evidence_weights(STREAM, "speech_presence", BK)["openai/whisper-large-v3"])
    assert seen == sorted(seen)  # monotone in the measurement
    assert seen[0] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    assert seen[-1] == pytest.approx(1.0)
    assert seen[3] == pytest.approx(0.5)  # identity above the floor — no unmeasured shaping


def test_zero_floor_is_refused() -> None:
    """A configurable zero floor is purging with extra steps."""
    with pytest.raises(ValueError, match="floor"):
        evidence_weight_from_corroboration(0.4, floor=0.0)


def test_attenuation_does_not_move_its_own_measurement() -> None:
    """The evidence pool excludes claimants, so re-measuring after acting yields the same number.

    If corroboration were read off ``p_voice`` — a weighted mean over *all* presence voters,
    the indicted one included — the source would partly protect itself and attenuating it would
    change the very number that indicted it. That is a same-round feedback path.
    """
    store = _store_with_lone_asr_claim()
    pool = ["frame_brouhaha_vad"]
    before = bucket_corroboration(store.active_votes(STREAM, "speech_presence", BK), evidence_signals=pool)
    _attenuate(store, before or 0.0)
    after = bucket_corroboration(store.active_votes(STREAM, "speech_presence", BK), evidence_signals=pool)
    assert before == after


def test_unmeasured_source_keeps_full_weight() -> None:
    """Absent is not zero: with no independent evidence in the bucket, nothing may be discounted.

    On a run with no ``frame_*``/``acoustic_*`` voter the mechanism must be inert, not universally
    condemning — a missing model must not look like a wrong one.
    """
    store = VoteStore()
    store.add_vote(
        Vote(
            axis="speech_presence",
            bucket=BK,
            source="openai/whisper-large-v3",
            stream=STREAM,
            scope="file",
            round=1,
            payload={"speaks": True, "native_confidence": 0.3},
        )
    )
    votes = store.active_votes(STREAM, "speech_presence", BK)
    assert bucket_corroboration(votes, evidence_signals=[]) is None
    assert store.evidence_weights(STREAM, "speech_presence", BK) == {}
    row = store.reaggregate_bucket("speech_presence", BK, aggregator="min")
    assert row["p_voice"] == pytest.approx(speech_presence_p_voice(votes))
    assert row["attenuated_sources"] == {}


def test_attenuated_source_stays_in_contributing_sources() -> None:
    """The parquet must still show who spoke up, and by how much they were discounted."""
    store = _store_with_lone_asr_claim()
    _attenuate(store, 0.02)
    row = store.reaggregate_bucket("speech_presence", BK, aggregator="min")
    assert "openai/whisper-large-v3" in row["contributing_sources"]
    assert row["attenuated_sources"]["openai/whisper-large-v3"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    state = BeliefState.from_store(store, aggregator="min", round_index=1)
    belief_row = state.axis_rows("speech_presence")[0]
    assert "openai/whisper-large-v3" in belief_row["attenuated_sources"]


def test_provenance_records_what_was_attenuated_and_why() -> None:
    """An audit must be able to re-derive the weight without re-running a model.

    Every factor is appended rather than overwritten: two rules may both have something to say
    about one vote, and an overwrite hides the first.
    """
    store = _store_with_lone_asr_claim()
    _attenuate(store, 0.12, round_idx=2)
    vote = next(v for v in store._votes.values() if v.source == "openai/whisper-large-v3")
    assert vote.status == "active"
    factors = vote.provenance["evidence_weight_factors"]
    assert len(factors) == 1
    factor = factors[0]
    assert factor["reason"] == "uncorroborated_speech_claim"  # observation, never a claimed cause
    assert factor["corroboration"] == pytest.approx(0.12)
    assert factor["corroboration_pooling"] == "max"
    assert factor["weight_map"] == "identity_floored"
    assert factor["evidence_sources"] == ["frame_brouhaha_vad"]
    assert factor["measured_on"] == {"axis": "speech_presence", "bucket": [0.0, 0.5]}
    assert factor["floor"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    assert factor["round"] == 2
    assert vote.to_record()["evidence_weight"] == pytest.approx(0.12)

    _attenuate(store, 0.5, round_idx=3)
    assert len(vote.provenance["evidence_weight_factors"]) == 2  # appended, not replaced


def test_no_purged_status_survives() -> None:
    """``status`` is a filter; anything not "active" vanishes from aggregation.

    Keeping a purge status around invites the next reader to add it back to a
    ``status != "active"`` test, which restores the erasure.
    """
    store = _store_with_lone_asr_claim()
    _attenuate(store, 0.0)
    assert {v.status for v in store._votes.values()} <= {"active", "shadowed"}
    assert not hasattr(VoteStore, "purge_source_in_bucket")


def test_asr_axis_attenuation_reaches_its_one_voter() -> None:
    """A weight must reach the asr axis's voter, and must not erase it.

    This used to check that attenuation reached the pairwise phoneme family, back when the axis
    carried per-model text plus a pairwise distance block and the weight reached two of the three
    sub-signals. There is one voter now — the consensus word fold — so "reaches all three" collapses
    to "reaches it", and the floor still has to keep its dissent visible.

    ``apply_aggregator`` scales a signal's *doubt* by its weight rather than taking a weighted mean,
    so a lone voter's attenuation does move the fold; that is the documented behaviour and the reason
    the floor is not zero.
    """
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis

    buckets = {"raw": [{"start": 0.0, "end": 0.1, "votes": {"consensus_words": {"value": 0.9}}}]}
    unweighted = fuse_axis(buckets, weights={}, aggregator="mean", snr_gate=None)[0]
    attenuated = fuse_axis(
        buckets,
        weights={"consensus_words": MIN_EVIDENCE_WEIGHT},
        aggregator="mean",
        weight_basis={"consensus_words": {"stability": MIN_EVIDENCE_WEIGHT, "support": 1.0}},
        snr_gate=None,
    )[0]
    assert attenuated["triage_score"] < unweighted["triage_score"]
    assert attenuated["triage_score"] > 0.0, "the floor keeps an attenuated voter's dissent visible"
    # The weight and the factors behind it are on the row, so a discounted voter records *why*.
    assert attenuated["signal_weights"] == {"consensus_words": MIN_EVIDENCE_WEIGHT}
    assert attenuated["weight_basis"]["consensus_words"]["stability"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    # ``uncertainty`` is the entropy measure and takes no weight, so it is unmoved — the split the
    # fold exists to keep.
    assert attenuated["uncertainty"] == pytest.approx(unweighted["uncertainty"])


def test_empty_weight_map_is_byte_identical_to_none() -> None:
    """Round-1 re-aggregation must stay bit-identical, or the replay proof proves nothing.

    ``replay_check`` re-aggregates from the persisted votes; if passing an empty
    weight map perturbed the fold, the proof would silently become a comparison of two quantities.
    """
    from senselab.audio.workflows.audio_analysis.fuse import per_signal_uncertainty

    presence = {"a": {"speaks": True, "native_confidence": 0.8}, "b": {"speaks": False, "native_confidence": 0.6}}
    assert speech_presence_p_voice(presence, weights={}) == speech_presence_p_voice(presence)

    # The per-axis folds this used to check are gone — they had no production caller. What the replay
    # proof actually re-derives from is ``per_signal_uncertainty``, which reads the votes and takes no
    # weight map at all: an empty map cannot perturb a function that never receives one, and stating
    # that is the property the deleted assertions were reaching for.
    asr = {"consensus_words": {"value": 0.42}}
    speaker = {"d::e": {"same_label_uncertainty": 0.4, "change_inconsistency_uncertainty": 0.2}}
    assert per_signal_uncertainty({"votes": asr}) == {"consensus_words": 0.42}
    assert per_signal_uncertainty({"votes": speaker}) == {"d::e": 0.4}


@pytest.mark.parametrize(
    ("site", "constant"),
    [
        ("rounds.MIN_REGIONAL_TRUST", "senselab.audio.workflows.audio_analysis.rounds:MIN_REGIONAL_TRUST"),
        ("reliability.MIN_RELIABILITY", "senselab.audio.workflows.audio_analysis.reliability:MIN_RELIABILITY"),
        ("support.SUPPORT_FLOOR", "senselab.audio.workflows.audio_analysis.support:SUPPORT_FLOOR"),
        ("invariance.MIN_INVARIANCE", "senselab.audio.workflows.audio_analysis.invariance:MIN_INVARIANCE"),
        (
            "identity_repair.MIN_WINDOW_WEIGHT",
            "senselab.audio.workflows.audio_analysis.adaptive.identity_repair:MIN_WINDOW_WEIGHT",
        ),
        (
            "speech_to_text_ensemble.MIN_CORROBORATION",
            "senselab.audio.tasks.speech_to_text_ensemble.api:MIN_CORROBORATION",
        ),
    ],
)
def test_one_floor_shared_by_every_withdrawal_site(site: str, constant: str) -> None:
    """Every site applies one argument; every restated literal is how they drift apart.

    Identity, not equality. Two modules that each write ``0.05`` are equal today and are exactly
    the arrangement this test exists to forbid — the value agrees until someone retunes one of
    them, and nothing then reports that the others did not follow.
    """
    from importlib import import_module

    module_path, name = constant.split(":")
    value = getattr(import_module(module_path), name)
    assert value is MIN_EVIDENCE_WEIGHT, (
        f"{site} restates the floor instead of binding to floors.MIN_EVIDENCE_WEIGHT "
        f"(got {value!r}). One argument, one definition."
    )


def test_the_gate_bottoms_out_at_the_shared_floor() -> None:
    """The one site with no constant of its own: influence's gate, floored inside the function."""
    from senselab.audio.workflows.audio_analysis import influence

    assert influence.effective_weight(1.0, uncertainty=1.0, derivation_gate=1.0) == pytest.approx(MIN_EVIDENCE_WEIGHT)


def test_policy_yaml_weight_floors_do_not_drift_from_the_shared_constant() -> None:
    """YAML cannot import a Python constant, so the link is a test or it is nothing.

    Reads the ``adaptive:`` section of ``data/run_config/default.yaml``.

    Both keys floor a *withdrawn weight* — the same quantity as ``MIN_EVIDENCE_WEIGHT``, reached by
    the same argument. They stay literals because an operator has to be able to read and override a
    policy, so what is enforced is the packaged default: ship the shared number, and if someone
    retunes the constant without the YAML (or the reverse) this is what says so.

    Driven off ``policy._WEIGHT_FLOOR_KEYS`` rather than a list repeated here, so a floor added to
    the policy is covered the moment it is registered for validation.
    """
    import yaml

    from senselab.audio.workflows.audio_analysis.adaptive import policy as policy_module
    from senselab.audio.workflows.audio_analysis.run_config import DEFAULT_CONFIG_PATH

    # The policy is the ``adaptive:`` section of the run config now, not a file of its own.
    raw = (yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {})["adaptive"]
    assert policy_module._WEIGHT_FLOOR_KEYS, "no floors registered for validation — the guard is inert"
    for path, key in policy_module._WEIGHT_FLOOR_KEYS:
        node: Any = raw
        for step in path:
            node = node[step]
        dotted = ".".join((*path, key))
        assert float(node[key]) == pytest.approx(MIN_EVIDENCE_WEIGHT), (
            f"policy default {dotted} = {node[key]} has drifted from "
            f"floors.MIN_EVIDENCE_WEIGHT = {MIN_EVIDENCE_WEIGHT}. Both floor a withdrawn weight; "
            "if the argument has genuinely changed for one of them, say so in floors.py."
        )


# ── the rule that measures ───────────────────────────────────────────────


def _p3_ctx() -> dict[str, Any]:
    """Ctx backed by a real VoteStore, as the rule reads it in a run."""
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    store = _store_with_lone_asr_claim()
    # Further buckets so `informative_evidence` has enough observations to judge the frame voter:
    # the pool is derived from the run rather than assumed, and a voter that never reports absence
    # is dropped from it.
    for bk, p in [((0.5, 1.0), 0.95), ((1.0, 1.5), 0.9), ((1.5, 2.0), 0.85)]:
        store.add_vote(_frame_vote(bk, p))
    rows = [
        {"start": s, "end": e, "meta": {"src_speech": 0.1}}
        for (s, e) in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    ]

    class _State:
        def axis_rows(self, axis: str) -> list[dict[str, Any]]:
            return rows if axis == "speech_presence" else []

    return {
        "state": _State(),
        "store": store,
        "policy": load_policy(),
        "passes": [STREAM],
        "asr_model_ids": {STREAM: {"openai/whisper-large-v3"}},
        "round_idx": 2,
    }


def test_p3_stops_firing_once_it_has_measured() -> None:
    """Without an idempotence guard the loop spins.

    Attenuation changes neither ``speaks`` nor corroboration, so the candidate set is stable and
    the rule re-fires every round for zero gain, ``epsilon`` never admits it, and convergence C4
    (``untried_actions``) never reaches zero. ``status`` used to supply this guard implicitly.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import (
        _adjudication_candidates,
        _p3_execute,
    )

    ctx = _p3_ctx()
    assert len(_adjudication_candidates(ctx, STREAM)) == 1
    result = _p3_execute({}, ctx)
    assert len(result["attenuated"]) == 1
    assert result["attenuated"][0]["evidence_weight"] == pytest.approx(MIN_EVIDENCE_WEIGHT)
    ctx["round_idx"] = 3
    assert _adjudication_candidates(ctx, STREAM) == []


def test_p3_needs_measured_evidence_before_it_may_act() -> None:
    """No independent evidence in the bucket ⇒ no candidate. Nothing measured, nothing discounted."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _adjudication_candidates

    ctx = _p3_ctx()
    for vote in list(ctx["store"]._votes.values()):
        if vote.source == "frame_brouhaha_vad":
            vote.status = "shadowed"  # the only pool member stops reporting
    assert _adjudication_candidates(ctx, STREAM) == []


def test_p3_rule_id_names_the_observation_not_a_cause() -> None:
    """A rule id claiming "hallucination" asserts a cause the evidence cannot reach.

    What is observed is non-corroboration, and a quiet or overlapped speaker produces it too.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import RULES

    ids = [r["id"] for r in RULES]
    assert "P3_uncorroborated_speech_attenuation" in ids
    assert not any("hallucination" in i for i in ids)
