"""Unit tests for the adaptive loop's pure parts (tasks.md T036).

Model-free: exercises vote shadowing/purging, region proposal, deterministic
planning + budget, and fusion voting on synthetic inputs.
"""

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.belief import Vote, VoteStore, bucket_key
from senselab.audio.workflows.audio_analysis.adaptive.fusion import fuse_words
from senselab.audio.workflows.audio_analysis.adaptive.policy import (
    BudgetLedger,
    family_weights,
    load_policy,
    plan_round,
)
from senselab.audio.workflows.audio_analysis.adaptive.regions import propose_regions

BK = bucket_key(0.0, 0.5)


def _vote(source: str, scope: str = "file", speaks: bool = True, round_idx: int = 1) -> Vote:
    return Vote(
        axis="presence",
        bucket=BK,
        source=source,
        stream="raw_16k",
        scope=scope,
        round=round_idx,
        payload={"speaks": speaks, "native_confidence": None},
    )


def test_region_scope_shadows_same_source_file_scope() -> None:
    """D5: region vote replaces same (source, stream) file vote; others coexist."""
    store = VoteStore()
    store.add_vote(_vote("model_a"))
    store.add_vote(_vote("model_b"))
    store.add_vote(_vote("model_a", scope="region:r2_x", speaks=False, round_idx=2))
    active = store.active_votes("raw_16k", "presence", BK)
    assert active["model_a"]["speaks"] is False  # region vote won
    assert active["model_b"]["speaks"] is True  # unrelated model untouched
    shadowed = [v for v in store._votes.values() if v.status == "shadowed"]
    assert len(shadowed) == 1 and shadowed[0].source == "model_a" and shadowed[0].scope == "file"


def test_purge_excludes_from_aggregation_but_keeps_row() -> None:
    """C10: purged votes leave aggregation on both axes yet persist for provenance."""
    store = VoteStore()
    store.add_vote(_vote("asr_x"))
    n = store.purge_source_in_bucket("raw_16k", BK, "asr_x", reason="hallucination", round_idx=2)
    assert n == 1
    assert "asr_x" not in store.active_votes("raw_16k", "presence", BK)
    assert any(v.status == "purged_hallucination" for v in store._votes.values())


def _rows(uncertainties: list[float], win: float = 0.5) -> list[dict]:
    return [
        {"start": i * win, "end": (i + 1) * win, "aggregated_uncertainty": u, "status": "open"}
        for i, u in enumerate(uncertainties)
    ]


def test_region_proposal_seed_expand_pad() -> None:
    """FR-010: seed ≥ θ_high, expand ≥ θ_low, pad, mass ranking."""
    policy = load_policy()
    rows = _rows([0.1, 0.4, 0.9, 0.5, 0.1, 0.1, 0.7, 0.1])
    regions = propose_regions(rows, axis="utterance", stream="raw_16k", policy=policy, round_idx=2, duration_s=4.0)
    assert len(regions) == 2
    r0 = regions[0]
    assert (r0["core_start"], r0["core_end"]) == (0.5, 2.0)  # expanded left+right over ≥0.33
    assert r0["crop_start"] == 0.0 and r0["crop_end"] == 3.0  # ±1.0 s pad, clipped
    assert r0["region_id"] == "r2_raw_utterance_0"
    assert regions[1]["core_start"] == 3.0  # isolated 0.7 seed


def test_planner_is_deterministic_and_budget_bounded() -> None:
    """FR-011/025: stable total order; budget defers instead of dropping."""
    policy = load_policy()
    policy["budget"]["medium_per_run"] = 1
    rules = [
        {
            "id": "R_med",
            "axes": ["utterance"],
            "cost": "medium",
            "trigger": lambda r, c: (True, {}),
            "guard": None,
            "gain": lambda r, c, t: r["uncertainty_mass"],
            "execute": lambda cand, c: {"touched": {}},
        }
    ]
    regions = [
        {
            "region_id": "rA",
            "axis": "utterance",
            "stream": "raw_16k",
            "core_start": 0.0,
            "core_end": 1.0,
            "crop_start": 0.0,
            "crop_end": 2.0,
            "uncertainty_mass": 0.2,
            "status": "open",
        },
        {
            "region_id": "rB",
            "axis": "utterance",
            "stream": "raw_16k",
            "core_start": 2.0,
            "core_end": 3.0,
            "crop_start": 1.0,
            "crop_end": 4.0,
            "uncertainty_mass": 0.9,
            "status": "open",
        },
    ]
    ledger = BudgetLedger(policy)
    admitted, deferred = plan_round(rules=rules, regions=regions, ctx={}, ledger=ledger, policy=policy, round_idx=2)
    assert [c["region_id"] for c in admitted] == ["rB"]  # higher mass wins the single slot
    assert [c["status"] for c in deferred] == ["deferred_budget"]
    ledger2 = BudgetLedger(policy)
    admitted2, _ = plan_round(rules=rules, regions=regions, ctx={}, ledger=ledger2, policy=policy, round_idx=2)
    assert [c["intervention_id"] for c in admitted] == [c["intervention_id"] for c in admitted2]


def test_family_weights_prevent_double_counting() -> None:
    """FR-008: two whisper-family models share one family's weight."""
    policy = load_policy()
    w = family_weights(
        ["openai/whisper-large-v3-turbo", "nyralabs/CrisperWhisper2.0_turbo", "Qwen/Qwen3-ASR-1.7B"], policy
    )
    assert w["openai/whisper-large-v3-turbo"] == pytest.approx(0.5)
    assert w["nyralabs/CrisperWhisper2.0_turbo"] == pytest.approx(0.5)
    assert w["Qwen/Qwen3-ASR-1.7B"] == pytest.approx(1.0)


def test_fusion_votes_and_abstention_penalty() -> None:
    """D9 + coverage rule: agreement wins slots; single-witness words can't score 1.0."""
    policy = load_policy()
    streams = {
        "Qwen/Qwen3-ASR-1.7B": [{"text": "hello", "start": 0.0, "end": 0.4}],
        "nvidia/canary-qwen-2.5b": [{"text": "hello", "start": 0.05, "end": 0.42}],
        "ibm-granite/granite-speech-3.3-8b": [
            {"text": "hello", "start": 0.02, "end": 0.41},
            {"text": "stray", "start": 2.0, "end": 2.3},
        ],
    }
    words = fuse_words(streams, policy=policy)
    assert [w["text"] for w in words] == ["hello", "stray"]
    hello, stray = words
    assert hello["confidence"] > 0.9 and hello["coverage"] == 1.0
    assert stray["confidence"] < 0.5 and "single_source" in stray["flags"]
    assert hello["alternates"] == []


def test_identity_repair_recovers_synthetic_speakers() -> None:
    """I1+I2 on synthetic embeddings: 3 planted speakers → 3 clusters, boundaries found."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.adaptive.identity_repair import repair_identity

    rng = np.random.default_rng(0)
    centroids = {0: rng.normal(0, 1, 32), 1: rng.normal(4, 1, 32), 2: rng.normal(-4, 1, 32)}
    # 0-2s speaker0, 2-4s speaker1, 4-6s speaker2; windows 0.5s / 0.25s hop.
    windows = []
    t = 0.0
    while t + 0.5 <= 6.0:
        spk = 0 if t < 1.9 else (1 if t < 3.9 else 2)
        vec = centroids[spk] + rng.normal(0, 0.05, 32)
        windows.append({"start_s": round(t, 2), "end_s": round(t + 0.5, 2), "vector": vec.tolist()})
        t += 0.25
    repaired = repair_identity(
        window_embeddings={"ecapa": windows, "resnet": windows},
        diar_boundaries=[],
        p_voice_at=lambda t: 0.9,
        duration_s=6.0,
        policy={
            "identity": {
                "cp_k": 1.0,
                "cp_floor": 0.15,
                "min_segment_s": 0.25,
                "recluster_cosine_threshold": 0.45,
                "voiced_threshold": 0.5,
            }
        },
    )
    assert repaired is not None
    assert repaired["n_clusters"] == 3
    cp_times = [c["time"] for c in repaired["change_points"]]
    assert any(abs(t - 2.0) < 0.3 for t in cp_times)
    assert any(abs(t - 4.0) < 0.3 for t in cp_times)
    # Deterministic: same input → same output.
    repaired2 = repair_identity(
        window_embeddings={"ecapa": windows, "resnet": windows},
        diar_boundaries=[],
        p_voice_at=lambda t: 0.9,
        duration_s=6.0,
        policy={
            "identity": {
                "cp_k": 1.0,
                "cp_floor": 0.15,
                "min_segment_s": 0.25,
                "recluster_cosine_threshold": 0.45,
                "voiced_threshold": 0.5,
            }
        },
    )
    assert repaired2 is not None
    assert [s["cluster_id"] for s in repaired["segments"]] == [s["cluster_id"] for s in repaired2["segments"]]


def test_triage_gates() -> None:
    """US1: silent → no speech; clean speech → no enhancement; noisy speech → enhancement."""
    from senselab.audio.workflows.audio_analysis.adaptive.triage import triage_decision

    hop = 0.017  # ~segmentation-3.0 frame hop
    silent = triage_decision(p_speech=[0.02] * 300, frame_hop_s=hop, snr_db=[30.0] * 100, snr_hop_s=0.05)
    assert silent["speech_present"] is False

    clean = triage_decision(p_speech=[0.95] * 300, frame_hop_s=hop, snr_db=[25.0] * 100, snr_hop_s=0.05)
    assert clean["speech_present"] is True and clean["needs_enhancement"] is False

    noisy = triage_decision(p_speech=[0.95] * 300, frame_hop_s=hop, snr_db=[3.0] * 100, snr_hop_s=0.05)
    assert noisy["speech_present"] is True and noisy["needs_enhancement"] is True

    unknown = triage_decision(p_speech=[0.95] * 300, frame_hop_s=hop, snr_db=None)
    assert unknown["speech_present"] is True and unknown["needs_enhancement"] is None

    empty = triage_decision(p_speech=[], frame_hop_s=hop)
    assert empty["speech_present"] is True and empty["inconclusive"] is True  # conservative on no evidence


def test_triage_brief_speech_below_minimum() -> None:
    """A 0.2 s blip stays below the 0.3 s min-speech gate; 0.5 s passes."""
    from senselab.audio.workflows.audio_analysis.adaptive.triage import triage_decision

    hop = 0.02
    blip = [0.0] * 100 + [0.9] * 10 + [0.0] * 100  # 0.2 s of speech
    assert triage_decision(p_speech=blip, frame_hop_s=hop)["speech_present"] is False
    longer = [0.0] * 100 + [0.9] * 25 + [0.0] * 100  # 0.5 s
    assert triage_decision(p_speech=longer, frame_hop_s=hop)["speech_present"] is True


def test_dsp_snr_series_orders_noise_levels() -> None:
    """The DSP fallback ranks a loud-speech-on-quiet-floor frame above noise frames."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.adaptive.triage import dsp_snr_series

    rng = np.random.default_rng(0)
    quiet = rng.normal(0, 0.001, 16000)
    loud = rng.normal(0, 0.3, 16000)
    snr, hop = dsp_snr_series(np.concatenate([quiet, loud, quiet]), 16000)
    assert hop > 0
    n = len(snr)
    assert np.mean(snr[n // 3 : 2 * n // 3]) > np.mean(snr[: n // 3]) + 20


def test_calibration_profiles() -> None:
    """T033: logistic and piecewise profiles map confidences; None disables."""
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import fuse_words, load_calibrator

    assert load_calibrator(None) is None
    logistic = load_calibrator({"type": "logistic", "a": 1.0, "b": 0.0})
    assert logistic(0.7) == pytest.approx(0.7, abs=1e-3)  # identity logistic
    piecewise = load_calibrator({"type": "piecewise", "x": [0.0, 1.0], "y": [0.0, 0.5]})
    assert piecewise(0.8) == pytest.approx(0.4)
    assert piecewise(-0.2) == 0.0 and piecewise(2.0) == 0.5  # clamped at knots
    with pytest.raises(ValueError):
        load_calibrator({"type": "nope"})

    policy = load_policy()
    streams = {
        "Qwen/Qwen3-ASR-1.7B": [{"text": "hello", "start": 0.0, "end": 0.4}],
        "nvidia/canary-qwen-2.5b": [{"text": "hello", "start": 0.05, "end": 0.42}],
    }
    raw = fuse_words(streams, policy=policy)[0]["confidence"]
    halved = fuse_words(streams, policy=policy, calibrator=piecewise)[0]["confidence"]
    assert halved == pytest.approx(raw / 2, abs=1e-3)


def test_policy_hash_stable_and_override(tmp_path: Path) -> None:
    """D10: identical policy → identical hash; overrides change it."""
    p1, p2 = load_policy(), load_policy()
    assert p1["policy_hash"] == p2["policy_hash"]
    override = tmp_path / "p.yaml"
    override.write_text("thresholds:\n  theta_high: 0.5\n")
    p3 = load_policy(override)
    assert p3["thresholds"]["theta_high"] == 0.5
    assert p3["policy_hash"] != p1["policy_hash"]
    assert p3["thresholds"]["theta_low"] == p1["thresholds"]["theta_low"]  # deep-merge kept defaults


def test_from_harvests_in_process_integration() -> None:
    """T044/T009: PassHarvest → VoteStore without a parquet round-trip; parity with aggregate_pass."""
    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore
    from senselab.audio.workflows.audio_analysis.aggregate import aggregate_presence
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        pass_label="raw_16k",
        presence_votes=[{"start": 0.0, "end": 0.5, "votes": {"m1": {"speaks": True}, "m2": {"speaks": False}}}],
        identity_votes=[{"start": 0.0, "end": 1.0, "votes": {"__cross_diar_label_disagreement__": {"value": 0.5}}}],
        utterance_votes=[],
        quality_by_bucket={(0.0, 0.5): {"quality_snr": 0.3, "_raw": {}}},
    )
    store = VoteStore.from_harvests({"raw_16k": harvest})
    votes = store.active_votes("raw_16k", "presence", (0.0, 0.5))
    assert set(votes) == {"m1", "m2"}
    row = store.reaggregate_bucket("raw_16k", "presence", (0.0, 0.5), aggregator="min")
    assert row["aggregated_uncertainty"] == pytest.approx(aggregate_presence(votes))
    assert store.row_meta[("raw_16k", "presence", (0.0, 0.5))]["quality_snr"] == 0.3
    ident = store.reaggregate_bucket("raw_16k", "identity", (0.0, 1.0), aggregator="min")
    assert ident["aggregated_uncertainty"] == pytest.approx(0.5)


# ── In-process ingest path (T040) ─────────────────────────────────────


def test_run_adaptive_loop_accepts_in_process_harvests(tmp_path: Path) -> None:
    """The loop can ingest PassHarvest objects directly, with no parquet round-trip.

    This is T040's integration point: analyze_audio hands over what it just
    computed instead of writing nine parquets and reading them back.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        pass_label="raw_16k",
        presence_votes=[
            {"start": 0.0, "end": 0.5, "votes": {"m1": {"speaks": True}}},
            {"start": 0.5, "end": 1.0, "votes": {"m1": {"speaks": False}}},
        ],
        identity_votes=[{"start": 0.0, "end": 1.0, "votes": {}}],
        utterance_votes=[{"start": 0.0, "end": 1.0, "votes": {"a": {"text": "hi"}}}],
        grids={"utterance": {"win_length": 1.0, "hop_length": 1.0}},
    )
    summary = {"passes": {"raw_16k": {"duration_s": 1.0, "audio_signature": "a" * 64}}}

    log = run_adaptive_loop(
        tmp_path,
        harvests={"raw_16k": harvest},
        summary=summary,
        max_rounds=1,
        aggregator="min",
    )
    assert (tmp_path / "rounds" / "1").is_dir(), "round 1 artifacts must still be emitted"
    assert isinstance(log, dict)


def test_in_process_path_reports_parity_as_skipped_not_passing(tmp_path: Path) -> None:
    """A vacuous parity check would be a misleading proof — it must say "skipped".

    parity_check compares re-aggregation against the *stored* parquet values. On
    the in-process path those don't exist yet, so every bucket would be "compared:
    0, mismatches: 0" — which reads as a pass while proving nothing.
    """
    import json as _json

    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        pass_label="raw_16k",
        presence_votes=[{"start": 0.0, "end": 0.5, "votes": {"m1": {"speaks": True}}}],
        grids={"utterance": {"win_length": 1.0, "hop_length": 1.0}},
    )
    run_adaptive_loop(
        tmp_path,
        harvests={"raw_16k": harvest},
        summary={"passes": {"raw_16k": {"duration_s": 1.0, "audio_signature": "b" * 64}}},
        max_rounds=1,
        aggregator="min",
    )
    round1 = _json.loads((tmp_path / "rounds" / "1" / "summary.json").read_text())
    assert round1["parity_check"]["status"] == "skipped"
    assert "stored parquet" in round1["parity_check"]["reason"]


def test_in_process_ingest_ignores_passes_absent_from_the_summary(tmp_path: Path) -> None:
    """Only passes the summary reports as completed may contribute votes."""
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    def _h(label: str) -> PassHarvest:
        return PassHarvest(
            pass_label=label,
            presence_votes=[{"start": 0.0, "end": 0.5, "votes": {"m1": {"speaks": True}}}],
            grids={"utterance": {"win_length": 1.0, "hop_length": 1.0}},
        )

    log = run_adaptive_loop(
        tmp_path,
        harvests={"raw_16k": _h("raw_16k"), "enhanced_16k": _h("enhanced_16k")},
        # enhancement failed, so the summary has no duration_s for it
        summary={
            "passes": {
                "raw_16k": {"duration_s": 1.0, "audio_signature": "c" * 64},
                "enhanced_16k": {"status": "failed"},
            }
        },
        max_rounds=1,
        aggregator="min",
    )
    assert isinstance(log, dict)
    belief = (tmp_path / "rounds" / "1").glob("belief*")
    assert any(belief), "round-1 belief artifacts should exist for the surviving pass"
