"""Unit tests for the adaptive loop's pure parts (tasks.md T036).

Model-free: exercises vote shadowing/purging, region proposal, deterministic
planning + budget, and fusion voting on synthetic inputs.
"""

from pathlib import Path
from typing import Any

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
from senselab.audio.workflows.audio_analysis.adaptive.types import Region
from senselab.audio.workflows.audio_analysis.axes import AXIS_NAMES
from senselab.audio.workflows.audio_analysis.layout import belief_dir, round_dir

BK = bucket_key(0.0, 0.5)


def _vote(source: str, scope: str = "file", speaks: bool = True, round_idx: int = 1) -> Vote:
    return Vote(
        axis="speech_presence",
        bucket=BK,
        source=source,
        stream="raw",
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
    active = store.active_votes("raw", "speech_presence", BK)
    assert active["model_a"]["speaks"] is False  # region vote won
    assert active["model_b"]["speaks"] is True  # unrelated model untouched
    shadowed = [v for v in store._votes.values() if v.status == "shadowed"]
    assert len(shadowed) == 1 and shadowed[0].source == "model_a" and shadowed[0].scope == "file"


def test_attenuated_votes_stay_in_aggregation() -> None:
    """C10 no longer erases: an uncorroborated claim is weighed down, not removed.

    The behaviour this replaces deleted the vote from ``active_votes``, so the bucket reported
    confident silence whenever the only witness to a quiet speaker was doubted.
    """
    store = VoteStore()
    store.add_vote(_vote("asr_x"))
    records = store.attenuate_source_in_bucket(
        "raw",
        BK,
        "asr_x",
        corroboration=0.0,
        evidence_sources=["frame_vad"],
        reason="uncorroborated_speech_claim",
        round_idx=2,
        measured_on=("speech_presence", BK),
    )
    assert len(records) == 1
    assert "asr_x" in store.active_votes("raw", "speech_presence", BK)
    assert store.evidence_weights("raw", "speech_presence", BK)["asr_x"] > 0.0
    assert all(v.status == "active" for v in store._votes.values())


def _rows(uncertainties: list[float], win: float = 0.5) -> list[dict]:
    return [
        {"start": i * win, "end": (i + 1) * win, "uncertainty": u, "status": "open"}
        for i, u in enumerate(uncertainties)
    ]


def test_region_proposal_seed_expand_pad() -> None:
    """FR-010: seed ≥ θ_high, expand ≥ θ_low, pad, mass ranking."""
    policy = load_policy()
    rows = _rows([0.1, 0.4, 0.9, 0.5, 0.1, 0.1, 0.7, 0.1])
    regions = propose_regions(rows, axis="asr", policy=policy, round_idx=2, duration_s=4.0)
    assert len(regions) == 2
    r0 = regions[0]
    assert (r0["core_start"], r0["core_end"]) == (0.5, 2.0)  # expanded left+right over ≥0.33
    assert r0["crop_start"] == 0.0 and r0["crop_end"] == 3.0  # ±1.0 s pad, clipped
    assert r0["region_id"] == "r2_asr_0"
    assert regions[1]["core_start"] == 3.0  # isolated 0.7 seed


def test_planner_is_deterministic_and_budget_bounded() -> None:
    """FR-011/025: stable total order; budget defers instead of dropping."""
    policy = load_policy()
    policy["budget"]["medium_per_run"] = 1
    rules = [
        {
            "id": "R_med",
            "axes": ["asr"],
            "cost": "medium",
            "trigger": lambda r, c: (True, {}),
            "guard": None,
            "gain": lambda r, c, t: r["uncertainty_mass"],
            "execute": lambda cand, c: {"touched": {}},
        }
    ]
    regions: list[Region] = [
        {
            "region_id": "rA",
            "axis": "asr",
            "core_start": 0.0,
            "core_end": 1.0,
            "crop_start": 0.0,
            "crop_end": 2.0,
            "uncertainty_mass": 0.2,
            "n_buckets": 2,
            "status": "open",
        },
        {
            "region_id": "rB",
            "axis": "asr",
            "core_start": 2.0,
            "core_end": 3.0,
            "crop_start": 1.0,
            "crop_end": 4.0,
            "uncertainty_mass": 0.9,
            "n_buckets": 2,
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
            "speaker": {
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
            "speaker": {
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
    assert logistic(0.7) == pytest.approx(0.7, abs=1e-3)  # speaker logistic
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
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        perturbation="raw",
        speech_presence_evidence=[
            {"start": 0.0, "end": 0.5, "evidence": {"m1": {"covered_fraction": 1.0}, "m2": {"covered_fraction": 0.0}}}
        ],
        speaker_votes=[{"start": 0.0, "end": 1.0, "votes": {"__cross_diar_label_disagreement__": {"value": 0.5}}}],
        asr_votes=[],
        quality_by_bucket={(0.0, 0.5): {"quality_snr": 0.3, "_raw": {}}},
    )
    # Every harvested axis is read off the harvest, the mask included, so nothing has to be handed
    # in beside it. This fixture measures no mask, and an axis with no evidence is *absent* from the
    # store rather than present-and-settled.
    store = VoteStore.from_harvests({"raw": harvest})
    votes = store.active_votes("raw", "speech_presence", (0.0, 0.5))
    assert set(votes) == {"m1", "m2"}
    row = store.reaggregate_bucket("speech_presence", (0.0, 0.5), aggregator="min")
    # ``p_voice`` is a probability about the world and still comes from ``speaks``: two voters
    # split evenly. ``uncertainty`` is the fold and is ``None`` here, because neither coverage
    # voter reported any doubt of its own — no signal spoke, which is not the same as agreement.
    assert row["p_voice"] == pytest.approx(0.5)
    assert row["uncertainty"] is None
    assert store.row_meta[("raw", "speech_presence", (0.0, 0.5))]["quality_snr"] == 0.3
    ident = store.reaggregate_bucket("speaker", (0.0, 1.0), aggregator="min")
    assert ident["uncertainty"] == pytest.approx(
        fuse_axis(
            {"raw": [{"start": 0.0, "end": 1.0, "votes": store.active_votes("raw", "speaker", (0.0, 1.0))}]},
            weights={},
        )[0]["uncertainty"]
    ), "the store's fold is fuse_axis, not a second implementation of it"


# ── In-process ingest path (T040) ─────────────────────────────────────


def test_run_adaptive_loop_accepts_in_process_harvests(tmp_path: Path) -> None:
    """The loop can ingest PassHarvest objects directly, with no parquet round-trip.

    This is T040's integration point: analyze_audio hands over what it just
    computed instead of writing nine parquets and reading them back.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        perturbation="raw",
        speech_presence_evidence=[
            {"start": 0.0, "end": 0.5, "evidence": {"m1": {"covered_fraction": 1.0}}},
            {"start": 0.5, "end": 1.0, "evidence": {"m1": {"covered_fraction": 0.0}}},
        ],
        speaker_votes=[{"start": 0.0, "end": 1.0, "votes": {}}],
        asr_votes=[{"start": 0.0, "end": 1.0, "votes": {"a": {"text": "hi"}}}],
        grids={"asr": {"win_length": 1.0, "hop_length": 1.0}},
    )
    summary = {"passes": {"raw": {"duration_s": 1.0, "audio_signature": "a" * 64}}}

    log = run_adaptive_loop(
        tmp_path,
        harvests={"raw": harvest},
        summary=summary,
        max_rounds=1,
        aggregator="min",
    )
    assert round_dir(tmp_path, 0).is_dir(), "the baseline round's artifacts must still be emitted"
    assert isinstance(log, dict)


def test_both_ingest_paths_run_the_replay_proof(tmp_path: Path) -> None:
    """The in-process path is checkable too, and reports real comparisons.

    The old parity check compared re-aggregation against ``within_pass_uncertainty`` on the L1
    parquet. That oracle was wrong twice: the quantity was a per-pass axis, which cannot exist,
    and it came from a *second implementation* of the fold, so a mismatch could not distinguish
    "the store missed an input" from "the two folds disagree". The in-process path could not run
    it at all and had to report "skipped". The replay proves the property that matters — every
    value is re-derivable from the persisted votes plus the recorded decisions — and runs on both
    paths.
    """
    import json as _json

    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    harvest = PassHarvest(
        perturbation="raw",
        speech_presence_evidence=[
            {
                "start": 0.0,
                "end": 0.5,
                "evidence": {"m1": {"covered_fraction": 1.0}, "vad": {"frame_mean": 0.8, "excess_db": 6.0}},
            }
        ],
        grids={"asr": {"win_length": 1.0, "hop_length": 1.0}},
    )
    run_adaptive_loop(
        tmp_path,
        harvests={"raw": harvest},
        summary={"passes": {"raw": {"duration_s": 1.0, "audio_signature": "b" * 64}}},
        max_rounds=1,
        aggregator="min",
    )
    # ``adaptive``, because a round's summary now carries a block per producer: fusion's fold log
    # and the loop's proofs are different facts about the same round, and the round has one
    # document rather than two writers taking turns erasing each other.
    round1 = _json.loads((round_dir(tmp_path, 0) / "summary.json").read_text())["adaptive"]
    report = round1["replay_check"]
    assert report, "the in-process path must be checkable, not skipped"
    assert all(entry["mismatches"] == 0 for entry in report.values())
    assert any(entry["compared"] > 0 for entry in report.values()), "a vacuous check proves nothing"


def test_in_process_ingest_ignores_passes_absent_from_the_summary(tmp_path: Path) -> None:
    """Only passes the summary reports as completed may contribute votes."""
    from senselab.audio.workflows.audio_analysis.adaptive.loop import run_adaptive_loop
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    def _h(label: str) -> PassHarvest:
        return PassHarvest(
            perturbation=label,
            speech_presence_evidence=[{"start": 0.0, "end": 0.5, "evidence": {"m1": {"covered_fraction": 1.0}}}],
            grids={"asr": {"win_length": 1.0, "hop_length": 1.0}},
        )

    log = run_adaptive_loop(
        tmp_path,
        harvests={"raw": _h("raw"), "enhanced": _h("enhanced")},
        # enhancement failed, so the summary has no duration_s for it
        summary={
            "passes": {
                "raw": {"duration_s": 1.0, "audio_signature": "c" * 64},
                "enhanced": {"status": "failed"},
            }
        },
        max_rounds=1,
        aggregator="min",
    )
    assert isinstance(log, dict)
    assert (round_dir(tmp_path, 0) / "summary.json").exists(), (
        "the baseline round's summary should exist for the surviving perturbation"
    )


# ── Policy override precedence (T040) ─────────────────────────────────


def test_cli_overrides_win_over_a_policy_file(tmp_path: Path) -> None:
    """Precedence is packaged default < --policy file < CLI flags."""
    import json as _json

    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    policy_file = tmp_path / "p.yaml"
    policy_file.write_text(_json.dumps({"budget": {"heavy_per_run": 9}}))  # YAML accepts JSON
    merged = load_policy(policy_file, {"budget": {"heavy_per_run": 0}})
    assert merged["budget"]["heavy_per_run"] == 0, "CLI must beat the file"


def test_none_overrides_leave_the_policy_untouched() -> None:
    """An unset flag must not clobber the policy's value with None."""
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    default = load_policy()
    merged = load_policy(None, {"budget": {"medium_per_run": None, "heavy_per_run": 2}})
    assert merged["budget"]["medium_per_run"] == default["budget"]["medium_per_run"]
    assert merged["budget"]["heavy_per_run"] == 2


def test_policy_hash_reflects_the_overrides() -> None:
    """Two runs differing only by --budget-heavy must not claim the same policy hash.

    The hash is provenance: it has to identify the policy that actually ran, not
    the file on disk.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    a = load_policy(None, {"budget": {"heavy_per_run": 1}})
    b = load_policy(None, {"budget": {"heavy_per_run": 2}})
    assert a["policy_hash"] != b["policy_hash"]


def test_empty_overrides_match_the_unmodified_policy() -> None:
    """No flags set → byte-identical policy (and hash) to loading with none."""
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    assert load_policy(None, {}) == load_policy()
    assert load_policy(None, {"budget": {"medium_per_run": None}})["policy_hash"] == load_policy()["policy_hash"]


# ── P2_fine_posteriors (T041) ─────────────────────────────────────────


def _p2_ctx(
    buckets: list[tuple[tuple[float, float], dict[str, dict[str, Any]], float | None]],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ctx backed by a real VoteStore.

    Vote payloads must come from the store — the belief row only carries
    ``contributing_sources`` (names). An earlier version of this fixture handed the
    trigger fabricated rows with a ``model_votes`` key, which does not exist on a
    real row; the trigger read nothing, never fired, and the unit tests still
    passed. Building the store for real is what stops that recurring.

    Each entry is ``(bucket, {source: payload}, frame_dispersion)``.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.belief import Vote, VoteStore
    from senselab.audio.workflows.audio_analysis.adaptive.policy import load_policy

    store = VoteStore()
    rows: list[dict[str, Any]] = []
    for bk, votes, instability in buckets:
        for source, payload in votes.items():
            store.add_vote(
                Vote(
                    axis="speech_presence",
                    bucket=bk,
                    source=source,
                    stream="raw",
                    scope="file",
                    round=1,
                    payload=payload,
                )
            )
        meta: dict[str, Any] = {}
        if instability is not None:
            meta["frame_dispersion"] = instability
        rows.append({"start": bk[0], "end": bk[1], "meta": meta})

    class _State:
        def axis_rows(self, axis: str) -> list[dict[str, Any]]:
            return rows if axis == "speech_presence" else []

    return {"state": _State(), "store": store, "policy": policy or load_policy(), "passes": ["raw"], "_rows": rows}


def _p2_region() -> dict[str, Any]:
    return {
        "axis": "speech_presence",
        "region_id": "r2_raw_speech_presence_0",
        "core_start": 0.0,
        "core_end": 1.0,
        "crop_start": 0.0,
        "crop_end": 2.0,
        "uncertainty_mass": 0.4,
    }


def test_p2_is_registered_before_i4() -> None:
    """I4's contract says it "else fires P2 first", so P2 must be plannable."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import RULES

    ids = [r["id"] for r in RULES]
    assert "P2_fine_posteriors" in ids
    assert ids.index("P2_fine_posteriors") < ids.index("I4_overlap_detection")


def test_p2_declares_speech_presence_axis_and_medium_cost() -> None:
    """contracts/interventions.md: speech_presence axis, medium cost."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import RULES

    rule = next(r for r in RULES if r["id"] == "P2_fine_posteriors")
    assert rule["axes"] == ["speech_presence"]
    assert rule["cost"] == "medium"


def test_p2_fires_when_coarse_voters_dominate() -> None:
    """Coarse voters cast one identical vote across every bucket they span.

    Their agreement is an artifact of window size, not evidence about this bucket,
    so a majority-coarse region is exactly what a finer grid should re-decide.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger

    ctx = _p2_ctx(
        [
            (
                (0.0, 0.5),
                {
                    "ast": {"speaks": True, "coarse": True},
                    "yamnet": {"speaks": True, "coarse": True},
                    "opensmile": {"speaks": True},
                },
                None,
            )
        ]
    )
    fires, info = _p2_trigger(_p2_region(), ctx)
    assert fires is True
    assert info["coarse_share"] == pytest.approx(2 / 3, abs=1e-4)
    assert info["reason"] == "coarse_dominance"


def test_p2_does_not_fire_on_fine_evidence() -> None:
    """All-fine voters with a stable posterior need no re-analysis."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger

    ctx = _p2_ctx([((0.0, 0.5), {"opensmile": {"speaks": True}, "ppg": {"speaks": True}}, 0.0)])
    fires, _ = _p2_trigger(_p2_region(), ctx)
    assert fires is False


def test_p2_fires_on_frame_dispersion_even_without_coarse_votes() -> None:
    """The second independent trigger: a bucket straddling an onset."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger

    ctx = _p2_ctx([((0.0, 0.5), {"opensmile": {"speaks": True}}, 0.6)])
    fires, info = _p2_trigger(_p2_region(), ctx)
    assert fires is True
    assert info["reason"] == "frame_dispersion"


def test_p2_ignores_non_speech_presence_regions() -> None:
    """A speech_presence-only rule must not claim speaker or asr regions."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger

    for axis in ("speaker", "asr"):
        fires, _ = _p2_trigger({**_p2_region(), "axis": axis}, _p2_ctx([]))
        assert fires is False


def test_p2_skips_inactive_votes_when_measuring_coarse_share() -> None:
    """Only votes that actually decided (`speaks` not None) count toward the share."""
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import _p2_trigger

    ctx = _p2_ctx(
        [
            (
                (0.0, 0.5),
                {
                    "ast": {"speaks": True, "coarse": True},
                    "abstained": {"speaks": None, "coarse": True},
                    "opensmile": {"speaks": True},
                },
                None,
            )
        ]
    )
    _, info = _p2_trigger(_p2_region(), ctx)
    assert info["n_active_votes"] == 2
    assert info["coarse_share"] == pytest.approx(0.5)


def test_p2_execute_replaces_votes_at_region_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fine posterior enters as a region-scoped vote, superseding coarse ones.

    Scoped rather than deleting the round-1 voters: the store keeps both and the
    later scope wins, which is what keeps the decision log auditable.
    """
    from senselab.audio.workflows.audio_analysis.adaptive import interventions as iv
    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore

    ctx = _p2_ctx(
        [
            ((0.0, 0.5), {"ast": {"speaks": True, "coarse": True}}, None),
            ((0.5, 1.0), {"ast": {"speaks": True, "coarse": True}}, None),
        ]
    )
    ctx.update({"round_idx": 2, "input_audio": "x.wav"})

    monkeypatch.setattr(iv, "region_buckets", lambda region, rws: {(0.0, 0.5), (0.5, 1.0)})
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.audio_io.get_stream_wav",
        lambda c, s: (object(), None),
    )
    # 0.1 s hop over a 2 s crop: speech for the first 0.5 s, silence after —
    # so bucket (0.0, 0.5) reads as speech and (0.5, 1.0) as silence.
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.backends.speech_posteriors",
        lambda wav, span: (
            {"frame_hop": 0.1, "speech": [0.9] * 5 + [0.05] * 15, "overlap": [0.2] * 20, "n_classes": 7},
            None,
        ),
    )

    result = iv._p2_execute({"region": _p2_region(), "trigger": {"stream": "raw"}}, ctx)
    assert result["votes_added"] == 2
    votes = [v for v in ctx["store"]._votes.values() if v.source == "frame_posterior_fine"]
    assert len(votes) == 2
    assert all(v.scope == "region:r2_raw_speech_presence_0" for v in votes)
    assert all(v.payload["coarse"] is False for v in votes)
    # first bucket sits in the speech half, second in the silence half
    by_bucket = {v.bucket: v for v in votes}
    assert by_bucket[(0.0, 0.5)].payload["speaks"] is True
    assert by_bucket[(0.5, 1.0)].payload["speaks"] is False


def test_p2_execute_no_longer_emits_an_overlap_posterior(monkeypatch: pytest.MonkeyPatch) -> None:
    """P2 measures speech with Brouhaha's VAD head, which is single-channel and reports no overlap.

    The contract previously let I4 run "light (reuses P2 output)". I4 now derives overlap from
    cross-diarizer spans, so the two are independent — and leaving a stale value on the row would be
    worse than leaving none, because a reader cannot tell an overlap of 0.0 from an unmeasured one.
    """
    from senselab.audio.workflows.audio_analysis.adaptive import interventions as iv
    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore

    ctx = _p2_ctx([((0.0, 0.5), {"ast": {"speaks": True, "coarse": True}}, None)])
    rows = ctx["_rows"]
    ctx.update({"round_idx": 2, "input_audio": "x.wav"})
    monkeypatch.setattr(iv, "region_buckets", lambda region, rws: {(0.0, 0.5)})
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.audio_io.get_stream_wav",
        lambda c, s: (object(), None),
    )
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.backends.speech_posteriors",
        lambda wav, span: ({"frame_hop": 0.1, "speech": [0.8] * 20, "overlap": [0.42] * 20, "n_classes": 7}, None),
    )
    iv._p2_execute({"region": _p2_region(), "trigger": {"stream": "raw"}}, ctx)
    assert "overlap_posterior" not in rows[0], "P2 must not leave a stale overlap value"
    assert rows[0].get("meta", {}).get("overlap_posterior") is None


def test_p2_execute_raises_when_posteriors_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed posterior must surface as a rule failure, not silent success."""
    from senselab.audio.workflows.audio_analysis.adaptive import interventions as iv
    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore

    ctx = _p2_ctx([])
    ctx.update({"round_idx": 2, "input_audio": "x.wav"})
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.audio_io.get_stream_wav",
        lambda c, s: (object(), None),
    )
    monkeypatch.setattr(
        "senselab.audio.workflows.audio_analysis.adaptive.backends.speech_posteriors",
        lambda wav, span: (None, "posteriors_failed (boom)"),
    )
    with pytest.raises(RuntimeError, match="posteriors_failed"):
        iv._p2_execute({"region": _p2_region(), "trigger": {"stream": "raw"}}, ctx)


def test_the_deliverable_presence_track_is_the_last_round_byte_for_byte(tmp_path: Path) -> None:
    """T042 restated as D-17 states it: ``final/`` extracts, so the columns are the round's.

    The contract columns — ``speech_presence_confidence``, ``contributing_passes``,
    ``overlap_posterior`` — used to exist only on a separately-built ``L2/speech_presence.parquet``
    that ``build_final_outputs`` constructed from the belief state. The deliverable therefore
    carried numbers *no round carried*, which is a value computed at the wrong stage and with
    nowhere to look for when it had been decided. They are on the estimate row now, and this
    asserts the two files are identical rather than merely agreeing.
    """
    pytest.importorskip("pandas")
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.adaptive.belief import BeliefState, Vote, VoteStore
    from senselab.audio.workflows.audio_analysis.adaptive.fusion import extract_final_estimates
    from senselab.audio.workflows.audio_analysis.adaptive.loop import _write_round_belief
    from senselab.audio.workflows.audio_analysis.layout import estimates_dir, final_dir

    store = VoteStore()
    store.add_vote(
        Vote(
            axis="speech_presence",
            bucket=(0.0, 0.5),
            source="m1",
            stream="raw",
            scope="file",
            round=1,
            payload={"speaks": True},
        )
    )
    state = BeliefState.from_store(store, aggregator="min")
    _write_round_belief(tmp_path, 1, state)
    extract_final_estimates(tmp_path, 1)
    deliverable = pd.read_parquet(final_dir(tmp_path) / "estimates" / "speech_presence.parquet")
    for col in ("speech_presence_confidence", "contributing_passes", "overlap_posterior"):
        assert col in list(deliverable.columns), f"contract column {col!r} missing from the deliverable"
    pd.testing.assert_frame_equal(deliverable, pd.read_parquet(estimates_dir(tmp_path, 1) / "speech_presence.parquet"))
    # And every active axis, not the three a hand-written list would have named.
    for axis in AXIS_NAMES:
        assert (final_dir(tmp_path) / "estimates" / f"{axis}.parquet").exists(), axis


# ── the store's fold is L2's fold (fused_parity) ─────────────────────────


def _parity_fixture() -> tuple[Any, dict[str, list[dict[str, Any]]]]:
    """A two-pass store plus the axes ``fuse_axis`` produces from the same votes."""
    from senselab.audio.workflows.audio_analysis.fuse import fuse_axis
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    def _harvest(label: str, value: float) -> PassHarvest:
        return PassHarvest(
            perturbation=label,
            speaker_votes=[
                {"start": 0.0, "end": 1.0, "votes": {"diar_a": {"same_label_uncertainty": value}}},
                {"start": 1.0, "end": 2.0, "votes": {"diar_a": {"same_label_uncertainty": 1.0 - value}}},
            ],
        )

    harvests = {"raw": _harvest("raw", 0.2), "enhanced": _harvest("enhanced", 0.6)}
    store = VoteStore.from_harvests(harvests)
    rows = fuse_axis({label: h.speaker_votes for label, h in harvests.items()}, weights={})
    return store, {"speaker": rows}


def test_the_store_and_l2_fold_the_same_evidence_to_the_same_number() -> None:
    """The parity check the store should always have been held to.

    Against ``fuse_axes``, not against a stored L1 fold: the old oracle was a per-pass axis, which
    cannot exist, produced by a *second implementation*, so a mismatch could not distinguish "the
    store missed an input" from "the two folds disagree". Now there is one fold, and any difference
    is a difference in evidence — which is the thing worth catching.
    """
    store, fused = _parity_fixture()
    report = store.fused_parity(fused, aggregator="min")["speaker"]
    assert report["compared"] == 2, "a check that compares nothing reports the same zero as one that passes"
    assert report["mismatches"] == 0 and report["not_in_l2"] == 0


def test_a_signal_the_ingest_drops_is_reported_as_a_mismatch() -> None:
    """The failure the check exists for: same arithmetic, different evidence.

    This is exactly what the artifact ingest did to ``__cross_diar_label_disagreement__`` — kept it
    on one path, filed it as a measurement on the other — and no test or artifact said so.
    """
    store, fused = _parity_fixture()
    dropped = [v for v in store.votes_for("enhanced", "speaker", (0.0, 1.0))]
    assert dropped, "fixture must have an enhanced-pass vote to drop"
    for vote in dropped:
        vote.status = "shadowed"
    report = store.fused_parity(fused, aggregator="min")["speaker"]
    assert report["mismatches"] >= 1, "losing one pass's signal must not fold to the same number"
