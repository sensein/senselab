"""Level 1 emits signals; level 2 produces the final uncertainty maps.

Level 1 harvests per-signal values and each signal's *own* uncertainty. It must not decide
the answer: the fold it was doing per pass is a within-pass diagnostic that pre-empts the
aggregation, and pre-empting it is how one saturated sub-signal came to pin an axis while
every other signal disagreed.

Level 2 aggregates across all signals and all passes, weighting each by what was measured
about it, and iterates. The maps it writes are the answer.
"""

from __future__ import annotations

import pathlib

import pytest

from senselab.audio.workflows.audio_analysis.fuse import (
    fuse_axis,
    per_signal_uncertainty,
)


def _bucket(start: float, values: dict[str, float | None]) -> dict:
    votes = {name: {"same_label_uncertainty": v} for name, v in values.items()}
    return {"start": start, "end": start + 0.5, "votes": votes}


# ── level 1: per-signal, no fold ───────────────────────────────────────


def test_level_one_reports_each_signal_separately() -> None:
    """The emission is per signal, so level 2 can weight them; a fold cannot be re-weighted."""
    out = per_signal_uncertainty(_bucket(0.0, {"a": 0.1, "b": 0.9}))
    assert out == {"a": 0.1, "b": 0.9}


def test_level_one_does_not_impute_a_missing_signal() -> None:
    """A silent signal is dropped, never zero-filled (FR-007). Zero is a confident claim."""
    assert per_signal_uncertainty(_bucket(0.0, {"a": 0.4, "b": None})) == {"a": 0.4}


def test_level_one_emits_nothing_when_no_signal_spoke() -> None:
    """An empty map is distinguishable from a confident zero."""
    assert per_signal_uncertainty(_bucket(0.0, {"a": None})) == {}


# ── level 2: aggregate across signals and passes ───────────────────────


def test_level_two_folds_every_pass_not_just_one() -> None:
    """The final map is over all evidence, so a bucket seen in both passes uses both.

    A per-pass fold answers "what did this pass think", which is a diagnostic. The question
    a consumer asks is "what do we believe", and that spans the passes.
    """
    fused = fuse_axis(
        {"raw": [_bucket(0.0, {"a": 0.0})], "enhanced": [_bucket(0.0, {"a": 1.0})]},
        weights={},
        aggregator="mean",
        snr_gate=None,
    )
    assert len(fused) == 1
    assert fused[0]["triage_score"] == pytest.approx(0.5)
    assert sorted(fused[0]["contributing_passes"]) == ["enhanced", "raw"]


def test_level_two_weights_signals_by_what_was_measured() -> None:
    """The weighting the per-pass fold could not apply, because it ran before measurement."""
    buckets = {"raw": [_bucket(0.0, {"trusted": 0.0, "doubtful": 1.0})]}
    unweighted = fuse_axis(buckets, weights={}, aggregator="min", snr_gate=None)
    weighted = fuse_axis(buckets, weights={"doubtful": 0.05, "trusted": 1.0}, aggregator="min", snr_gate=None)
    # The weights act on the policy fold, which is what the adaptive loop ranks on.
    assert unweighted[0]["triage_score"] == pytest.approx(1.0)
    assert weighted[0]["triage_score"] < 0.2


def test_level_two_records_which_signals_it_used() -> None:
    """A final number with no attribution cannot be acted on (FR-006)."""
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.2, "b": 0.4})]}, weights={}, aggregator="mean", snr_gate=None)
    assert sorted(fused[0]["contributing_signals"]) == ["a", "b"]


def test_level_two_reports_the_weight_each_signal_carried() -> None:
    """Otherwise a low final uncertainty is indistinguishable from a suppressed dissenter."""
    fused = fuse_axis(
        {"raw": [_bucket(0.0, {"a": 0.2, "b": 0.9})]},
        weights={"b": 0.1},
        aggregator="mean",
        snr_gate=None,
    )
    assert fused[0]["signal_weights"] == {"a": 1.0, "b": 0.1}


def test_a_bucket_no_signal_spoke_in_is_not_given_an_answer() -> None:
    """``None`` says "not measured here"; 0.0 would assert confidence nobody expressed."""
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": None})]}, weights={}, aggregator="mean", snr_gate=None)
    assert fused[0]["uncertainty"] is None


def test_buckets_come_out_in_time_order() -> None:
    """The final maps are asserted byte-identical across runs (SC-004)."""
    fused = fuse_axis(
        {"b": [_bucket(1.0, {"x": 0.1})], "a": [_bucket(0.0, {"x": 0.2})]},
        weights={},
        aggregator="mean",
        snr_gate=None,
    )
    assert [f["start"] for f in fused] == [0.0, 1.0]


def test_fusion_is_deterministic() -> None:
    """Same inputs, byte-identical output — no dict-order dependence."""
    args = ({"raw": [_bucket(0.0, {"a": 0.3, "b": 0.7})]}, {"b": 0.5})
    first = fuse_axis(args[0], weights=args[1], aggregator="mean", snr_gate=None)
    second = fuse_axis(args[0], weights=args[1], aggregator="mean", snr_gate=None)
    assert first == second


# ── the written artifacts ──────────────────────────────────────────────


def _harvest(label: str, buckets: list[dict]) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(perturbation=label, speech_presence_evidence=[], speaker_votes=buckets, asr_votes=[])


def test_the_final_maps_are_written_for_every_axis(tmp_path) -> None:  # noqa: ANN001
    """A consumer must find all three where the answer lives, even an axis with no rows."""
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty

    written = write_final_uncertainty(
        tmp_path,
        harvests={"raw": _harvest("raw", [_bucket(0.0, {"a": 0.3})])},
        weights_by_axis={},
        snr_floor_db=10.0,
        snr_gated_passes=frozenset(),
    )
    assert {"speech_presence", "speaker", "asr"} <= set(written)
    for axis in ("speech_presence", "speaker", "asr"):
        pd.read_parquet(written[axis])


def test_the_final_map_carries_its_attribution(tmp_path) -> None:  # noqa: ANN001
    """A final number without the signals and weights behind it cannot be audited (FR-006)."""
    import json

    import pandas as pd

    from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty

    written = write_final_uncertainty(
        tmp_path,
        harvests={
            "raw": _harvest("raw", [_bucket(0.0, {"a": 0.2, "b": 0.9})]),
            "enhanced": _harvest("enhanced", [_bucket(0.0, {"a": 0.2, "b": 0.9})]),
        },
        weights_by_axis={"speaker": {"b": 0.1}},
        snr_floor_db=10.0,
        snr_gated_passes=frozenset(),
    )
    frame = pd.read_parquet(written["speaker"])
    row = frame.iloc[0]
    assert sorted(row["contributing_signals"]) == ["a", "b"]
    assert sorted(row["contributing_passes"]) == ["enhanced", "raw"]
    assert json.loads(row["signal_weights"]) == {"a": 1.0, "b": 0.1}


def test_the_final_map_is_byte_identical_across_runs(tmp_path) -> None:  # noqa: ANN001
    """SC-004 applies to the answer, not only to the diagnostics."""
    from senselab.audio.workflows.audio_analysis.fuse import write_final_uncertainty

    harvests = {"raw": _harvest("raw", [_bucket(0.5, {"b": 0.4}), _bucket(0.0, {"a": 0.1})])}
    first = tmp_path / "one"
    second = tmp_path / "two"
    a = write_final_uncertainty(
        first,
        harvests=harvests,
        weights_by_axis={},
        snr_floor_db=10.0,
        snr_gated_passes=frozenset(),
    )
    b = write_final_uncertainty(
        second,
        harvests=harvests,
        weights_by_axis={},
        snr_floor_db=10.0,
        snr_gated_passes=frozenset(),
    )
    assert pathlib.Path(a["speaker"]).read_bytes() == pathlib.Path(b["speaker"]).read_bytes()
    assert set(a) == set(b)


# ── L2 reports three distinct quantities ───────────────────────────────


def test_level_two_reports_confidence_uncertainty_and_variability() -> None:
    """Three questions, three numbers, each with its own estimator.

    Confidence is a probability, variability a dispersion in the units of the quantity, and
    uncertainty a normalised entropy. Collapsing them into one column is what let a max-doubt
    fold, an entropy and a max-minus-min spread all be called "uncertainty".
    """
    fused = fuse_axis(
        {"raw": [_bucket(0.0, {"a": 0.1, "b": 0.9})]},
        weights={},
        aggregator="mean",
        snr_gate=None,
    )
    row = fused[0]
    assert 0.0 <= row["uncertainty"] <= 1.0
    assert 0.0 <= row["confidence"] <= 1.0
    assert row["variability"] is not None and row["variability"] > 0.0


def test_signals_that_agree_have_no_variability_and_low_uncertainty() -> None:
    """Agreement collapses the dispersion and the entropy together."""
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.0, "b": 0.0})]}, weights={}, aggregator="mean", snr_gate=None)
    assert fused[0]["variability"] == pytest.approx(0.0)
    assert fused[0]["uncertainty"] == pytest.approx(0.0)


def test_a_lone_signal_has_no_variability_but_can_still_be_uncertain() -> None:
    """The distinction the two statistics exist to draw.

    One signal cannot disagree with anyone, so dispersion is undefined — but a signal reporting
    0.5 is still maximally undetermined about its own answer.
    """
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.5})]}, weights={}, aggregator="mean", snr_gate=None)
    assert fused[0]["variability"] is None
    assert fused[0]["uncertainty"] > 0.5


def test_epistemic_uncertainty_is_reported_separately() -> None:
    """The reducible part is what tells the loop another measurement could help.

    Two signals each internally certain but disagreeing: all of the doubt is reducible.
    """
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.0, "b": 1.0})]}, weights={}, aggregator="mean", snr_gate=None)
    assert fused[0]["uncertainty"] == pytest.approx(1.0, abs=0.05)
    assert fused[0]["epistemic_uncertainty"] == pytest.approx(fused[0]["uncertainty"], abs=0.05)


def test_shared_doubt_is_not_reported_as_reducible() -> None:
    """Shared doubt is irreducible.

    Signals agreeing at 0.5 face noise, not disagreement; sending the loop after it would
    waste budget on evidence that cannot resolve anything.
    """
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.5, "b": 0.5})]}, weights={}, aggregator="mean", snr_gate=None)
    assert fused[0]["epistemic_uncertainty"] == pytest.approx(0.0, abs=0.05)
    assert fused[0]["uncertainty"] > 0.5


def test_the_weight_basis_names_which_factor_discounted_a_signal() -> None:
    """A weight of 0.05 must say whether the signal was unstable, unsupported, or both."""
    fused = fuse_axis(
        {"raw": [_bucket(0.0, {"a": 0.3})]},
        weights={"a": 0.2},
        aggregator="mean",
        weight_basis={"a": {"stability": 1.0, "support": 0.2}},
        snr_gate=None,
    )
    assert fused[0]["weight_basis"]["a"] == {"stability": 1.0, "support": 0.2}


def test_the_round_is_recorded() -> None:
    """Later rounds refine earlier ones, so a value without its round is not comparable."""
    fused = fuse_axis({"raw": [_bucket(0.0, {"a": 0.3})]}, weights={}, aggregator="mean", round_index=2, snr_gate=None)
    assert fused[0]["round"] == 2


def test_every_axis_vote_shape_yields_a_per_signal_uncertainty() -> None:
    """The three axes name their fields differently, and all three must be readable.

    On a real run speaker fused 85 buckets while speech_presence fused 0 of 1070 and asr 0 of
    41, because only speaker happens to name its field like an uncertainty. A level that
    silently covers one axis of three is worse than one covering none — the gap is invisible.
    """
    speaker = {"votes": {"a": {"same_label_uncertainty": 0.4}}}
    speech_presence = {"votes": {"b": {"speaks": True, "native_confidence": 0.9}}}
    asr = {"votes": {"c": {"text": "hello", "avg_logprob": -0.2}}}

    assert per_signal_uncertainty(speaker)["a"] == pytest.approx(0.4)
    assert per_signal_uncertainty(speech_presence)["b"] == pytest.approx(0.1)
    assert per_signal_uncertainty(asr)["c"] == pytest.approx(1.0 - 0.8187, abs=0.01)


def test_a_declared_uncertainty_wins_over_a_derived_one() -> None:
    """A signal is authoritative about its own uncertainty.

    Deriving one from a confidence in the same entry would silently override what the signal
    said.
    """
    bucket = {"votes": {"a": {"same_label_uncertainty": 0.9, "native_confidence": 0.95}}}
    assert per_signal_uncertainty(bucket)["a"] == pytest.approx(0.9)


def test_a_synthetic_cross_signal_block_is_never_read_as_a_signal() -> None:
    """``__``-prefixed keys are cross-signal records, not voters, and must not enter the fold.

    The asr axis's pairwise phoneme distances used to be one such block *and* a fallback source of
    per-signal doubt, which is what made this distinction slippery. Nothing derives a signal from a
    ``__`` block now, so the property is simple and worth pinning: a bucket holding only synthetic
    blocks contributes no signals at all rather than inventing names from them.
    """
    bucket = {"votes": {"a": {"same_label_uncertainty": 0.05}, "__cross_diar_label_disagreement__": {"value": 0.9}}}
    out = per_signal_uncertainty(bucket)
    assert out["a"] == pytest.approx(0.05)
    assert out["__cross_diar_label_disagreement__"] == pytest.approx(0.9), (
        "the speaker axis's cross-model block *is* read as a signal by name — it is a measurement "
        "over the diarizers, and `_signal_rows_from_buckets` keeps it out of L1 separately"
    )
