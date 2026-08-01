"""D-11: each round asks whether the axes should change, given everything the loop knows.

The categories are coupled — a speaker ambiguity is frequently a presence ambiguity, and D-7 makes
speaker and presence explicitly joint — so running each axis's rounds to completion before starting
the next one structurally prevents the coupling from ever acting. The axes are interleaved instead:
every axis folds round 0, then each later round re-derives the shared structure from the previous
round's axes and re-estimates every axis against it.

**The derivatives are the channel.** A speaker ambiguity reaches the presence axis by changing the
mask and the speaker allocation both axes are estimated against — never by one axis averaging in a
number another axis reported. An earlier draft did the latter, and needed an arbitrary discount to
keep the extra voter from moving the mean. That was a band-aid on the wrong channel: a discount
bounds double-counting without removing it, and convergence cannot detect it either, since a biased
fixed point is still a fixed point. Conditioning on a shared latent has neither problem, so there is
nothing left for a gate to do.

Convergence is no axis changing, or the loop entering a periodic one.

The driver is N-axis: the design names four (``speech_presence``, ``speaker``, ``asr``,
``background_mask``) with ``task`` punted, and nothing here may assume three.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.fuse import (
    Derivatives,
    derive_mask_from_axes,
    fuse_axes,
    fuse_rounds,
)


def _b(start: float, values: dict[str, float]) -> dict:
    return {
        "start": start,
        "end": start + 0.5,
        "votes": {k: {"same_label_uncertainty": v} for k, v in values.items()},
    }


def _axes(**per_axis: dict[str, float]) -> dict:
    return {axis: {"raw_16k": [_b(0.0, values)]} for axis, values in per_axis.items()}


def _weights(by_axis: dict) -> dict:
    return {axis: {s: 1.0 for p in passes.values() for b in p for s in b["votes"]} for axis, passes in by_axis.items()}


def _run(by_axis: dict, **kw: object) -> dict:
    rows, _logs = fuse_axes(by_axis, weights_by_axis=_weights(by_axis), max_rounds=3, **kw)  # type: ignore[arg-type]
    return rows


# ── the derivatives are the coupling channel ─────────────────────────────


def test_a_settled_presence_axis_reaches_the_speaker_axis_through_the_mask() -> None:
    """The point of D-11, and the shape of the coupling: shared structure, not a borrowed vote.

    Presence settling that a region is quiet re-derives the mask, and the mask discounts a signal
    still claiming a speaker there. Nothing averages one axis's number into another's.
    """
    axes = {
        "speech_presence": {"raw_16k": [_b(0.0, {"p": 0.0, "q": 0.0})]},
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.9, "b": 0.1})]},
    }
    claims = {"a": [(0.0, 0.5)]}
    coupled, _ = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, speaker_claims=claims)
    isolated, _ = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, speaker_claims=claims, derive=None)
    assert coupled["speaker"][0]["signal_weights"]["a"] < isolated["speaker"][0]["signal_weights"]["a"]


def test_the_previous_round_s_axes_are_inputs_to_this_round() -> None:
    """All three inputs feed every round: the signals, the derivatives, and the other axes."""
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    with_axes = _run(axes)
    signals_only, _ = fuse_axes({"speech_presence": axes["speech_presence"]}, weights_by_axis=_weights(axes))
    assert "axis::speaker" in with_axes["speech_presence"][0]["contributing_signals"]
    assert with_axes["speech_presence"][0]["uncertainty"] != signals_only["speech_presence"][0]["uncertainty"]


def test_a_cross_axis_input_carries_no_assigned_discount() -> None:
    """Weights in this module are *measured*, and a factor never measured must not discount.

    An earlier draft multiplied cross-axis inputs by a hand-set 0.4 to stop them dominating the
    fold. That is precisely the unmeasured factor the weighting rule exists to exclude — the
    correlation it was standing in for is something to measure, not to assume.
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    row = _run(axes)["speech_presence"][0]
    assert row["signal_weights"]["axis::speaker"] == pytest.approx(row["signal_weights"]["c"])


def test_an_axis_is_not_an_input_to_itself() -> None:
    """Its previous value is the thing being updated, not evidence about it."""
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    row = _run(axes)["speaker"][0]
    assert "axis::speaker" not in row["contributing_signals"]


def test_the_coupling_is_recorded_on_the_rows_it_moved() -> None:
    """A value the shared structure moved must be distinguishable from one reached alone."""
    axes = {
        "speech_presence": {"raw_16k": [_b(0.0, {"p": 0.0, "q": 0.0})]},
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.9, "b": 0.1})]},
    }
    rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, speaker_claims={"a": [(0.0, 0.5)]})
    assert "speech_presence" in rows["speaker"][0]["coupled_from"]
    assert any(e.get("derivatives_refreshed") for e in logs["speaker"] if e["round"] >= 1)


def test_an_axis_is_not_listed_as_coupling_to_itself() -> None:
    """Shared structure an axis contributed to is not evidence that axis gave itself."""
    axes = {
        "speech_presence": {"raw_16k": [_b(0.0, {"p": 0.0, "q": 0.0})]},
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.9, "b": 0.1})]},
    }
    rows, _ = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, speaker_claims={"a": [(0.0, 0.5)]})
    assert "speech_presence" not in rows["speech_presence"][0]["coupled_from"]


def test_a_round_that_revised_the_shared_structure_is_not_credited_with_the_drop() -> None:
    """The self-confirmation guard, reached through ``overwrote_values``.

    A round that re-derived its own derivatives produced any subsequent fall itself, so it must not
    buy the loop more rounds on the strength of it.
    """
    axes = {
        "speech_presence": {"raw_16k": [_b(0.0, {"p": 0.0, "q": 0.0})]},
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.9, "b": 0.1})]},
    }
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, speaker_claims={"a": [(0.0, 0.5)]})
    later = [e for e in logs["speaker"] if e["round"] >= 1]
    assert later, "the coupled round must actually run"
    assert all((e["credited_epistemic_change"] or 0.0) >= 0 for e in later)


def test_an_unsettled_axis_does_not_re_derive_the_mask() -> None:
    """Only a region the previous round settled may withdraw trust.

    A bucket nobody resolved is not evidence of quiet, and a mask that filled those in would
    discount signals on the strength of a gap.
    """
    unsettled = {"speech_presence": [{"start": 0.0, "end": 0.5, "uncertainty": 0.95}]}
    assert derive_mask_from_axes(unsettled, Derivatives()) is None


def test_an_unmeasured_bucket_yields_no_mask_region() -> None:
    """Absence of a claim is not a claim of absence."""
    unmeasured = {"speech_presence": [{"start": 0.0, "end": 0.5, "uncertainty": None}]}
    assert derive_mask_from_axes(unmeasured, Derivatives()) is None


def test_a_tentative_mask_withdraws_proportionally_less_trust() -> None:
    """The mask's own confidence already gates how far it may act."""
    sure = derive_mask_from_axes({"speech_presence": [{"start": 0.0, "end": 0.5, "uncertainty": 0.0}]}, Derivatives())
    unsure = derive_mask_from_axes({"speech_presence": [{"start": 0.0, "end": 0.5, "uncertainty": 0.3}]}, Derivatives())
    assert sure is not None and unsure is not None
    assert sure.mask_regions[0]["confidence"] > unsure.mask_regions[0]["confidence"]


def test_the_settled_threshold_is_policy_rather_than_a_literal() -> None:
    """Named and replaceable, like every other L2 threshold."""
    rows = {"speech_presence": [{"start": 0.0, "end": 0.5, "uncertainty": 0.5}]}
    assert derive_mask_from_axes(rows, Derivatives()) is None
    assert derive_mask_from_axes(rows, Derivatives(), settled_below=0.6) is not None


def test_several_axes_can_be_run_fully_isolated() -> None:
    """A coupling that cannot be turned off cannot be evaluated against anything.

    Isolation has to be reachable with the *same* axis set, not by running one axis at a time —
    otherwise the comparison changes two things at once and says nothing about the coupling.
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, derive=None, couple_axes=False)
    assert set(rows) == {"speaker", "speech_presence"}
    for axis_rows in rows.values():
        for row in axis_rows:
            assert row["coupled_from"] == []
            assert not any(str(s).startswith("axis::") for s in row["contributing_signals"])
    assert all(e.get("derivatives_refreshed", False) is False for log in logs.values() for e in log)


def test_isolation_is_what_the_coupling_is_measured_against() -> None:
    """Same axes, same signals, coupling the only difference."""
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    coupled = _run(axes)
    isolated, _ = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, derive=None, couple_axes=False)
    assert coupled["speech_presence"][0]["uncertainty"] != isolated["speech_presence"][0]["uncertainty"]


# ── N axes, not three ────────────────────────────────────────────────────


def test_the_driver_takes_any_number_of_axes() -> None:
    """The design names four with a fifth punted; hard-coding three would break on the fourth."""
    four = _axes(
        speech_presence={"a": 0.2, "b": 0.2},
        speaker={"c": 0.5, "d": 0.5},
        asr={"e": 0.3, "f": 0.3},
        background_mask={"g": 0.5, "h": 0.5},
    )
    rows = _run(four)
    assert set(rows) == {"speech_presence", "speaker", "asr", "background_mask"}
    assert rows["speech_presence"][0]["coupled_from"] == ["asr", "background_mask", "speaker"]


def test_cross_axis_voters_are_ordered_so_output_stays_byte_identical() -> None:
    """Mutual influence applied in a different sequence can reach a different fixed point."""
    four = _axes(
        speech_presence={"a": 0.2},
        speaker={"c": 0.5},
        asr={"e": 0.3},
        background_mask={"g": 0.5},
    )
    row = _run(four)["speech_presence"][0]
    assert row["coupled_from"] == sorted(row["coupled_from"])


# ── interleaving ─────────────────────────────────────────────────────────


def test_every_axis_completes_round_zero_before_any_axis_runs_round_one() -> None:
    """The structural fix. Running one axis to completion first is why D-11 could not act.

    An axis's value only exists once it has folded round 0, so an axis that finished all its rounds
    beforehand could never have seen it.
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3)
    for axis_log in logs.values():
        assert axis_log[0]["round"] == 0
    assert {e["round"] for log in logs.values() for e in log} >= {0, 1}


def test_round_one_reads_round_zero_not_the_same_round() -> None:
    """Within a round the axes must not see each other's fresh values.

    Reading a partially-updated round makes the result depend on axis order, which breaks the
    byte-reproducibility the convergence outputs promise (FR-011f).
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    forward, _ = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3)
    reversed_order = {k: axes[k] for k in reversed(list(axes))}
    backward, _ = fuse_axes(reversed_order, weights_by_axis=_weights(axes), max_rounds=3)
    assert forward["speech_presence"][0]["uncertainty"] == pytest.approx(backward["speech_presence"][0]["uncertainty"])


def test_c4_counts_a_pending_re_examination_as_an_untried_action() -> None:
    """A flagged region nobody has revisited is exactly the "untried action" C4 asks about."""
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3)
    presence_log = logs["speech_presence"]
    assert presence_log[-1]["action_scope"] == "cross_axis"
    assert "c4" not in presence_log[-1]["blocking"], "the re-examination was performed, and counted"


# ── derivatives are round outputs, not fixed inputs ──────────────────────


def test_a_round_re_derives_the_mask_from_the_previous_round_s_axes() -> None:
    """L2 round N takes round N-1's outputs *and* the signals, and re-derives what it derives.

    The mask and the speaker claims are estimates, not givens. Frozen at whatever round 0 produced,
    every later round withdraws trust on the strength of a judgement the loop had already improved
    on — the same staleness the per-axis driver had for the axes themselves.
    """
    from senselab.audio.workflows.audio_analysis.fuse import Derivatives

    seen: list[int] = []

    def _derive(rows_by_axis: dict, current: Derivatives) -> Derivatives:
        seen.append(len(rows_by_axis))
        return Derivatives(
            mask_regions=({"start": 0.0, "end": 0.5, "state": "target_free", "confidence": 1.0},),
            speaker_claims={"a": [(0.0, 0.5)]},
        )

    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, derive=_derive)
    assert seen, "the derivative stage must actually run"
    assert seen[0] == 2, "it is handed every axis's previous-round rows"
    assert any(e.get("derivatives_refreshed") for e in logs["speaker"] if e["round"] >= 1)


def test_a_stale_derivative_is_reported_as_stale_rather_than_looking_current() -> None:
    """No `derive` means round 0's judgement is reused, and the log has to say so.

    A reader cannot otherwise tell a mask the loop refreshed from one it never revisited.
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, derive=None)
    later = [e for e in logs["speaker"] if e["round"] >= 1]
    assert later and all(e["derivatives_refreshed"] is False for e in later)


def test_a_derive_hook_returning_nothing_leaves_the_derivatives_alone() -> None:
    """Having no better estimate must not be read as the estimate now being empty."""
    from senselab.audio.workflows.audio_analysis.fuse import Derivatives

    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    _rows, logs = fuse_axes(
        axes,
        weights_by_axis=_weights(axes),
        max_rounds=3,
        derive=lambda _rows, _current: None,
    )
    later = [e for e in logs["speaker"] if e["round"] >= 1]
    assert later and all(e["derivatives_refreshed"] is False for e in later)
    assert isinstance(Derivatives(), Derivatives)


def test_the_single_axis_driver_is_the_same_loop_with_one_axis() -> None:
    """One round loop, not two. Two implementations could disagree about the same history."""
    single, log = fuse_rounds({"raw_16k": [_b(0.0, {"a": 0.3})]}, weights={"a": 1.0}, max_rounds=3)
    assert len(single) == 1
    assert log[0]["round"] == 0


# ── the gate that replaced the assigned one, measured ────────────────────


def test_evidence_overlap_between_axes_is_measured_not_assumed() -> None:
    """The successor to the deleted constant: measure what two axes hold in common.

    `speaker_identity` already had this lesson — its gate is `claim.support`, a measured quantity,
    after a *declared* source-kind gate suppressed the source that matched the spoken names on a
    real recording. A hand-set 0.4 was that same construct one module over.
    """
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    target = [{"contributing_signals": ["shared", "own"]}]
    source = [{"contributing_signals": ["shared", "theirs"]}]
    assert measure_axis_overlap(target, source) == pytest.approx(0.5)


def test_an_axis_telling_us_only_what_we_already_have_is_attenuated() -> None:
    """Fully-overlapping evidence is one computation counted twice, so it earns little weight."""
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    rows = [{"contributing_signals": ["a", "b"]}]
    assert measure_axis_overlap(rows, rows) == pytest.approx(1.0)


def test_an_axis_with_independent_evidence_is_not_attenuated() -> None:
    """Disjoint evidence is a genuine second observation."""
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    assert measure_axis_overlap([{"contributing_signals": ["a"]}], [{"contributing_signals": ["z"]}]) == 0.0


def test_a_prior_round_s_coupling_is_not_counted_as_the_source_s_own_evidence() -> None:
    """Otherwise the overlap measure feeds on itself and drifts every round."""
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    source = [{"contributing_signals": ["z", "axis::speaker"]}]
    assert measure_axis_overlap([{"contributing_signals": ["a"]}], source) == 0.0


def test_unmeasurable_overlap_applies_no_discount() -> None:
    """A factor never measured must not act as a discount — the rule the constant broke."""
    from senselab.audio.workflows.audio_analysis.fuse import measure_axis_overlap

    assert measure_axis_overlap([{"contributing_signals": ["a"]}], [{"contributing_signals": []}]) is None


def test_the_measured_overlap_reaches_the_weight_and_is_recorded() -> None:
    """Auditable: a reader can see which factor discounted a cross-axis input, and by how much."""
    axes = _axes(speaker={"a": 0.5, "shared": 0.5}, speech_presence={"shared": 0.0, "d": 0.0})
    rows = _run(axes)
    row = rows["speech_presence"][0]
    assert row["signal_weights"]["axis::speaker"] < 1.0, "half its evidence is already ours"
    assert row["weight_basis"]["axis::speaker"]["evidence_overlap"] > 0.0


# ── D-10: a round may re-measure, not only re-weight ─────────────────────


def _unsettled() -> dict:
    return {
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.5, "b": 0.5})]},
        "speech_presence": {"raw_16k": [_b(0.0, {"c": 0.5, "d": 0.5})]},
    }


def test_a_round_can_re_measure_an_unsettled_region() -> None:
    """D-10: an estimate that improves a signal may re-measure it, not merely re-weight it.

    Re-weighting can only redistribute the evidence already gathered. A region no amount of
    re-weighting resolves is exactly the one that needs a finer look.
    """
    calls: list[tuple[str, int]] = []

    def _remeasure(axis: str, regions: list, rows_by_axis: dict) -> dict | None:
        calls.append((axis, len(regions)))
        return {"finer": [_b(0.0, {f"{axis}_refined": 0.0})]}

    axes = _unsettled()
    rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, remeasure=_remeasure)
    assert calls, "an unsettled region must reach the re-measurement hook"
    assert any("speaker_refined" in r["contributing_signals"] for r in rows["speaker"])
    assert any(e.get("remeasured") for e in logs["speaker"] if e["round"] >= 1)


def test_a_settled_region_is_not_re_measured() -> None:
    """Re-measurement is expensive and pointless where the answer already holds."""
    calls: list[str] = []

    def _remeasure(axis: str, regions: list, rows_by_axis: dict) -> dict | None:
        calls.append(axis)
        return None

    axes = _axes(speaker={"a": 0.0, "b": 0.0}, speech_presence={"c": 0.0, "d": 0.0})
    fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, remeasure=_remeasure)
    assert calls == [], "nothing was unsettled, so nothing needed a finer look"


def test_a_region_is_not_re_measured_twice() -> None:
    """The same finer look repeated is not new evidence, and C4 would never reach zero."""
    calls: list[int] = []

    def _remeasure(axis: str, regions: list, rows_by_axis: dict) -> dict | None:
        calls.append(len(regions))
        return None

    axes = _unsettled()
    fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=5, remeasure=_remeasure)
    assert calls and calls[0] > 0
    assert all(c == 0 for c in calls[2:]) or len(calls) <= 2, "a re-measured region stops being pending"


def test_a_re_measured_round_is_not_credited_with_the_drop_it_caused() -> None:
    """A re-measurement *replaces* a value, so C1 must not read the fall as independent agreement."""
    axes = _unsettled()
    _rows, logs = fuse_axes(
        axes,
        weights_by_axis=_weights(axes),
        max_rounds=3,
        remeasure=lambda axis, regions, rows: {"finer": [_b(0.0, {f"{axis}_refined": 0.0})]},
    )
    later = [e for e in logs["speaker"] if e["round"] >= 1]
    assert later and all((e["credited_epistemic_change"] or 0.0) >= 0 for e in later)


def test_pending_re_measurements_block_c4() -> None:
    """A region nobody has looked at again is the untried action C4 exists to notice."""
    axes = _unsettled()
    _rows, logs = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, remeasure=lambda a, r, rows: None)
    assert logs["speaker"][0]["action_scope"] == "remeasure"


def test_the_unsettled_threshold_is_policy_rather_than_a_literal() -> None:
    """Which regions deserve a finer look is a decision, so it is named and replaceable."""
    calls: list[str] = []
    axes = _axes(speaker={"a": 0.3, "b": 0.3}, speech_presence={"c": 0.3, "d": 0.3})

    def _hook(axis: str, regions: list, rows: dict) -> dict | None:
        calls.append(axis)
        return None

    fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=2, remeasure=_hook, unsettled_above=0.99)
    assert calls == []
    fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=2, remeasure=_hook, unsettled_above=0.1)
    assert calls != []


# ── wiring the real pipeline's shapes ────────────────────────────────────


def test_mask_regions_carry_uncertainty_and_are_converted_not_defaulted() -> None:
    """The mask reports `uncertainty`; `regional_weights` reads `confidence`.

    Passing the rows through unconverted lets `region.get("confidence", 1.0)` default to *fully
    confident*, so a mask that was unsure would withdraw maximum trust — the same
    absent-reads-as-fine failure this design keeps hitting.
    """
    from senselab.audio.workflows.audio_analysis.fuse import mask_regions_from_rows

    rows = [{"start": 0.0, "end": 1.0, "state": "target_free", "uncertainty": 0.25}]
    out = mask_regions_from_rows(rows)
    assert out[0]["confidence"] == pytest.approx(0.75)


def test_a_mask_row_with_no_uncertainty_is_dropped_rather_than_assumed_certain() -> None:
    """An unmeasured mask row must not act with full authority."""
    from senselab.audio.workflows.audio_analysis.fuse import mask_regions_from_rows

    assert mask_regions_from_rows([{"start": 0.0, "end": 1.0, "state": "target_free"}]) == []


def test_speaker_claims_are_the_spans_each_model_actually_named_a_speaker() -> None:
    """Regional trust discounts a signal where it *claimed* a speaker, so the claim must be real."""
    from senselab.audio.workflows.audio_analysis.fuse import speaker_claims_from_votes

    votes = [
        {"start": 0.0, "end": 0.5, "votes": {"diarA": {"speaker_label": "SPEAKER_00"}}},
        {"start": 0.5, "end": 1.0, "votes": {"diarA": {"speaker_label": "<silent>"}}},
        {"start": 1.0, "end": 1.5, "votes": {"diarA": {"speaker_label": "SPEAKER_01"}}},
    ]
    claims = speaker_claims_from_votes(votes)
    assert claims["diarA"] == [(0.0, 0.5), (1.0, 1.5)], "silence is not a speaker claim"


def test_a_model_that_named_nobody_makes_no_claim() -> None:
    """Silence about a region is not a violation in it — the rule regional trust already follows."""
    from senselab.audio.workflows.audio_analysis.fuse import speaker_claims_from_votes

    votes = [{"start": 0.0, "end": 0.5, "votes": {"quiet": {"speaker_label": None}}}]
    assert speaker_claims_from_votes(votes) == {}


def test_cross_axis_input_cannot_create_buckets_the_axis_never_measured() -> None:
    """An axis emits on its own grid; coupling informs it, it does not extend it.

    Found on a real run: the mask contributed one whole-clip region, and the fused
    `background_mask` axis emitted 1197 buckets — every one of them sourced from the other axes'
    finer grid. An axis with one datum of its own cannot contribute anything back, so it only
    echoes, and the echo is indistinguishable in the output from a measurement. The overlap gate
    does not catch it either: overlap is measured on signal names, and this axis's name collides
    with nothing, so a fully-derived axis reads as fully independent.
    """
    by_axis = {
        "background_mask": {"mask": [_b(0.0, {"mask": 0.0})]},
        "speaker": {"raw_16k": [_b(0.0, {"a": 0.5}), _b(0.5, {"a": 0.5}), _b(1.0, {"a": 0.5})]},
    }
    rows, _logs = fuse_axes(
        by_axis,
        weights_by_axis={"background_mask": {"mask": 1.0}, "speaker": {"a": 1.0}},
        max_rounds=3,
    )
    assert len(rows["background_mask"]) == 1, "the mask measured one bucket and must emit one"


def test_every_round_s_rows_are_returned_not_only_the_last() -> None:
    """`L2/round<N>/` promises one directory per round; only the final one was ever written.

    `fuse_axes` returned final rows only, and every row carried the final round index, so the
    writer's per-round loop saw one distinct round. A single map cannot distinguish "settled
    immediately" from "moved a long way and then settled" — which is the exact question the
    oscillation verdict on a real run left unanswerable.
    """
    axes = _axes(speaker={"a": 0.5, "b": 0.5}, speech_presence={"c": 0.0, "d": 0.0})
    rows, logs, history = fuse_axes(axes, weights_by_axis=_weights(axes), max_rounds=3, return_history=True)
    assert set(history) == set(rows), "every axis carries a per-round history"
    for axis, log in logs.items():
        assert sorted(history[axis]) == [e["round"] for e in log], f"{axis}: one snapshot per round run"
    assert history["speaker"][0][0]["round"] == 0
    assert rows["speaker"] == history["speaker"][max(history["speaker"])], "final rows are the last round's"
