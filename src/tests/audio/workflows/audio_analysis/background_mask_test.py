"""Background mask construction (T024-T030, FR-031 to FR-045).

The mask marks regions free of **target activity** — activity from the near-microphone
participant — not regions free of speech. That distinction is the whole point, and it has
a sharp consequence: for a breathing or cough task, speech detection reports *no activity*
during the target event. A mask built from speech activity alone would admit the target
breaths, and since AudioSet maps ``Breathing`` and ``Cough`` to the ``people`` category,
they would be reported as a background human-sound source — the signal being collected,
misattributed as an environmental finding.

That is the failure SC-024 exists to catch, and it is why FR-033a forbids building the
mask from speech detection alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.background_mask import (
    MASK_STATES,
    BackgroundMask,
    build_mask,
    requires_label_detection,
    target_event_types_for,
    target_labels_for,
)
from senselab.audio.workflows.audio_analysis.calibration import DEFAULT_DETECTION_MARGIN

PROFILE = DEFAULT_DETECTION_MARGIN


def _buckets(spec: list[tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    """Build bucket rows from ``(start, end, target_confidence, uncertainty)`` tuples."""
    return [{"start": s, "end": e, "target_confidence": c, "uncertainty": u} for s, e, c, u in spec]


def _quiet(n: int, *, dur: float = 0.5, uncertainty: float = 0.05) -> list[dict[str, Any]]:
    return _buckets([(i * dur, (i + 1) * dur, 0.0, uncertainty) for i in range(n)])


# ── task metadata drives the target definition (T026, FR-033/FR-033b) ──


def test_recognized_task_selects_its_target_events() -> None:
    """A breathing task's target is breath, not speech."""
    types, provenance = target_event_types_for("breath", PROFILE)
    assert "breath" in types
    assert provenance == "recognized"


def test_speech_task_includes_participant_vocal_activity() -> None:
    """In a speech task the participant's breaths and mouth noise are still target."""
    types, _ = target_event_types_for("speech", PROFILE)
    assert "speech" in types and "breath" in types


def test_absent_task_metadata_falls_back_conservatively() -> None:
    """Unknown task ⟹ treat any participant vocal activity as target, and say so."""
    types, provenance = target_event_types_for(None, PROFILE)
    assert provenance == "fallback"
    assert {"speech", "breath", "cough"} <= set(types)


def test_unrecognized_task_type_falls_back_and_is_recorded() -> None:
    """A typo must not silently produce a speech-only mask."""
    _types, provenance = target_event_types_for("bretahing", PROFILE)
    assert provenance == "fallback"


# ── FR-033a: speech detection alone is not enough ──────────────────────


def test_non_speech_target_requires_label_detection() -> None:
    """The decisive requirement. Speech detection is silent during a breath.

    Without this, a breath task's targets enter the mask and are reported as
    background ``people`` sounds — the collected signal read as an environmental finding.
    """
    assert requires_label_detection(["breath"]) is True
    assert requires_label_detection(["cough"]) is True


def test_speech_only_target_does_not_require_label_detection() -> None:
    """A pure speech target is adequately served by voice activity + diarization."""
    assert requires_label_detection(["speech"]) is False


def test_target_labels_for_breath_are_audioset_classes() -> None:
    """The labels must be ones the scene classifier can actually emit."""
    labels = target_labels_for(["breath"])
    assert "Breathing" in labels


def test_target_labels_for_cough_include_related_events() -> None:
    """Throat clearing accompanies coughing closely enough to count as target."""
    labels = target_labels_for(["cough"])
    assert "Cough" in labels and "Throat clearing" in labels


def test_breath_and_cough_labels_map_to_the_people_category() -> None:
    """Why misattribution is the specific risk, not a generic one."""
    from senselab.audio.workflows.audio_analysis.sound_sources import load_source_category_map

    mapping = load_source_category_map()["map"]
    for label in ("Breathing", "Cough"):
        assert mapping[label] == "people"


# ── three states (T024, FR-032) ───────────────────────────────────────


def test_every_state_is_reachable() -> None:
    """Each state is a distinct outcome, including non-target interest.

    ``nontarget_active`` needs non-target evidence to be reachable at all, which is why it
    takes an extra field rather than falling out of the target confidence alone.
    """
    buckets = _buckets(
        [
            (0.0, 0.5, 0.95, 0.05),  # clearly active
            (10.0, 10.5, 0.02, 0.05),  # clearly free, nothing else there
            (20.0, 20.5, 0.50, 0.90),  # cannot tell
        ]
    )
    buckets.append(
        {"start": 30.0, "end": 30.5, "target_confidence": 0.02, "uncertainty": 0.05, "nontarget_confidence": 0.9}
    )
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    assert {r.state for r in mask.regions} == set(MASK_STATES)


def test_states_are_limited_to_the_declared_set() -> None:
    """No fourth state may appear."""
    mask = build_mask(_quiet(4), task_type="speech", profile=PROFILE)
    assert all(r.state in MASK_STATES for r in mask.regions)


def test_uncertainty_is_in_unit_range() -> None:
    """Mask uncertainty shares the axes' [0, 1] convention."""
    mask = build_mask(_quiet(4), task_type="speech", profile=PROFILE)
    assert all(0.0 <= r.uncertainty <= 1.0 for r in mask.regions)


def test_high_uncertainty_blocks_target_free_even_when_confidence_is_low() -> None:
    """Low confidence plus high uncertainty is not a usable background region."""
    mask = build_mask(_buckets([(0.0, 0.5, 0.01, 0.95)]), task_type="speech", profile=PROFILE)
    assert mask.regions[0].state == "indeterminate"


def test_contiguous_same_state_buckets_share_a_region_id() -> None:
    """Regions are contiguous runs, so a consumer reasons about spans not buckets."""
    mask = build_mask(_quiet(5), task_type="speech", profile=PROFILE)
    assert len({r.region_id for r in mask.regions}) == 1


# ── guard interval (T025, FR-034) ─────────────────────────────────────


def test_bucket_adjacent_to_target_activity_is_not_target_free() -> None:
    """Reverberant tail and classifier context contaminate the interval after activity."""
    buckets = _buckets([(0.0, 0.5, 0.95, 0.05)] + [(0.5 + i * 0.5, 1.0 + i * 0.5, 0.0, 0.05) for i in range(6)])
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    first_after = next(r for r in mask.regions if r.start >= 0.5)
    assert first_after.state != "target_free"


def test_guard_interval_is_recorded_per_region() -> None:
    """The trimmed duration is attributable, not just the state change."""
    buckets = _buckets([(0.0, 0.5, 0.95, 0.05)] + [(0.5 + i * 0.5, 1.0 + i * 0.5, 0.0, 0.05) for i in range(8)])
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    assert any(r.guard_trimmed_s > 0.0 for r in mask.regions)


def test_far_from_activity_is_still_target_free() -> None:
    """The guard is an interval, not a blanket veto."""
    buckets = _buckets([(0.0, 0.5, 0.95, 0.05)] + [(0.5 + i * 0.5, 1.0 + i * 0.5, 0.0, 0.05) for i in range(20)])
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    assert any(r.state == "target_free" for r in mask.regions)


# ── non-target speech stays in the mask (T028, FR-033c) ───────────────


def test_distant_talker_stays_masked_and_is_flagged() -> None:
    """Target-free but not speech-free. It is a finding, not contamination."""
    buckets = _buckets([(0.0, 0.5, 0.0, 0.05)] * 1)
    buckets[0]["nontarget_speech"] = True
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    assert mask.regions[0].state == "target_free"
    assert mask.regions[0].contains_nontarget_speech is True


def test_nontarget_speech_defaults_to_false_when_unknown() -> None:
    """Absent evidence is not evidence of a distant talker."""
    mask = build_mask(_quiet(2), task_type="speech", profile=PROFILE)
    assert mask.regions[0].contains_nontarget_speech is False


# ── empty / negligible mask (T029, FR-038/FR-040) ─────────────────────


def test_continuous_target_activity_yields_an_empty_mask() -> None:
    """Reported as a stated limitation, not an omitted field."""
    mask = build_mask(_buckets([(i * 0.5, (i + 1) * 0.5, 0.95, 0.05) for i in range(20)]), "speech", profile=PROFILE)
    assert mask.is_empty is True
    assert mask.total_masked_s == pytest.approx(0.0)


def test_totals_are_always_reported() -> None:
    """Background findings can never be read without knowing how much supports them."""
    mask = build_mask(_quiet(10), task_type="speech", profile=PROFILE)
    assert mask.total_masked_s > 0.0
    assert 0.0 <= mask.masked_fraction <= 1.0


def test_tiny_mask_is_flagged_as_negligible() -> None:
    """A mask too small to support conclusions says so."""
    buckets = _buckets([(0.0, 0.5, 0.0, 0.05)] + [(0.5 + i * 0.5, 1.0 + i * 0.5, 0.95, 0.05) for i in range(40)])
    mask = build_mask(buckets, task_type="speech", profile=PROFILE)
    assert mask.negligible_fraction is True


def test_entirely_target_free_recording_masks_everything() -> None:
    """No target activity means the whole recording is usable background."""
    mask = build_mask(_quiet(20), task_type="speech", profile=PROFILE)
    assert mask.masked_fraction == pytest.approx(1.0)
    assert mask.is_empty is False


# ── long-window support (T030, FR-045) ────────────────────────────────


def test_short_region_cannot_support_a_long_window_decision() -> None:
    """A region mostly zero-padding gives a gain-dependent pad/signal contrast."""
    buckets = _buckets([(0.0, 0.5, 0.0, 0.05), (0.5, 1.0, 0.95, 0.05)])
    mask = build_mask(buckets, task_type="speech", profile=PROFILE, long_window_s=10.24)
    free = next(r for r in mask.regions if r.state == "target_free")
    assert free.supports_long_window is False


def test_long_region_supports_a_long_window_decision() -> None:
    """A region at least as long as the window needs no padding."""
    mask = build_mask(_quiet(40), task_type="speech", profile=PROFILE, long_window_s=10.24)
    free = next(r for r in mask.regions if r.state == "target_free")
    assert free.supports_long_window is True


def test_mask_reports_how_many_regions_support_the_long_window() -> None:
    """So a consumer knows whether long-window results are available at all (SC-032)."""
    mask = build_mask(_quiet(4), task_type="speech", profile=PROFILE, long_window_s=10.24)
    assert mask.regions_supporting_long_window == 0
    assert mask.regions_total >= 1


# ── provenance and serialization ──────────────────────────────────────


def test_mask_records_metadata_provenance() -> None:
    """A mask built via fallback is never mistaken for one built with task context."""
    assert build_mask(_quiet(4), task_type="breath", profile=PROFILE).metadata_provenance == "recognized"
    assert build_mask(_quiet(4), task_type=None, profile=PROFILE).metadata_provenance == "fallback"


def test_mask_serializes_to_the_contract_shape() -> None:
    """contracts/background-mask.md - every field a consumer reads."""
    doc = build_mask(_quiet(6), task_type="breath", profile=PROFILE).to_json()
    for key in (
        "task_type",
        "target_event_types",
        "metadata_provenance",
        "guard_interval_s",
        "total_masked_s",
        "masked_fraction",
        "is_empty",
        "negligible_fraction",
        "regions_supporting_long_window",
        "regions_total",
    ):
        assert key in doc, f"missing {key}"


def test_mask_rows_carry_every_contract_column() -> None:
    """background_mask.parquet columns."""
    rows = build_mask(_quiet(6), task_type="breath", profile=PROFILE).to_rows()
    for col in (
        "region_id",
        "start",
        "end",
        "state",
        "uncertainty",
        "guard_trimmed_s",
        "contains_nontarget_speech",
        "supports_long_window",
        "target_event_types",
    ):
        assert col in rows[0], f"missing {col}"


def test_empty_bucket_list_yields_an_empty_mask() -> None:
    """A pass that produced no buckets is not an error."""
    mask = build_mask([], task_type="speech", profile=PROFILE)
    assert isinstance(mask, BackgroundMask)
    assert mask.is_empty is True and mask.regions == []


# ── target-activity evidence (T033, FR-033a) ───────────────────────────


def _yamnet(windows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"diarization": {"by_model": {}}, "yamnet": {"status": "ok", "result": [windows]}}


def test_absent_target_label_is_low_score_evidence_not_missing_evidence() -> None:
    """A target label outside a top-k window bounds its score from above.

    Treating absence as "no evidence" instead marks every quiet bucket ``indeterminate``
    and leaves the mask permanently empty — for exactly the tasks that need it most. This
    was caught by running the stage rather than by reading it.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    # Realistic top-k: the classifier reports several labels, so the smallest is a
    # genuinely informative bound on the absent target.
    summary = _yamnet(
        [{"start": 0.0, "end": 4.0, "label_scores": [{"Silence": 0.95}, {"Inside, small room": 0.04}, {"Hum": 0.01}]}]
    )
    rows = target_confidence_by_bucket(summary, [(0.0, 0.5), (0.5, 1.0)], ["breath"])
    assert all(r["uncertainty"] < 1.0 for r in rows), "absence must not read as unexamined"
    assert all(r["target_confidence"] < 0.6 for r in rows), "an informative bound must read as inactive"


def test_present_target_label_drives_confidence_up() -> None:
    """A detected breath is target activity, evidenced by the label not by voice activity."""
    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    summary = _yamnet([{"start": 0.0, "end": 1.0, "label_scores": [{"Breathing": 0.88}]}])
    rows = target_confidence_by_bucket(summary, [(0.0, 0.5)], ["breath"])
    assert rows[0]["target_confidence"] == pytest.approx(0.88)


def test_no_evidence_source_at_all_is_maximally_uncertain() -> None:
    """A bucket nothing examined must not read as confidently target-free."""
    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    rows = target_confidence_by_bucket({"diarization": {"by_model": {}}}, [(0.0, 0.5)], ["breath"])
    assert rows[0]["uncertainty"] == pytest.approx(1.0)
    assert build_mask(rows, "breath", profile=PROFILE).regions[0].state == "indeterminate"


def test_breath_task_masks_the_quiet_stretch() -> None:
    """The end-to-end property: label-evidenced target activity, then usable background."""
    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    summary = _yamnet(
        [
            {"start": 0.0, "end": 2.0, "label_scores": [{"Breathing": 0.85}]},
            {"start": 2.0, "end": 20.0, "label_scores": [{"Silence": 0.95}, {"Inside, small room": 0.02}]},
        ]
    )
    buckets = [(i * 0.5, (i + 1) * 0.5) for i in range(40)]
    mask = build_mask(target_confidence_by_bucket(summary, buckets, ["breath"]), "breath", profile=PROFILE)
    assert mask.is_empty is False
    assert mask.masked_fraction > 0.5
    # No non-target evidence is supplied here, so ``nontarget_active`` is unreachable by
    # construction — the states this fixture can produce are the other three.
    assert {r.state for r in mask.regions} == {"target_active", "target_free", "indeterminate"}


def test_uninformative_absent_label_bound_yields_cannot_tell() -> None:
    """A bound above the active threshold is true but says nothing.

    One confident label in a window bounds an absent target only by that label's score.
    Reporting it as the confidence would read a quiet bucket as target-active, so the
    honest answer is ``indeterminate`` rather than a fabricated activity claim.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    summary = _yamnet([{"start": 0.0, "end": 4.0, "label_scores": [{"Silence": 0.95}]}])
    rows = target_confidence_by_bucket(summary, [(0.0, 0.5)], ["breath"], active_threshold=0.6)
    assert rows[0]["uncertainty"] == pytest.approx(1.0)
    assert build_mask(rows, "breath", profile=PROFILE).regions[0].state == "indeterminate"


# ── segment shapes, found only by a real run ────────────────────────────


def test_nested_diarization_result_is_flattened() -> None:
    """``diarize_audios`` returns one entry per input audio, so a single call nests.

    Every fixture here had used the flat shape, so this failed only on real audio: the
    inner lists were handed to the segment reader where segments were expected.
    """
    from types import SimpleNamespace

    from senselab.audio.workflows.audio_analysis.background_mask import _flatten_segments

    nested = [[SimpleNamespace(start=0.0, end=1.0), SimpleNamespace(start=2.0, end=3.0)]]
    assert len(_flatten_segments(nested)) == 2


def test_segment_bounds_read_from_object_and_dict() -> None:
    """In-memory segments are objects; cache-deserialized ones are dicts."""
    from types import SimpleNamespace

    from senselab.audio.workflows.audio_analysis.background_mask import _seg_bounds

    assert _seg_bounds(SimpleNamespace(start=1.0, end=2.0)) == (1.0, 2.0)
    assert _seg_bounds({"start": 1.0, "end": 2.0}) == (1.0, 2.0)


def test_segment_bounds_of_an_unreadable_value_is_none() -> None:
    """An unreadable segment degrades to None rather than raising.

    The eager-default trap: ``getattr(x, "start", x.get("start"))`` evaluates the fallback
    even when the attribute exists, and raises on anything that is neither an object with
    the attribute nor a mapping. Returning None keeps one odd segment from failing a run.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import _seg_bounds

    assert _seg_bounds([1, 2]) is None
    assert _seg_bounds(None) is None
    assert _seg_bounds({"start": "not-a-number", "end": 2.0}) is None


def test_speech_activity_from_the_real_nested_shape() -> None:
    """End-to-end on the shape a real diarizer produces."""
    from types import SimpleNamespace

    from senselab.audio.workflows.audio_analysis.background_mask import _speech_activity_by_bucket

    nested = [[SimpleNamespace(start=0.0, end=1.0), SimpleNamespace(start=2.0, end=3.0)]]
    summary = {"diarization": {"by_model": {"m": {"status": "ok", "result": nested}}}}
    assert _speech_activity_by_bucket(summary, [(0.0, 0.5), (1.5, 1.9), (2.5, 2.9)]) == [1.0, 0.0, 1.0]


def test_mask_is_skipped_on_a_non_unmodified_variant() -> None:
    """The mask is only meaningful on unmodified audio (found on a real run).

    Measured: the enhanced pass masked 50% of a recording against the unmodified pass's
    17.9%, because enhancement removes the non-speech evidence target activity is read
    from. That inflates the mask exactly where the background was destroyed, so the pass
    is skipped with the reason recorded rather than producing a generous-looking mask.
    """
    from types import SimpleNamespace

    import torch

    import senselab.audio.workflows.audio_analysis.stages as stages_mod
    from senselab.audio.workflows.audio_analysis.stage_context import PassPlan, StageContext

    ctx = StageContext(perturbation="enhanced", audio_signature="e" * 64, variant="speech_enhanced")
    audio = SimpleNamespace(waveform=torch.zeros((1, 16000)), sampling_rate=16000)
    # A waveform/sampling_rate stand-in: this path skips before touching anything else.
    summary = stages_mod.run_pass(audio, ctx, PassPlan(background_mask=True))  # type: ignore[arg-type]
    assert summary["background_mask"]["status"] == "skipped"
    assert "unmodified" in summary["background_mask"]["reason"]


def test_enhanced_perturbation_maps_to_the_enhanced_variant() -> None:
    """The gate is only as good as the variant the pass actually declares.

    An end-to-end run showed the mask still being written on the enhanced pass: the skip
    logic was correct, but the CLI never set the variant, so every pass reported
    ``unmodified``. The unit test had passed because it constructed the context with the
    variant set by hand — the wiring was the untested part.
    """
    import ast
    import pathlib

    src = pathlib.Path("scripts/analyze_audio.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_stage_context")
    body = ast.get_source_segment(src, fn) or ""
    assert "speech_enhanced" in body, "_stage_context must derive the variant from the pass label"
    assert "variant=" in body, "_stage_context must pass the variant to StageContext"


def test_participant_speech_is_target_activity_in_every_task() -> None:
    """The near-field participant talking is target activity whatever they were asked to do.

    Found on a real cough recording: with ``cough`` mapping to cough events only, the
    spoken tail was target-FREE and would have been reported as a background ``speech``
    source. That is the same misattribution the task mapping exists to prevent, arriving
    from the other direction.
    """
    for task in ("speech", "breath", "cough"):
        types, provenance = target_event_types_for(task, PROFILE)
        assert provenance == "recognized"
        assert "speech" in types, f"{task} task omits participant speech from its targets"


def test_task_specific_events_are_still_task_specific() -> None:
    """Adding speech everywhere must not blur the tasks into one another."""
    cough, _ = target_event_types_for("cough", PROFILE)
    breath, _ = target_event_types_for("breath", PROFILE)
    assert "cough" in cough and "cough" not in breath
    assert "breath" in breath and "breath" not in cough


# ── graded evidence: the mask must be able to be uncertain (D-24) ──────────────
#
# The defect these pin, measured on ``english_conversation_higgs_audio_v2_20260804-145231``:
# ``L2/background_mask.parquet`` held **one** region spanning 0-21 s, ``target_active``, at
# ``uncertainty`` 0.0, while the ``background_mask`` axis over the same recording had 1070 buckets
# averaging 0.0949. ``build_mask`` is not the problem — it classifies per bucket and run-length
# encodes — the evidence reaching it was boolean, so every bucket scored an identical, maximal
# confidence and the encoding correctly collapsed them into one.
#
# Two independent saturators produced that. Both are the same error in different clothes: a
# yes/no answer to a question whose measurement is graded.


def test_partially_covered_bucket_is_partially_active() -> None:
    """Coverage is a proportion, not a hit test.

    ``_speech_activity_by_bucket`` asked "does any segment overlap this bucket" per diarizer, so a
    bucket a segment clips by 10 ms and one it fills entirely both scored 1.0. ``diar_covered_
    fraction`` already exists for exactly this reason and says so: "A segment overlapping 5% of a
    bucket and one covering all of it are not the same evidence, and a bool cannot tell them
    apart — which matters most at segment boundaries."
    """
    from types import SimpleNamespace

    from senselab.audio.workflows.audio_analysis.background_mask import _speech_activity_by_bucket

    nested = [[SimpleNamespace(start=0.0, end=1.0)]]
    summary = {"diarization": {"by_model": {"m": {"status": "ok", "result": nested}}}}
    # 0.8-1.0 is covered only up to 1.0, so one fifth of the 1.0-1.8 bucket.
    covered = _speech_activity_by_bucket(summary, [(0.0, 0.5), (0.8, 1.8), (2.0, 2.5)])

    assert covered[0] == pytest.approx(1.0), "a fully covered bucket is fully active"
    assert covered[2] == pytest.approx(0.0), "an uncovered bucket is inactive"
    assert 0.0 < (covered[1] or 0.0) < 1.0, f"a partially covered bucket must be graded, got {covered[1]}"


def test_word_evidence_does_not_pin_a_bucket_to_absolute_certainty() -> None:
    """A word touching a bucket raises confidence; it does not make the bucket certain.

    ``apply_span_evidence`` set ``uncertainty`` to ``min(u, 0.0)`` — which is 0.0 for every bucket
    a span touches, however slightly. Zero is the confident claim "nothing about this bucket is in
    doubt", asserted from a single overlapping word, and on a conversation that is most buckets.
    """
    from senselab.audio.workflows.audio_analysis.background_mask import apply_span_evidence

    rows = [
        {"start": 0.0, "end": 1.0, "target_confidence": 0.3, "uncertainty": 0.8},
        {"start": 1.0, "end": 2.0, "target_confidence": 0.3, "uncertainty": 0.8},
    ]
    # A word covering 5% of the first bucket, and one covering all of the second.
    out = apply_span_evidence(rows, target_spans=[(0.0, 0.05), (1.0, 2.0)])

    assert out[0]["target_confidence"] > 0.3, "a word is evidence the target was active"
    assert out[0]["target_confidence"] < out[1]["target_confidence"], (
        "a word clipping a bucket is weaker evidence than a bucket full of words"
    )
    assert out[0]["uncertainty"] > 0.0, f"a 5% word overlap left the bucket absolutely certain: {out[0]}"
    assert out[1]["uncertainty"] < 0.8, "a bucket full of words is more certain than before"


def test_continuous_conversation_still_has_uncertainty_at_its_boundaries() -> None:
    """The reported symptom, at the smallest scale that reproduces it.

    Diarization covering a whole recording made every bucket identical, so the mask was one region
    with nothing to be uncertain about. Speaker turns have boundaries, and a bucket straddling one
    is partly covered — so a real conversation cannot produce a single perfectly certain verdict.
    """
    from types import SimpleNamespace

    from senselab.audio.workflows.audio_analysis.background_mask import target_confidence_by_bucket

    # Two turns with a 0.15 s gap: continuous-looking coverage with real boundaries in it.
    nested = [[SimpleNamespace(start=0.0, end=1.05), SimpleNamespace(start=1.2, end=2.0)]]
    summary = {"diarization": {"by_model": {"m": {"status": "ok", "result": nested}}}}
    buckets = [(round(i * 0.1, 6), round((i + 1) * 0.1, 6)) for i in range(20)]

    rows = target_confidence_by_bucket(summary, buckets, ("speech",))
    mask = build_mask(rows, "conversation", profile=PROFILE)

    assert len({r.state for r in mask.regions}) > 1, "a recording with a pause is not one uniform state"
    assert any(r.uncertainty > 0.0 for r in mask.regions), (
        f"every region is perfectly certain: {[(r.state, r.uncertainty) for r in mask.regions]}"
    )
