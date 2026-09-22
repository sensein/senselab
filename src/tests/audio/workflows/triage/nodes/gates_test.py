"""The gates: three layers, resolved family-first, applied in VERDICT. Nothing here loads a model."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from senselab.audio.workflows.triage.config import (
    DATA_MAP_PATHS,
    TriageConfig,
    UnknownConfigKey,
    load_triage_config,
)
from senselab.audio.workflows.triage.nodes import gates as gates_module
from senselab.audio.workflows.triage.nodes.branches import (
    DETECT_GROUP,
    EXPECTATIONS,
    PARAM_KEYS,
    BranchParams,
    Result,
    branch_params,
)
from senselab.audio.workflows.triage.nodes.gates import (
    AT_LEAST,
    AT_MOST,
    CONFORMANCE_GATES,
    DEFAULT_LAYER,
    FAMILY_LAYER,
    GATE_KEYS,
    GATE_SECTION,
    GATE_SPECS,
    GROUP_LAYER,
    LAYERS,
    REQUIRED_COUNT,
    TYPICAL_COUNT,
    UNDETERMINED,
    UNGATEABLE_READINGS,
    GateBounds,
    GateSpec,
    Pattern,
    _gate_specs,
    apply_gates,
    conformance_gate_names,
    load_gate_bounds,
)

MOVED = (
    "continuity_min",
    "coverage_min",
    "dominant_segment_min_fraction",
    "echo_overlap_max",
    "f0_spread_max_semitones",
    "gap_off_task_min_s",
    "interval_max_s",
    "monotone_tolerance_semitones",
    "production_min_s",
    "rate_prominence_min",
    "repeat_overlap_min",
    "response_min_s",
    "score_min",
    "train_min_s",
    "verbatim_overlap_max",
    "voiced_fraction_min",
)
"""The sixteen keys ``specs/20260921-gates-in-verdict/design.md`` moves out of ``branch:``."""

STAYED = (
    "burst_window_ms",
    "echo_ngram_n",
    "effort_split_hz",
    "event_min_s",
    "f0_spread_window_s",
    "label_sets",
    "modulation_band_hz",
    "pause_min_s",
    "peak_prominence_db",
    "phoneme_place_classes",
    "phoneme_vowel_classes",
    "run_gap_max_s",
    "smoothing_window_s",
    "trough_return_db",
    "voiced_strength_min",
)
"""The fifteen instrument settings the same table keeps in ``branch:``."""


def _bounds(group: Pattern, **gates: object) -> GateBounds:
    """A resolved gate table with exactly the gates named, all from the group layer.

    Args:
        group: The task group.
        **gates: Gate name to bound.

    Returns:
        The bounds.
    """
    return GateBounds(
        group=group,
        family=None,
        bounds=dict(gates),
        layers=dict.fromkeys(gates, GROUP_LAYER),
    )


def _layered(tmp_path: Path, body: str) -> TriageConfig:
    """The packaged configuration with one partial YAML over it.

    Args:
        tmp_path: Where the override is written.
        body: The partial YAML.

    Returns:
        The resolved configuration.
    """
    path = tmp_path / f"override-{abs(hash(body)) % 10**10}.yaml"
    path.write_text(body)
    return load_triage_config(path)


class TestTheSplitIsTheDesignsTable:
    """Sixteen gates moved and fifteen instrument settings stayed, and neither list moved twice."""

    def test_every_moved_key_left_the_branch_section(self, config: TriageConfig) -> None:
        """A gate is not an operating point a branch body may ask for any more."""
        packaged = set(config.require("branch"))
        assert packaged.isdisjoint(MOVED)
        assert set(PARAM_KEYS).isdisjoint(MOVED)

    def test_every_instrument_setting_stayed(self, config: TriageConfig) -> None:
        """The right-hand column is definitional and is still read from ``branch:``."""
        assert set(config.require("branch")) == set(STAYED) == set(PARAM_KEYS)

    def test_every_moved_key_is_a_gate(self) -> None:
        """Nothing was dropped on the way across."""
        assert set(MOVED) <= set(GATE_KEYS)

    def test_the_gate_table_is_one_vocabulary_in_two_places(self, config: TriageConfig) -> None:
        """Every name the packaged layers spell is a declared gate, and every gate is used."""
        spelled = {name for name in (config.get(f"{GATE_SECTION}.{DEFAULT_LAYER}") or {})}
        spelled |= {
            name for group in Pattern for name in (config.get(f"{GATE_SECTION}.{GROUP_LAYER}.{group.name}") or {})
        }
        assert spelled <= set(GATE_SPECS)
        assert spelled == set(GATE_SPECS), sorted(set(GATE_SPECS) - spelled)

    def test_each_layer_is_a_data_mapping_so_an_override_may_add_a_gate(self) -> None:
        """Expressing a gate for a task that has none is the point; an override must be able to."""
        assert f"{GATE_SECTION}.{DEFAULT_LAYER}" in DATA_MAP_PATHS
        assert f"{GATE_SECTION}.{FAMILY_LAYER}" in DATA_MAP_PATHS
        for group in Pattern:
            assert f"{GATE_SECTION}.{GROUP_LAYER}.{group.name}" in DATA_MAP_PATHS


class TestAGroupThatNamesNoValueDoesNotApplyTheGate:
    """The design's mechanism, and its headline case."""

    def test_glide_is_not_bound_by_a_held_vowels_spread(self, config: TriageConfig) -> None:
        """``GLIDE``'s purpose is that pitch moves, so the spread bound is simply not configured."""
        assert "f0_spread_max_semitones" in (config.get(f"{GATE_SECTION}.{GROUP_LAYER}.SUSTAINED") or {})
        assert "f0_spread_max_semitones" not in (config.get(f"{GATE_SECTION}.{GROUP_LAYER}.GLIDE") or {})
        assert "f0_spread_max_semitones" not in (config.get(f"{GATE_SECTION}.{DEFAULT_LAYER}") or {})

    def test_an_unconfigured_gate_is_not_applied_at_all(self) -> None:
        """Not configured is not the same as configured and unreadable."""
        bounds = _bounds(Pattern.GLIDE, production_min_s=0.5)
        conformance, applied = apply_gates(
            ("production_min_s", "f0_spread_max_semitones"), bounds, {"carrier_duration_s": 1.0}
        )
        assert conformance is True
        assert [gate.name for gate in applied] == ["production_min_s"]

    def test_a_group_configuring_none_of_its_patterns_gates_answers_undetermined(self) -> None:
        """No gate applied is no answer, never a pass."""
        conformance, applied = apply_gates(("production_min_s",), _bounds(Pattern.GLIDE), {})
        assert conformance == UNDETERMINED
        assert applied == []

    def test_sound_coverage_declares_no_conformance_gate(self) -> None:
        """Its instruction states no conformable expectation, so it answers nothing."""
        assert CONFORMANCE_GATES[Pattern.SOUND_COVERAGE] == ()


class TestAnAbsentReadingIsUndeterminedAndNeverFalse:
    """The rule three separate defects came from, enforced in one place."""

    def test_a_reading_nothing_wrote_yields_undetermined(self) -> None:
        """A gate whose instrument never measured has not failed; it has not been answered."""
        conformance, applied = apply_gates(("production_min_s",), _bounds(Pattern.SUSTAINED, production_min_s=0.5), {})
        assert conformance == UNDETERMINED
        assert applied[0].passed == UNDETERMINED
        assert applied[0].value is None

    def test_a_bound_nobody_measured_yields_undetermined(self) -> None:
        """A refusal is a decision; an unmeasured bound decides nothing."""
        conformance, applied = apply_gates(
            ("production_min_s",), _bounds(Pattern.SUSTAINED, production_min_s=None), {"carrier_duration_s": 0.1}
        )
        assert conformance == UNDETERMINED
        assert applied[0].passed == UNDETERMINED

    def test_one_unanswerable_gate_makes_the_whole_conformance_undetermined(self) -> None:
        """Answering on the gates that could be read would report a partial reading as a whole one."""
        bounds = _bounds(Pattern.SUSTAINED, production_min_s=0.5, continuity_min=0.5)
        conformance, _ = apply_gates(("production_min_s", "continuity_min"), bounds, {"carrier_duration_s": 1.0})
        assert conformance == UNDETERMINED

    def test_a_reading_that_is_present_and_short_is_a_false(self) -> None:
        """The distinction the rule protects: absent is not the same as measured and failing."""
        conformance, applied = apply_gates(
            ("production_min_s",), _bounds(Pattern.SUSTAINED, production_min_s=0.5), {"carrier_duration_s": 0.12}
        )
        assert conformance is False
        assert applied[0].passed is False


class TestTheComparisonRunsTheWayTheGateDeclares:
    """``at_least`` and ``at_most`` are declared once, in ``GATE_SPECS``."""

    @pytest.mark.parametrize(
        ("gate", "reading", "value", "bound", "expected"),
        [
            ("production_min_s", "carrier_duration_s", 0.5, 0.5, True),
            ("production_min_s", "carrier_duration_s", 0.49, 0.5, False),
            ("f0_spread_max_semitones", "carrier_f0_spread_semitones", 2.0, 2.0, True),
            ("f0_spread_max_semitones", "carrier_f0_spread_semitones", 2.01, 2.0, False),
            ("omissions_max", "expected_tokens_omitted", 0, 0, True),
            ("omissions_max", "expected_tokens_omitted", 1, 0, False),
        ],
    )
    def test_the_boundary_is_inclusive_on_both_sides(
        self, gate: str, reading: str, value: float, bound: float, expected: bool
    ) -> None:
        """At the bound passes, whichever direction the gate reads."""
        group = Pattern.SUSTAINED if gate != "omissions_max" else Pattern.ORDERED_TOKENS
        conformance, _ = apply_gates((gate,), _bounds(group, **{gate: bound}), {reading: value})
        assert conformance is expected

    def test_every_gate_declares_one_of_the_two_comparisons(self) -> None:
        """A third comparison would be a rule with no table entry."""
        assert {spec.op for spec in GATE_SPECS.values()} == {AT_LEAST, AT_MOST}


class TestEveryConformanceGateHasAReading:
    """A gate VERDICT applies must name what it reads; a located gate must not be applied there."""

    def test_each_pattern_names_only_gates_that_read_something(self) -> None:
        """VERDICT locates nothing, so it can only apply a gate with a scalar reading."""
        for pattern, names in CONFORMANCE_GATES.items():
            for name in names:
                assert GATE_SPECS[name].reading is not None, f"{pattern.name}/{name}"

    def test_a_located_gate_applied_as_a_conformance_raises(self) -> None:
        """Applying ``gap_off_task_min_s`` in VERDICT would be a rule with nowhere to point."""
        with pytest.raises(ValueError, match="carries no reading"):
            apply_gates(("gap_off_task_min_s",), _bounds(Pattern.ORDERED_TOKENS, gap_off_task_min_s=1.0), {})

    def test_every_pattern_has_a_conformance_rule_and_a_configured_group(self, config: TriageConfig) -> None:
        """A group nobody configured would raise at the first recording that declared it."""
        for pattern in Pattern:
            assert pattern in CONFORMANCE_GATES
            assert load_gate_bounds(config, pattern).group is pattern

    def test_a_recall_is_gated_on_coverage_rather_than_on_how_long_it_ran(self) -> None:
        """``verbatim_source`` reassigns the conformance term, as the old branch code did."""
        assert conformance_gate_names(Pattern.FREE_RESPONSE) == ("response_min_s",)
        assert conformance_gate_names(Pattern.FREE_RESPONSE, anti_pattern="verbatim_source") == ("coverage_min",)
        assert conformance_gate_names(Pattern.FREE_RESPONSE, anti_pattern="verbatim_prompt") == ("response_min_s",)


class TestTheBoundsAreReadFromOnePlace:
    """``load_gate_bounds`` is the only reader of ``verdict.gates``."""

    def test_a_group_the_packaged_file_does_not_spell_raises(self, config: TriageConfig) -> None:
        """A typo in a group name is a programming error, never a missing measurement."""
        values = {**config.values, "verdict": {**config.values["verdict"], "gates": {}}}
        stripped = TriageConfig(name=config.name, version=config.version, config_hash="x", values=values)
        with pytest.raises(UnknownConfigKey):
            load_gate_bounds(stripped, Pattern.SUSTAINED)

    def test_a_layer_naming_something_that_is_not_a_gate_raises(self, config: TriageConfig) -> None:
        """The gate vocabulary is closed; an unknown name is not silently ignored."""
        gates = {**config.values["verdict"]["gates"], DEFAULT_LAYER: {"not_a_gate": 1.0}}
        values = {**config.values, "verdict": {**config.values["verdict"], "gates": gates}}
        broken = TriageConfig(name=config.name, version=config.version, config_hash="x", values=values)
        with pytest.raises(ValueError, match="not gates"):
            load_gate_bounds(broken, Pattern.SUSTAINED)

    def test_an_override_may_add_a_gate_to_a_group(self, tmp_path: Path) -> None:
        """Expressing a gate for a group that has none is what the change is for."""
        settings = _layered(
            tmp_path, "verdict:\n  gates:\n    by_group:\n      GLIDE:\n        f0_spread_max_semitones: 12.0\n"
        )
        bounds = load_gate_bounds(settings, Pattern.GLIDE)
        assert bounds.bound("f0_spread_max_semitones") == 12.0
        assert bounds.bound("production_min_s") == 0.5

    def test_the_bounds_record_names_the_group_the_family_and_every_layer(self, config: TriageConfig) -> None:
        """A decision must be readable backwards without rerunning anything."""
        record = load_gate_bounds(config, Pattern.SUSTAINED, "maximum-phonation-time").record()
        assert record["group"] == "sustained"
        assert record["family"] == "maximum-phonation-time"
        assert record["bounds"]["f0_spread_max_semitones"] == 2.0
        assert record["layers"]["f0_spread_max_semitones"] == GROUP_LAYER


class TestAGateResolvesFamilyThenGroupThenDefault:
    """A task group alone is too coarse; the layers are how a family says so."""

    def test_the_family_layer_ships_empty(self, config: TriageConfig) -> None:
        """Every value moved at its current setting; an empty layer says no difference is derived."""
        assert config.get(f"{GATE_SECTION}.{FAMILY_LAYER}") in (None, {})

    def test_the_default_layer_ships_empty_because_nothing_is_universal(self, config: TriageConfig) -> None:
        """``gap_off_task_min_s`` reaches six groups of twelve; no gate reaches all of them."""
        assert config.get(f"{GATE_SECTION}.{DEFAULT_LAYER}") in (None, {})
        reach = {
            name: sum(
                1 for group in Pattern if name in (config.get(f"{GATE_SECTION}.{GROUP_LAYER}.{group.name}") or {})
            )
            for name in GATE_KEYS
        }
        assert max(reach.values()) < len(Pattern), "a gate every group names would belong in the default"

    def test_a_family_overrides_its_group_key_by_key_and_not_wholesale(self, tmp_path: Path) -> None:
        """The obvious bug: replacing the group's whole mapping and losing the gates it kept."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n    by_family:\n      maximum-phonation-time:\n        production_min_s: 4.0\n",
        )
        bounds = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time")
        assert bounds.bound("production_min_s") == 4.0
        assert bounds.layer("production_min_s") == FAMILY_LAYER
        # The three the family said nothing about are still the group's, and still applied.
        assert bounds.bound("voiced_fraction_min") == 0.5
        assert bounds.bound("f0_spread_max_semitones") == 2.0
        assert bounds.bound("continuity_min") == 0.5
        assert {
            bounds.layer(name) for name in ("voiced_fraction_min", "f0_spread_max_semitones", "continuity_min")
        } == {GROUP_LAYER}

    def test_one_family_of_a_group_is_overridden_and_its_sibling_is_not(self, tmp_path: Path) -> None:
        """The case the layers exist for: two families, one group, one of them different."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n    by_family:\n      maximum-phonation-time:\n        production_min_s: 4.0\n",
        )
        overridden = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time")
        sibling = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time-v2")
        assert overridden.bound("production_min_s") == 4.0
        assert sibling.bound("production_min_s") == 0.5
        assert sibling.layer("production_min_s") == GROUP_LAYER

    def test_a_family_may_add_a_gate_its_group_does_not_name(self, tmp_path: Path) -> None:
        """A per-family bound need not be a per-group one first."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n    by_family:\n      glides-low-to-high:\n        f0_spread_max_semitones: 12.0\n",
        )
        bounds = load_gate_bounds(settings, Pattern.GLIDE, "glides-low-to-high")
        assert bounds.bound("f0_spread_max_semitones") == 12.0
        assert bounds.layer("f0_spread_max_semitones") == FAMILY_LAYER
        assert not load_gate_bounds(settings, Pattern.GLIDE, "glides-high-to-low").names("f0_spread_max_semitones")

    def test_the_default_layer_is_taken_where_neither_family_nor_group_names_a_gate(self, tmp_path: Path) -> None:
        """The third layer, exercised: nothing ships in it, so an override is how it is tested."""
        settings = _layered(tmp_path, "verdict:\n  gates:\n    default:\n      items_min: 7\n")
        bounds = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time")
        assert bounds.bound("items_min") == 7
        assert bounds.layer("items_min") == DEFAULT_LAYER

    def test_a_group_beats_the_default_and_a_family_beats_both(self, tmp_path: Path) -> None:
        """Most specific first, and the record says which one won."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n"
            "    default:\n      production_min_s: 9.0\n"
            "    by_family:\n      maximum-phonation-time:\n        production_min_s: 4.0\n",
        )
        family = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time")
        group = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time-v2")
        ungrouped = load_gate_bounds(settings, Pattern.ITEM_LIST, "animal-fluency")
        assert (family.bound("production_min_s"), family.layer("production_min_s")) == (4.0, FAMILY_LAYER)
        assert (group.bound("production_min_s"), group.layer("production_min_s")) == (0.5, GROUP_LAYER)
        assert (ungrouped.bound("production_min_s"), ungrouped.layer("production_min_s")) == (9.0, DEFAULT_LAYER)

    def test_the_out_of_family_mode_reads_no_family_layer(self, tmp_path: Path) -> None:
        """``detect_*`` declares no family, so a family bound cannot reach it."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n    by_family:\n      maximum-phonation-time:\n        production_min_s: 4.0\n",
        )
        bounds = load_gate_bounds(settings, Pattern.SUSTAINED, None)
        assert bounds.bound("production_min_s") == 0.5
        assert bounds.family is None

    def test_a_family_may_clear_a_bound_its_group_measured(self, tmp_path: Path) -> None:
        """Nulling at the family layer is how one family says the group's bound is not derived for it."""
        settings = _layered(
            tmp_path,
            "verdict:\n  gates:\n    by_family:\n      maximum-phonation-time:\n        production_min_s:\n",
        )
        bounds = load_gate_bounds(settings, Pattern.SUSTAINED, "maximum-phonation-time")
        assert bounds.names("production_min_s")
        assert bounds.bound("production_min_s") is None
        assert bounds.layer("production_min_s") == FAMILY_LAYER

    def test_the_layers_are_declared_most_specific_first(self) -> None:
        """The resolution order is data, not a sequence of ifs a reader has to trace."""
        assert LAYERS == (FAMILY_LAYER, GROUP_LAYER, DEFAULT_LAYER)

    def test_every_applied_gate_records_the_layer_that_supplied_it(self, config: TriageConfig) -> None:
        """A reader must tell a family-specific bound from an inherited one without the config."""
        bounds = load_gate_bounds(config, Pattern.SUSTAINED, "maximum-phonation-time")
        _, applied = apply_gates(("production_min_s",), bounds, {"carrier_duration_s": 8.0})
        assert applied[0].record()["layer"] == GROUP_LAYER
        assert applied[0].record()["keyed_under"] == "SUSTAINED"


class TestNoGateReadsACountNobodyGave:
    """A measured median is reported beside a reading; no bound may ever be put on it."""

    def test_no_gate_reads_a_typical_count(self) -> None:
        """Ten `/pa/` was never spoken to anyone; a participant giving eight did the task."""
        assert TYPICAL_COUNT in UNGATEABLE_READINGS
        assert TYPICAL_COUNT not in {spec.reading for spec in GATE_SPECS.values()}

    def test_a_gate_table_that_reads_a_typical_count_is_refused_at_import(self) -> None:
        """Structural, not a convention: the table itself refuses the binding."""
        with pytest.raises(ValueError, match="which no gate may be bound to"):
            _gate_specs({"typical_min": GateSpec(TYPICAL_COUNT, AT_LEAST, int)})

    def test_a_required_count_is_not_refused(self) -> None:
        """A number the instruction spoke may be gated once a tolerance is derived."""
        assert REQUIRED_COUNT not in UNGATEABLE_READINGS
        assert _gate_specs({"required_min": GateSpec(REQUIRED_COUNT, AT_LEAST, int)})

    def test_the_shipped_table_is_built_through_the_refusal(self) -> None:
        """A table assigned around the factory would make the refusal unreachable."""
        tree = ast.parse(Path(gates_module.__file__).read_text())
        assigned = [
            node
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "GATE_SPECS"
        ]
        [statement] = assigned
        assert isinstance(statement.value, ast.Call)
        assert isinstance(statement.value.func, ast.Name) and statement.value.func.id == "_gate_specs"

    def test_no_gate_reads_either_count_today(self) -> None:
        """Deriving a tolerance for the required kind is separate work this change enables."""
        readings = {spec.reading for spec in GATE_SPECS.values()}
        assert not readings & {REQUIRED_COUNT, TYPICAL_COUNT}

    def test_the_airway_event_gate_reads_what_was_found_and_not_what_was_declared(self) -> None:
        """``events_min`` asks whether the sound happened at all, not whether the count was met."""
        assert GATE_SPECS["events_min"].reading == "airway_events_found"
        bounds = _bounds(Pattern.EVENT_SERIES, events_min=1)
        assert apply_gates(("events_min",), bounds, {"airway_events_found": 2})[0] is True
        assert apply_gates(("events_min",), bounds, {"airway_events_found": 9})[0] is True

    def test_the_syllable_gate_reads_repetitions_found_and_not_the_ten_nobody_asked_for(self) -> None:
        """A DDK train is measured by its rate; the ten in the row is not a target."""
        assert GATE_SPECS["repetitions_min"].reading == "ddk_repetitions_found"
        bounds = _bounds(Pattern.SYLLABLE_TRAIN, repetitions_min=1)
        assert apply_gates(("repetitions_min",), bounds, {"ddk_repetitions_found": 8})[0] is True


class TestABranchReadsItsOwnGroupsGates:
    """``BranchParams.gate`` is bound to one group and refuses to guess."""

    def test_reading_a_gate_before_a_group_is_bound_raises(self, config: TriageConfig) -> None:
        """Which bound applies is a property of the task, and a body may not assume one."""
        with pytest.raises(KeyError, match="no task group is bound"):
            branch_params(config).gate("production_min_s")

    def test_a_name_that_is_not_a_gate_raises(self, config: TriageConfig) -> None:
        """A misspelled gate is a typo in the calling code."""
        with pytest.raises(KeyError, match="not a gate"):
            branch_params(config).bind(Pattern.SUSTAINED).gate("voiced_strength_min")

    def test_an_unconfigured_gate_returns_none_and_is_not_an_unmeasured_ask(self, config: TriageConfig) -> None:
        """A group that names no value for a gate is not asking for one."""
        params = branch_params(config).bind(Pattern.GLIDE)
        assert params.gate("f0_spread_max_semitones") is None
        assert params.missing == []

    def test_a_gate_configured_null_is_recorded_as_unmeasured(self, tmp_path: Path) -> None:
        """A bound nobody has measured is an ask the report carries, under its full path."""
        path = tmp_path / "over.yaml"
        path.write_text("verdict:\n  gates:\n    by_group:\n      SUSTAINED:\n        production_min_s:\n")
        params = branch_params(load_triage_config(path)).bind(Pattern.SUSTAINED)
        assert params.gate("production_min_s") is None
        assert params.missing == ["verdict.gates.by_group.SUSTAINED.production_min_s"]

    def test_binding_returns_the_same_instance_so_misses_accumulate(self, config: TriageConfig) -> None:
        """One record of what a branch asked for, whichever group it was bound to."""
        params = branch_params(config)
        assert params.bind(Pattern.SUSTAINED) is params


class TestABranchCannotReportAConformance:
    """The contract is structural: ``Result`` has no field to write a verdict into."""

    def test_the_result_carries_two_things(self) -> None:
        """``done`` is gone; a branch reports spans and findings and nothing else."""
        assert Result._fields == ("components", "deviations")

    def test_every_branch_has_a_detect_group(self) -> None:
        """The out-of-family mode still needs its instruments' bounds, keyed to a real group."""
        assert set(DETECT_GROUP) == set(EXPECTATIONS)
        for group in DETECT_GROUP.values():
            assert isinstance(group, Pattern)
