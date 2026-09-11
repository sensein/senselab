"""The family taxonomy ruleset, over synthetic feature records rather than corpus files."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.features import TRANSCRIPT_CAP, RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    FAMILY_SETS,
    GateOutcome,
    Ruleset,
    evaluate_gate,
    evaluate_routes,
    load_ruleset,
    max_token_repeat,
    tally_families,
)
from senselab.audio.workflows.triage.vocabulary import BRANCHES


@pytest.fixture(scope="module")
def ruleset() -> Ruleset:
    """The packaged ruleset.

    Returns:
        The ruleset as ``data/config/default.yaml`` declares it.
    """
    return load_ruleset(load_triage_config())


def _features(family: str, **overrides: object) -> RecordingFeatures:
    """A feature record whose every gate is readable and silent, before the overrides.

    Args:
        family: The task family.
        overrides: Fields to replace.

    Returns:
        The record.
    """
    record = RecordingFeatures(
        stem=f"sub-1_ses-1_task-{family}",
        run_root="run",
        task_id=family,
        family=family,
        duration_s=10.0,
        words={"agreement": 0, "total": 0, "lexical": 0},
        consensus_present=True,
        transcript="",
        residual={"energy_fraction": 0.0},
        span_longest_s={"amplitude": 0.0},
        span_stats={"all.peak_over_floor_db_max": 0.0},
        classifier_streams=["plain|yamnet"],
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


class TestTheDeclaredRoute:
    """Which branch a family's instruction declares, and which measurement confirms it."""

    def test_every_branch_declares_a_family_set_that_exists(self, ruleset: Ruleset) -> None:
        """A branch naming a set the families module does not have would route nothing, silently."""
        assert set(ruleset.declared_family_set) == set(BRANCHES)
        for set_name in ruleset.declared_family_set.values():
            assert set_name in FAMILY_SETS

    def test_the_declared_branch_comes_off_the_task_entity(self, ruleset: Ruleset) -> None:
        """One family per branch, read through the family sets rather than restated here."""
        assert ruleset.declared_branches("harvard-sentences-list") == ("SPEECH",)
        assert ruleset.declared_branches("voluntary-cough") == ("AIRWAY",)
        assert ruleset.declared_branches("prolonged-vowel") == ("VOICE",)
        assert ruleset.declared_branches("diadochokinesis-pataka") == ("DDK",)

    def test_a_family_no_set_carries_declares_nothing(self, ruleset: Ruleset) -> None:
        """An unrecognised task id is not silently swept into a branch."""
        assert ruleset.declared_branches("unknown") == ()


class TestEachGateFiresAndDoesNot:
    """Every gate, at and under its own threshold."""

    @pytest.mark.parametrize(
        ("gate", "field", "firing", "silent"),
        [
            ("speech.declared", "words", {"agreement": 4}, {"agreement": 3}),
            ("speech.intrusion", "words", {"agreement": 2}, {"agreement": 1}),
            ("voice.sustained", "span_longest_s", {"amplitude": 3.0}, {"amplitude": 2.99}),
            ("airway.breath", "residual", {"energy_fraction": 0.10}, {"energy_fraction": 0.09}),
            (
                "airway.cough",
                "span_stats",
                {"all.peak_over_floor_db_max": 50.0},
                {"all.peak_over_floor_db_max": 49.9},
            ),
        ],
    )
    def test_a_scalar_gate_fires_at_its_threshold_and_not_below(
        self, ruleset: Ruleset, gate: str, field: str, firing: dict[str, float], silent: dict[str, float]
    ) -> None:
        """The comparison is inclusive, so the threshold itself is a firing value."""
        rule = ruleset.gates[gate]
        assert evaluate_gate(_features("free-speech", **{field: firing}), rule) is GateOutcome.FIRED
        assert evaluate_gate(_features("free-speech", **{field: silent}), rule) is GateOutcome.SILENT

    def test_the_glide_gate_reads_the_yamnet_singing_union(self, ruleset: Ruleset) -> None:
        """Any member of the union carries the gate, not only the bare ``Singing`` label."""
        rule = ruleset.gates["voice.glide"]
        assert evaluate_gate(_features("glides-low-to-high", peaks={"plain|yamnet|Humming": 0.05}), rule) is (
            GateOutcome.FIRED
        )
        assert evaluate_gate(_features("glides-low-to-high", peaks={"plain|yamnet|Humming": 0.04}), rule) is (
            GateOutcome.SILENT
        )

    def test_the_ddk_gate_counts_repeated_transcript_tokens(self, ruleset: Ruleset) -> None:
        """Three repeats of one syllable carries it; two do not."""
        rule = ruleset.gates["ddk.declared"]
        assert evaluate_gate(_features("diadochokinesis-pa", transcript="pa pa pa"), rule) is GateOutcome.FIRED
        assert evaluate_gate(_features("diadochokinesis-pa", transcript="pa pa"), rule) is GateOutcome.SILENT


class TestAMissingMeasurementIsNotANegative:
    """``unavailable`` is carried through the gate, the evaluation and the tally."""

    def test_a_gate_whose_feature_was_never_written_reads_unavailable(self, ruleset: Ruleset) -> None:
        """An absent residual measurement is not a residual of zero."""
        record = _features("breath-sounds", residual={})
        assert evaluate_gate(record, ruleset.gates["airway.breath"]) is GateOutcome.UNAVAILABLE

    def test_a_gate_whose_classifier_never_ran_reads_unavailable(self, ruleset: Ruleset) -> None:
        """A stream with no YAMNet summary cannot be scored against the singing union."""
        record = _features("glides-low-to-high", classifier_streams=[])
        assert evaluate_gate(record, ruleset.gates["voice.glide"]) is GateOutcome.UNAVAILABLE

    def test_a_gate_with_no_consensus_transcript_reads_unavailable(self, ruleset: Ruleset) -> None:
        """No transcript is no repeat count, which is not a repeat count of zero."""
        record = _features("diadochokinesis-pa", consensus_present=False)
        assert evaluate_gate(record, ruleset.gates["ddk.declared"]) is GateOutcome.UNAVAILABLE

    def test_an_unavailable_declared_branch_is_neither_confirmed_nor_unconfirmed(self, ruleset: Ruleset) -> None:
        """A breath recording with no residual and no span statistic is unmeasured, not empty."""
        record = _features("breath-sounds", residual={}, span_stats={})
        result = evaluate_routes(record, ruleset)
        assert result.declared == ("AIRWAY",)
        assert result.unavailable == ("AIRWAY",)
        assert result.confirmed == ()
        assert result.unconfirmed == ()
        assert result.fell_through is True

    def test_the_tally_counts_unavailable_separately_from_a_silent_gate(self, ruleset: Ruleset) -> None:
        """Collapsing the two would report a broken store as a negative measurement."""
        unmeasured = _features("breath-sounds", residual={}, span_stats={})
        silent = _features("breath-sounds")
        tallies = tally_families([evaluate_routes(unmeasured, ruleset), evaluate_routes(silent, ruleset)])
        row = tallies["breath-sounds"]
        assert row.recordings == 2
        assert row.declared["AIRWAY"] == 2
        assert row.unavailable["AIRWAY"] == 1
        assert row.confirmed["AIRWAY"] == 0
        assert row.fell_through == 2


class TestTheDiscoveredRouteAdds:
    """Discovered speech is an addition to the declared route, never a replacement."""

    def test_a_sustained_vowel_with_an_intruding_voice_routes_to_both(self, ruleset: Ruleset) -> None:
        """The vowel keeps VOICE; the intrusion adds SPEECH."""
        record = _features("prolonged-vowel", span_longest_s={"amplitude": 6.0}, words={"agreement": 2})
        result = evaluate_routes(record, ruleset)
        assert result.declared == ("VOICE",)
        assert result.confirmed == ("VOICE",)
        assert result.discovered == ("SPEECH",)
        assert result.routed == ("SPEECH", "VOICE")
        assert result.fell_through is False

    def test_a_declared_speech_route_is_not_also_reported_as_discovered(self, ruleset: Ruleset) -> None:
        """A confirmed branch is confirmed; the discovery gate does not double-count it."""
        record = _features("free-speech", words={"agreement": 9})
        result = evaluate_routes(record, ruleset)
        assert result.confirmed == ("SPEECH",)
        assert result.discovered == ()
        assert result.routed == ("SPEECH",)


class TestARecordingRoutesToNothing:
    """The fall-through count is a first-class output."""

    def test_a_silent_recording_lands_in_no_bucket(self, ruleset: Ruleset) -> None:
        """Every gate readable, every gate silent: declared, unconfirmed, routed nowhere."""
        result = evaluate_routes(_features("prolonged-vowel"), ruleset)
        assert result.declared == ("VOICE",)
        assert result.unconfirmed == ("VOICE",)
        assert result.routed == ()
        assert result.fell_through is True

    def test_the_tally_counts_the_fall_through_per_family(self, ruleset: Ruleset) -> None:
        """Two families, one fall-through each, counted where the owner asked to read them."""
        evaluations = [
            evaluate_routes(_features("prolonged-vowel"), ruleset),
            evaluate_routes(_features("prolonged-vowel", span_longest_s={"amplitude": 4.0}), ruleset),
            evaluate_routes(_features("voluntary-cough"), ruleset),
        ]
        tallies = tally_families(evaluations)
        assert tallies["prolonged-vowel"].fell_through == 1
        assert tallies["prolonged-vowel"].confirmed["VOICE"] == 1
        assert tallies["voluntary-cough"].fell_through == 1
        assert tallies["voluntary-cough"].declared["AIRWAY"] == 1
        assert set(tallies["voluntary-cough"].declared) == set(BRANCHES)


class TestTheTokenRepeatCounter:
    """The one feature the ruleset reads that no extraction writes."""

    def test_a_transcript_with_no_repeat_counts_one(self) -> None:
        """Every token distinct is a maximum of one, which no threshold above one clears."""
        assert max_token_repeat("the quick brown fox") == 1

    def test_an_empty_transcript_counts_zero(self) -> None:
        """No token is not one token."""
        assert max_token_repeat("") == 0

    def test_case_and_punctuation_do_not_split_a_repeat(self) -> None:
        """``Pa,`` and ``pa`` are the same syllable, and a hyphenated run is four of them."""
        assert max_token_repeat("Pa, pa. PA! pa") == 4
        assert max_token_repeat("pa-pa-pa-pa") == 4

    def test_a_token_the_cap_cut_in_half_is_not_another_instance(self) -> None:
        """The 300-character cap truncates mid-token, and the fragment counts as its own token."""
        transcript = ("diadochokinesis " * 25)[:TRANSCRIPT_CAP]
        assert len(transcript) == TRANSCRIPT_CAP
        assert transcript.endswith("diadochokine")
        assert max_token_repeat(transcript) == 18
