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
    score_branches,
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


class TestTheReferenceStandardIsNotARouter:
    """The declared family scores the routed set; it never decides which gates run."""

    def test_every_branch_names_a_family_set_that_exists(self, ruleset: Ruleset) -> None:
        """A branch naming a set the families module does not have would score against nothing."""
        assert set(ruleset.reference_family_set) == set(BRANCHES)
        for set_name in ruleset.reference_family_set.values():
            assert set_name in FAMILY_SETS

    def test_the_reference_branch_comes_off_the_task_entity(self, ruleset: Ruleset) -> None:
        """One family per branch, read through the family sets rather than restated here."""
        assert ruleset.reference_branches("harvard-sentences-list") == ("SPEECH",)
        assert ruleset.reference_branches("voluntary-cough") == ("AIRWAY",)
        assert ruleset.reference_branches("prolonged-vowel") == ("VOICE",)
        assert ruleset.reference_branches("diadochokinesis-pataka") == ("DDK",)

    def test_a_family_no_set_carries_is_a_reference_positive_for_nothing(self, ruleset: Ruleset) -> None:
        """An unrecognised task id is not silently swept into a branch."""
        assert ruleset.reference_branches("unknown") == ()

    def test_every_branch_has_gates_and_they_all_exist(self, ruleset: Ruleset) -> None:
        """A branch with no gate could never be routed to, whatever a recording carries."""
        assert set(ruleset.branch_gates) == set(BRANCHES)
        for names in ruleset.branch_gates.values():
            assert names
            assert all(name in ruleset.gates for name in names)


class TestEachGateFiresAndDoesNot:
    """Every gate, at and under its own threshold."""

    @pytest.mark.parametrize(
        ("gate", "field", "firing", "silent"),
        [
            ("speech.lexical", "words", {"lexical": 2}, {"lexical": 1}),
            ("speech.transcript_agreement", "words", {"agreement": 3}, {"agreement": 2}),
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

    def test_the_chant_gate_reads_the_yamnet_chant_label_alone(self, ruleset: Ruleset) -> None:
        """The union's own cut is 0.05; chant carries VOICE from 0.02 on its single label."""
        rule = ruleset.gates["voice.chant"]
        assert evaluate_gate(_features("prolonged-vowel", peaks={"plain|yamnet|Chant": 0.02}), rule) is (
            GateOutcome.FIRED
        )
        assert evaluate_gate(_features("prolonged-vowel", peaks={"plain|yamnet|Chant": 0.019}), rule) is (
            GateOutcome.SILENT
        )

    def test_the_ddk_gate_counts_repeated_transcript_tokens(self, ruleset: Ruleset) -> None:
        """Three repeats of one syllable carries it; two do not."""
        rule = ruleset.gates["ddk.lexical_repetition"]
        assert evaluate_gate(_features("diadochokinesis-pa", transcript="pa pa pa"), rule) is GateOutcome.FIRED
        assert evaluate_gate(_features("diadochokinesis-pa", transcript="pa pa"), rule) is GateOutcome.SILENT


class TestContentRoutesWithoutTheInstruction:
    """Every branch's gates run on every recording, whatever the task asked for."""

    def test_a_branch_the_family_never_declared_still_routes(self, ruleset: Ruleset) -> None:
        """A cough recording that carries five lexical words routes to SPEECH as well as AIRWAY."""
        record = _features(
            "voluntary-cough",
            words={"agreement": 1, "total": 5, "lexical": 5},
            span_stats={"all.peak_over_floor_db_max": 55.0},
        )
        result = evaluate_routes(record, ruleset)
        assert result.declared == ("AIRWAY",)
        assert result.routed == ("AIRWAY", "SPEECH")
        assert result.agreed == ("AIRWAY",)
        assert result.extra == ("SPEECH",)
        assert result.missed == ()
        assert result.fell_through is False

    def test_a_declared_branch_that_does_not_route_is_missed(self, ruleset: Ruleset) -> None:
        """A prolonged vowel with no held span, no singing and no chant is a miss, not a pass."""
        result = evaluate_routes(_features("prolonged-vowel"), ruleset)
        assert result.declared == ("VOICE",)
        assert result.routed == ()
        assert result.missed == ("VOICE",)
        assert result.agreed == ()
        assert result.extra == ()
        assert result.fell_through is True

    def test_speech_routes_on_lexical_words_alone_when_one_recogniser_disagreed(self, ruleset: Ruleset) -> None:
        """The harvard case: nine lexical words, three agreed, unambiguous read speech."""
        record = _features("harvard-sentences-list", words={"agreement": 3, "total": 9, "lexical": 9})
        assert evaluate_gate(record, ruleset.gates["speech.lexical"]) is GateOutcome.FIRED
        result = evaluate_routes(record, ruleset)
        assert result.routed == ("SPEECH",)
        assert result.agreed == ("SPEECH",)

    def test_the_four_harvard_recordings_that_the_old_declared_gate_discarded_all_route(self, ruleset: Ruleset) -> None:
        """Every measured row of the harvard fall-through, at its own lexical and agreement counts."""
        for lexical, agreement in ((4, 3), (6, 3), (9, 3), (21, 3)):
            record = _features(
                "harvard-sentences-list", words={"agreement": agreement, "total": lexical, "lexical": lexical}
            )
            assert evaluate_routes(record, ruleset).routed == ("SPEECH",)

    def test_a_recording_can_route_to_more_than_one_branch(self, ruleset: Ruleset) -> None:
        """A held vowel with an intruding voice routes to both, and neither suppresses the other."""
        record = _features(
            "prolonged-vowel",
            span_longest_s={"amplitude": 6.0},
            words={"agreement": 3, "total": 6, "lexical": 6},
        )
        result = evaluate_routes(record, ruleset)
        assert result.routed == ("SPEECH", "VOICE")
        assert result.extra == ("SPEECH",)
        assert result.agreed == ("VOICE",)


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
        assert evaluate_gate(record, ruleset.gates["ddk.lexical_repetition"]) is GateOutcome.UNAVAILABLE

    def test_an_unreadable_gate_is_named_rather_than_counted_as_silent(self, ruleset: Ruleset) -> None:
        """A breath recording with no residual and no span statistic is unmeasured, not empty."""
        record = _features("breath-sounds", residual={}, span_stats={})
        result = evaluate_routes(record, ruleset)
        assert result.unavailable["AIRWAY"] == ("airway.breath", "airway.cough")
        assert result.routed == ()
        assert result.missed == ("AIRWAY",)
        assert result.gate_outcomes["airway.breath"] is GateOutcome.UNAVAILABLE
        assert result.gate_outcomes["voice.sustained"] is GateOutcome.SILENT

    def test_a_silent_branch_carries_no_unavailable_entry(self, ruleset: Ruleset) -> None:
        """Silent is a measurement; only an unread feature is named in ``unavailable``."""
        result = evaluate_routes(_features("breath-sounds"), ruleset)
        assert result.unavailable == {}
        assert result.routed == ()

    def test_a_branch_routes_on_a_readable_gate_beside_an_unreadable_one(self, ruleset: Ruleset) -> None:
        """One unread gate does not withhold the branch a second gate fired for."""
        record = _features("voluntary-cough", residual={}, span_stats={"all.peak_over_floor_db_max": 55.0})
        result = evaluate_routes(record, ruleset)
        assert result.routed == ("AIRWAY",)
        assert result.unavailable["AIRWAY"] == ("airway.breath",)

    def test_the_tally_counts_unavailable_separately_from_a_silent_gate(self, ruleset: Ruleset) -> None:
        """Collapsing the two would report a broken store as a negative measurement."""
        unmeasured = _features("breath-sounds", residual={}, span_stats={})
        silent = _features("breath-sounds")
        tallies = tally_families([evaluate_routes(unmeasured, ruleset), evaluate_routes(silent, ruleset)])
        row = tallies["breath-sounds"]
        assert row.recordings == 2
        assert row.declared["AIRWAY"] == 2
        assert row.unavailable["AIRWAY"] == 1
        assert row.routed["AIRWAY"] == 0
        assert row.missed["AIRWAY"] == 2
        assert row.fell_through == 2


class TestARecordingRoutesToNothing:
    """The fall-through count is a first-class output."""

    def test_the_tally_counts_the_fall_through_per_family(self, ruleset: Ruleset) -> None:
        """Two families, one fall-through each, counted where the owner asked to read them."""
        evaluations = [
            evaluate_routes(_features("prolonged-vowel"), ruleset),
            evaluate_routes(_features("prolonged-vowel", span_longest_s={"amplitude": 4.0}), ruleset),
            evaluate_routes(_features("voluntary-cough"), ruleset),
        ]
        tallies = tally_families(evaluations)
        assert tallies["prolonged-vowel"].fell_through == 1
        assert tallies["prolonged-vowel"].routed["VOICE"] == 1
        assert tallies["prolonged-vowel"].agreed["VOICE"] == 1
        assert tallies["voluntary-cough"].fell_through == 1
        assert tallies["voluntary-cough"].declared["AIRWAY"] == 1
        assert set(tallies["voluntary-cough"].declared) == set(BRANCHES)


class TestScoringContentAgainstTheDeclaredFamily:
    """Can content-only routing recover the overarching families?"""

    def test_the_two_by_two_counts_every_recording_once_per_branch(self, ruleset: Ruleset) -> None:
        """Each branch sees every recording, as a positive or a negative, predicted or not."""
        evaluations = [
            evaluate_routes(_features("harvard-sentences-list", words={"agreement": 3, "lexical": 9}), ruleset),
            evaluate_routes(_features("prolonged-vowel"), ruleset),
            evaluate_routes(_features("prolonged-vowel", peaks={"plain|yamnet|Chant": 0.5}), ruleset),
        ]
        scores = score_branches(evaluations)
        assert list(scores) == list(BRANCHES)
        for table in scores.values():
            assert table.tp + table.fp + table.tn + table.fn == 3

    def test_sensitivity_and_specificity_read_off_the_raw_counts(self, ruleset: Ruleset) -> None:
        """One declared vowel routed, one declared vowel missed, one speech recording not a vowel."""
        evaluations = [
            evaluate_routes(_features("prolonged-vowel", peaks={"plain|yamnet|Chant": 0.5}), ruleset),
            evaluate_routes(_features("prolonged-vowel"), ruleset),
            evaluate_routes(_features("harvard-sentences-list", words={"agreement": 3, "lexical": 9}), ruleset),
        ]
        voice = score_branches(evaluations)["VOICE"]
        assert (voice.tp, voice.fn, voice.fp, voice.tn) == (1, 1, 0, 1)
        assert voice.sensitivity == 0.5
        assert voice.specificity == 1.0

    def test_a_branch_no_recording_declares_has_no_sensitivity(self, ruleset: Ruleset) -> None:
        """No reference positive is not a sensitivity of zero."""
        ddk = score_branches([evaluate_routes(_features("prolonged-vowel"), ruleset)])["DDK"]
        assert ddk.sensitivity is None
        assert ddk.specificity == 1.0


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


class TestSpeechRoutesOnAsrWordsAlone:
    """The SPEECH gate is lexical content; agreement is not part of entering the branch."""

    def test_the_branch_has_exactly_one_gate_and_it_reads_lexical_words(self, ruleset: Ruleset) -> None:
        """A second entry gate would let something other than words decide that speech was said."""
        assert ruleset.branch_gates["SPEECH"] == ("speech.lexical",)
        assert ruleset.gates["speech.lexical"].feature == ("words", "lexical")

    def test_two_lexical_words_with_no_agreement_at_all_route(self, ruleset: Ruleset) -> None:
        """Nothing the recognisers agreed on, and the recording is still speech."""
        record = _features("free-speech", words={"agreement": 0, "total": 2, "lexical": 2})
        result = evaluate_routes(record, ruleset)
        assert result.routed == ("SPEECH",)
        assert result.gate_outcomes["speech.transcript_agreement"] is GateOutcome.SILENT

    def test_one_lexical_word_does_not_route(self, ruleset: Ruleset) -> None:
        """A single spurious token on a held vowel is the artifact the cut of 2 exists to drop."""
        record = _features("prolonged-vowel", words={"agreement": 0, "total": 1, "lexical": 1})
        assert evaluate_routes(record, ruleset).routed == ()

    def test_agreement_alone_no_longer_routes(self, ruleset: Ruleset) -> None:
        """Agreed bracketed tokens are not lexical words and carry no branch on their own."""
        record = _features("free-speech", words={"agreement": 3, "total": 3, "bracketed": 3, "lexical": 0})
        assert evaluate_routes(record, ruleset).routed == ()

    def test_the_agreement_gate_is_not_a_branch_gate_anywhere(self, ruleset: Ruleset) -> None:
        """It was deleted from ``branch_gates``, not moved to another branch's entry list."""
        for names in ruleset.branch_gates.values():
            assert "speech.transcript_agreement" not in names


class TestAgreementIsAFlagAndNotAGate:
    """Agreement records doubt about what was said, so it is carried and never routes."""

    def test_the_flag_is_declared_on_the_speech_branch(self, ruleset: Ruleset) -> None:
        """A flag is per branch, beside the gates it annotates rather than inside them."""
        assert ruleset.branch_flags["SPEECH"] == ("speech.transcript_agreement",)

    def test_a_routed_recording_carries_the_flag(self, ruleset: Ruleset) -> None:
        """Nine lexical words, three agreed: routed on the words, annotated by the agreement."""
        record = _features("harvard-sentences-list", words={"agreement": 3, "total": 9, "lexical": 9})
        result = evaluate_routes(record, ruleset)
        assert result.routed == ("SPEECH",)
        assert result.flags == {"SPEECH": ("speech.transcript_agreement",)}

    def test_the_flag_does_not_route_a_recording_that_no_gate_carried(self, ruleset: Ruleset) -> None:
        """Three agreed bracketed tokens raise the flag and leave ``routed`` empty."""
        record = _features("free-speech", words={"agreement": 3, "total": 3, "bracketed": 3, "lexical": 0})
        result = evaluate_routes(record, ruleset)
        assert result.flags == {"SPEECH": ("speech.transcript_agreement",)}
        assert result.routed == ()
        assert result.fell_through is True

    def test_a_silent_flag_is_absent_rather_than_keyed_empty(self, ruleset: Ruleset) -> None:
        """Only a fired flag is reported, so a branch's absence from the mapping is the negative."""
        record = _features("free-speech", words={"agreement": 0, "total": 4, "lexical": 4})
        assert evaluate_routes(record, ruleset).flags == {}

    def test_the_tally_counts_the_flag_per_branch(self, ruleset: Ruleset) -> None:
        """A flag is reported at corpus scale beside the axes it does not belong to."""
        flagged = _features("free-speech", words={"agreement": 3, "total": 9, "lexical": 9})
        plain = _features("free-speech", words={"agreement": 0, "total": 9, "lexical": 9})
        row = tally_families([evaluate_routes(flagged, ruleset), evaluate_routes(plain, ruleset)])["free-speech"]
        assert row.flagged["SPEECH"] == 1
        assert row.routed["SPEECH"] == 2


def _classified(family: str, enhanced: float, residual: float, **overrides: object) -> RecordingFeatures:
    """A record whose enhanced and residual YAMNet summaries both exist and carry a top score.

    Args:
        family: The task family.
        enhanced: The highest tracked-label score on the enhanced stream.
        residual: The same on the residual stream.
        overrides: Further fields to replace.

    Returns:
        The record.
    """
    record = _features(
        family,
        classifier_streams=["plain|yamnet", "enhanced|yamnet", "residual|yamnet"],
        peaks={"enhanced|yamnet|Speech": enhanced, "residual|yamnet|Speech": residual},
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


class TestEmptinessIsCheckedBeforeRouting:
    """A recording that carried nothing is not a recording whose content matched no gate."""

    def test_both_streams_under_the_floor_is_empty(self, ruleset: Ruleset) -> None:
        """The short-recording case: the enhanced and the residual stream are both silent."""
        result = evaluate_routes(_classified("prolonged-vowel", 0.003, 0.0), ruleset)
        assert result.empty is True
        assert result.routed == ()
        assert result.fell_through is False

    def test_one_stream_at_the_floor_is_not_empty(self, ruleset: Ruleset) -> None:
        """The floor is inclusive on the content side, so a stream at 0.2 carries something."""
        assert evaluate_routes(_classified("prolonged-vowel", 0.2, 0.0), ruleset).empty is False
        assert evaluate_routes(_classified("prolonged-vowel", 0.0, 0.2), ruleset).empty is False

    def test_an_empty_recording_is_not_a_fall_through(self, ruleset: Ruleset) -> None:
        """The owner's number is the content that landed nowhere, not the silence."""
        blank = evaluate_routes(_classified("prolonged-vowel", 0.001, 0.0), ruleset)
        content = evaluate_routes(_classified("prolonged-vowel", 0.9, 0.3), ruleset)
        assert (blank.empty, blank.fell_through) == (True, False)
        assert (content.empty, content.fell_through) == (False, True)

    def test_the_tally_counts_the_two_separately(self, ruleset: Ruleset) -> None:
        """One empty and one content-bearing miss, in a family whose total is two."""
        blank = evaluate_routes(_classified("prolonged-vowel", 0.001, 0.0), ruleset)
        content = evaluate_routes(_classified("prolonged-vowel", 0.9, 0.3), ruleset)
        row = tally_families([blank, content])["prolonged-vowel"]
        assert row.recordings == 2
        assert row.empty == 1
        assert row.fell_through == 1

    def test_an_empty_recording_runs_no_branch_gate(self, ruleset: Ruleset) -> None:
        """A precondition is ahead of routing, so a lexical word in a silent file routes nothing."""
        record = _classified("prolonged-vowel", 0.0, 0.0, words={"agreement": 0, "total": 4, "lexical": 4})
        result = evaluate_routes(record, ruleset)
        assert result.empty is True
        assert result.routed == ()
        assert result.gate_outcomes == {}
        assert result.missed == ("VOICE",)

    def test_a_stream_with_no_summary_cannot_be_called_empty(self, ruleset: Ruleset) -> None:
        """An absent classifier summary is not a stream that scored zero."""
        assert evaluate_routes(_features("prolonged-vowel"), ruleset).empty is False


class TestBracketedTokensAreAirwayEvidence:
    """``[breath]`` is a detection with a timing, and ``[um]`` is a filler."""

    def test_the_bracket_gate_is_an_airway_branch_gate(self, ruleset: Ruleset) -> None:
        """It enters AIRWAY beside the residual and span gates rather than annotating it."""
        assert "airway.bracketed_event" in ruleset.branch_gates["AIRWAY"]

    def test_the_airway_bracket_types_come_off_the_configuration(self, ruleset: Ruleset) -> None:
        """Which brackets are airway is data, resolved into the gate at load."""
        assert ruleset.gates["airway.bracketed_event"].feature == (
            "bracketed_set",
            "breath",
            "cough",
            "sniff",
            "throatclearing",
        )

    def test_a_breath_bracket_routes_airway(self, ruleset: Ruleset) -> None:
        """One bracketed breath is one airway event."""
        record = _features("breath-sounds", bracketed_types={"breath": 1})
        assert evaluate_routes(record, ruleset).routed == ("AIRWAY",)

    def test_a_filler_bracket_does_not_route_airway(self, ruleset: Ruleset) -> None:
        """``[um]`` is speech, and it is not in the airway set."""
        record = _features("free-speech", bracketed_types={"um": 4})
        assert evaluate_routes(record, ruleset).routed == ()

    def test_every_airway_bracket_type_carries_the_gate_on_its_own(self, ruleset: Ruleset) -> None:
        """The gate is a union, so no member depends on another being present."""
        rule = ruleset.gates["airway.bracketed_event"]
        for token in ("breath", "cough", "throatclearing", "sniff"):
            assert evaluate_gate(_features("breath-sounds", bracketed_types={token: 1}), rule) is GateOutcome.FIRED

    def test_no_consensus_transcript_is_not_a_bracket_count_of_zero(self, ruleset: Ruleset) -> None:
        """A store that never ran consensus has an unread gate, not a silent one."""
        record = _features("breath-sounds", consensus_present=False)
        assert evaluate_gate(record, ruleset.gates["airway.bracketed_event"]) is GateOutcome.UNAVAILABLE
