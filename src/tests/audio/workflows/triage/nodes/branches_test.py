"""The shared foundation the four branches sit on: the two modes, the table, the propose path.

Nothing here runs a branch. What is pinned is the contract four branch implementations are written
against: that a proposal without its derivation is refused, that a proposer cannot reach another
branch's family, that all 58 expectation rows are data rather than code, that each shared helper
does what a hand-checked fixture says it does, and that the declared task family is derived in one
place and returns nothing rather than raising when no carrier names one.
"""

import ast
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import DATA_MAP_PATHS, TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import branches as branches_module
from senselab.audio.workflows.triage.nodes.branches import (
    AIRWAY_EXPECTATIONS,
    BRANCH_FAMILY,
    BRANCHES,
    DEVIATION_TYPES,
    EXPECTATIONS,
    PARAM_KEYS,
    PARAM_SECTION,
    POINT_TYPES,
    PROPOSERS,
    REFERENCE_FAMILY_SET,
    RESERVED_SPAN_ATTRIBUTES,
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    UNMEASURED_POINTS,
    VOICE_EXPECTATIONS,
    VOICE_EXPECTATIONS_PENDING_DECLARATION,
    BranchParams,
    ContinuityTrack,
    EnvelopeTrack,
    Expectation,
    Finding,
    Pattern,
    PhonationTracks,
    Proposal,
    Result,
    SpectrogramBlock,
    branch_params,
    content_coverage,
    contest,
    count,
    declared_duration_count,
    declared_task_family,
    deviation,
    dispatch,
    duration,
    envelope_slice,
    events_in_extent,
    gaps,
    group_by_breaks,
    hull,
    inter_word_gaps,
    lexical,
    lexical_runs,
    longest_monotone_run,
    max_windowed_spread,
    measured,
    merge,
    mode_of,
    ngram_echo_fraction,
    off_task,
    ordered_run,
    overlaps,
    peak_over_floor_db,
    propose_span,
    propose_spans,
    proposer,
    read_envelope_track,
    semitones,
    sounds_like,
    spans_by_measure,
    spectral_balance_db,
    stream_extent,
    token_occurrences,
    touches_edge,
    trace_slice,
    track_slice,
    train_rate_hz,
    voiced_extent,
    write_findings,
)
from senselab.audio.workflows.triage.routing_analysis.families import (
    AIRWAY_ELICITING,
    SPEECH_ELICITING,
    SYLLABLE_REPETITION,
    VOICE_ELICITING,
)
from senselab.utils.prov_store import Entity, ProvStore

BIDS_STEM = "sub-abc123_ses-1_task-harvard-sentences-list-10-3"
"""A real stem shape: two trailing numeric segments, which ``task_family`` collapses."""


def _store(*, path: str | None = BIDS_STEM + ".wav", duration_s: float = 20.0) -> ProvStore:
    """A store holding only what the mode selector reads.

    Args:
        path: The ``recording`` stream's source path, or None to omit the stream entirely.
        duration_s: The recording's duration.

    Returns:
        The store.
    """
    store = ProvStore(run_id="branches-test")
    if path is not None:
        store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": "recording", "path": path, "sampling_rate": 16000, "channels": 1},
        )
    return store


def _word(index: int, text: str, extent: tuple[float, float], *, bracketed: bool = False) -> Entity:
    """One consensus word, with only the attributes the helpers read.

    Args:
        index: Its position in the consensus stream.
        text: Its text.
        extent: Its extent, in seconds.
        bracketed: Whether it is a bracketed non-lexical token.

    Returns:
        The entity.
    """
    return Entity(
        id=f"word-{index}",
        prov_type="word",
        extent=extent,
        attributes={"index": index, "text": text, "bracketed": bracketed},
    )


def _span(span_id: str, measure: str, extent: tuple[float, float], **extra: Any) -> Entity:  # noqa: ANN401
    """One PREPROCESS span.

    Args:
        span_id: Its id.
        measure: ``amplitude``, ``continuity``, ``asr`` or ``gap``.
        extent: Its extent, in seconds.
        **extra: Further attributes.

    Returns:
        The entity.
    """
    return Entity(
        id=span_id, prov_type="span", extent=extent, attributes={"measure": measure, "signal": "plain", **extra}
    )


def _params(**values: Any) -> BranchParams:  # noqa: ANN401
    """The packaged operating points, with a fixture's own values overriding some of them.

    Args:
        **values: ``branch.*`` keys to override, e.g. to set a null a test wants to read as missing.

    Returns:
        The record over a configuration carrying those values.
    """
    config = load_triage_config()
    merged = dict(config.values)
    merged[PARAM_SECTION] = {**config.values[PARAM_SECTION], **values}
    return branch_params(TriageConfig(config.name, config.version, config.config_hash, merged))


# --------------------------------------------------------------------- the propose path


class TestAProposalNamesItsEvidence:
    """Under propose-only the derivation is the whole record of where an extent came from."""

    def test_a_proposal_with_no_derivation_is_refused_at_minting(self) -> None:
        """A branch that forgets it loses information rather than omitting a detail."""
        with pytest.raises(ValueError, match="names its evidence"):
            PROPOSERS["VOICE"]("task_extent", (1.0, 2.0))

    def test_the_message_says_why_rather_than_only_that(self) -> None:
        """The author has to learn that nothing else carries the relationship."""
        with pytest.raises(ValueError, match="the whole record"):
            PROPOSERS["AIRWAY"]("task_extent", (0.0, 1.0))

    def test_a_proposal_with_no_derivation_is_refused_at_writing_too(self) -> None:
        """The write path is checked independently: a hand-built Proposal reaches it too."""
        store = _store()
        activity = store.activity(node="VOICE", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        naked = Proposal("voice", "task_extent", 1.0, 2.0, (), {})
        with pytest.raises(ValueError, match="names its evidence"):
            propose_span(store, activity, agent, naked)
        assert not store.entities("span"), "nothing may be written before the check"

    def test_a_zero_length_proposal_is_refused(self) -> None:
        """A span with no duration names no extent."""
        with pytest.raises(ValueError, match="positive duration"):
            PROPOSERS["SPEECH"]("task_extent", (2.0, 2.0), "evidence-1")

    def test_a_reversed_proposal_is_refused(self) -> None:
        """An end before its start is a bug, not a degenerate case to tolerate."""
        with pytest.raises(ValueError, match="positive duration"):
            PROPOSERS["SPEECH"]("task_extent", (3.0, 1.0), "evidence-1")


class TestAProposerMintsOnlyIntoItsOwnFamily:
    """The family is fixed by the branch. A caller cannot ask for another's."""

    def test_each_branchs_proposer_stamps_its_own_family(self) -> None:
        """Four proposers, four families, no parameter that changes one."""
        for branch, family in BRANCH_FAMILY.items():
            assert PROPOSERS[branch]("task_extent", (0.0, 1.0), "e1").family == family

    def test_an_attribute_may_not_shadow_the_family_or_the_role(self) -> None:
        """Both are stamped from the proposer's own arguments; a second value could disagree."""
        for shadow in ({"family": "speech"}, {"role": "elsewhere"}):
            with pytest.raises(ValueError, match="may not shadow"):
                PROPOSERS["VOICE"]("task_extent", (0.0, 1.0), "e1", **shadow)
        assert RESERVED_SPAN_ATTRIBUTES == {"family", "role"}

    def test_no_attribute_key_at_all_can_set_the_family(self) -> None:
        """Not only ``family``: no aliasing key may become a back door into another's family."""
        for key in ("family", "Family", "_family", "span_family", "family_", "own_family", "branch"):
            try:
                proposal = PROPOSERS["VOICE"]("task_extent", (0.0, 1.0), "e1", **{key: "speech"})
            except ValueError:
                continue
            assert proposal.family == "voice", f"{key} reached the family"
            assert proposal.attributes.get(key) == "speech", f"{key} was consumed rather than stored"

    def test_role_and_extent_are_positional_only(self) -> None:
        """So a ``role`` keyword reaches the reserved check instead of colliding with the parameter."""
        with pytest.raises(TypeError):
            PROPOSERS["VOICE"](role="task_extent", extent=(0.0, 1.0))  # type: ignore[call-arg]

    def test_a_hand_built_proposals_attributes_cannot_displace_the_stamps(self) -> None:
        """The proposer refuses it, so only a hand-built record can carry it; the write path wins."""
        proposal = Proposal("voice", "task_extent", 0.0, 1.0, ("e1",), {"family": "speech", "role": "other"})
        store = _store()
        activity = store.activity(node="VOICE", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        span = store.get_entity(propose_span(store, activity, agent, proposal))
        assert (span.attributes["family"], span.attributes["role"]) == ("voice", "task_extent")

    def test_dispatch_refuses_a_branch_that_proposed_into_another_family(self) -> None:
        """The enforcement is not only at minting: a hand-built Proposal is caught at the seam."""
        intruder = Proposal("speech", "task_extent", 0.0, 1.0, ("e1",), {})

        def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
            return Result(True, [intruder], [])

        def detect(store: ProvStore, params: BranchParams) -> Result:
            return Result(UNDETERMINED, [], [])

        with pytest.raises(ValueError, match="mints only into 'voice'"):
            dispatch("VOICE", _store(path="x_task-prolonged-vowel.wav"), _params(), align=align, detect=detect)

    def test_the_proposer_table_covers_every_branch_and_no_more(self) -> None:
        """QUALITY has only the detect mode and is deliberately not a routed branch."""
        assert set(PROPOSERS) == set(BRANCHES) == set(BRANCH_FAMILY) == set(EXPECTATIONS)


class TestWhatTheProposedSpanCarries:
    """A written proposal is a span entity in the branch's family, named by role and derivation."""

    def test_the_span_carries_family_role_extent_and_every_derivation(self) -> None:
        """All four, because all four are what a downstream reader has to go on."""
        store = _store()
        activity = store.activity(node="AIRWAY", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        proposal = PROPOSERS["AIRWAY"]("cough_event", (1.5, 2.25), "env-1", "span-3", peak_over_floor_db=41.0)
        span_id = propose_span(store, activity, agent, proposal)
        span = store.get_entity(span_id)
        assert span.prov_type == "span"
        assert span.extent == (1.5, 2.25)
        assert span.attributes["family"] == "airway"
        assert span.attributes["role"] == "cough_event"
        assert span.attributes["peak_over_floor_db"] == 41.0
        assert set(store.derived_from(span_id)) == {"env-1", "span-3"}
        assert store.generated_by(span_id) == activity

    def test_many_proposals_are_written_in_order(self) -> None:
        """The report reads them positionally, so the order is part of the contract."""
        store = _store()
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        proposals = [PROPOSERS["SPEECH"]("task_extent", (float(i), float(i) + 1.0), "e1") for i in range(3)]
        ids = propose_spans(store, activity, agent, proposals)
        assert [store.get_entity(each).extent for each in ids] == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]


class TestAFindingNamesTheEvidenceItWasReadOff:
    """An extent beside an entity is coincidence; ``wasDerivedFrom`` is the edge a reader follows."""

    def test_an_assertion_carries_one_edge_per_source(self) -> None:
        """A deviation is written beside the region it was read off, not merely over the same seconds."""
        store = _store()
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        [entity_id] = write_findings(
            store, activity, agent, [deviation("filler", 1.0, 2.0, "word-7", "word-8", text="um")], signal="plain"
        )
        assert set(store.derived_from(entity_id)) == {"word-7", "word-8"}

    def test_a_measurement_carries_one_edge_per_source(self) -> None:
        """The measurement path writes the same edges as the assertion path, not fewer."""
        store = _store()
        activity = store.activity(node="VOICE", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        [entity_id] = write_findings(
            store, activity, agent, [measured("voiced_duration_s", 0.5, 3.0, 2.5, "span-4", "tracks-1")], signal="plain"
        )
        assert set(store.derived_from(entity_id)) == {"span-4", "tracks-1"}

    def test_the_contested_span_is_the_edge_rather_than_an_attribute(self) -> None:
        """``of_span`` was a field a reader had to know to look for; the edge is the record."""
        store = _store()
        activity = store.activity(node="AIRWAY", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        [entity_id] = write_findings(
            store, activity, agent, [contest("span-9", (3.0, 4.0), "cough", "no_event")], signal="plain"
        )
        assert store.derived_from(entity_id) == ["span-9"]
        assert "of_span" not in store.get_entity(entity_id).attributes

    def test_the_folded_counts_measurement_derives_from_the_union_of_its_entries(self) -> None:
        """One entity stands for every entry, so every entry's evidence has to be reachable from it."""
        store = _store()
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        findings = [
            count("syllables", 12, 12, "span-1", "envelope-1"),
            count("realised_cycles", 4, None, "span-1", "wideband-1"),
        ]
        [entity_id] = write_findings(store, activity, agent, findings, signal="plain")
        assert store.derived_from(entity_id) == ["span-1", "envelope-1", "wideband-1"], "union, first-seen order"

    def test_a_finding_over_an_extent_naming_no_source_is_refused(self) -> None:
        """The same rule ``propose_span`` holds a proposal to: an unrecorded relationship is lost."""
        store = _store()
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        with pytest.raises(ValueError, match="names its evidence"):
            write_findings(store, activity, agent, [deviation("truncation", 1.0, 2.0)], signal="plain")

    def test_a_measurement_over_an_extent_naming_no_source_is_refused_too(self) -> None:
        """The check is on the extent, not on the kind, so the measurement path cannot slip past it."""
        store = _store()
        activity = store.activity(node="VOICE", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        with pytest.raises(ValueError, match="names its evidence"):
            write_findings(store, activity, agent, [measured("glide_semitones", 0.0, 1.0, 4.0)], signal="plain")

    def test_a_finding_with_no_extent_is_accepted_without_a_source(self) -> None:
        """An omission claims something is absent; there is no region to point at, so none is demanded.

        The count beside it names one, so the exemption is on the missing extent rather than on the
        kind: a source-bearing count in the same batch still reaches the store as an edge.
        """
        store = _store()
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        findings = [deviation("omission", None, None, expected="rainbow"), count("items", 0, 3, "word-2")]
        assertion_id, counts_id = write_findings(store, activity, agent, findings, signal="plain")
        assert store.derived_from(assertion_id) == []
        assert store.derived_from(counts_id) == ["word-2"]


class TestFindingsAreWrittenBesideSpansNotOntoThem:
    """``deviate`` and ``contest`` are assertions; counts fold into one ``counts`` measurement."""

    def test_each_kind_lands_in_the_form_its_kind_calls_for(self) -> None:
        """Four kinds, three entity shapes, and one count measurement however many counts there are."""
        store = _store()
        activity = store.activity(node="AIRWAY", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        findings = [
            Finding("deviation", "off_task_extent", 1.0, 2.0, {"measure": "gap"}, ("gap-1",)),
            Finding("contest", "cough", 3.0, 4.0, {"reason": "no_event"}, ("span-1",)),
            Finding("measure", "event_rate_hz", 0.0, 5.0, {"value": 1.2, "uncalibrated": True}, ("span-1",)),
            Finding("count", "events", None, None, {"found": 4, "declared": 5}),
            Finding("count", "declared_duration_s", None, None, {"found": 20.0, "declared": 20.0}),
        ]
        written = write_findings(store, activity, agent, findings, signal="plain")
        assert len(written) == 4, "two counts became one measurement"
        assertions = store.entities("assertion")
        assert [each.attributes["verb"] for each in assertions] == ["deviate", "contest"]
        assert assertions[0].attributes["deviation_type"] == "off_task_extent"
        assert assertions[1].attributes["claim"] == "cough"
        counts = [each for each in store.entities("measurement") if each.attributes["name"] == "counts"]
        assert len(counts) == 1
        assert counts[0].attributes["entries"] == {
            "events": {"found": 4, "declared": 5},
            "declared_duration_s": {"found": 20.0, "declared": 20.0},
        }

    def test_an_unknown_kind_is_refused_rather_than_dropped(self) -> None:
        """A finding nothing writes is a finding nobody sees."""
        store = _store()
        activity = store.activity(node="VOICE", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        with pytest.raises(ValueError, match="unknown finding kinds"):
            write_findings(store, activity, agent, [Finding("refine", "x", None, None, {})], signal="plain")


# --------------------------------------------------------------------- the table


class TestTheExpectationTableIsData:
    """48 rows over 48 families. A row that cannot round-trip is carrying behaviour."""

    def test_there_are_forty_eight_rows(self) -> None:
        """The count is the design's own, and the per-branch split is the reference family sets'."""
        assert {branch: len(table) for branch, table in EXPECTATIONS.items()} == {
            "AIRWAY": 11,
            "SPEECH": 31,
            "VOICE": 6,
        }
        assert sum(len(table) for table in EXPECTATIONS.values()) == 48

    def test_each_table_is_exactly_its_branchs_reference_family_set(self) -> None:
        """A row for a family the branch does not own would never be reached; a gap returns detect."""
        for branch, families in (
            ("AIRWAY", AIRWAY_ELICITING),
            ("SPEECH", SPEECH_ELICITING),
            ("VOICE", VOICE_ELICITING),
        ):
            assert set(EXPECTATIONS[branch]) == set(families), branch
            assert REFERENCE_FAMILY_SET[branch] == families

    def test_the_reference_sets_match_the_ones_the_packaged_config_names(self) -> None:
        """``taxonomy.ruleset.reference_family_set`` names the same three sets by key."""
        declared = load_triage_config().require("taxonomy.ruleset.reference_family_set")
        assert declared == {"AIRWAY": "airway", "SPEECH": "speech", "VOICE": "voice"}

    def test_the_ten_syllable_families_are_speechs_with_the_instructions_own_expectation(self) -> None:
        """A syllable train is a speaking task, so its row says what the instruction asked for."""
        assert set(SYLLABLE_REPETITION) <= set(SPEECH_EXPECTATIONS)
        assert SPEECH_EXPECTATIONS["diadochokinesis-pa"].pattern is Pattern.SYLLABLE_TRAIN
        assert SPEECH_EXPECTATIONS["diadochokinesis-pa"].sequence == ("p", "aa")
        assert SPEECH_EXPECTATIONS["diadochokinesis-pataka"].pattern is Pattern.SYLLABLE_SEQUENCE
        assert SPEECH_EXPECTATIONS["diadochokinesis-buttercup"].pattern is Pattern.SYLLABLE_SEQUENCE

    def test_every_row_round_trips_through_plain_data(self) -> None:
        """All 48, field for field, so the table can be recorded in a run and read back."""
        seen = 0
        for table in EXPECTATIONS.values():
            for family, expectation in table.items():
                mapping = expectation.as_mapping()
                assert set(mapping) == {field.name for field in fields(Expectation)}, family
                assert Expectation.from_mapping(mapping) == expectation, family
                seen += 1
        assert seen == 48

    def test_the_mapping_is_json_shaped_rather_than_python_shaped(self) -> None:
        """A tuple or an Enum in a stored attribute is not something a reader can rely on."""
        mapping = SPEECH_EXPECTATIONS["loudness"].as_mapping()
        assert mapping["pattern"] == "ordered_tokens"
        assert mapping["tokens"] == ["hey", "hey", "hey"]
        assert AIRWAY_EXPECTATIONS["breath-sounds"].as_mapping()["unviable"] == [["route", "as `fivebreaths`"]]

    def test_the_v1_v2_pairs_differ_in_exactly_the_field_the_design_names(self) -> None:
        """Written as data the pair cannot drift; written as two functions it repeatedly did.

        The DDK pair gained a third difference when the template became the token's phonemes: the
        v2 stimulus says "puh" where v1 says "pa", which is a different vowel. The place-and-height
        template could not record that and the two rows were identical; the phoneme template can.
        """

        def difference(a: Expectation, b: Expectation) -> set[str]:
            return {f.name for f in fields(Expectation) if getattr(a, f.name) != getattr(b, f.name)}

        mpt = difference(VOICE_EXPECTATIONS["maximum-phonation-time"], VOICE_EXPECTATIONS["maximum-phonation-time-v2"])
        assert mpt == {"expect_inhale"}
        free = difference(SPEECH_EXPECTATIONS["free-speech"], SPEECH_EXPECTATIONS["free-speech-v2"])
        assert "anti_pattern" in free
        ddk = difference(SPEECH_EXPECTATIONS["diadochokinesis-pa"], SPEECH_EXPECTATIONS["diadochokinesis-v2-puh"])
        assert ddk == {"expected_event_count", "declared_duration_s", "sequence"}
        assert SPEECH_EXPECTATIONS["diadochokinesis-pa"].sequence == ("p", "aa")
        assert SPEECH_EXPECTATIONS["diadochokinesis-v2-puh"].sequence == ("p", "ah")

    def test_the_pending_declaration_rows_are_not_in_the_dispatch_table(self) -> None:
        """CAPE-V and loudness are ``LEXICAL_SPEECH``, so ``align_voice`` must not reach them."""
        assert set(VOICE_EXPECTATIONS_PENDING_DECLARATION) == {
            "cape-v-sentences",
            "cape-v-sentences-v2",
            "loudness",
            "loudness-v2",
        }
        assert not set(VOICE_EXPECTATIONS_PENDING_DECLARATION) & set(VOICE_EXPECTATIONS)
        assert set(VOICE_EXPECTATIONS_PENDING_DECLARATION) <= set(SPEECH_EXPECTATIONS)

    def test_every_pattern_kind_is_used_by_at_least_one_row(self) -> None:
        """Thirteen matcher branches replace 32 functions; an unused kind is a matcher nothing needs."""
        used = {
            expectation.pattern
            for table in list(EXPECTATIONS.values()) + [VOICE_EXPECTATIONS_PENDING_DECLARATION]
            for expectation in table.values()
        }
        assert used == set(Pattern)

    def test_an_unviable_entry_names_a_measurement_and_a_reason(self) -> None:
        """A determination with no viable approach is carried as data, not silently omitted."""
        rows = [
            expectation for table in EXPECTATIONS.values() for expectation in table.values() if expectation.unviable
        ]
        assert rows, "the design carries ten unviable sites"
        for expectation in rows:
            for name, why in expectation.unviable:
                assert name and why


# --------------------------------------------------------------------- the task family


class TestTheDeclaredFamilyIsDerivedInOnePlace:
    """Nothing passes a task family to a branch, so the mode test derives its own."""

    def test_the_recording_path_yields_the_family_with_its_trailing_index_collapsed(self) -> None:
        """The one carrier available today: ADMIT's ``path`` on the ``recording`` stream."""
        assert declared_task_family(_store()) == "harvard-sentences-list"

    def test_a_non_bids_path_names_no_family(self) -> None:
        """A path with no ``task-`` entity is not a declaration; None takes the safe arm."""
        assert declared_task_family(_store(path="/tmp/some-recording.wav")) is None

    def test_an_absent_recording_entity_does_not_raise(self) -> None:
        """A branch reached before ADMIT wrote, or on a store read back without it, concludes nothing."""
        assert declared_task_family(_store(path=None)) is None

    def test_a_recording_entity_with_no_path_names_no_family(self) -> None:
        """The attribute may be absent; that is not an error either."""
        store = ProvStore(run_id="no-path")
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording"})
        assert declared_task_family(store) is None

    def test_the_hint_metadata_carrier_is_read_first(self) -> None:
        """The clean route works the moment something populates it, with no branch changing."""
        hint = AudioHints(metadata={"task_token": "prolonged-vowel-2"})
        assert declared_task_family(_store(), hint) == "prolonged-vowel"

    def test_an_empty_hint_token_falls_through_to_the_path(self) -> None:
        """A blank carrier is not a declaration of "no family"; the other carrier still speaks."""
        hint = AudioHints(metadata={"task_token": "   "})
        assert declared_task_family(_store(), hint) == "harvard-sentences-list"

    def test_a_hint_with_no_metadata_at_all_is_fine(self) -> None:
        """``AudioHints.metadata`` may be None."""
        assert declared_task_family(_store(), AudioHints()) == "harvard-sentences-list"

    def test_a_stem_whose_task_is_literally_unknown_names_no_family(self) -> None:
        """``task_id_of`` spells an absent task ``"unknown"``; so must an explicit one."""
        assert declared_task_family(_store(path="sub-a_task-unknown.wav")) is None


class TestTheModeSelector:
    """The declaration picks the mode; it never supplies the answer."""

    def test_an_in_family_declaration_reaches_align_with_its_family(self) -> None:
        """The family reaching ``align_*`` is a key of that branch's own table."""
        seen: list[Any] = []

        def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
            seen.append(task_family)
            return Result(True, [], [])

        def detect(store: ProvStore, params: BranchParams) -> Result:
            seen.append("detect")
            return Result(UNDETERMINED, [], [])

        result = dispatch("VOICE", _store(path="sub-a_task-prolonged-vowel.wav"), _params(), align=align, detect=detect)
        assert seen == ["prolonged-vowel"]
        assert result.done is True

    def test_another_branchs_family_takes_the_out_of_family_mode(self) -> None:
        """A CAPE-V recording is out of family for VOICE under the reference family set."""
        assert mode_of("VOICE", _store(path="sub-a_task-cape-v-sentences.wav")) == ("detect", "cape-v-sentences")
        assert mode_of("SPEECH", _store(path="sub-a_task-cape-v-sentences.wav")) == ("align", "cape-v-sentences")

    def test_no_declaration_takes_the_out_of_family_mode(self) -> None:
        """The safe arm: annotate the speciality, conclude nothing. There is no third arm."""
        assert mode_of("AIRWAY", _store(path="/tmp/plain.wav")) == ("detect", None)

    def test_a_syllable_family_is_in_family_for_speech_and_no_other_branch(self) -> None:
        """One recording, one align mode. There is no second branch to divide the task with."""
        store = _store(path="sub-a_task-diadochokinesis-pa.wav")
        assert mode_of("SPEECH", store)[0] == "align"
        assert mode_of("VOICE", store)[0] == "detect"
        assert mode_of("AIRWAY", store)[0] == "detect"

    def test_the_out_of_family_mode_may_answer_only_undetermined(self) -> None:
        """``done = UNDETERMINED`` there is a rule, not a default: no pattern was expected of it."""

        def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
            return Result(True, [], [])

        def detect(store: ProvStore, params: BranchParams) -> Result:
            return Result(False, [], [])

        with pytest.raises(ValueError, match="evaluates no task"):
            dispatch("SPEECH", _store(path="/tmp/plain.wav"), _params(), align=align, detect=detect)

    def test_the_in_family_mode_may_answer_false(self) -> None:
        """A declared family that produced none of its patterns returns not-done."""

        def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
            return Result(False, [], [])

        def detect(store: ProvStore, params: BranchParams) -> Result:
            return Result(UNDETERMINED, [], [])

        store = _store(path="sub-a_task-prolonged-vowel.wav")
        assert dispatch("VOICE", store, _params(), align=align, detect=detect).done is False

    def test_an_unknown_branch_is_refused(self) -> None:
        """A misspelled branch must not silently take the detect arm of nothing."""

        def align(task_family: str, store: ProvStore, hint: AudioHints | None, params: BranchParams) -> Result:
            return Result(True, [], [])

        def detect(store: ProvStore, params: BranchParams) -> Result:
            return Result(UNDETERMINED, [], [])

        with pytest.raises(KeyError):
            dispatch("PROSODY", _store(), _params(), align=align, detect=detect)


# --------------------------------------------------------------------- the operating points


class TestEveryOperatingPointIsAConfigKey:
    """A branch does not decide, and a refusal is a decision: ``point()`` never raises for a null."""

    def test_point_types_param_keys_and_the_packaged_section_all_match(self) -> None:
        """A key in one but not the other two is a drift nothing else would catch."""
        assert set(POINT_TYPES) == set(PARAM_KEYS) == set(load_triage_config().values[PARAM_SECTION])

    def test_there_are_no_numeric_p_star_properties_any_more(self) -> None:
        """The only accessor is ``point()``. ``p_normalise`` survives because it is a function."""
        p_properties = {
            name[2:]
            for name in dir(BranchParams)
            if name.startswith("p_") and isinstance(getattr(BranchParams, name), property)
        }
        assert p_properties == {"normalise"}

    def test_every_key_ships_a_value_and_point_reads_it_without_refusing(self) -> None:
        """The design fitted every key; none may still ship null, and ``point`` never raises for one."""
        config = load_triage_config()
        mappings = {key for key in PARAM_KEYS if f"{PARAM_SECTION}.{key}" in DATA_MAP_PATHS}
        numeric = [key for key in PARAM_KEYS if key not in mappings]
        assert mappings == {"label_sets", "phoneme_place_classes", "phoneme_vowel_classes"}
        assert len(numeric) == 29
        params = branch_params(config)
        for key in numeric:
            assert config.values[PARAM_SECTION][key] is not None, key
            assert params.point(key) is not None, key
        assert params.missing == []

    def test_reading_a_null_point_records_it_and_returns_none(self) -> None:
        """One overridden null must not fail a body that needs another key."""
        params = _params(event_min_s=None)
        assert params.point("event_min_s") is None
        assert params.missing == ["event_min_s"]
        assert params.point("score_min") == pytest.approx(0.2)

    def test_point_raises_only_for_a_name_outside_point_types(self) -> None:
        """A typo in the calling code is a programming error, not a measurement the graph is missing."""
        with pytest.raises(KeyError, match="PARAM_KEYS"):
            branch_params(load_triage_config()).point("not_a_branch_key")

    def test_record_names_every_null_key_in_read_order(self) -> None:
        """Which point a body reached for first is what says where the evaluation stopped."""
        params = _params(peak_prominence_db=None, event_min_s=None, score_min=None)
        params.point("event_min_s")
        params.point("peak_prominence_db")
        params.point("score_min")
        [finding] = params.record()
        assert finding.name == UNMEASURED_POINTS
        assert finding.evidence["value"] == ["event_min_s", "peak_prominence_db", "score_min"]
        assert finding.evidence["section"] == PARAM_SECTION

    def test_the_label_sets_ship_the_airway_decision_already_taken(self) -> None:
        """The one non-numeric key: set membership derived from ``airway.labels_of_interest``."""
        config = load_triage_config()
        assert branch_params(config).point("label_sets") == {"cough": ("Cough",), "breath": ("Breathe",)}
        assert set(config.require("airway.labels_of_interest")) == {"Cough", "Breathe"}

    def test_a_campaign_may_add_a_label_set_without_editing_the_package(self) -> None:
        """It is data, not schema, so it is in ``DATA_MAP_PATHS``."""
        assert f"{PARAM_SECTION}.label_sets" in DATA_MAP_PATHS

    def test_normalise_is_the_normalisation_the_transcript_was_built_under(self) -> None:
        """Not a config key: a function. A second spelling would compare against another normal form."""
        normalise = branch_params(load_triage_config()).p_normalise
        assert normalise("Hey,") == "hey"
        assert normalise("BUTTERCUP") == "buttercup"
        assert "normalise" not in load_triage_config().values[PARAM_SECTION]

    def test_a_tuple_valued_key_comes_back_as_a_tuple(self) -> None:
        """YAML gives a list; the body's arithmetic wants two floats."""
        assert _params(modulation_band_hz=[1.5, 9.0]).point("modulation_band_hz") == (1.5, 9.0)
        assert _params(phoneme_place_classes={"labial": ["p", "b"]}).point("phoneme_place_classes") == {
            "labial": ("p", "b")
        }

    def test_a_count_key_comes_back_as_an_integer(self) -> None:
        """``point("echo_ngram_n")`` indexes a slice; a float there is a TypeError at the call site."""
        assert isinstance(_params(echo_ngram_n=3).point("echo_ngram_n"), int)
        assert isinstance(_params(echo_ngram_n=3.0).point("echo_ngram_n"), int)


# --------------------------------------------------------------------- the shared helpers


class TestExtentArithmetic:
    """Hand-checkable, because every branch's boundaries are built out of these five."""

    def test_overlaps_is_open_at_the_boundary(self) -> None:
        """Two spans that merely touch do not overlap, or every adjacent pair would."""
        assert overlaps((0.0, 1.0), (0.5, 1.5))
        assert not overlaps((0.0, 1.0), (1.0, 2.0))
        assert not overlaps((2.0, 3.0), (0.0, 1.0))

    def test_duration_of_nothing_is_zero_rather_than_an_error(self) -> None:
        """A span with no extent is common in the store and is not a failure to read."""
        assert duration((1.25, 3.75)) == pytest.approx(2.5)
        assert duration(None) == 0.0

    def test_hull_covers_every_extent_and_is_none_when_there_are_none(self) -> None:
        """An empty hull is a result; a zero-length one at the origin would be a fabrication."""
        assert hull([(3.0, 4.0), (1.0, 2.0), (1.5, 9.0)]) == (1.0, 9.0)
        assert hull([]) is None

    def test_merge_joins_only_what_touches(self) -> None:
        """Touching counts; a gap of any size does not."""
        assert merge([(1.0, 2.0), (2.0, 3.0), (5.0, 6.0)]) == [(1.0, 3.0), (5.0, 6.0)]
        assert merge([(0.0, 1.0), (0.5, 0.75)]) == [(0.0, 1.0)]
        assert merge([]) == []

    def test_touches_edge_is_inclusive_at_both_ends(self) -> None:
        """A production running to the end of the file is truncated evidence, so the test is ``>=``."""
        assert touches_edge((0.0, 1.0), (0.0, 10.0))
        assert touches_edge((9.0, 10.0), (0.0, 10.0))
        assert not touches_edge((1.0, 9.0), (0.0, 10.0))

    def test_stream_extent_reads_the_recordings_own_extent(self) -> None:
        """The measured duration every "was it done" test needs, and None when nothing carries it."""
        assert stream_extent(_store(duration_s=12.5)) == (0.0, 12.5)
        assert stream_extent(_store(path=None)) is None


class TestReadingSpansAndWords:
    """The four span measures and the bracketed/lexical split are read once, here."""

    def test_each_measure_selects_only_its_own_spans_earliest_first(self) -> None:
        """``peak_over_floor_db`` exists only on amplitude, so a body must not mix the measures."""
        spans = [
            _span("s3", "gap", (5.0, 7.0)),
            _span("s1", "amplitude", (1.0, 2.0), peak_over_floor_db=30.0),
            _span("s2", "asr", (2.0, 4.0)),
            _span("s4", "amplitude", (0.0, 0.5)),
        ]
        assert [each.id for each in spans_by_measure(spans, "amplitude")] == ["s4", "s1"]
        assert [each.id for each in gaps(spans)] == ["s3"]
        assert spans_by_measure(spans, "continuity") == []

    def test_a_span_with_no_extent_is_dropped_rather_than_placed_at_the_origin(self) -> None:
        """Sorting an extentless span to ``(0, 0)`` would make it the first carrier every time."""
        spans = [Entity(id="s0", prov_type="span", extent=None, attributes={"measure": "amplitude"})]
        assert spans_by_measure(spans, "amplitude") == []

    def test_lexical_drops_the_bracketed_tokens(self) -> None:
        """``[breath]`` is airway evidence, not a word the participant said."""
        words = [_word(0, "one", (0.0, 0.4)), _word(1, "[breath]", (0.5, 1.0), bracketed=True)]
        assert [each.attributes["text"] for each in lexical(words)] == ["one"]


class TestLexicalRunsIsNotMerge:
    """The defect the design draft carried: ``merge`` returns one extent per word."""

    def test_non_touching_words_group_into_one_run(self) -> None:
        """Ordinary speech has a gap between every pair, so ``merge`` would group nothing."""
        words = [_word(0, "the", (1.00, 1.20)), _word(1, "quick", (1.35, 1.60)), _word(2, "fox", (1.70, 1.95))]
        assert merge([w.extent for w in words if w.extent]) == [(1.0, 1.2), (1.35, 1.6), (1.7, 1.95)]
        assert lexical_runs(words, max_gap_s=0.2) == [(1.0, 1.95)]

    def test_a_gap_wider_than_the_maximum_starts_a_new_run(self) -> None:
        """The grouping is a decision about the gap, so the gap has to be able to break it."""
        words = [_word(0, "a", (0.0, 0.2)), _word(1, "b", (0.3, 0.5)), _word(2, "c", (2.0, 2.3))]
        assert lexical_runs(words, max_gap_s=0.2) == [(0.0, 0.5), (2.0, 2.3)]

    def test_the_boundary_gap_is_inclusive(self) -> None:
        """A gap exactly at the maximum stays one run; the comparison is ``<=``."""
        words = [_word(0, "a", (0.0, 1.0)), _word(1, "b", (1.5, 2.0))]
        assert lexical_runs(words, max_gap_s=0.5) == [(0.0, 2.0)]
        assert lexical_runs(words, max_gap_s=0.49) == [(0.0, 1.0), (1.5, 2.0)]

    def test_no_words_is_no_runs(self) -> None:
        """An empty list is a result, not a degenerate extent."""
        assert lexical_runs([], max_gap_s=0.5) == []

    def test_an_overlapping_pair_does_not_shorten_the_run(self) -> None:
        """Two recognizers can place words out of order; the run must still cover both."""
        words = [_word(0, "a", (0.0, 2.0)), _word(1, "b", (1.0, 1.5))]
        assert lexical_runs(words, max_gap_s=0.2) == [(0.0, 2.0)]


class TestGapsAndGrouping:
    """Breath groups and inter-word gaps, on a fixture whose answer can be read off."""

    def test_only_gaps_at_or_over_the_minimum_are_reported(self) -> None:
        """The boundary is inclusive: a gap exactly at the minimum is a pause."""
        words = [_word(0, "a", (0.0, 1.0)), _word(1, "b", (1.5, 2.0)), _word(2, "c", (2.1, 2.5))]
        assert inter_word_gaps(words, min_gap_s=0.5) == [(1.0, 1.5)]
        assert inter_word_gaps(words, min_gap_s=0.05) == [(1.0, 1.5), (2.0, 2.1)]

    def test_a_break_between_two_words_splits_the_group(self) -> None:
        """A HeAR ``Breathe`` window or a ``[breath]`` token lands in exactly this position."""
        words = [_word(0, "a", (0.0, 1.0)), _word(1, "b", (2.0, 3.0)), _word(2, "c", (3.1, 4.0))]
        assert group_by_breaks(words, [(1.2, 1.8)]) == [(0.0, 1.0), (2.0, 4.0)]

    def test_a_break_outside_every_inter_word_interval_splits_nothing(self) -> None:
        """A breath before the first word is not a group boundary inside the production."""
        words = [_word(0, "a", (1.0, 2.0)), _word(1, "b", (2.1, 3.0))]
        assert group_by_breaks(words, [(0.0, 0.5)]) == [(1.0, 3.0)]

    def test_no_words_is_no_groups(self) -> None:
        """Nothing to group is not one group covering nothing."""
        assert group_by_breaks([], [(1.0, 2.0)]) == []

    def test_off_task_reports_an_uncovered_gap_only(self) -> None:
        """Off-task material is the absence of the speciality, so it is a deviation, never a span."""
        spans = [_span("g1", "gap", (0.0, 3.0)), _span("g2", "gap", (5.0, 5.2)), _span("g3", "gap", (8.0, 12.0))]
        components = [PROPOSERS["SPEECH"]("task_extent", (0.0, 2.0), "e1")]
        found = off_task(components, spans, p_gap_off_task_min_s=1.0)
        assert [(each.name, each.start, each.end) for each in found] == [("off_task_extent", 8.0, 12.0)]
        assert {each.kind for each in found} == {"deviation"}

    def test_declared_duration_is_a_count_beside_the_declaration(self) -> None:
        """A count asserts no discrepancy; which of the two is wrong is a separate question."""
        [finding] = declared_duration_count(_store(duration_s=19.004), 20.0)
        assert finding.kind == "count"
        assert finding.evidence == {"found": 19.0, "declared": 20.0}
        assert declared_duration_count(_store(), None) == []


class TestLexicalArithmetic:
    """The ordered match, the echo fractions and the occurrence index."""

    def test_the_ordered_match_is_greedy_left_to_right_and_reports_omissions(self) -> None:
        """Out-of-order material is an omission plus an unclaimed word, never a silent reorder."""
        words = [_word(0, "One", (0.0, 0.3)), _word(1, "three", (0.4, 0.7))]
        matched, omissions = ordered_run(["one", "two", "three"], words, lambda token: token.casefold())
        assert [token for token, _ in matched] == ["one", "three"]
        assert omissions == ["two"]

    def test_a_repeated_expected_token_consumes_a_second_word(self) -> None:
        """``loudness`` expects ``hey`` three times; the cursor must advance past each hit."""
        words = [_word(i, "hey", (float(i), float(i) + 0.2)) for i in range(2)]
        matched, omissions = ordered_run(["hey", "hey", "hey"], words, lambda token: token)
        assert len(matched) == 2
        assert omissions == ["hey"]

    def test_the_echo_fraction_is_over_the_sources_own_ngrams(self) -> None:
        """A prompt echo is measured against the prompt, so the denominator is the source."""
        source = ["a", "b", "c", "d"]
        assert ngram_echo_fraction(source, ["a", "b", "c", "d"], 2) == pytest.approx(1.0)
        assert ngram_echo_fraction(source, ["a", "b"], 2) == pytest.approx(1.0 / 3.0)
        assert ngram_echo_fraction(source, ["z"], 2) == 0.0
        assert ngram_echo_fraction([], ["a", "b"], 2) == 0.0

    def test_content_coverage_is_over_distinct_tokens(self) -> None:
        """Saying one word twice does not cover two."""
        assert content_coverage(["a", "b", "c"], ["a", "a", "b"]) == pytest.approx(2.0 / 3.0)
        assert content_coverage([], ["a"]) == 0.0

    def test_token_occurrences_indexes_every_word_of_each_token(self) -> None:
        """A repeated item is a deviation over that word's own extent, so all of them are kept."""
        words = [_word(0, "Dog", (0.0, 0.3)), _word(1, "cat", (0.5, 0.8)), _word(2, "dog", (1.0, 1.3))]
        found = token_occurrences(words, lambda token: token.casefold())
        assert sorted(found) == ["cat", "dog"]
        assert [each.id for each in found["dog"]] == ["word-0", "word-2"]


class TestArrayHelpers:
    """Slices, spreads and runs, each on a series whose answer is arithmetic."""

    def test_the_envelope_slice_reports_where_it_started(self) -> None:
        """A found index has to be put back into seconds, so the offset travels with the slice."""
        envelope = EnvelopeTrack(envelope_dbfs=np.arange(100.0), floor_dbfs=-60.0, sampling_rate=10.0)
        values, offset = envelope_slice(envelope, (2.0, 4.0))
        assert offset == 20
        assert values.tolist() == list(np.arange(20.0, 40.0))

    def test_a_slice_outside_the_envelope_is_empty_rather_than_clamped(self) -> None:
        """Clamping would place a measurement on ground the extent does not name."""
        envelope = EnvelopeTrack(envelope_dbfs=np.arange(10.0), floor_dbfs=-60.0, sampling_rate=10.0)
        values, _ = envelope_slice(envelope, (5.0, 6.0))
        assert values.size == 0

    def test_peak_over_floor_is_measured_against_the_one_global_floor(self) -> None:
        """PREPROCESS broadcasts one value; a local floor would make every event prominent."""
        envelope = EnvelopeTrack(envelope_dbfs=np.array([-50.0, -20.0, -45.0]), floor_dbfs=-60.0, sampling_rate=1.0)
        assert peak_over_floor_db(envelope, (0.0, 3.0)) == pytest.approx(40.0)
        assert np.isnan(peak_over_floor_db(envelope, (9.0, 10.0)))

    def test_semitones_are_about_the_voiced_median_and_nan_elsewhere(self) -> None:
        """An unvoiced frame must not read as a pitch of zero, which is an octave excursion."""
        out = semitones(np.array([100.0, 200.0, 400.0, 0.0, np.nan]))
        assert out[0] == pytest.approx(-12.0, abs=1e-9), "the reference is the voiced median, 200 Hz"
        assert out[1] == pytest.approx(0.0, abs=1e-9)
        assert out[2] == pytest.approx(12.0, abs=1e-9)
        assert np.isnan(out[3]) and np.isnan(out[4])
        assert semitones(np.array([110.0, 220.0]), ref_hz=110.0).tolist() == pytest.approx([0.0, 12.0])
        assert np.isnan(semitones(np.array([0.0, 0.0]))).all()

    def test_the_windowed_spread_finds_the_worst_window_not_the_whole_series(self) -> None:
        """A slow drift over a long vowel is not instability; a local jump is."""
        drift = np.linspace(0.0, 10.0, 101)
        assert max_windowed_spread(drift, hop_s=0.01, window_s=0.10) < 2.0
        jump = np.concatenate([np.zeros(50), np.full(51, 10.0)])
        assert max_windowed_spread(jump, hop_s=0.01, window_s=0.10) > 8.0

    def test_a_series_shorter_than_one_window_falls_back_to_the_whole_spread(self) -> None:
        """Returning zero there would report a two-frame excursion as perfect stability."""
        assert max_windowed_spread(np.array([0.0, 10.0]), hop_s=0.01, window_s=1.0) == pytest.approx(9.0)

    def test_the_longest_monotone_run_tolerates_a_reversal_under_the_tolerance(self) -> None:
        """A glide is not strictly monotone; a jitter of a few cents must not cut it in two."""
        rising = np.array([0.0, 1.0, 0.9, 2.0, 3.0])
        assert longest_monotone_run(rising, tolerance=0.5) == (0, 4, 1)
        strict = longest_monotone_run(rising, tolerance=0.0)
        assert strict is not None and strict[0] == 2, "with no tolerance the reversal cuts the run"

    def test_the_run_reports_the_direction_it_found(self) -> None:
        """A downward glide declared upward is the finding, so the sign has to come back."""
        assert longest_monotone_run(np.array([5.0, 4.0, 3.0, 2.0]), tolerance=0.1) == (0, 3, -1)
        assert longest_monotone_run(np.array([np.nan]), tolerance=0.1) is None

    def test_the_track_slice_decides_voicing_and_reports_the_hop(self) -> None:
        """Every voiced-fraction test reads ``voiced``, so the threshold is applied once."""
        tracks = PhonationTracks(
            times_s=np.array([0.0, 0.01, 0.02, 0.03]),
            f0_hz=np.array([100.0, 110.0, 0.0, 105.0]),
            strength=np.array([0.9, 0.8, 0.1, 0.7]),
        )
        sliced = track_slice(tracks, (0.0, 0.03), p_voiced_strength_min=0.5)
        assert sliced.times_s.tolist() == [0.0, 0.01, 0.02]
        assert sliced.voiced.tolist() == [True, True, False]
        assert sliced.hop_s == pytest.approx(0.01)

    def test_the_voiced_extent_is_the_production_not_the_carrier(self) -> None:
        """The carrier span holds count-in and silence; the measurement wants the vowel."""
        tracks = PhonationTracks(
            times_s=np.array([0.0, 0.5, 1.0, 1.5]),
            f0_hz=np.array([0.0, 120.0, 120.0, 0.0]),
            strength=np.array([0.1, 0.9, 0.9, 0.1]),
        )
        sliced = track_slice(tracks, (0.0, 2.0), p_voiced_strength_min=0.5)
        assert voiced_extent((0.0, 2.0), sliced) == pytest.approx((0.5, 1.5))

    def test_an_unvoiced_carrier_keeps_its_own_extent(self) -> None:
        """A carrier with nothing voiced inside is a finding, not a zero-length span."""
        tracks = PhonationTracks(times_s=np.array([0.0]), f0_hz=np.array([0.0]), strength=np.array([0.1]))
        sliced = track_slice(tracks, (0.0, 1.0), p_voiced_strength_min=0.5)
        assert voiced_extent((0.0, 1.0), sliced) == (0.0, 1.0)

    def test_the_continuity_slice_is_empty_outside_the_trace(self) -> None:
        """A stationarity of nothing is not a stationarity of zero."""
        trace = ContinuityTrack(continuity=np.linspace(0.0, 1.0, 11), sampling_rate=10.0)
        assert trace_slice(trace, (0.0, 0.5)).size == 5
        assert trace_slice(trace, (5.0, 6.0)).size == 0

    def test_the_spectral_balance_is_high_over_low_in_db(self) -> None:
        """A flat spectrum splits at zero when the split is the band's midpoint."""
        flat = SpectrogramBlock(spectrogram=np.ones((9, 4)), n_fft=16, hop_length=8, sampling_rate=16.0)
        assert spectral_balance_db(flat, (0.0, 2.0), split_hz=4.0) == pytest.approx(0.0, abs=1e-9)

    def test_energy_only_in_the_low_band_reads_far_negative(self) -> None:
        """The direction is the finding, so the sign has to be the one a reader expects."""
        low_only = SpectrogramBlock(
            spectrogram=np.vstack([np.ones((4, 4)), np.full((5, 4), 1e-8)]),
            n_fft=16,
            hop_length=8,
            sampling_rate=16.0,
        )
        assert spectral_balance_db(low_only, (0.0, 2.0), split_hz=4.0) < -40.0

    def test_the_high_band_stops_below_nyquist_rather_than_at_it(self) -> None:
        """``[split, sr/2)`` excludes the Nyquist bin.

        One bin of many at a real transform length, but a split AT Nyquist therefore selects
        nothing and the balance is NaN rather than zero.
        """
        flat = SpectrogramBlock(spectrogram=np.ones((9, 4)), n_fft=16, hop_length=8, sampling_rate=16.0)
        assert np.isnan(spectral_balance_db(flat, (0.0, 2.0), split_hz=8.0))

    def test_an_extent_outside_the_spectrogram_reads_as_nan(self) -> None:
        """A band power of nothing is not a band power of zero."""
        flat = SpectrogramBlock(spectrogram=np.ones((9, 4)), n_fft=16, hop_length=8, sampling_rate=16.0)
        assert np.isnan(spectral_balance_db(flat, (100.0, 101.0), split_hz=4.0))


class TestTheThreeInstruments:
    """The event walk, the label read and the rate, each on a synthetic signal."""

    @staticmethod
    def _bump(size: int, centres: tuple[int, ...], peak_dbfs: float, half_width: int) -> np.ndarray:
        """A floor with one triangular rise per centre, which is what an event looks like.

        Args:
            size: Samples.
            centres: Where each rise peaks.
            peak_dbfs: The value at each peak.
            half_width: Samples from a peak back down to the floor.

        Returns:
            The envelope.
        """
        envelope = np.full(size, -60.0)
        for centre in centres:
            for offset in range(-half_width + 1, half_width):
                envelope[centre + offset] = -60.0 + (peak_dbfs + 60.0) * (1.0 - abs(offset) / half_width)
        return envelope

    def test_two_maxima_inside_one_extent_become_two_events(self) -> None:
        """The whole point: ``by_label`` counts a span holding three coughs as one today."""
        track = EnvelopeTrack(
            envelope_dbfs=self._bump(500, (100, 300), -10.0, 40), floor_dbfs=-60.0, sampling_rate=100.0
        )
        params = _params(smoothing_window_s=0.01, peak_prominence_db=10.0, trough_return_db=6.0, event_min_s=0.05)
        events = events_in_extent(track, (0.0, 5.0), params)
        assert len(events) == 2
        assert events[0][0] < 1.0 < events[0][1]
        assert events[1][0] < 3.0 < events[1][1]

    def test_a_peak_under_the_prominence_is_not_an_event(self) -> None:
        """The prominence is measured twice: over the global floor and over the flanking troughs."""
        track = EnvelopeTrack(envelope_dbfs=self._bump(300, (150,), -57.0, 40), floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(smoothing_window_s=0.01, peak_prominence_db=10.0, trough_return_db=6.0, event_min_s=0.05)
        assert events_in_extent(track, (0.0, 3.0), params) == []

    def test_a_rise_under_the_global_floor_prominence_is_not_an_event(self) -> None:
        """The two prominence gates are independent: a rise can clear its troughs and not the floor.

        PREPROCESS's floor is one global value for the whole recording, so an event in a quiet
        stretch of a loud file sits below it while still standing well over its own surroundings.
        """
        track = EnvelopeTrack(envelope_dbfs=self._bump(300, (150,), -20.0, 40), floor_dbfs=-10.0, sampling_rate=100.0)
        params = _params(smoothing_window_s=0.01, peak_prominence_db=10.0, trough_return_db=6.0, event_min_s=0.05)
        assert events_in_extent(track, (0.0, 3.0), params) == []

    def test_a_flat_topped_plateau_yields_no_event(self) -> None:
        """A digitally flat rise reports nothing.

        The ported walk needs a trough on BOTH sides (``max(left_min, right_min)``), and a
        plateau's last sample has none on its left.
        """
        envelope = np.full(300, -60.0)
        envelope[120:180] = -10.0
        track = EnvelopeTrack(envelope_dbfs=envelope, floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(smoothing_window_s=0.0, peak_prominence_db=10.0, trough_return_db=6.0, event_min_s=0.05)
        assert events_in_extent(track, (0.0, 3.0), params) == []

    def test_an_event_shorter_than_the_minimum_is_dropped(self) -> None:
        """The minimum is what keeps a single-sample spike out of a cough count."""
        envelope = np.full(300, -60.0)
        envelope[150] = -10.0
        track = EnvelopeTrack(envelope_dbfs=envelope, floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(smoothing_window_s=0.0, peak_prominence_db=10.0, trough_return_db=6.0, event_min_s=0.5)
        assert events_in_extent(track, (0.0, 3.0), params) == []

    def test_too_few_samples_is_no_events_rather_than_an_error(self) -> None:
        """A one-sample extent has no maximum to walk away from."""
        track = EnvelopeTrack(envelope_dbfs=np.array([-10.0, -20.0]), floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(smoothing_window_s=0.01, peak_prominence_db=1.0, trough_return_db=1.0, event_min_s=0.0)
        assert events_in_extent(track, (0.0, 0.02), params) == []

    def test_sounds_like_reads_raw_scores_of_this_spans_windows_only(self) -> None:
        """``labels`` is a top-K decision the shipped config leaves unmade; ``raw_scores`` is always there."""
        span = _span("s1", "amplitude", (0.0, 2.0))
        mine = Entity(
            id="w1",
            prov_type="measurement",
            extent=(0.0, 2.0),
            attributes={"name": "span_hear", "span_id": "s1", "raw_scores": {"Cough": 0.42, "Speech": 0.01}},
        )
        someone_elses = Entity(
            id="w2",
            prov_type="measurement",
            extent=(0.0, 2.0),
            attributes={"name": "span_hear", "span_id": "s2", "raw_scores": {"Cough": 0.99}},
        )
        assert sounds_like(span, [mine, someone_elses], ["Cough"], p_score_min=0.4)
        assert not sounds_like(span, [mine, someone_elses], ["Cough"], p_score_min=0.5)
        assert not sounds_like(span, [someone_elses], ["Cough"], p_score_min=0.4)

    def test_a_window_with_no_scores_at_all_is_not_a_crash(self) -> None:
        """SQUIM and the classifiers both refuse sometimes, and a refusal carries no scores."""
        span = _span("s1", "amplitude", (0.0, 2.0))
        refused = Entity(
            id="w1",
            prov_type="measurement",
            extent=(0.0, 2.0),
            attributes={"name": "span_hear", "span_id": "s1", "unmeasured": "no_native_window"},
        )
        assert not sounds_like(span, [refused], ["Cough"], p_score_min=0.1)

    def test_the_train_rate_finds_a_synthesised_modulation(self) -> None:
        """A 4 Hz envelope modulation reads back as 4 Hz, within the transform's own resolution."""
        rate = 100.0
        times = np.arange(0.0, 4.0, 1.0 / rate)
        envelope = -30.0 + 10.0 * np.sin(2.0 * np.pi * 4.0 * times)
        track = EnvelopeTrack(envelope_dbfs=envelope, floor_dbfs=-60.0, sampling_rate=rate)
        params = _params(modulation_band_hz=[1.0, 12.0], rate_prominence_min=2.0)
        assert train_rate_hz(track, (0.0, 4.0), params) == pytest.approx(4.0, abs=0.3)

    def test_an_unmodulated_envelope_has_no_readable_rate(self) -> None:
        """None is the honest answer; a peak of noise would be a rate nobody produced."""
        track = EnvelopeTrack(envelope_dbfs=np.full(400, -30.0), floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(modulation_band_hz=[1.0, 12.0], rate_prominence_min=2.0)
        assert train_rate_hz(track, (0.0, 4.0), params) is None

    def test_too_short_an_extent_has_no_readable_rate(self) -> None:
        """Eight samples cannot carry a modulation spectrum."""
        track = EnvelopeTrack(envelope_dbfs=np.arange(4.0), floor_dbfs=-60.0, sampling_rate=100.0)
        params = _params(modulation_band_hz=[1.0, 12.0], rate_prominence_min=2.0)
        assert train_rate_hz(track, (0.0, 0.04), params) is None

    def test_a_band_selecting_nothing_has_no_readable_rate(self) -> None:
        """A band above the envelope's own Nyquist selects no bin, which is not a rate of zero."""
        track = EnvelopeTrack(envelope_dbfs=np.arange(100.0), floor_dbfs=-60.0, sampling_rate=10.0)
        params = _params(modulation_band_hz=[100.0, 200.0], rate_prominence_min=1.0)
        assert train_rate_hz(track, (0.0, 10.0), params) is None


class TestReadingADerivativeBack:
    """A derivative the store never received is absent, not zero."""

    def test_an_absent_measurement_reads_as_none(self) -> None:
        """Four branches would otherwise each learn this separately, three of them wrongly."""
        assert read_envelope_track(_store(), Path("/nonexistent")) is None

    def test_a_measurement_whose_sidecar_is_gone_reads_as_none(self) -> None:
        """A run directory can be moved or pruned; that is not an envelope of silence."""
        store = _store()
        activity = store.activity(node="PREPROCESS", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        entity = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "energy_envelope",
                "signal": "plain",
                "path": "derivatives/energy_envelope.npz",
                "sampling_rate": 16000,
            },
        )
        store.was_generated_by(entity, activity)
        store.was_attributed_to(entity, agent)
        assert read_envelope_track(store, Path("/nonexistent")) is None

    def test_the_global_floor_comes_back_as_one_number(self, tmp_path: Path) -> None:
        """PREPROCESS broadcasts it with ``np.full_like``; a body wants the scalar."""
        (tmp_path / "derivatives").mkdir()
        np.savez(
            tmp_path / "derivatives" / "energy_envelope.npz",
            envelope_dbfs=np.array([-30.0, -20.0, -25.0]),
            floor_dbfs=np.full(3, -58.5),
        )
        store = _store()
        activity = store.activity(node="PREPROCESS", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        entity = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "energy_envelope",
                "signal": "plain",
                "path": "derivatives/energy_envelope.npz",
                "sampling_rate": 100,
            },
        )
        store.was_generated_by(entity, activity)
        store.was_attributed_to(entity, agent)
        track = read_envelope_track(store, tmp_path)
        assert track is not None
        assert track.floor_dbfs == pytest.approx(-58.5)
        assert track.sampling_rate == pytest.approx(100.0)
        assert peak_over_floor_db(track, (0.0, 0.03)) == pytest.approx(38.5)


class TestTheDeviationVocabularyIsClosed:
    """Eleven deviation types were emitted against three declared, because nothing checked.

    The write now refuses an undeclared name, and this sweep refuses a declaration nobody emits —
    so the two cannot drift apart again in either direction.
    """

    @staticmethod
    def _emitted() -> dict[str, list[str]]:
        """Every ``deviation(...)`` name in the triage tree, by site.

        An AST walk rather than a grep: the calls that hid this defect wrap across lines, and a
        grep for ``deviation("name"`` matched only the single-line ones.
        """
        found: dict[str, list[str]] = {}
        root = Path(branches_module.__file__ or "").parent.parent
        for source in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(source.read_text())):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "deviation" or not node.args:
                    continue
                first = node.args[0]
                assert isinstance(first, ast.Constant) and isinstance(first.value, str), (
                    f"{source.name}:{node.lineno} builds a deviation name at runtime; the sweep "
                    "cannot check it and VERDICT cannot fold it"
                )
                found.setdefault(first.value, []).append(f"{source.name}:{node.lineno}")
        return found

    def test_every_emitted_deviation_is_declared(self) -> None:
        """A new deviation type fails here until someone declares what it observes."""
        undeclared = {name: sites for name, sites in self._emitted().items() if name not in DEVIATION_TYPES}
        assert undeclared == {}, f"emitted but not in DEVIATION_TYPES: {undeclared}"

    def test_every_declared_deviation_is_emitted(self) -> None:
        """A declared type nobody writes is a claim about the graph that is not true."""
        unemitted = sorted(set(DEVIATION_TYPES) - set(self._emitted()))
        assert unemitted == [], f"declared in DEVIATION_TYPES but emitted nowhere: {unemitted}"

    def test_every_type_says_what_it_observes(self) -> None:
        """The declaration carries a meaning, not just a name."""
        for name, what in DEVIATION_TYPES.items():
            assert what and not what.endswith("."), name

    def test_the_write_refuses_an_undeclared_deviation(self) -> None:
        """The mechanism that stops the next drift, at the only function that writes."""
        store = _store()
        agent = store.agent(agent_type="software", version="test")
        activity = store.activity(node="VOICE", step="branch", parameters={})
        finding = Finding("deviation", "invented_type", 0.0, 1.0, {}, ("word-1",))
        with pytest.raises(ValueError, match="undeclared deviation types"):
            write_findings(store, activity, agent, [finding], signal="plain")

    def test_a_declared_deviation_still_writes(self) -> None:
        """The guard refuses the undeclared name and nothing else."""
        store = _store()
        agent = store.agent(agent_type="software", version="test")
        activity = store.activity(node="VOICE", step="branch", parameters={})
        finding = Finding("deviation", "truncation", 0.0, 1.0, {}, ("word-1",))
        assert len(write_findings(store, activity, agent, [finding], signal="plain")) == 1
