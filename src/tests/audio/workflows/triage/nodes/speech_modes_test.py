"""SPEECH's two modes, over hand-built stores: no model runs and no node machinery is involved.

``align_speech`` evaluates a declared speech task against what its instruction asked for;
``detect_speech`` finds lexical speech on a recording of another branch's kind and evaluates
nothing. What is pinned here is which mode a declaration selects, what each mode proposes, and the
four defects the design was executed to find: a designed-empty family scoring its own transcript as
a departure, ``free-speech``'s two versions sharing an expectation, Stroop matched against the words
it displays rather than the answers it asks for, and ``detect_speech`` grouping words with ``merge``.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.audio.data_structures import AudioHints, ExpectedSpeech
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.branches import (
    SPEECH_EXPECTATIONS,
    UNDETERMINED,
    UNMEASURED_POINTS,
    Finding,
    Proposal,
    Result,
    branch_params,
    dispatch,
    merge,
    mode_of,
    write_findings,
)
from senselab.audio.workflows.triage.nodes.speech import (
    align_speech,
    detect_speech,
)
from senselab.utils.prov_store import ProvStore

HARVARD = "The birch canoe slid on the smooth planks."
"""One Harvard sentence, as ``stimulus_text`` spells it."""


def _config(tmp_path: Path, branch: dict[str, Any] | None = None) -> TriageConfig:
    """The packaged config, with the named ``branch`` operating points measured.

    Args:
        tmp_path: Where the override file is written.
        branch: The ``branch.*`` keys to supply, or None for the packaged config as shipped.

    Returns:
        The resolved configuration.
    """
    values = {"branch": dict(branch)} if branch else {}
    path = tmp_path / f"override-{abs(hash(yaml.safe_dump(values))) % 10**10}.yaml"
    path.write_text(yaml.safe_dump(values))
    return load_triage_config(path)


def _store(*, family: str | None = None, duration_s: float = 20.0) -> ProvStore:
    """A store carrying only the ``recording`` stream the mode selector reads.

    Args:
        family: The declared task family, written into a BIDS stem, or None for no declaration.
        duration_s: The recording's duration.

    Returns:
        The store.
    """
    store = ProvStore(run_id="speech-modes-test")
    path = "input.wav" if family is None else f"sub-abc123_ses-1_task-{family}.wav"
    store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={"name": "recording", "path": path, "sampling_rate": 16000, "channels": 1},
    )
    return store


def _transcript(store: ProvStore, placed: list[tuple[str, float, float]]) -> str:
    """Write the consensus words and the measurement that names them.

    Args:
        store: The store to write into.
        placed: ``(text, start, end)`` per word, in stream order. A bracketed text is bracketed.

    Returns:
        The ``consensus_transcript`` measurement's id.
    """
    word_ids: list[str] = []
    for index, (text, start, end) in enumerate(placed):
        word_ids.append(
            store.entity(
                prov_type="word",
                extent=(start, end),
                attributes={
                    "text": text,
                    "index": index,
                    "bracketed": text.startswith("[") and text.endswith("]"),
                    "agreement": 1.0,
                    "variants": [],
                },
            )
        )
    return store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "consensus_transcript",
            "signal": "plain",
            "text": " ".join(text for text, _, _ in placed),
            "word_ids": word_ids,
            "sources": [{"name": "whisper"}],
        },
    )


def _stimulus_measurement(store: ProvStore, **counts: int) -> str:
    """Write PREPROCESS's ``stimulus_alignment`` measurement, with whatever counts a test pins.

    Args:
        store: The store to write into.
        **counts: The provenance counts the derivative recorded, if any.

    Returns:
        The measurement's id.
    """
    return store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "stimulus_alignment",
            "signal": "plain",
            "path": "derivatives/stimulus_alignment.npz",
            **counts,
        },
    )


def _asr_span(store: ProvStore, extent: tuple[float, float]) -> str:
    """Write one of PREPROCESS's ASR spans, which a free response is measured over.

    Args:
        store: The store to write into.
        extent: The span's extent.

    Returns:
        The span entity's id.
    """
    return store.entity(
        prov_type="span",
        extent=extent,
        attributes={"signal": "plain", "measure": "asr", "merged_proposals": 1},
    )


def _hint(*texts: str) -> AudioHints:
    """A declaration carrying these prompts and nothing else.

    Args:
        *texts: The ``expected_speech`` entries, in declared order.

    Returns:
        The hints.
    """
    return AudioHints(expected_speech=[ExpectedSpeech(text=text) for text in texts])


def _roles(result: Result) -> list[str]:
    """The roles of every span a result proposed.

    Args:
        result: What a mode returned.

    Returns:
        The roles, in the order proposed.
    """
    return [proposal.role for proposal in result.components]


def _of_kind(result: Result, kind: str, name: str) -> list[Finding]:
    """Every finding of one kind and name.

    Args:
        result: What a mode returned.
        kind: The finding kind.
        name: The deviation type, count name or measurement name.

    Returns:
        The findings.
    """
    return [finding for finding in result.deviations if finding.kind == kind and finding.name == name]


def _unmeasured(result: Result) -> list[str]:
    """The ``branch.*`` keys the evaluation asked for and could not read.

    Args:
        result: What a mode returned.

    Returns:
        The keys, sorted, or an empty list when every key was read.
    """
    found = _of_kind(result, "measure", UNMEASURED_POINTS)
    return list(found[0].evidence["value"]) if found else []


class TestTheDeclarationPicksTheMode:
    """The declared family selects the mode and never supplies the answer."""

    @pytest.mark.parametrize("family", sorted(SPEECH_EXPECTATIONS))
    def test_every_in_family_declaration_takes_align(self, family: str) -> None:
        """All 31 rows are reachable: a family in the table never falls through to detect."""
        assert mode_of("SPEECH", _store(family=family)) == ("align", family)

    def test_another_branchs_kind_takes_detect(self) -> None:
        """A breath task is out of family for SPEECH, so it annotates and concludes nothing."""
        assert mode_of("SPEECH", _store(family="respiration-and-cough-cough")) == (
            "detect",
            "respiration-and-cough-cough",
        )

    def test_no_declaration_takes_detect(self) -> None:
        """A path that is not a BIDS stem names no family, which is the safe arm."""
        assert mode_of("SPEECH", _store()) == ("detect", None)

    def test_dispatch_runs_align_on_a_declared_speech_task(self, tmp_path: Path) -> None:
        """The in-family arm evaluates the task, so `done` is an answer rather than UNDETERMINED."""
        store = _store(family="harvard-sentences-list")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(HARVARD.split())])
        _stimulus_measurement(store)
        result = dispatch(
            "SPEECH",
            store,
            branch_params(_config(tmp_path)),
            _hint(HARVARD),
            align=align_speech,
            detect=detect_speech,
        )
        assert result.done is True
        assert "task_extent" in _roles(result)

    def test_dispatch_runs_detect_on_another_branchs_kind(self, tmp_path: Path) -> None:
        """The out-of-family arm evaluates no task, so its only answer is UNDETERMINED."""
        store = _store(family="respiration-and-cough-cough")
        _transcript(store, [("okay", 1.0, 1.4), ("ready", 1.5, 1.9)])
        result = dispatch(
            "SPEECH",
            store,
            branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})),
            None,
            align=align_speech,
            detect=detect_speech,
        )
        assert result.done == UNDETERMINED
        assert _roles(result) == ["lexical_run_0"]

    def test_align_refuses_a_family_that_is_not_its_own(self, tmp_path: Path) -> None:
        """The caller owes detect for an out-of-family task; align does not guess one."""
        with pytest.raises(KeyError, match="not a SPEECH family"):
            align_speech("respiration-and-cough-cough", _store(), None, branch_params(_config(tmp_path)))


class TestAFullySpecifiedFamilyAlignsAgainstTheDerivative:
    """S3: the declared text against what was said, read off PREPROCESS's stimulus alignment."""

    def _read(self, spoken: list[str], **counts: int) -> tuple[ProvStore, AudioHints]:
        """A Harvard recording whose transcript is ``spoken``.

        Args:
            spoken: The words the consensus produced, in order.
            **counts: Provenance counts to write onto the derivative.

        Returns:
            The store and the declaration.
        """
        store = _store(family="harvard-sentences-list")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(spoken)])
        _stimulus_measurement(store, **counts)
        return store, _hint(HARVARD)

    def test_a_faithful_reading_realises_every_token(self, tmp_path: Path) -> None:
        """Nine tokens declared, nine realised: the task was done and nothing departed."""
        store, hint = self._read(HARVARD.split())
        result = align_speech("harvard-sentences-list", store, hint, branch_params(_config(tmp_path)))
        assert result.done is True
        assert _of_kind(result, "deviation", "stimulus_mismatch") == []
        assert _of_kind(result, "deviation", "omission") == []
        [task_extent] = [proposal for proposal in result.components if proposal.role == "task_extent"]
        assert task_extent.attributes == {"words_n": 8, "expected_n": 8}

    def test_a_substitution_is_reported_with_both_surfaces(self, tmp_path: Path) -> None:
        """A substituted token: the deviation carries what was expected and what was read."""
        spoken = HARVARD.split()
        spoken[2] = "canoes"
        store, hint = self._read(spoken)
        result = align_speech("harvard-sentences-list", store, hint, branch_params(_config(tmp_path)))
        [mismatch] = _of_kind(result, "deviation", "stimulus_mismatch")
        [task_extent] = [proposal for proposal in result.components if proposal.role == "task_extent"]
        assert task_extent.attributes == {"words_n": 7, "expected_n": 8}, "a substitution realised no token"
        assert mismatch.evidence["expected"] == "canoe"
        assert mismatch.evidence["read"] == "canoes"
        assert (mismatch.start, mismatch.end) == (3.0, 3.5)

    def test_an_omission_is_placed_where_it_should_have_been(self, tmp_path: Path) -> None:
        """A skipped word has no extent of its own, so it is anchored at the last realised token."""
        spoken = [word for word in HARVARD.split() if word != "smooth"]
        store, hint = self._read(spoken)
        result = align_speech("harvard-sentences-list", store, hint, branch_params(_config(tmp_path)))
        [omission] = _of_kind(result, "deviation", "omission")
        assert omission.evidence["expected"] == "smooth"
        assert omission.start == omission.end == 6.5, "the end of `the`, the last token realised before it"
        assert result.done is False, "a token nothing realised is the task not being done"
        assert omission.derived_from, "an anchored omission carries an extent, so it names its evidence"
        activity = store.activity(node="SPEECH", step=None, parameters={})
        agent = store.agent(agent_type="software", version="test")
        written = write_findings(store, activity, agent, result.deviations, signal="plain")
        anchored = [
            entity_id for entity_id in written if store.get_entity(entity_id).attributes.get("expected") == "smooth"
        ]
        assert anchored, "the anchored omission has to reach the store for the edge to be checkable"
        assert store.derived_from(anchored[0]) == list(omission.derived_from)

    def test_the_declared_structure_becomes_one_span_per_sentence(self, tmp_path: Path) -> None:
        """One declared string carrying four terminators yields four structure spans, not one."""
        passage = "When the sunlight strikes raindrops. There is a rainbow. People look. Others pass by."
        store = _store(family="rainbow-passage")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(passage.split())])
        _stimulus_measurement(store)
        result = align_speech("rainbow-passage", store, _hint(passage), branch_params(_config(tmp_path)))
        structure = [proposal for proposal in result.components if proposal.role.startswith("structure_")]
        assert [proposal.role for proposal in structure] == [
            "structure_0",
            "structure_1",
            "structure_2",
            "structure_3",
        ]
        assert [proposal.attributes["expected_n"] for proposal in structure] == [5, 4, 2, 3]

    def test_the_absent_derivative_proposes_nothing_rather_than_guessing(self, tmp_path: Path) -> None:
        """No alignment and a per-recording expectation: no extent can be placed at all."""
        store = _store(family="harvard-sentences-list")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(HARVARD.split())])
        result = align_speech("harvard-sentences-list", store, _hint(HARVARD), branch_params(_config(tmp_path)))
        assert result.done == UNDETERMINED
        assert result.components == []
        [unviable] = _of_kind(result, "measure", "expected_token_sequence")
        assert "stimulus_alignment is absent" in unviable.evidence["why"]

    def test_a_rebuild_disagreeing_with_the_recorded_counts_is_said_so(self, tmp_path: Path) -> None:
        """The projections are rebuilt from the same inputs; a divergence is a finding, not silence."""
        store, hint = self._read(HARVARD.split(), n_expected=99)
        result = align_speech("harvard-sentences-list", store, hint, branch_params(_config(tmp_path)))
        [disagreement] = _of_kind(result, "measure", "stimulus_alignment_rebuild_agrees")
        assert disagreement.evidence["value"] is False

    def test_a_reading_touching_the_recordings_edge_is_a_truncation(self, tmp_path: Path) -> None:
        """The recording may have started after the speaker did; that is the deviation."""
        store = _store(family="harvard-sentences-list", duration_s=9.0)
        _transcript(store, [(word, 0.0 + index, 0.5 + index) for index, word in enumerate(HARVARD.split())])
        _stimulus_measurement(store)
        result = align_speech("harvard-sentences-list", store, _hint(HARVARD), branch_params(_config(tmp_path)))
        assert _of_kind(result, "deviation", "truncation")

    def test_a_literal_token_row_needs_no_derivative(self, tmp_path: Path) -> None:
        """`loudness` prescribes "hey" three times in its instruction, not in a stimulus text."""
        store = _store(family="loudness")
        _transcript(store, [("hey", 1.0, 1.3), ("hey", 3.0, 3.3), ("hey", 5.0, 5.3)])
        result = align_speech("loudness", store, None, branch_params(_config(tmp_path)))
        assert result.done is True
        [counted] = _of_kind(result, "count", "expected_event_count")
        assert counted.evidence == {"found": 3, "declared": 3}


class TestADesignedEmptyFamilyScoresNothingAsADeparture:
    """`picture-description` carries `stimulus_text: ''` on all 1,591: nothing lexical is expected."""

    def _described(self, family: str = "picture-description") -> ProvStore:
        """A picture description: three ASR spans and nine words, and no declared text.

        Args:
            family: The declared family.

        Returns:
            The store.
        """
        store = _store(family=family)
        _transcript(store, [(f"word{index}", 1.0 + index, 1.6 + index) for index in range(9)])
        for start in (1.0, 4.0, 7.0):
            _asr_span(store, (start, start + 2.6))
        return store

    def test_every_lexical_word_is_the_response(self, tmp_path: Path) -> None:
        """The whole transcript is the lexical complement, because nothing was expected of it."""
        result = align_speech(
            "picture-description", self._described(), None, branch_params(_config(tmp_path, {"response_min_s": 1.0}))
        )
        [task_extent] = [proposal for proposal in result.components if proposal.role == "task_extent"]
        assert task_extent.attributes["words_n"] == 9
        assert (task_extent.start, task_extent.end) == (1.0, 9.6)
        assert result.done is True

    def test_no_departure_from_an_expectation_that_does_not_exist(self, tmp_path: Path) -> None:
        """An absent expectation is a typed outcome; it is not nine mismatches."""
        result = align_speech(
            "picture-description", self._described(), None, branch_params(_config(tmp_path, {"response_min_s": 1.0}))
        )
        assert _of_kind(result, "deviation", "stimulus_mismatch") == []
        assert _of_kind(result, "deviation", "omission") == []
        assert _of_kind(result, "deviation", "filler") == []

    def test_the_measurement_that_needs_no_cut_is_taken_anyway(self, tmp_path: Path) -> None:
        """Speech rate over the response needs no operating point, so an absent one costs nothing."""
        result = align_speech("picture-description", self._described(), None, branch_params(_config(tmp_path)))
        [rate] = _of_kind(result, "measure", "speech_rate_from_consensus_words_per_s")
        assert rate.evidence == {"value": 1.047, "support_words": 9}

    def test_an_unmeasured_operating_point_is_named_rather_than_defaulted(self, tmp_path: Path) -> None:
        """Every `branch` key now ships a value; nulling one is still named rather than defaulted."""
        params = branch_params(_config(tmp_path, {"response_min_s": None, "breath_group_min_gap_s": None}))
        result = align_speech("picture-description", self._described(), None, params)
        assert result.done == UNDETERMINED
        assert "response_min_s" in _unmeasured(result)
        assert "breath_group_min_gap_s" in _unmeasured(result)

    def test_the_family_whose_source_is_a_physical_book_says_so(self, tmp_path: Path) -> None:
        """`cinderella-story` has an empty `stimulus_text` on all 258; the overlap is unviable."""
        result = align_speech(
            "cinderella-story", self._described("cinderella-story"), None, branch_params(_config(tmp_path))
        )
        [unviable] = _of_kind(result, "measure", "source_overlap")
        assert "physical storybook" in unviable.evidence["why"]


class TestFreeSpeechsTwoVersionsExpectOppositeThings:
    """v1's instruction says not to read the prompt, so the prompt in the transcript is the finding."""

    PROMPT = "Tell me about a happy memory from your childhood"

    def _spoken(self, family: str, spoken: str) -> tuple[ProvStore, AudioHints]:
        """A free-speech recording whose transcript is ``spoken``.

        Args:
            family: `free-speech` or `free-speech-v2`.
            spoken: What the participant said.

        Returns:
            The store and the declaration.
        """
        store = _store(family=family)
        words = spoken.split()
        _transcript(store, [(word, 1.0 + 0.4 * index, 1.3 + 0.4 * index) for index, word in enumerate(words)])
        _asr_span(store, (1.0, 1.0 + 0.4 * len(words)))
        _stimulus_measurement(store)
        return store, _hint(self.PROMPT)

    def test_reading_the_prompt_back_is_the_deviation_in_v1(self, tmp_path: Path) -> None:
        """The negative pattern: the instruction forbids it, so its presence is reported."""
        store, hint = self._spoken("free-speech", self.PROMPT)
        params = branch_params(_config(tmp_path, {"echo_ngram_n": 2, "echo_overlap_max": 0.5, "response_min_s": 1.0}))
        result = align_speech("free-speech", store, hint, params)
        [overlap] = _of_kind(result, "measure", "verbatim_overlap_fraction")
        assert overlap.evidence["value"] == 1.0
        [mismatch] = _of_kind(result, "deviation", "stimulus_mismatch")
        assert mismatch.evidence["reading"] == "verbatim_prompt"

    def test_answering_the_prompt_in_v1_is_not_a_deviation(self, tmp_path: Path) -> None:
        """The anti-pattern is the prompt itself, not any particular content."""
        store, hint = self._spoken("free-speech", "we went to the lake every summer with my grandmother")
        params = branch_params(_config(tmp_path, {"echo_ngram_n": 2, "echo_overlap_max": 0.5, "response_min_s": 1.0}))
        result = align_speech("free-speech", store, hint, params)
        assert _of_kind(result, "deviation", "stimulus_mismatch") == []

    def test_v2_drops_the_clause_so_the_same_transcript_is_clean(self, tmp_path: Path) -> None:
        """Same family name, opposite expectation: v2's instruction no longer forbids the prompt."""
        store, hint = self._spoken("free-speech-v2", self.PROMPT)
        params = branch_params(_config(tmp_path, {"echo_ngram_n": 2, "echo_overlap_max": 0.5, "response_min_s": 1.0}))
        result = align_speech("free-speech-v2", store, hint, params)
        assert _of_kind(result, "deviation", "stimulus_mismatch") == []
        assert _of_kind(result, "measure", "verbatim_overlap_fraction") == []

    def test_story_recall_reads_coverage_rather_than_echo(self, tmp_path: Path) -> None:
        """Recall in your own words: coverage is expected and verbatim reproduction is not."""
        store, hint = self._spoken("story-recall", self.PROMPT)
        params = branch_params(_config(tmp_path, {"echo_ngram_n": 2, "verbatim_overlap_max": 0.5, "coverage_min": 0.5}))
        result = align_speech("story-recall", store, hint, params)
        [covered] = _of_kind(result, "measure", "source_content_coverage")
        assert covered.evidence["value"] == 1.0
        assert result.done is True


class TestStroopIsMatchedAgainstItsAnswersNotItsDisplay:
    """`word-color-stroop`'s `stimulus_text` is the colour sequence asked for, not the words shown."""

    ANSWERS = "red blue green"
    DISPLAYED = "blue green red"

    def _said(self, declared: str) -> tuple[ProvStore, AudioHints]:
        """A Stroop recording where the participant gave the ink colours correctly.

        Args:
            declared: What the declaration carries as ``expected_speech``.

        Returns:
            The store and the declaration.
        """
        store = _store(family="word-color-stroop")
        _transcript(
            store,
            [
                *[(word, 1.0 + 2 * index, 1.4 + 2 * index) for index, word in enumerate(self.ANSWERS.split())],
                ("[uh]", 6.0, 6.2),
            ],
        )
        _stimulus_measurement(store)
        return store, _hint(declared)

    def test_the_answers_declared_score_a_correct_response_as_correct(self, tmp_path: Path) -> None:
        """Three answers asked for, three given: no departure."""
        store, hint = self._said(self.ANSWERS)
        result = align_speech("word-color-stroop", store, hint, branch_params(_config(tmp_path)))
        assert _of_kind(result, "deviation", "stimulus_mismatch") == []
        assert result.done is True

    def test_the_display_declared_would_score_it_backwards(self, tmp_path: Path) -> None:
        """Matching the words on the card against the answers spoken inverts the task."""
        store, hint = self._said(self.DISPLAYED)
        result = align_speech("word-color-stroop", store, hint, branch_params(_config(tmp_path)))
        departures = _of_kind(result, "deviation", "stimulus_mismatch") + _of_kind(result, "deviation", "omission")
        assert len(departures) == 2, "an answer nothing displayed, and a display nothing answered"
        assert result.done is False

    def test_hesitation_is_the_dependent_variable_and_never_a_filler(self, tmp_path: Path) -> None:
        """`emit_filler` is False on this row: a filled pause here is the measurement, not a fault."""
        store, hint = self._said(self.ANSWERS)
        result = align_speech("word-color-stroop", store, hint, branch_params(_config(tmp_path)))
        assert _of_kind(result, "deviation", "filler") == []

    def test_a_read_task_does_report_a_filled_pause(self, tmp_path: Path) -> None:
        """The discriminator: the same bracketed token on a Harvard reading is a filler."""
        store = _store(family="harvard-sentences-list")
        placed = [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(HARVARD.split())]
        _transcript(store, [*placed, ("[uh]", 11.0, 11.2), ("[breath]", 12.0, 12.4)])
        _stimulus_measurement(store)
        result = align_speech("harvard-sentences-list", store, _hint(HARVARD), branch_params(_config(tmp_path)))
        [filler] = _of_kind(result, "deviation", "filler")
        assert filler.evidence["text"] == "[uh]", "`[breath]` is breath-group structure, not a disfluency"


class TestTheItemListAndTheCategoryItsRuleDependsOn:
    """`animal-fluency` forbids repeats; `random-item-generation` takes the rule from its category."""

    def _listed(self, family: str, items: list[str]) -> ProvStore:
        """A recording of a spoken item list.

        Args:
            family: The declared family.
            items: The items said, in order.

        Returns:
            The store.
        """
        store = _store(family=family)
        _transcript(store, [(item, 1.0 + index, 1.5 + index) for index, item in enumerate(items)])
        return store

    def test_a_repeat_where_none_is_allowed_is_a_deviation(self, tmp_path: Path) -> None:
        """One span over the list, and the repetition is a deviation over the repeated word."""
        store = self._listed("animal-fluency", ["cat", "dog", "cat"])
        result = align_speech("animal-fluency", store, None, branch_params(_config(tmp_path)))
        [repeated] = _of_kind(result, "deviation", "repeated_item")
        assert repeated.evidence["text"] == "cat"
        assert repeated.evidence["first_at"] == 1.0
        assert _roles(result) == ["task_extent"]

    def test_category_membership_is_recorded_as_unviable_rather_than_guessed(self, tmp_path: Path) -> None:
        """Scoring an item as belonging to the category needs a lexicon nothing in the graph has."""
        store = self._listed("animal-fluency", ["cat", "dog"])
        result = align_speech("animal-fluency", store, None, branch_params(_config(tmp_path)))
        assert _of_kind(result, "measure", "category_membership")

    def test_an_unreadable_category_concludes_nothing(self, tmp_path: Path) -> None:
        """Two of ten categories allow repetition, so a family-scoped rule would invert those."""
        store = self._listed("random-item-generation", ["a", "a"])
        result = align_speech("random-item-generation", store, None, branch_params(_config(tmp_path)))
        assert result.done == UNDETERMINED
        assert _of_kind(result, "measure", "repetition_rule")
        assert result.components == []

    def test_the_category_that_allows_repetition_reports_none(self, tmp_path: Path) -> None:
        """`Letters` says repetition is allowed, so a repeat is not a departure."""
        store = self._listed("random-item-generation", ["a", "a"])
        hint = AudioHints(metadata={"category": "Letters"})
        result = align_speech("random-item-generation", store, hint, branch_params(_config(tmp_path)))
        assert _of_kind(result, "deviation", "repeated_item") == []
        [allowed] = _of_kind(result, "count", "repetition_allowed")
        assert allowed.evidence["found"] is True

    def test_a_category_that_forbids_it_reports_the_repeat(self, tmp_path: Path) -> None:
        """The discriminator: eight of the ten categories say do not repeat any item."""
        store = self._listed("random-item-generation", ["a", "a"])
        hint = AudioHints(metadata={"category": "Animals"})
        result = align_speech("random-item-generation", store, hint, branch_params(_config(tmp_path)))
        assert len(_of_kind(result, "deviation", "repeated_item")) == 1


class TestASyllableFamilyIsEvaluatedAsTheTrainItsInstructionAsksFor:
    """The ten ``diadochokinesis-*`` families are SPEECH's own rows, served by the syllable body."""

    def test_a_train_family_reaches_the_syllable_body_rather_than_a_no_lexical_body(self, tmp_path: Path) -> None:
        """With no envelope the syllable body says which instrument was absent; a no-lexical body would not."""
        store = _store(family="diadochokinesis-pa")
        _transcript(store, [])
        result = align_speech("diadochokinesis-pa", store, None, branch_params(_config(tmp_path)))
        absent = [
            (finding.name, finding.evidence.get("unavailable"))
            for finding in result.deviations
            if finding.kind == "measure" and finding.evidence.get("unavailable")
        ]
        assert ("ddk_syllable_rate_from_envelope_modulation_hz", "energy_envelope") in absent
        assert result.done == UNDETERMINED

    def test_the_row_carries_the_instructions_own_expectation(self) -> None:
        """A syllable family's row says what it asks for; ``no lexical content`` is not a task."""
        assert SPEECH_EXPECTATIONS["diadochokinesis-pa"].sequence == ("labial",)
        assert SPEECH_EXPECTATIONS["diadochokinesis-pa"].expected_event_count == 10
        assert SPEECH_EXPECTATIONS["diadochokinesis-pataka"].sequence == ("labial", "alveolar", "velar")

    def test_a_lexical_word_on_a_syllable_train_is_not_a_departure(self, tmp_path: Path) -> None:
        """The removed claim: a word here was an ``off_task_extent``, which is what the owner rejected."""
        store = _store(family="diadochokinesis-pa")
        _transcript(store, [("buttercup", 2.0, 2.6)])
        result = align_speech("diadochokinesis-pa", store, None, branch_params(_config(tmp_path)))
        assert _of_kind(result, "deviation", "off_task_extent") == []

    def test_buttercup_takes_the_lexical_token_path(self, tmp_path: Path) -> None:
        """``buttercup`` is a word, so the consensus words serve the family directly."""
        store = _store(family="diadochokinesis-buttercup")
        _transcript(store, [("buttercup", 1.0 + 0.5 * index, 1.4 + 0.5 * index) for index in range(10)])
        result = align_speech("diadochokinesis-buttercup", store, None, branch_params(_config(tmp_path)))
        [counted] = [finding for finding in result.deviations if finding.name == "expected_event_count"]
        assert counted.evidence["found"] == 10
        assert counted.evidence["declared"] == 10
        [train] = [span for span in result.components if span.role == "task_extent"]
        assert train.attributes["production"] == "lexical_repetition"
        assert train.attributes["token"] == "buttercup"
        assert result.done is True

    def test_a_buttercup_recording_with_no_such_word_did_not_perform_the_task(self, tmp_path: Path) -> None:
        """The discriminator for the lexical path: the count is of that token, not of any word."""
        store = _store(family="diadochokinesis-buttercup")
        _transcript(store, [("the", 1.0, 1.2), ("birch", 1.3, 1.6)])
        result = align_speech("diadochokinesis-buttercup", store, None, branch_params(_config(tmp_path)))
        assert result.components == []
        assert result.done is False

    def test_the_syllable_spans_are_minted_into_speechs_own_family(self, tmp_path: Path) -> None:
        """One minting family per branch is what ``dispatch`` enforces; a second would weaken it."""
        store = _store(family="diadochokinesis-buttercup")
        _transcript(store, [("buttercup", 1.0, 1.4)])
        result = align_speech("diadochokinesis-buttercup", store, None, branch_params(_config(tmp_path)))
        assert {span.family for span in result.components} == {"speech"}


class TestDetectSpeechNeedsNoAlignmentAndGroupsByGap:
    """Out of family, every lexical word is the finding, and the grouping is not `merge`."""

    PLACED = [("one", 1.0, 1.2), ("two", 1.3, 1.5), ("three", 1.6, 1.8)]

    def _store_with_words(self, family: str = "prolonged-vowel") -> ProvStore:
        """A held-vowel recording carrying a spoken count-in.

        Args:
            family: The declared family, which is VOICE's rather than SPEECH's.

        Returns:
            The store.
        """
        store = _store(family=family)
        _transcript(store, list(self.PLACED))
        return store

    def test_a_non_touching_run_is_one_span_and_not_one_per_word(self, tmp_path: Path) -> None:
        """Ordinary speech has a gap between every pair of words, so `merge` would give three."""
        result = detect_speech(self._store_with_words(), branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        assert _roles(result) == ["lexical_run_0"]
        [run] = result.components
        assert (run.start, run.end) == (1.0, 1.8)
        assert run.attributes["words_n"] == 3
        assert run.attributes["text"] == "one two three"
        assert len(merge([(start, end) for _, start, end in self.PLACED])) == 3, "what `merge` would have given"

    def test_a_gap_wider_than_the_cut_separates_the_runs(self, tmp_path: Path) -> None:
        """The discriminator: the grouping reads the cut rather than adjacency."""
        result = detect_speech(self._store_with_words(), branch_params(_config(tmp_path, {"run_gap_max_s": 0.05})))
        assert _roles(result) == ["lexical_run_0", "lexical_run_1", "lexical_run_2"]

    def test_it_evaluates_no_task(self, tmp_path: Path) -> None:
        """No pattern was expected of it, so its only answer is UNDETERMINED and each span says so."""
        result = detect_speech(self._store_with_words(), branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        assert result.done == UNDETERMINED
        assert all(proposal.attributes["evaluates_no_task"] for proposal in result.components)

    def test_it_reads_no_stimulus_alignment_at_all(self, tmp_path: Path) -> None:
        """Nothing lexical is expected on another branch's task, so there is nothing to align."""
        store = self._store_with_words()
        result = detect_speech(store, branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        assert _of_kind(result, "measure", "expected_token_sequence") == []
        assert [proposal.derived_from for proposal in result.components][0][0].startswith("measurement-")

    def test_an_unmeasured_gap_proposes_nothing_rather_than_one_span_per_word(self, tmp_path: Path) -> None:
        """A nulled `branch.run_gap_max_s`; falling back to adjacency is the defect, not the fix."""
        params = branch_params(_config(tmp_path, {"run_gap_max_s": None}))
        result = detect_speech(self._store_with_words(), params)
        assert result.components == []
        assert _unmeasured(result) == ["run_gap_max_s"]

    def test_it_counts_what_it_found_without_declaring_it_wrong(self, tmp_path: Path) -> None:
        """A count asserts no discrepancy: there is no declared number of words on a held vowel."""
        result = detect_speech(self._store_with_words(), branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        [counted] = _of_kind(result, "count", "lexical_words")
        assert counted.evidence == {"found": 3, "declared": None}

    def test_a_speech_span_no_word_supports_is_contested(self, tmp_path: Path) -> None:
        """Somebody else's claim of speech where the consensus found none is contested, not edited."""
        store = self._store_with_words()
        span_id = store.entity(prov_type="span", extent=(8.0, 9.0), attributes={"family": "speech", "role": "x"})
        result = detect_speech(store, branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        [contested] = _of_kind(result, "contest", "speech")
        assert contested.evidence == {"reason": "no_consensus_word_inside"}
        assert contested.derived_from == (span_id,), "the contested span is the edge, not an attribute"


class TestOnlySpeechsOwnFamilyIsProposedInto:
    """A branch mints into its own family and can reach no other."""

    def test_every_proposal_carries_the_speech_family(self, tmp_path: Path) -> None:
        """Both modes go through SPEECH's own proposer, which fixes the family."""
        store = _store(family="harvard-sentences-list")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(HARVARD.split())])
        _stimulus_measurement(store)
        aligned = align_speech("harvard-sentences-list", store, _hint(HARVARD), branch_params(_config(tmp_path)))
        detected = detect_speech(store, branch_params(_config(tmp_path, {"run_gap_max_s": 0.3})))
        proposals: list[Proposal] = [*aligned.components, *detected.components]
        assert proposals, "both modes proposed something on this store"
        assert {proposal.family for proposal in proposals} == {"speech"}

    def test_every_proposal_names_its_evidence(self, tmp_path: Path) -> None:
        """Under propose-only the derivation is the whole record of where an extent came from."""
        store = _store(family="harvard-sentences-list")
        _transcript(store, [(word, 1.0 + index, 1.5 + index) for index, word in enumerate(HARVARD.split())])
        _stimulus_measurement(store)
        result = align_speech("harvard-sentences-list", store, _hint(HARVARD), branch_params(_config(tmp_path)))
        assert all(proposal.derived_from for proposal in result.components)
