"""The stimulus alignment: one alignment, three projections, and what an absent expectation is."""

from senselab.audio.data_structures.audio_hints import ExpectedSpeech
from senselab.audio.workflows.triage.stimulus import (
    LexicalWord,
    align_stimulus,
    split_prompts,
)

TERMINATORS = ".?!"


def _words(*items: tuple[str, float, float]) -> list[LexicalWord]:
    """The lexical consensus stream, one word per item, full agreement."""
    return [
        LexicalWord(index=index, text=text, extent=(start, end), agreement=1.0)
        for index, (text, start, end) in enumerate(items)
    ]


def _spoken(text: str, *, start: float = 0.0, step: float = 0.3, width: float = 0.2) -> list[LexicalWord]:
    """One word per whitespace token, evenly spaced."""
    return _words(
        *[
            (token, round(start + i * step, 6), round(start + i * step + width, 6))
            for i, token in enumerate(text.split())
        ]
    )


HARVARD = [ExpectedSpeech(text="The pencils have all been used.")]


class TestTheFourOutcomes:
    """Realised, substituted, absent, and the unexpected complement."""

    def test_an_exact_reading_realises_every_token_with_its_extent(self) -> None:
        """Every expected token carries the extent of the word that realised it, and nothing is left over."""
        alignment = align_stimulus(HARVARD, _spoken("the pencils have all been used"), terminators=TERMINATORS)
        assert [token.realisation for token in alignment.expected] == ["realised"] * 6
        assert [token.text for token in alignment.expected] == ["The", "pencils", "have", "all", "been", "used."]
        assert [token.word_index for token in alignment.expected] == [0, 1, 2, 3, 4, 5]
        assert alignment.expected[0].extent == (0.0, 0.2)
        assert alignment.expected[5].extent == (1.5, 1.7)
        assert alignment.unexpected == ()
        assert alignment.substitutions == ()
        assert alignment.omissions == ()
        assert alignment.provenance["realised_fraction"] == 1.0

    def test_a_substitution_is_paired_and_names_both_surfaces(self) -> None:
        """A word aligned to an expected token under a different key is a substitution, not an omission."""
        alignment = align_stimulus(HARVARD, _spoken("the pencils have all bean used"), terminators=TERMINATORS)
        [substituted] = alignment.substitutions
        assert substituted.index == 4
        assert substituted.text == "been"
        assert substituted.read == "bean"
        assert substituted.extent == (1.2, 1.4)
        assert substituted.agreement == 1.0
        assert alignment.omissions == ()
        assert alignment.unexpected == ()
        assert alignment.provenance["n_substituted"] == 1
        assert alignment.provenance["n_realised"] == 5

    def test_an_omission_is_an_expected_token_nothing_was_paired_with(self) -> None:
        """The skipped token carries no word, no extent and no agreement; its neighbours keep theirs."""
        alignment = align_stimulus(HARVARD, _spoken("the pencils have all used"), terminators=TERMINATORS)
        [omitted] = alignment.omissions
        assert omitted.index == 4
        assert omitted.text == "been"
        assert (omitted.word_index, omitted.read, omitted.extent, omitted.agreement) == (None, None, None, None)
        assert alignment.expected[5].realisation == "realised"
        assert alignment.expected[5].read == "used"
        assert alignment.unexpected == ()
        assert alignment.provenance["n_absent"] == 1

    def test_an_insertion_is_the_lexical_complement_and_keeps_its_place(self) -> None:
        """A word matching no expected token is unexpected, and records which token it follows."""
        alignment = align_stimulus(HARVARD, _spoken("the pencils have all been mostly used"), terminators=TERMINATORS)
        [inserted] = alignment.unexpected
        assert inserted.text == "mostly"
        assert inserted.index == 5
        assert inserted.after == 4
        assert inserted.extent == (1.5, 1.7)
        assert [token.realisation for token in alignment.expected] == ["realised"] * 6
        assert alignment.provenance["n_unexpected"] == 1

    def test_a_word_before_the_first_expected_token_follows_minus_one(self) -> None:
        """``after`` is the token the word follows, and nothing precedes the first."""
        alignment = align_stimulus(HARVARD, _spoken("okay the pencils have all been used"), terminators=TERMINATORS)
        [inserted] = alignment.unexpected
        assert (inserted.text, inserted.index, inserted.after) == ("okay", 0, -1)


class TestAnEmptyExpectation:
    """A declared expectation of no particular words is an alignment, not an absence."""

    def test_a_declared_empty_prompt_puts_every_word_in_the_complement(self) -> None:
        """This is AIRWAY's projection: lexical material where none was expected."""
        alignment = align_stimulus(
            [ExpectedSpeech(text="")], _spoken("i think there is a cough"), terminators=TERMINATORS
        )
        assert alignment.expected == ()
        assert alignment.units == ()
        assert [word.text for word in alignment.unexpected] == ["i", "think", "there", "is", "a", "cough"]
        assert all(word.after == -1 for word in alignment.unexpected)
        assert alignment.provenance["n_expected"] == 0
        assert alignment.provenance["n_unexpected"] == 6
        assert alignment.provenance["realised_fraction"] is None

    def test_no_prompts_at_all_is_the_same_alignment_with_no_prompt_counted(self) -> None:
        """The distinction between the two lives in the node, which refuses to align on no prompts."""
        alignment = align_stimulus([], _spoken("one two"), terminators=TERMINATORS)
        assert alignment.provenance["n_prompts"] == 0
        assert alignment.provenance["n_unexpected"] == 2

    def test_a_declared_prompt_against_an_empty_transcript_omits_every_token(self) -> None:
        """No consensus word is not the same as no expectation; every expected token is absent."""
        alignment = align_stimulus(HARVARD, [], terminators=TERMINATORS)
        assert [token.realisation for token in alignment.expected] == ["absent"] * 6
        assert alignment.units[0].extent is None
        assert alignment.provenance["realised_fraction"] == 0.0


class TestTheDeclaredStructure:
    """A unit is a prompt, or a sentence inside one; its extent is the hull of what realised it."""

    def test_separate_prompts_are_separate_units_with_no_splitting(self) -> None:
        """A caller who declares the sentences separately gets them back unsplit."""
        prompts = [ExpectedSpeech(text="he helped her"), ExpectedSpeech(text="we were away")]
        alignment = align_stimulus(prompts, _spoken("he helped her we were away"), terminators=TERMINATORS)
        assert [unit.prompt for unit in alignment.units] == [0, 1]
        assert [unit.text for unit in alignment.units] == ["he helped her", "we were away"]
        assert [unit.token_indices for unit in alignment.units] == [(0, 1, 2), (3, 4, 5)]
        assert alignment.units[0].extent == (0.0, 0.8)
        assert alignment.units[1].extent == (0.9, 1.7)

    def test_one_multi_sentence_prompt_splits_into_the_same_units(self) -> None:
        """The rainbow-passage shape: four sentences in one declared string, four unit extents."""
        prompts = [ExpectedSpeech(text="He helped her. We were away! Were you there?")]
        alignment = align_stimulus(
            prompts, _spoken("he helped her we were away were you there"), terminators=TERMINATORS
        )
        assert [unit.text for unit in alignment.units] == ["He helped her.", "We were away!", "Were you there?"]
        assert [unit.prompt for unit in alignment.units] == [0, 0, 0]
        assert [unit.extent for unit in alignment.units] == [(0.0, 0.8), (0.9, 1.7), (1.8, 2.6)]
        assert len(alignment.structure_spans()) == 3

    def test_a_unit_nothing_realised_carries_no_extent_and_is_not_a_structure_span(self) -> None:
        """The second sentence was skipped: it is a unit, it has no boundaries, and it is not selectable."""
        prompts = [ExpectedSpeech(text="He helped her. We were away.")]
        alignment = align_stimulus(prompts, _spoken("he helped her"), terminators=TERMINATORS)
        assert len(alignment.units) == 2
        assert alignment.units[1].extent is None
        assert alignment.units[1].n_realised == 0
        assert [unit.index for unit in alignment.structure_spans()] == [0]

    def test_a_unit_counts_its_own_realisations_and_substitutions(self) -> None:
        """Per-unit counts, so a consumer can see which sentence the reading went wrong in."""
        prompts = [ExpectedSpeech(text="He helped her. We were away.")]
        alignment = align_stimulus(prompts, _spoken("he helped her we were about"), terminators=TERMINATORS)
        assert (alignment.units[0].n_realised, alignment.units[0].n_substituted) == (3, 0)
        assert (alignment.units[1].n_realised, alignment.units[1].n_substituted) == (2, 1)

    def test_an_unpunctuated_sequence_is_one_unit(self) -> None:
        """The stroop shape: fifteen colours, no terminator, one unit and fifteen tokens."""
        colours = "red purple red green brown blue red brown purple blue green purple green purple green"
        alignment = align_stimulus([ExpectedSpeech(text=colours)], _spoken(colours), terminators=TERMINATORS)
        assert len(alignment.units) == 1
        assert alignment.provenance["n_expected"] == 15
        assert alignment.provenance["n_realised"] == 15

    def test_a_prompt_of_punctuation_alone_contributes_no_unit(self) -> None:
        """A token normalising to the empty key is dropped, as the consensus drops one."""
        assert split_prompts([ExpectedSpeech(text="... ---")], terminators=TERMINATORS) == []
        alignment = align_stimulus([ExpectedSpeech(text="... ---")], _spoken("hello"), terminators=TERMINATORS)
        assert alignment.units == ()
        assert alignment.provenance["n_unexpected"] == 1


class TestRunFor:
    """The ordered sub-run projection the count-in and the loudness token are read through."""

    def test_a_run_reports_its_hull_and_its_departures(self) -> None:
        """``one two three`` realised inside a longer expectation: three tokens, one extent."""
        prompts = [ExpectedSpeech(text="one two three aah")]
        alignment = align_stimulus(prompts, _spoken("one to three aah"), terminators=TERMINATORS)
        run = alignment.run_for(["one", "two", "three"])
        assert run is not None
        assert run.token_indices == (0, 1, 2)
        assert run.extent == (0.0, 0.8)
        assert [token.read for token in run.substitutions] == ["to"]
        assert run.omissions == ()

    def test_a_run_the_expectation_does_not_declare_is_none(self) -> None:
        """The digits the corpus actually prescribes are not the spelled-out tokens."""
        alignment = align_stimulus([ExpectedSpeech(text="1, 2, 3 aah")], _spoken("one two three aah"), terminators=".")
        assert alignment.run_for(["one", "two", "three"]) is None
        assert alignment.run_for(["1", "2", "3"]) is not None
        assert [token.realisation for token in alignment.expected[:3]] == ["substituted"] * 3

    def test_tokens_present_but_not_contiguous_are_not_a_run(self) -> None:
        """A run is an ordered contiguous match, not a membership test over the expected stream."""
        prompts = [ExpectedSpeech(text="one and two and three aah")]
        alignment = align_stimulus(prompts, _spoken("one and two and three aah"), terminators=TERMINATORS)
        assert {"one", "two", "three"} <= {token.key for token in alignment.expected}
        assert alignment.run_for(["one", "two", "three"]) is None
        assert alignment.run_for(["and", "three", "aah"]) is not None

    def test_a_run_later_in_the_stream_reports_its_own_indices(self) -> None:
        """The run's indices are positions in the expected stream, not offsets within the run."""
        prompts = [ExpectedSpeech(text="please say one two three now")]
        alignment = align_stimulus(prompts, _spoken("please say one two three now"), terminators=TERMINATORS)
        run = alignment.run_for(["one", "two", "three"])
        assert run is not None
        assert run.token_indices == (2, 3, 4)
        assert run.extent == (0.6, 1.4)

    def test_an_empty_key_list_is_no_run(self) -> None:
        """Asking for nothing finds nothing, rather than matching everywhere."""
        alignment = align_stimulus(HARVARD, _spoken("the pencils have all been used"), terminators=TERMINATORS)
        assert alignment.run_for([]) is None


class TestProvenance:
    """What the measurement records about how the alignment was produced."""

    def test_the_counts_partition_the_expected_stream(self) -> None:
        """Realised, substituted and absent are the only three states, and they sum."""
        alignment = align_stimulus(HARVARD, _spoken("the pencil have all bean it used"), terminators=TERMINATORS)
        provenance = alignment.provenance
        total = provenance["n_realised"] + provenance["n_substituted"] + provenance["n_absent"]
        assert total == provenance["n_expected"] == 6
        assert provenance["n_lexical_words"] == 7
        assert provenance["sentence_terminators"] == TERMINATORS
        assert provenance["algorithm"] == "weighted_levenshtein_alignment"
        assert provenance["normalisation"] == "casefold; keep alphanumerics and apostrophe"
