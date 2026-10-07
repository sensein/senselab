"""The proper nouns a faithful performance of a task is expected to contain, as a declaration.

Owner, 2026-09-23: "analyze the task to determine expected names." The census behind which family
gets a list and which cannot have one is in
``specs/20260923-pii-near-match-and-expected-names/near-match-and-expected-names.md``.
"""

from __future__ import annotations

from dataclasses import fields

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes.branches import (
    SPEECH_EXPECTATIONS,
    Expectation,
    expected_names,
)
from senselab.audio.workflows.triage.nodes.speech import declared_content, in_stimulus
from senselab.audio.workflows.triage.stimulus import near_match

_NEAR = near_match(load_triage_config())


class TestTheDeclarationLivesOnTheExpectationRow:
    """A declaration about what the instruction asked for sits beside `sequence` and `tokens`."""

    def test_the_expectation_carries_the_field(self) -> None:
        """Not a fitted threshold and not a code literal: the same kind as the row's other fields."""
        assert "expected_names" in {field_.name for field_ in fields(Expectation)}

    def test_the_field_round_trips_through_the_recorded_mapping(self) -> None:
        """A run records which expectation it applied; a field it drops is one it cannot audit."""
        row = SPEECH_EXPECTATIONS["cinderella-story"]
        assert Expectation.from_mapping(row.as_mapping()) == row

    def test_cinderella_declares_its_cast(self) -> None:
        """A closed, knowable cast; 80.54% of the family's findings sit within one edit of it."""
        cast = expected_names("cinderella-story")
        assert "cinderella" in cast
        assert "godmother" in cast
        assert "stepsisters" in cast
        assert cast == tuple(sorted(cast)), "sorted, so a diff over the row reads as a diff"
        assert all(name == name.casefold() for name in cast), "normalised, like every other key here"

    def test_picture_description_declares_none_because_its_stimulus_is_an_image(self) -> None:
        """No text list can cover an image; a list here would turn 'never checked' into a false 'no'."""
        for family in ("picture-description", "picture-description-option1", "picture-description-option2"):
            assert expected_names(family) == ()

    def test_a_family_that_declares_stimulus_text_needs_no_cast(self) -> None:
        """The haystack is the declaration; a second one could disagree with it."""
        for family in ("rainbow-passage", "caterpillar-passage", "harvard-sentences-list", "story-recall"):
            assert expected_names(family) == ()

    def test_a_free_response_family_declares_none(self) -> None:
        """The instruction asks the participant to choose the content; a cast would be invented."""
        for family in ("free-speech", "free-speech-v2", "animal-fluency", "random-item-generation"):
            assert expected_names(family) == ()

    def test_an_undeclared_family_declares_none(self) -> None:
        """An unknown family is an absence, not an empty list that was checked."""
        assert expected_names(None) == ()
        assert expected_names("not-a-family") == ()


class TestTheDeclarationReachesTheStimulusCheck:
    """A name the task itself puts in the speaker's mouth is not a disclosure."""

    def test_a_cast_name_reads_as_accounted_for(self) -> None:
        """5,833 findings on this family read `null` because the question was never asked."""
        content = declared_content(AudioHints(), "cinderella-story")
        assert in_stimulus("Cinderella", content, _NEAR) is True

    def test_a_respelled_cast_name_reads_as_accounted_for(self) -> None:
        """The cast is matched under the same fitted bound as a declared stimulus."""
        content = declared_content(AudioHints(), "cinderella-story")
        assert in_stimulus("cindarela", content, _NEAR) is True

    def test_a_name_outside_the_cast_still_reads_as_a_disclosure(self) -> None:
        """A retelling that names the speaker's own sister is exactly what must survive."""
        content = declared_content(AudioHints(), "cinderella-story")
        assert in_stimulus("Springfield", content, _NEAR) is False

    def test_a_syllable_task_s_own_carrier_reads_as_accounted_for(self) -> None:
        """`buttercup` withheld recordings; the carrier is as declared as a prompt is."""
        content = declared_content(AudioHints(), "diadochokinesis-buttercup")
        assert in_stimulus("buttercup", content, _NEAR) is True

    def test_a_picture_description_still_reads_as_never_checked(self) -> None:
        """The tri-state's whole point: an image declares nothing to check against."""
        content = declared_content(AudioHints(), "picture-description")
        assert content is None
        assert in_stimulus("anything", content, _NEAR) is None

    def test_a_declared_prompt_and_a_declared_cast_are_both_searched(self) -> None:
        """A family could carry both; neither declaration shadows the other."""
        from senselab.audio.data_structures.audio_hints import ExpectedSpeech

        hint = AudioHints(expected_speech=[ExpectedSpeech(text="The rainbow is a division of white light.")])
        content = declared_content(hint, "cinderella-story")
        assert in_stimulus("rainbow", content, _NEAR) is True
        assert in_stimulus("Cinderella", content, _NEAR) is True
