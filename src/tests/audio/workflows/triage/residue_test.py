"""The lexical residue: what reaches the PII pathway, and what never does.

Every spelling here is one a recognizer wrote over the r4 corpus. The design and the replay are in
``specs/20260925-lexical-only-pii-pathway/design.md``.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.residue import (
    ALIGNED,
    FREE,
    SYLLABLE,
    VOCAL,
    ResidueRule,
    is_non_lexical,
    residue_method,
    residue_rule,
    task_residue,
)

RAINBOW = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. "
    "The rainbow is a division of white light into many beautiful colors."
)


@pytest.fixture(scope="module")
def rule() -> ResidueRule:
    """The packaged rule."""
    return residue_rule(load_triage_config())


def _residue(text: str, family: str | None, rule: ResidueRule, stimulus: str = "") -> list[str]:
    """The residue words of one whitespace-tokenised transcript."""
    words = text.split()
    return [words[position] for position in task_residue(words, family, stimulus.split(), rule).positions]


def _scanned(text: str, family: str | None, rule: ResidueRule, stimulus: str = "") -> bool:
    """Whether SPEECH would hand this transcript to the detectors."""
    residue = task_residue(text.split(), family, stimulus.split(), rule)
    return bool(residue.positions) and residue.content


class TestTheMethodFollowsTheFamily:
    """Which reading a family gets is a property of its declaration."""

    @pytest.mark.parametrize(
        ("family", "method"),
        [
            ("diadochokinesis-pataka", SYLLABLE),
            ("diadochokinesis-v2-tuh", SYLLABLE),
            ("prolonged-vowel", VOCAL),
            ("maximum-phonation-time", VOCAL),
            ("respiration-and-cough-cough", VOCAL),
            ("loudness", VOCAL),
            ("harvard-sentences-list", ALIGNED),
            ("word-color-stroop", ALIGNED),
            ("free-speech", FREE),
            ("animal-fluency", FREE),
            (None, FREE),
        ],
    )
    def test_each_family_is_read_by_its_own_method(self, family: str | None, method: str) -> None:
        """Ten syllable families, the vocal tasks, the read tasks and the free ones."""
        assert residue_method(family) == method


class TestASyllableTrainNeverReachesThePathway:
    """A DDK take in any spelling the recognizers wrote is the task, not words."""

    @pytest.mark.parametrize(
        ("family", "text"),
        [
            ("diadochokinesis-ta", "Ta-ta ta ta ta ta-ta-ta"),
            ("diadochokinesis-v2-tuh", "Ta ta-ta ta-ta-ta tuh tuh"),
            ("diadochokinesis-v2-puhtuhkuh", "Parika-parikha-parika parica parika"),
            ("diadochokinesis-pataka", "Parca-paraca, paracaparaca pataka"),
            ("diadochokinesis-buttercup", "Buttercup, butter cup, buttercup"),
            ("diadochokinesis-ka", "咔 咔 咔 咔 咔"),
            ("diadochokinesis-pa", "Pap-pot-pot-pot-pot-pot-pot-pot"),
            ("diadochokinesis-v2-puh", "Pup-p-p-p-p-p-p-p puh puh"),
            ("diadochokinesis-ka", "k-ka caw- caw caw- kah"),
        ],
    )
    def test_the_train_is_non_lexical(self, family: str, text: str, rule: ResidueRule) -> None:
        """Hyphen chains, ``ta`` for ``tuh``, ``paraca`` for ``pataka``, CJK syllables, run-ons."""
        assert _residue(text, family, rule) == []
        assert not _scanned(text, family, rule)

    def test_a_disclosure_over_a_take_is_the_residue(self, rule: ResidueRule) -> None:
        """The invariant: a name said inside a train still reaches the detectors, and only it."""
        residue = _residue("pa pa pa my name is Alice Smith pa pa", "diadochokinesis-pa", rule)
        assert residue == ["my", "name", "is", "Alice", "Smith"]

    def test_an_aside_after_a_take_is_the_residue(self, rule: ResidueRule) -> None:
        """What the examiner said after a pataka take is lexical and outside the task."""
        residue = _residue("Parca-paraca, paracaparaca That's very good.", "diadochokinesis-pataka", rule)
        assert "very" in residue
        assert "Parca-paraca," not in residue


class TestAVocalTaskIsItsSoundAndItsDeclaredWords:
    """Loudness asks for "hey", the prolonged vowel for "1, 2, 3 aah"; neither is a disclosure."""

    @pytest.mark.parametrize(
        ("family", "text"),
        [
            ("loudness", "Hey. Hey. Hey."),
            ("loudness-v2", "Hey. HEY!"),
            ("prolonged-vowel", "One, two, three, aaaaah"),
            ("prolonged-vowel", "1 2 3 ahh"),
            ("maximum-phonation-time", "Eeeee e hee"),
            ("respiration-and-cough-breath", "嗯 嗯 呼 呼 Ha ha"),
        ],
    )
    def test_the_task_is_non_lexical(self, family: str, text: str, rule: ResidueRule) -> None:
        """The declared tokens and the vocalisation are both the task."""
        assert not _scanned(text, family, rule)

    def test_a_disclosure_during_a_vocal_task_is_the_residue(self, rule: ResidueRule) -> None:
        """Speech the task did not ask for is scanned; a vowel-and-glide word like "my" may fall away."""
        residue = _residue("Hey. Hey. My name is Alice Smith. Hey.", "loudness", rule)
        assert residue[-4:] == ["name", "is", "Alice", "Smith."]


class TestAReadTaskIsAlignedToItsStimulus:
    """A read passage is the task wherever the transcript is a reading of it."""

    @pytest.mark.parametrize(
        ("stimulus", "text"),
        [
            ("The odor of spring makes young hearts jump.", "The odor of spring makes young heart stump."),
            ("Plead to the council to free the poor thief.", "Plead to the consul to free the poor thief."),
            ("It is hard to erase blue or red ink.", "It's hard to erase blue or red ink."),
            ("The wall phone rang loud and often.", "The wall phone the wall phone phone rang loud and often."),
            ("The tree top waved in a graceful way.", "The treetop top waved in a graceful way."),
            ("Two blue fish swam in the tank.", "Two blue fish swim in the tank."),
            ("New pants lack cuffs and pockets.", "New pants lack ca- cuffs and pockets."),
            ("Act on these orders with great speed.", "Oh, act on those orders with great speed."),
        ],
    )
    def test_a_reading_of_the_stimulus_is_not_scanned(self, stimulus: str, text: str, rule: ResidueRule) -> None:
        """Respellings, homophones, contractions, restarts, compounds, fragments and fillers."""
        assert not _scanned(text, "harvard-sentences-list", rule, stimulus)

    def test_a_disclosure_inside_a_passage_is_the_residue(self, rule: ResidueRule) -> None:
        """The invariant: an aside said mid-passage is outside the target and is scanned."""
        text = RAINBOW.replace("The rainbow is", "The rainbow, my name is Alice Smith, is")
        residue = _residue(text, "rainbow-passage", rule, RAINBOW)
        assert "Alice" in residue and "Smith," in residue
        assert _scanned(text, "rainbow-passage", rule, RAINBOW)

    def test_a_different_sentence_is_the_residue(self, rule: ResidueRule) -> None:
        """A recording whose words are not the declared sentence at all."""
        residue = _residue("The game runs in Brooklyn.", "harvard-sentences-list", rule, "He lay prone.")
        assert "Brooklyn." in residue

    def test_a_named_ink_colour_is_the_stroop_task(self, rule: ResidueRule) -> None:
        """Stroop names the ink, so a colour the word list does not hold is still the task."""
        assert not _scanned("Purple or blue. Yellow. Beige.", "word-color-stroop", rule, "blue purple red")

    def test_an_undeclared_family_with_a_prompt_is_aligned_to_it(self, rule: ResidueRule) -> None:
        """A prompt is a declaration even where no family is."""
        assert not _scanned("buttercup buttercup buttercup", None, rule, "buttercup")


class TestWhatIsNeverLexical:
    """Markers, fillers and fragments are not words anyone could be identified by."""

    @pytest.mark.parametrize("token", ["um", "Uh,", "hmm", "Mm-hmm", "ahh", "[throatclearing].", "[UH]", "wa-", "啊"])
    def test_the_token_is_non_lexical(self, token: str) -> None:
        """Every shape the corpus's scanned-for-nothing residue carried."""
        assert is_non_lexical(token)

    @pytest.mark.parametrize("token", ["a", "I", "Alice", "one", "Hey"])
    def test_the_token_is_lexical(self, token: str) -> None:
        """A single article, a pronoun, a name, a number and a greeting are words."""
        assert not is_non_lexical(token)


class TestTheContentTest:
    """A residue of closed-class words alone is not scanned; a number or a name is content."""

    def test_function_words_alone_are_not_content(self, rule: ResidueRule) -> None:
        """An inserted article in a read sentence."""
        residue = task_residue("A the big wet stain".split(), "harvard-sentences-list", "A big wet stain".split(), rule)
        assert residue.positions
        assert not residue.content

    def test_a_number_is_content(self, rule: ResidueRule) -> None:
        """A number can identify: an age, a date, a phone digit."""
        assert _scanned("One two three", None, rule)


class TestAFreeResponseIsAllResidue:
    """Open response is where disclosure lives: every lexical word is read."""

    def test_every_word_but_the_fillers(self, rule: ResidueRule) -> None:
        """The filler is taken out, nothing else."""
        words: Sequence[str] = "um I live in Springfield uh".split()
        assert _residue(" ".join(words), "free-speech", rule) == ["I", "live", "in", "Springfield"]
