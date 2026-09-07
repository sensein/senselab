"""The consensus text stream: sequence alignment decides, time is metadata, nothing is re-sorted."""

import inspect
import random
from dataclasses import fields
from typing import Sequence

import pytest

from senselab.audio.workflows.triage.consensus import (
    Consensus,
    ConsensusWord,
    SourceHypothesis,
    align_sources,
    bracketed_form,
    is_bracketed,
    isotonic_median_fit,
    render_transcript,
    vocabulary_key,
)


def _source(
    name: str, text: str, *, first_s: float = 0.0, step_s: float = 0.5, length_s: float = 0.4
) -> SourceHypothesis:
    """A hypothesis from whitespace-separated tokens at a fixed spacing."""
    tokens = text.split()
    words = tuple((token, first_s + i * step_s, first_s + i * step_s + length_s) for i, token in enumerate(tokens))
    return SourceHypothesis(name=name, words=words, timestamp_source="native", timestamp_model=None)


def _timed(name: str, words: Sequence[tuple[str, float, float]]) -> SourceHypothesis:
    """A hypothesis from explicit ``(text, start, end)`` triples."""
    return SourceHypothesis(name=name, words=tuple(words), timestamp_source="native", timestamp_model=None)


def _align(*sources: SourceHypothesis, onomatopoeic: set[str] | None = None) -> Consensus:
    return align_sources(list(sources), onomatopoeic=onomatopoeic or set())


def _outcomes(consensus: Consensus) -> list[str]:
    return [word.outcome for word in consensus.words]


def _texts(consensus: Consensus) -> list[str]:
    return [word.text for word in consensus.words]


class TestAgreementVariantInsertion:
    """The three outcomes are set relations over normalised keys; nothing is thresholded."""

    def test_identical_streams_agree_everywhere_and_keep_each_source_time_verbatim(self) -> None:
        """Test 1: agreement with per-source timings preserved unrounded."""
        a = _timed("a", [("hello", 0.1234, 0.4321), ("world", 0.5, 0.9)])
        b = _timed("b", [("hello", 0.1234, 0.4321), ("world", 0.5, 0.9)])
        consensus = _align(a, b)
        assert _outcomes(consensus) == ["agreement", "agreement"]
        assert all(word.agreement == 1.0 for word in consensus.words)
        assert consensus.words[0].readings == {"a": "hello", "b": "hello"}
        assert consensus.words[0].timings == {"a": (0.1234, 0.4321), "b": (0.1234, 0.4321)}
        assert consensus.words[0].extent == (0.1234, 0.4321)
        for word in consensus.words:
            assert word.onset_spread_s == 0.0 and word.offset_spread_s == 0.0
            assert word.temporal_uncertainty_s == 0.0

    def test_a_variant_records_every_reading_with_its_share(self) -> None:
        """Test 2: hi Jon / hi John."""
        consensus = _align(_source("a", "hi Jon"), _source("b", "hi John"))
        jon = consensus.words[1]
        assert jon.outcome == "variant"
        assert jon.text == "Jon" == jon.variants[0].text
        assert [(v.text, list(v.sources), v.share) for v in jon.variants] == [("Jon", ["a"], 0.5), ("John", ["b"], 0.5)]
        assert jon.agreement == 0.5
        assert jon.readings == {"a": "Jon", "b": "John"}

    def test_insertions_on_either_side_of_an_agreement(self) -> None:
        """Test 3: I uh think / I think so."""
        consensus = _align(_source("a", "I uh think"), _source("b", "I think so"))
        assert _texts(consensus) == ["I", "uh", "think", "so"]
        assert _outcomes(consensus) == ["agreement", "insertion", "agreement", "insertion"]
        assert consensus.words[1].sources == ("a",) and consensus.words[3].sources == ("b",)

    def test_an_agreed_repetition_is_two_agreement_words(self) -> None:
        """Test 5."""
        consensus = _align(_source("a", "that that"), _source("b", "that that"))
        assert _outcomes(consensus) == ["agreement", "agreement"]

    def test_a_variant_beside_a_repetition(self) -> None:
        """Test 6: the the / a."""
        consensus = _align(_source("a", "the the"), _source("b", "a"))
        assert _outcomes(consensus) == ["variant", "insertion"]
        assert [(v.text, list(v.sources)) for v in consensus.words[0].variants] == [("the", ["a"]), ("a", ["b"])]
        assert consensus.words[1].sources == ("a",)

    def test_empty_key_tokens_are_dropped_and_counted_per_source(self) -> None:
        """Test 11."""
        consensus = _align(_source("a", "... hello"), _source("b", "hello"))
        assert _texts(consensus) == ["hello"]
        assert consensus.provenance["empty_tokens_dropped"] == {"a": 1, "b": 0}

    def test_three_sources(self) -> None:
        """Test 12: majority variant, two-source insertion, three-way tie."""
        majority = _align(_source("a", "cat"), _source("b", "cat"), _source("c", "cot"))
        assert majority.words[0].outcome == "variant"
        assert majority.words[0].agreement == pytest.approx(2 / 3)
        assert list(majority.words[0].variants[0].sources) == ["a", "b"]

        insertion = _align(_source("a", "x y"), _source("b", "x y"), _source("c", "y"))
        assert _outcomes(insertion) == ["insertion", "agreement"]
        assert insertion.words[0].agreement == pytest.approx(2 / 3)
        assert insertion.words[0].sources == ("a", "b")

        tie = _align(_source("a", "cat"), _source("b", "cot"), _source("c", "cut"))
        assert len(tie.words[0].variants) == 3
        assert tie.words[0].agreement == pytest.approx(1 / 3)


class TestTheRepetitionFinding:
    """Test 4: the columns are emitted verbatim; nothing collapses or re-sorts a repeated token."""

    def test_and_the_and_the_d_the_is_six_words_in_column_order(self) -> None:
        """4a: six words, in column order, the duplicates real."""
        consensus = _align(_source("a", "and the and the d- the"), _source("b", "and the"))
        assert _texts(consensus) == ["and", "the", "and", "the", "d-", "the"]
        assert _outcomes(consensus) == ["insertion", "insertion", "agreement", "insertion", "insertion", "agreement"]
        assert render_transcript(consensus.words) == "and the **and** the d- **the**"
        assert render_transcript(consensus.words, strong=("", "")) == "and the and the d- the"

    def test_conflicting_times_keep_stream_order_and_derive_monotone_times(self) -> None:
        """4b: b opens boy before a's stumble; the stream does not move, the fit does not either."""
        a = _timed("a", [("the", 1.00, 1.20), ("d-", 1.20, 1.35), ("boy", 1.40, 1.70)])
        b = _timed("b", [("the", 1.00, 1.18), ("boy", 1.10, 1.70)])
        consensus = _align(a, b)
        assert _texts(consensus) == ["the", "d-", "boy"]
        assert [word.extent[0] for word in consensus.words] == pytest.approx([1.0, 1.2, 1.25])
        boy = consensus.words[2]
        assert boy.onset_spread_s == pytest.approx(0.30) == boy.temporal_uncertainty_s
        assert boy.timings["b"] == (1.10, 1.70)
        assert consensus.provenance["n_words_time_shifted"] == 0

    def test_the_gets_case_pools_onto_a_member_reading(self) -> None:
        """4c: the median of {9.60, 9.88, 9.99} is 9.88, so a- keeps its own span and gets moves onto it."""
        a = _timed("a", [("gets", 9.66, 9.74), ("a-", 9.88, 9.99), ("gets", 9.99, 10.10)])
        b = _timed("b", [("gets", 9.60, 9.84)])
        consensus = _align(a, b)
        assert _texts(consensus) == ["gets", "a-", "gets"]
        assert _outcomes(consensus) == ["insertion", "insertion", "agreement"]
        assert consensus.words[1].extent == (9.88, 9.99)
        assert consensus.words[2].extent == pytest.approx((9.88, 9.99))
        assert consensus.words[2].temporal_uncertainty_s == pytest.approx(0.39)
        assert consensus.words[1].temporal_uncertainty_s == 0.0
        assert consensus.provenance["n_words_time_shifted"] == 2
        assert consensus.provenance["max_time_shift_s"] == pytest.approx(0.085)

    def test_the_four_second_conflict_stays_aligned_and_carries_its_uncertainty(self) -> None:
        """4d: R-5's case; the pairing is kept and the 4.1 s spread is the record."""
        a = _timed(
            "a",
            [
                ("The", 31.30, 31.52),
                ("[UM]", 31.52, 31.70),
                ("the", 35.30, 35.36),
                ("little", 35.36, 35.58),
                ("boy", 35.58, 36.06),
            ],
        )
        b = _timed("b", [("the", 31.20, 31.84), ("little", 35.28, 35.60), ("boy", 35.60, 36.16)])
        consensus = _align(a, b)
        assert _outcomes(consensus) == ["insertion", "insertion", "agreement", "agreement", "agreement"]
        the = consensus.words[2]
        assert the.extent[0] == pytest.approx(33.25)
        assert the.temporal_uncertainty_s == pytest.approx(4.10)
        assert consensus.provenance["n_words_time_shifted"] == 0
        assert consensus.words[0].temporal_uncertainty_s == 0.0
        assert consensus.words[1].temporal_uncertainty_s == 0.0

    def test_derived_times_are_monotone_and_bounded_by_the_uncertainty(self) -> None:
        """4e: the property, over a long random pair with jittered times."""
        rng = random.Random(7)
        vocabulary = ["the", "boy", "and", "dog", "ran", "a-", "[UM]", "home", "to", "his"]
        base = [rng.choice(vocabulary) for _ in range(200)]
        a_words, b_words = [], []
        cursor = 0.0
        for token in base:
            a_start = cursor + rng.uniform(-0.2, 0.2)
            a_words.append((token, a_start, a_start + 0.3 + rng.uniform(-0.1, 0.1)))
            if rng.random() > 0.15:
                b_start = cursor + rng.uniform(-0.6, 0.6)
                b_words.append((token, b_start, b_start + 0.3 + rng.uniform(-0.2, 0.2)))
            cursor += 0.4
        consensus = _align(_timed("a", a_words), _timed("b", b_words))
        onsets = [word.extent[0] for word in consensus.words]
        offsets = [word.extent[1] for word in consensus.words]
        assert all(later >= earlier for earlier, later in zip(onsets, onsets[1:]))
        assert all(later >= earlier for earlier, later in zip(offsets, offsets[1:]))
        assert all(word.extent[0] <= word.extent[1] for word in consensus.words)
        assert all(
            word.temporal_uncertainty_s >= max(word.onset_spread_s, word.offset_spread_s) for word in consensus.words
        )
        assert [word.index for word in consensus.words] == list(range(len(consensus.words)))


class TestTheMedianFit:
    """The fit is PAVA over medians of the stacked member readings; even counts take the midpoint."""

    def test_two_readings_take_their_midpoint_when_nothing_pools(self) -> None:
        """4f."""
        consensus = _align(_timed("a", [("hi", 1.00, 1.20)]), _timed("b", [("hi", 1.10, 1.30)]))
        assert consensus.words[0].extent == pytest.approx((1.05, 1.25))

    def test_an_odd_pooled_block_lands_on_a_member_reading(self) -> None:
        """4f: the median of three stacked readings is the middle one, not a mean."""
        assert isotonic_median_fit([[1.0, 3.0], [1.5]]) == ([1.5, 1.5], [True, True])

    def test_output_is_non_decreasing_for_any_input(self) -> None:
        """PAVA's invariant, held over random inputs."""
        rng = random.Random(3)
        for _ in range(500):
            readings = [[rng.uniform(0, 10) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 15))]
            fitted, pooled = isotonic_median_fit(readings)
            assert all(later >= earlier for earlier, later in zip(fitted, fitted[1:]))
            assert len(pooled) == len(readings)

    def test_unpooled_positions_reproduce_their_own_value_exactly(self) -> None:
        """A block of one position is its own median, bit for bit."""
        fitted, pooled = isotonic_median_fit([[1.0, 1.2], [2.0], [3.0, 3.5, 4.0]])
        assert fitted == [1.1, 2.0, 3.5]
        assert pooled == [False, False, False]

    def test_an_empty_position_is_refused(self) -> None:
        """The precondition every caller meets, stated as a raise rather than a silent guess."""
        with pytest.raises(ValueError, match="position 1 carries no reading"):
            isotonic_median_fit([[1.0], []])

    def test_one_outlier_costs_one_neighbour_and_moves_no_other_position(self) -> None:
        """4g: pooling stops at the offending position."""
        words = [(f"w{i}", 1.0 + 0.5 * i, 1.3 + 0.5 * i) for i in range(20)]
        clean = _align(_timed("a", words), _timed("b", words))
        late = list(words)
        late[7] = (late[7][0], late[7][1] + 15.0, late[7][2] + 15.0)
        skewed = _align(_timed("a", words), _timed("b", late))
        own = [1.0 + 0.5 * i for i in range(20)]
        for index, word in enumerate(skewed.words):
            if index == 7:
                assert word.extent[0] == pytest.approx(own[8])
            else:
                assert word.extent[0] == pytest.approx(own[index])
                assert word.extent == pytest.approx(clean.words[index].extent)
        assert skewed.provenance["n_words_time_shifted"] == 2


class TestSourceOrderAndBrackets:
    """R-1 and §3.2: order by name, override and insertion are two mechanisms."""

    def test_source_order_does_not_depend_on_argument_order(self) -> None:
        """Test 7."""
        a, b = _source("a", "and the and the d- the"), _source("b", "and the")
        forward, backward = _align(a, b), _align(b, a)
        assert [(w.text, w.outcome, w.sources, w.extent) for w in forward.words] == [
            (w.text, w.outcome, w.sources, w.extent) for w in backward.words
        ]
        assert [s["name"] for s in forward.provenance["sources"]] == ["a", "b"]
        assert [s["name"] for s in backward.provenance["sources"]] == ["a", "b"]

    def test_a_bracket_override_is_one_agreement_word_with_the_bracketed_surface(self) -> None:
        """Test 8."""
        for sources in (
            (_source("a", "[COUGH] hi"), _source("b", "cough hi")),
            (_source("b", "cough hi"), _source("a", "[COUGH] hi")),
        ):
            consensus = _align(*sources)
            cough = consensus.words[0]
            assert cough.outcome == "agreement"
            assert cough.text == "[COUGH]" and cough.bracketed
            assert cough.readings == {"a": "[COUGH]", "b": "cough"}
            assert consensus.provenance["bracket_overrides_n"] == 1

    def test_a_bracketed_insertion_is_not_an_override(self) -> None:
        """Test 9."""
        consensus = _align(_source("a", "I [UM] think"), _source("b", "I think"))
        um = consensus.words[1]
        assert um.outcome == "insertion" and um.bracketed and um.sources == ("a",)
        assert consensus.provenance["bracket_overrides_n"] == 0
        assert render_transcript(consensus.words) == "**I** [UM] **think**"

    def test_an_onomatopoeic_token_becomes_a_bracketed_word(self) -> None:
        """Test 10."""
        both = _align(_source("a", "hello khh world"), _source("b", "hello khh world"), onomatopoeic={"khh"})
        assert both.words[1].outcome == "agreement" and both.words[1].text == "[KHH]"
        assert both.words[1].readings == {"a": "khh", "b": "khh"}
        one = _align(_source("a", "hello khh world"), _source("b", "hello world"), onomatopoeic={"khh"})
        assert one.words[1].outcome == "insertion" and one.words[1].text == "[KHH]"
        plain = _align(_source("a", "hello khh world"), _source("b", "hello khh world"))
        assert plain.words[1].text == "khh" and not plain.words[1].bracketed

    def test_the_bracket_helpers(self) -> None:
        """The three small helpers the node and the consensus share."""
        assert is_bracketed("[COUGH]") and is_bracketed(" [um] ")
        assert not is_bracketed("cough") and not is_bracketed("[") and not is_bracketed("[a")
        assert bracketed_form("Khh,", {"khh"}) == "[KHH]"
        assert bracketed_form("[BREATH]", set()) == "[BREATH]"
        assert bracketed_form("hello", {"khh"}) is None
        assert vocabulary_key("Ahem!") == "ahem"


class TestGuardsAndWordlessSources:
    """R-2 and the wordless cases."""

    def test_fewer_than_two_sources_raise_and_the_message_names_the_count(self) -> None:
        """Test 13."""
        with pytest.raises(LookupError, match=r"found 1: \['a'\]"):
            _align(_source("a", "hello"))
        with pytest.raises(LookupError, match=r"found 0: \[\]"):
            align_sources([], onomatopoeic=set())

    def test_a_wordless_source_still_counts(self) -> None:
        """Test 14."""
        consensus = _align(_source("a", "hello"), _source("b", ""))
        assert _outcomes(consensus) == ["insertion"]
        assert consensus.words[0].agreement == 0.5
        assert consensus.provenance["n_sources"] == 2
        assert [(s["name"], s["n_words"]) for s in consensus.provenance["sources"]] == [("a", 1), ("b", 0)]
        assert consensus.provenance["reference_source"] == "a"
        empty = _align(_source("a", ""), _source("b", ""))
        assert empty.words == ()
        assert empty.provenance["n_words"] == 0
        assert empty.provenance["reference_source"] is None
        assert render_transcript(empty.words) == ""


class TestRenderingAndProvenance:
    """Tests 15-17."""

    def test_render_marks_agreement_and_joins_variants(self) -> None:
        """Test 15: bold marks agreement; the slash form is display only."""
        consensus = _align(_source("a", "hi Jon"), _source("b", "hi John"))
        assert render_transcript(consensus.words) == "**hi** Jon/John"
        assert render_transcript(consensus.words, strong=("", "")) == "hi Jon"
        assert render_transcript(consensus.words, strong=("<b>", "</b>")) == "<b>hi</b> Jon/John"

    def test_render_follows_index_not_argument_order(self) -> None:
        """The renderer sorts by index, so a caller's order cannot reorder the text."""
        consensus = _align(_source("a", "one two"), _source("b", "one two"))
        assert render_transcript(list(reversed(consensus.words))) == "**one** **two**"

    def test_the_provenance_field_set(self) -> None:
        """Test 16: exactly the fields the module owns, none of the retired knobs."""
        consensus = _align(_source("a", "hello"), _source("b", "hello"))
        assert set(consensus.provenance) == {
            "algorithm",
            "routine",
            "normalisation",
            "source_order",
            "sources",
            "n_sources",
            "reference_source",
            "n_words",
            "outcomes",
            "bracket_overrides_n",
            "empty_tokens_dropped",
            "time_fit",
            "n_words_time_shifted",
            "max_time_shift_s",
        }
        assert consensus.provenance["algorithm"] == "star_sequence_alignment"
        assert consensus.provenance["time_fit"] == "weighted_isotonic_median"
        assert consensus.provenance["source_order"] == "lexicographic_by_source_name"
        assert consensus.provenance["outcomes"] == {"agreement": 1, "variant": 0, "insertion": 0}
        assert set(consensus.provenance["sources"][0]) == {"name", "n_words", "timestamp_source", "timestamp_model"}
        for retired in ("slot_overlap", "slot_mid_tol_s", "winner_margin", "alternate_min_share"):
            assert retired not in consensus.provenance

    def test_the_consensus_emits_words_and_provenance_only(self) -> None:
        """Test 17 (R-4): no slot, no span, no grouping parameter."""
        assert [f.name for f in fields(Consensus)] == ["words", "provenance"]
        names = {f.name for f in fields(ConsensusWord)}
        assert not any("slot" in name or "span" in name for name in names)
        assert set(inspect.signature(align_sources).parameters) == {"sources", "onomatopoeic"}
