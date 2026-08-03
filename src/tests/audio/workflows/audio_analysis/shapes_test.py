"""The six shapes an L1 measurement can have, and why one row type could not hold them.

`SignalRow(measurement: Mapping[str, float])` fits only the scalar-per-bucket case, and forcing the
other five through it is what produced every reduction D-18 found: a probability matrix stored as its
mean, 527 label scores stored as a hand-picked sum, a span set stored as a covered fraction, all on a
0.1 s grid none of them were measured at.

These tests encode the distinctions that made each reduction lossy, so a future collapse fails here
rather than in a run nobody inspects.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.shapes import (
    Categorical,
    Embedding,
    GridRelation,
    LabelScore,
    Matrix,
    Measurement,
    Series,
    Span,
    Spans,
    Tree,
    Window,
)

# ── Series ─────────────────────────────────────────────────────────────


def test_a_series_reports_the_hop_it_was_measured_at() -> None:
    """The defect this replaces recorded ``resolution_s: 0.0169`` on a row spanning 0.0 → 0.1.

    Provenance that describes a measurement the file does not contain is worse than absent
    provenance, because it reads as corroboration.
    """
    s = Series(values=(0.1, 0.2, 0.3), hop_s=0.0169, window_s=0.0619, units="probability")
    assert s.hop_s == 0.0169
    assert s.duration_s == pytest.approx(3 * 0.0169)


def test_an_unmeasured_frame_is_none_not_zero() -> None:
    """Zero is a confident claim. A tool that said nothing must not be read as having said zero."""
    s = Series(values=(0.4, None, 0.6), hop_s=0.01, window_s=0.06, units="dB")
    assert s.values[1] is None
    assert s.measured_count == 2


# ── Matrix ─────────────────────────────────────────────────────────────


def test_a_matrix_keeps_its_channels_rather_than_pooling_them() -> None:
    """The collapse that returned 1.0000 in 100% of frames on a half-silent clip.

    Pooling to one p(speech) is a choice among mean / max / noisy-or that changes the answer, so it
    is an L2 derivative and the channels have to survive L1 for it to be makeable at all.
    """
    m = Matrix(
        rows=((0.1, 0.9), (0.2, 0.8)),
        channels=("band_0_100", "band_100_200"),
        hop_s=0.1,
        window_s=0.1,
        units="dB",
    )
    assert m.n_channels == 2
    assert m.channel("band_100_200") == (0.9, 0.8)


def test_named_and_arbitrary_channels_are_different_kinds_of_matrix() -> None:
    """Brouhaha's heads have fixed meaning; a diarizer's speaker columns do not.

    Averaging across arbitrary channels is meaningless and across named ones is merely a choice, so
    a consumer has to be able to tell which it holds (D-5).
    """
    named = Matrix(rows=((0.1, 0.2),), channels=("snr", "c50"), hop_s=0.1, window_s=0.1, units="dB")
    arbitrary = Matrix(
        rows=((0.1, 0.2),),
        channels=("spk_0", "spk_1"),
        channel_semantics="arbitrary",
        hop_s=0.1,
        window_s=0.1,
        units="probability",
    )
    assert named.channels_are_comparable_across_frames is True
    assert arbitrary.channels_are_comparable_across_frames is False


def test_a_matrix_row_must_match_its_channel_count() -> None:
    """A ragged matrix would silently misalign every channel after the short row."""
    with pytest.raises(ValueError, match="channel"):
        Matrix(rows=((0.1, 0.2), (0.3,)), channels=("a", "b"), hop_s=0.1, window_s=0.1, units="dB")


# ── Categorical ────────────────────────────────────────────────────────


def _window(start: float, scores: list[tuple[str, float]]) -> Window:
    return Window(start=start, end=start + 0.96, scores=tuple(LabelScore(label=n, score=v) for n, v in scores))


def test_a_categorical_carries_its_top_k_so_a_zero_is_distinguishable_from_a_cutoff() -> None:
    """Label mass over a set whose members fell outside the top k is not recoverable.

    Without k on the row, "this label scored below the 7th" and "this label scored nothing" read
    identically — the absent-vs-zero distinction, at the vocabulary level.
    """
    c = Categorical(
        windows=(_window(0.0, [("Speech", 0.8), ("Music", 0.1)]),),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    assert c.top_k == 7
    assert c.mass_is_truncated is True


def test_a_categorical_over_the_whole_vocabulary_is_not_truncated() -> None:
    """When k covers the vocabulary, a missing label really did score nothing."""
    c = Categorical(
        windows=(_window(0.0, [("a", 0.5), ("b", 0.5)]),),
        vocabulary_id="tiny",
        vocabulary_size=2,
        top_k=2,
    )
    assert c.mass_is_truncated is False


def test_a_categorical_keeps_the_window_it_was_scored_over() -> None:
    """A 0.96 s window projected onto 0.1 s buckets asserted 10 independent values inside one."""
    c = Categorical(
        windows=(_window(0.0, [("Speech", 0.8)]), _window(0.48, [("Speech", 0.7)])),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    assert c.windows[0].end == pytest.approx(0.96)
    assert c.windows_overlap is True, "0.48 hop under a 0.96 window overlaps, and L2 must know"


# ── Spans ──────────────────────────────────────────────────────────────


def test_spans_have_no_grid() -> None:
    """A span set is not on any grid, so there is no resolution to record and none to get wrong."""
    s = Spans(spans=(Span(start=0.0, end=1.5, label="SPEAKER_00"),), capacity=None)
    assert not hasattr(s, "hop_s")


def test_capacity_distinguishes_unbounded_from_bounded_from_inapplicable() -> None:
    """Three states, all meaningful (D-19).

    Without the distinction a reader cannot tell "3 speakers active" from "3 active and the model
    had no fourth column", and a count posterior fused across tools of differing capacity is biased
    toward the smallest.
    """
    community = Spans(spans=(), capacity="unbounded")
    sortformer = Spans(spans=(), capacity=4)
    words = Spans(spans=(), capacity=None)
    assert community.is_censored_at(3) is False
    assert sortformer.is_censored_at(4) is True
    assert sortformer.is_censored_at(3) is False
    assert words.is_censored_at(4) is False, "capacity does not apply to a word span set"


def test_a_span_may_carry_the_tool_s_own_confidence() -> None:
    """A model's confidence is a measurement the model made; refusing it discards information."""
    s = Spans(spans=(Span(start=0.0, end=1.0, label="spk0", confidence=0.9),), capacity=4)
    assert s.spans[0].confidence == 0.9


def test_a_span_with_no_confidence_is_not_a_span_with_zero_confidence() -> None:
    """The tool did not report one. Defaulting to 0.0 would assert maximal doubt on its behalf."""
    s = Spans(spans=(Span(start=0.0, end=1.0, label="spk0"),), capacity=4)
    assert s.spans[0].confidence is None


def test_a_span_must_not_end_before_it_starts() -> None:
    """A backwards span would project onto no bucket and read as a tool that said nothing."""
    with pytest.raises(ValueError, match="end"):
        Span(start=1.0, end=0.5, label="spk0")


# ── Embedding ──────────────────────────────────────────────────────────


def test_embeddings_are_fixed_width() -> None:
    """A ragged embedding set cannot be compared, and the raggedness would surface as a distance."""
    e = Embedding(vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)), window_s=2.0, hop_s=0.05)
    assert e.dims == 3
    with pytest.raises(ValueError, match="width"):
        Embedding(vectors=((0.1, 0.2), (0.3,)), window_s=2.0, hop_s=0.05)


# ── Tree ───────────────────────────────────────────────────────────────


def test_a_tree_records_where_its_timings_came_from() -> None:
    """An ASR output *is* a time-aligned output, and which aligner timed it is provenance.

    Two transcripts timed by one aligner have correlated word boundaries, and the ASR axis compares
    word boundaries — so this field is what lets the correlation be measured rather than assumed
    absent.
    """
    t = Tree(script_line={"text": "hello"}, timestamp_source="bundled_aligner")
    assert t.timestamp_source == "bundled_aligner"


def test_an_unknown_timestamp_source_is_refused() -> None:
    """The set is closed: native, bundled_aligner, external_aligner."""
    with pytest.raises(ValueError, match="timestamp_source"):
        Tree(script_line={"text": "hi"}, timestamp_source="guessed")  # type: ignore[arg-type]


# ── the union ──────────────────────────────────────────────────────────


def test_every_shape_reports_whether_a_bucket_grid_is_meaningful_for_it() -> None:
    """Each shape reports what a bucket grid does to it.

    A grid is meaningful for Series and Matrix, a projection for Categorical and Embedding, and a
    reduction for Spans and Tree. Conflating the three is what made one row type look sufficient.
    """
    cases: list[tuple[Measurement, GridRelation]] = [
        (Series(values=(0.1,), hop_s=0.1, window_s=0.1, units="dB"), GridRelation.RESAMPLE),
        (Matrix(rows=((0.1,),), channels=("a",), hop_s=0.1, window_s=0.1, units="dB"), GridRelation.RESAMPLE),
        (
            Categorical(windows=(_window(0.0, [("a", 1.0)]),), vocabulary_id="v", vocabulary_size=1, top_k=1),
            GridRelation.PROJECT,
        ),
        (Embedding(vectors=((0.1,),), window_s=2.0, hop_s=0.05), GridRelation.PROJECT),
        (Spans(spans=(), capacity=None), GridRelation.REDUCE),
        (Tree(script_line={}, timestamp_source="native"), GridRelation.REDUCE),
    ]
    for shape, expected in cases:
        assert shape.grid_relation is expected, f"{type(shape).__name__} misclassified"


def test_grid_relation_is_a_class_constant_not_an_overridable_field() -> None:
    """A shape must not be able to claim a relation it does not have.

    As a ``Final`` *field* it landed in ``__init__``, so ``Tree(..., grid_relation=RESAMPLE)``
    constructed a transcript asserting it was resampleable — the shape lying about its own kind.
    """
    with pytest.raises(TypeError, match="grid_relation"):
        Tree(script_line={}, timestamp_source="native", grid_relation=GridRelation.RESAMPLE)  # type: ignore[call-arg]
