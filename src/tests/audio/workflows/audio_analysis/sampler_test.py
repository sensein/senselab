"""Consumers pull from native-resolution signals through a memoizing sampler (D-25).

A producer that resamples to a target grid has made an L2 decision — which grid, and which reduction
onto it — and destroyed the alternative before anyone could ask. So L1 emits native resolution and the
consumer asks for the samples it wants.

The load-bearing detail is that **the cache key is the derivative key**: D-21 already names every
projection `(Target, Operator, Source)`, so a query is one of those plus an interval. That makes
D-22's "materialisation is a caching decision, not a semantic one" literal rather than analogical, and
it makes `GridRelation` the dispatch rather than a description.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.keys import DerivativeKey, Operator, Route, SignalKey
from senselab.audio.workflows.audio_analysis.sampler import Sampler, UnknownOperator
from senselab.audio.workflows.audio_analysis.shapes import (
    Categorical,
    LabelScore,
    Matrix,
    Series,
    Span,
    Spans,
    Tree,
    Window,
)

SNR = SignalKey("snr", "pyannote/brouhaha", Route())
LABELS = SignalKey("scene_labels", "MIT/ast-finetuned-audioset", Route())
DIAR = SignalKey("speaker_spans", "pyannote/speaker-diarization-community-1", Route())
TRANSCRIPT = SignalKey("transcript", "openai/whisper-large-v3-turbo", Route())


def _resample(source: SignalKey = SNR, how: str = "mean") -> DerivativeKey:
    return DerivativeKey("snr", Operator("resample", how), sources=(source,))


# ── native resolution is preserved; the reduction happens on demand ────


def test_a_query_reduces_native_frames_without_the_producer_resampling() -> None:
    """The producer stores its own hop; the consumer names the interval and the reduction.

    The defect this replaces stored a value at a grid the model never reported, with the native
    window recorded beside it as if it described that value.
    """
    s = Series(values=(1.0, 3.0, 5.0, 7.0, 9.0), hop_s=0.02, window_s=0.02, units="dB")
    sampler = Sampler({SNR: s})
    # frames at 0.00,0.02,0.04,0.06,0.08 — a 0.1 s bucket covers all five
    assert sampler.at(_resample(), 0.0, 0.1) == pytest.approx(5.0)
    # a 0.04 s bucket covers the first two
    assert sampler.at(_resample(), 0.0, 0.04) == pytest.approx(2.0)


def test_the_reduction_is_named_in_the_key_so_two_choices_are_two_answers() -> None:
    """`mean` and `max` are different derivatives of one signal, not one value with a flag."""
    sampler = Sampler({SNR: Series(values=(1.0, 9.0), hop_s=0.05, window_s=0.05, units="dB")})
    assert sampler.at(_resample(how="mean"), 0.0, 0.1) == pytest.approx(5.0)
    assert sampler.at(_resample(how="max"), 0.0, 0.1) == pytest.approx(9.0)


def test_an_interval_with_no_frames_yields_none_not_zero() -> None:
    """Nothing measured there. Zero would assert a measurement nobody made."""
    sampler = Sampler({SNR: Series(values=(1.0,), hop_s=0.02, window_s=0.02, units="dB")})
    assert sampler.at(_resample(), 5.0, 5.1) is None


def test_unmeasured_frames_are_skipped_rather_than_counted_as_zero() -> None:
    """A None frame contributes nothing to the mean and does not drag it toward zero."""
    s = Series(values=(2.0, None, 4.0), hop_s=0.02, window_s=0.02, units="dB")
    assert Sampler({SNR: s}).at(_resample(), 0.0, 0.1) == pytest.approx(3.0)


def test_a_bucket_where_every_frame_is_unmeasured_is_none() -> None:
    """All-null frames leave nothing to reduce, so the answer is None rather than a default."""
    s = Series(values=(None, None), hop_s=0.02, window_s=0.02, units="dB")
    assert Sampler({SNR: s}).at(_resample(), 0.0, 0.1) is None


# ── memoization ────────────────────────────────────────────────────────


def test_the_same_query_is_computed_once() -> None:
    """The point of the sampler: the same sample or operation is not recomputed."""
    sampler = Sampler({SNR: Series(values=(1.0, 2.0), hop_s=0.05, window_s=0.05, units="dB")})
    sampler.at(_resample(), 0.0, 0.1)
    sampler.at(_resample(), 0.0, 0.1)
    assert sampler.stats == {"hits": 1, "misses": 1}


def test_a_different_interval_is_a_different_entry() -> None:
    """The interval is part of the cache key, so two windows cannot share one answer."""
    sampler = Sampler({SNR: Series(values=(1.0, 2.0), hop_s=0.05, window_s=0.05, units="dB")})
    sampler.at(_resample(), 0.0, 0.05)
    sampler.at(_resample(), 0.05, 0.1)
    assert sampler.stats == {"hits": 0, "misses": 2}


def test_a_different_operator_variant_is_a_different_entry() -> None:
    """Otherwise the cache would return a mean where a max was asked for."""
    sampler = Sampler({SNR: Series(values=(1.0, 9.0), hop_s=0.05, window_s=0.05, units="dB")})
    sampler.at(_resample(how="mean"), 0.0, 0.1)
    sampler.at(_resample(how="max"), 0.0, 0.1)
    assert sampler.stats["misses"] == 2


# ── GridRelation is the dispatch ───────────────────────────────────────


def test_a_categorical_projects_its_window_rather_than_being_resampled() -> None:
    """A 0.96 s window is not ten 0.1 s measurements; its value is assigned to the buckets it spans."""
    c = Categorical(
        windows=(Window(0.0, 0.96, (LabelScore("Speech", 0.8), LabelScore("Music", 0.1))),),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    key = DerivativeKey("speech", Operator("project_labels", "speech_v3"), sources=(LABELS,))
    sampler = Sampler({LABELS: c}, label_sets={"speech_v3": ("Speech",)})
    assert sampler.at(key, 0.0, 0.1) == pytest.approx(0.8)
    assert sampler.at(key, 0.5, 0.6) == pytest.approx(0.8), "same window, so same value"
    assert sampler.at(key, 2.0, 2.1) is None, "no window covers this"


def test_a_label_outside_the_set_does_not_contribute() -> None:
    """The selection is the operator's variant, so it is named and replaceable."""
    c = Categorical(
        windows=(Window(0.0, 0.96, (LabelScore("Speech", 0.8), LabelScore("Music", 0.1))),),
        vocabulary_id="audioset",
        vocabulary_size=527,
        top_k=7,
    )
    key = DerivativeKey("music", Operator("project_labels", "music_v1"), sources=(LABELS,))
    sampler = Sampler({LABELS: c}, label_sets={"music_v1": ("Music",)})
    assert sampler.at(key, 0.0, 0.1) == pytest.approx(0.1)


def test_spans_reduce_to_coverage() -> None:
    """A span set has no grid, so a per-bucket value is a reduction with a named choice."""
    s = Spans(spans=(Span(0.0, 0.05, "spk0"),), capacity="unbounded")
    key = DerivativeKey("occupancy", Operator("cover"), sources=(DIAR,))
    sampler = Sampler({DIAR: s})
    assert sampler.at(key, 0.0, 0.1) == pytest.approx(0.5)
    assert sampler.at(key, 0.1, 0.2) is None, "no span here: absent, not zero coverage"


def test_a_matrix_resamples_per_channel() -> None:
    """Channels stay separate; pooling them is a different named derivative."""
    m = Matrix(
        rows=((1.0, 10.0), (3.0, 30.0)),
        channels=("band_a", "band_b"),
        hop_s=0.05,
        window_s=0.05,
        units="dB",
    )
    key = DerivativeKey("noise_floor", Operator("resample", "mean"), sources=(SNR,))
    sampler = Sampler({SNR: m})
    assert sampler.at(key, 0.0, 0.1) == {"band_a": pytest.approx(2.0), "band_b": pytest.approx(20.0)}


def test_a_tree_reduces_to_word_overlap_seconds() -> None:
    """A transcript has no per-bucket value, so the reduction is named."""
    t = Tree(
        script_line={"chunks": [{"text": "hello", "start": 0.0, "end": 0.06}]},
        timestamp_source="native",
    )
    key = DerivativeKey("speech", Operator("word_coverage"), sources=(TRANSCRIPT,))
    sampler = Sampler({TRANSCRIPT: t})
    assert sampler.at(key, 0.0, 0.1) == pytest.approx(0.06)
    assert sampler.at(key, 0.5, 0.6) is None


# ── refusals ───────────────────────────────────────────────────────────


def test_an_unknown_operator_raises_rather_than_returning_none() -> None:
    """`None` means "measured nothing here". A typo must not borrow that meaning."""
    sampler = Sampler({SNR: Series(values=(1.0,), hop_s=0.05, window_s=0.05, units="dB")})
    with pytest.raises(UnknownOperator, match="invent"):
        sampler.at(DerivativeKey("snr", Operator("invent"), sources=(SNR,)), 0.0, 0.1)


def test_a_query_for_an_absent_signal_raises() -> None:
    """A signal that never ran is not a signal that measured nothing."""
    sampler = Sampler({})
    with pytest.raises(KeyError, match="snr"):
        sampler.at(_resample(), 0.0, 0.1)


def test_a_projection_query_must_have_exactly_one_source() -> None:
    """A fold is not a sample; asking the sampler for one is a category error."""
    sampler = Sampler({SNR: Series(values=(1.0,), hop_s=0.05, window_s=0.05, units="dB")})
    fold = DerivativeKey("snr", Operator("resample", "mean"), sources=(SNR, LABELS))
    with pytest.raises(ValueError, match="a sample takes one"):
        sampler.at(fold, 0.0, 0.1)


# ── whole-grid convenience, and independence ───────────────────────────


def test_a_non_overlapping_grid_gives_independent_buckets() -> None:
    """Window equals hop, so no two rows share a frame — what DEFAULT_TIME_GRID exists for."""
    s = Series(values=tuple(float(i) for i in range(10)), hop_s=0.02, window_s=0.02, units="dB")
    sampler = Sampler({SNR: s})
    rows = sampler.on_grid(_resample(), duration_s=0.2, win_length=0.1, hop_length=0.1)
    assert [r["start"] for r in rows] == [pytest.approx(0.0), pytest.approx(0.1)]
    assert rows[0]["value"] == pytest.approx(2.0)
    assert rows[1]["value"] == pytest.approx(7.0)


def test_the_grid_the_consumer_asks_for_is_not_a_property_of_the_producer() -> None:
    """One native signal, two consumers, two grids, no producer change."""
    s = Series(values=tuple(float(i) for i in range(10)), hop_s=0.02, window_s=0.02, units="dB")
    sampler = Sampler({SNR: s})
    fine = sampler.on_grid(_resample(), duration_s=0.2, win_length=0.05, hop_length=0.05)
    coarse = sampler.on_grid(_resample(), duration_s=0.2, win_length=0.2, hop_length=0.2)
    assert len(fine) == 4
    assert len(coarse) == 1
