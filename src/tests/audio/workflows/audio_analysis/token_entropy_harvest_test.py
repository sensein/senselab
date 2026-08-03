"""Token-entropy harvesting into asr buckets (T030, FR-017).

Covers ``mean_token_entropy_in_window`` and the ``token_entropy`` vote field that
``harvest_asr_votes`` attaches. Uses ``SimpleNamespace`` transcripts to
exercise both shapes the harvesters see: in-memory Pydantic-like objects and
cache-deserialized dicts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.asr import harvest_asr_votes
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import mean_token_entropy_in_window


def _line(**kwargs: Any) -> SimpleNamespace:  # noqa: ANN401 — mirrors ScriptLine's heterogeneous fields
    """A transcript line with harvester-visible attributes defaulted to None."""
    base: dict[str, Any] = {
        "text": None,
        "start": None,
        "end": None,
        "chunks": None,
        "token_entropy": None,
        "avg_logprob": None,
        "no_speech_prob": None,
        "score": None,
        "language": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


# ── mean_token_entropy_in_window ──────────────────────────────────────


def test_none_when_backend_supplied_no_entropy() -> None:
    """Backends without token logits contribute nothing, not a zero."""
    line = _line(text="hello", start=0.0, end=1.0)
    assert mean_token_entropy_in_window([line], 0.0, 1.0) is None


def test_line_level_list_collapses_to_mean() -> None:
    """A per-sequence entropy list is averaged for the overlapping window."""
    line = _line(text="hello", start=0.0, end=1.0, token_entropy=[0.0, 2.0])
    assert mean_token_entropy_in_window([line], 0.0, 1.0) == pytest.approx(1.0)


def test_line_level_scalar_used_directly() -> None:
    """A pre-collapsed scalar is taken as-is."""
    line = _line(text="hello", start=0.0, end=1.0, token_entropy=0.75)
    assert mean_token_entropy_in_window([line], 0.0, 1.0) == pytest.approx(0.75)


def test_line_outside_window_ignored() -> None:
    """A line that doesn't overlap the bucket contributes nothing."""
    line = _line(text="hello", start=5.0, end=6.0, token_entropy=2.0)
    assert mean_token_entropy_in_window([line], 0.0, 1.0) is None


def test_word_level_entropy_selected_by_midpoint() -> None:
    """Word-level entropy uses the midpoint rule, so each word lands in one bucket.

    Words at 0.0-0.4 (mid 0.2) and 0.6-1.0 (mid 0.8): querying [0.0, 0.5) must see
    only the first.
    """
    parent = _line(
        text="a b",
        start=0.0,
        end=1.0,
        chunks=[
            _line(text="a", start=0.0, end=0.4, token_entropy=1.0),
            _line(text="b", start=0.6, end=1.0, token_entropy=3.0),
        ],
    )
    assert mean_token_entropy_in_window([parent], 0.0, 0.5) == pytest.approx(1.0)
    assert mean_token_entropy_in_window([parent], 0.5, 1.0) == pytest.approx(3.0)


def test_word_level_preferred_over_line_level() -> None:
    """When words carry entropy, the coarser line value is not double-counted."""
    parent = _line(
        text="a b",
        start=0.0,
        end=1.0,
        token_entropy=99.0,  # would dominate if line-level leaked in
        chunks=[
            _line(text="a", start=0.0, end=0.4, token_entropy=1.0),
            _line(text="b", start=0.5, end=1.0, token_entropy=2.0),
        ],
    )
    assert mean_token_entropy_in_window([parent], 0.0, 1.0) == pytest.approx(1.5)


def test_dict_shaped_transcripts_supported() -> None:
    """Cache reads deserialize to plain dicts; the harvester handles both shapes."""
    line = {"text": "hello", "start": 0.0, "end": 1.0, "token_entropy": [1.0, 2.0]}
    assert mean_token_entropy_in_window([line], 0.0, 1.0) == pytest.approx(1.5)


def test_multiple_lines_averaged() -> None:
    """Two overlapping lines average their entropies."""
    lines = [
        _line(text="a", start=0.0, end=0.5, token_entropy=1.0),
        _line(text="b", start=0.5, end=1.0, token_entropy=3.0),
    ]
    assert mean_token_entropy_in_window(lines, 0.0, 1.0) == pytest.approx(2.0)


def test_non_numeric_entropy_ignored() -> None:
    """Garbage in the field degrades to None rather than raising."""
    line = _line(text="hello", start=0.0, end=1.0, token_entropy="not-a-number")
    assert mean_token_entropy_in_window([line], 0.0, 1.0) is None


# ── harvest_asr_votes wiring ────────────────────────────────────


def _pass_summary(lines: list[Any]) -> dict[str, Any]:
    return {
        "duration_s": 1.0,
        "asr": {"by_model": {"whisper": {"status": "ok", "result": lines}}},
    }


def test_harvest_attaches_token_entropy_vote() -> None:
    """The per-model vote carries token_entropy alongside the existing fields."""
    lines = [_line(text="hello", start=0.0, end=1.0, token_entropy=[1.0, 2.0])]
    buckets = harvest_asr_votes(
        pass_summary=_pass_summary(lines),
        grid=BucketGrid(win_length=1.0, hop_length=1.0),
        alignment_by_model={},
    )
    assert buckets, "expected at least one bucket"
    vote = buckets[0]["votes"]["whisper"]
    assert vote["token_entropy"] == pytest.approx(1.5)


def test_harvest_token_entropy_none_for_backend_without_logits() -> None:
    """Graceful degradation: the key exists but is None (FR-017)."""
    lines = [_line(text="hello", start=0.0, end=1.0)]
    buckets = harvest_asr_votes(
        pass_summary=_pass_summary(lines),
        grid=BucketGrid(win_length=1.0, hop_length=1.0),
        alignment_by_model={},
    )
    assert buckets[0]["votes"]["whisper"]["token_entropy"] is None
