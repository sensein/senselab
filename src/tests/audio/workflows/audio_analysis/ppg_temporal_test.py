"""Focused tests for the PPG sequence-edit-distance harvester (2026-05-09 rewrite)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.harvesters import (
    arpabet_to_ppg_inventory,
    asr_phoneme_sequence_in_window,
    asr_text_in_window,
    ppg_argmax_per_frame,
    ppg_argmax_runs_in_window,
    ppg_sequence_per_in_window,
)


def test_arpabet_to_ppg_inventory_strips_case_and_stress() -> None:
    """``g2p_en`` returns ``"AH0"``; PPG inventory uses ``"ah"``."""
    assert arpabet_to_ppg_inventory("AH0") == "ah"
    assert arpabet_to_ppg_inventory("EY1") == "ey"
    assert arpabet_to_ppg_inventory("IY2") == "iy"
    assert arpabet_to_ppg_inventory("HH") == "hh"  # no stress marker


def test_ppg_argmax_per_frame_normalizes_phonemes_x_frames_shape() -> None:
    """``extract_ppgs_from_audios`` returns ``(phonemes, frames)``; we transpose internally."""
    n_phonemes, n_frames = 40, 100
    ppg = np.zeros((n_phonemes, n_frames), dtype=np.float32)
    ppg[0, :50] = 1.0
    ppg[6, 50:] = 1.0
    per_frame, frame_hop = ppg_argmax_per_frame([ppg], None, duration_s=1.0)
    assert frame_hop == pytest.approx(0.01)
    assert per_frame[0] == "aa"
    assert per_frame[49] == "aa"
    assert per_frame[50] == "b"
    assert per_frame[99] == "b"


def test_ppg_argmax_per_frame_handles_batch_dim() -> None:
    """Tensor ``(1, phonemes, frames)`` (the most common ppgs output) collapses cleanly."""
    ppg = np.zeros((1, 40, 50), dtype=np.float32)
    ppg[0, 18, :] = 1.0  # all frames argmax to "jh"
    per_frame, hop = ppg_argmax_per_frame([ppg], None, duration_s=0.5)
    assert all(p == "jh" for p in per_frame)
    assert hop == pytest.approx(0.01)


def test_ppg_argmax_runs_collapses_consecutive_same_phoneme_frames() -> None:
    """Runs walk consecutive same-phoneme frames into single ``(start, end, phoneme)`` tuples."""
    n_phonemes = 40
    ppg = np.zeros((n_phonemes, 100), dtype=np.float32)
    # frames 0–29 → "aa" (idx 0); 30–69 → "b" (idx 6); 70–99 → "aa" again.
    ppg[0, :30] = 1.0
    ppg[6, 30:70] = 1.0
    ppg[0, 70:] = 1.0
    per_frame, hop = ppg_argmax_per_frame([ppg], None, duration_s=1.0)
    runs = ppg_argmax_runs_in_window(per_frame, hop, 0.0, 1.0)
    assert len(runs) == 3
    assert runs[0][2] == "aa"
    assert runs[0][0] == pytest.approx(0.0)
    assert runs[0][1] == pytest.approx(0.30)
    assert runs[1][2] == "b"
    assert runs[2][2] == "aa"


def test_asr_phoneme_sequence_in_window_uses_fully_contained_words() -> None:
    """Boundary-straddling words are dropped; fully-contained words contribute their G2P phonemes."""
    chunks = [
        SimpleNamespace(start=0.10, end=0.40, text="hello"),
        SimpleNamespace(start=0.90, end=1.20, text="world"),  # straddles 1.0 boundary
        SimpleNamespace(start=1.50, end=1.90, text="cat"),
    ]
    line = SimpleNamespace(text="hello world cat", chunks=chunks, start=0.10, end=1.90)
    asr = [line]
    seq_a = asr_phoneme_sequence_in_window(asr, 0.0, 1.0)
    seq_b = asr_phoneme_sequence_in_window(asr, 1.0, 2.0)
    assert seq_a, "fully-contained 'hello' should yield phonemes"
    assert seq_b, "fully-contained 'cat' should yield phonemes"
    # All phonemes lowercase, no stress markers.
    for p in seq_a + seq_b:
        assert p == p.lower()
        assert not any(ch.isdigit() for ch in p)


def test_ppg_sequence_per_zero_when_sequences_match() -> None:
    """Identical PPG argmax sequence and ASR-implied phoneme sequence → PER 0."""
    # Build PPG that always argmaxes to "k" then "ae" then "t" — matches "cat".
    n_phonemes = 40
    ppg = np.zeros((n_phonemes, 90), dtype=np.float32)
    # idx 19 = "k", idx 1 = "ae", idx 30 = "t"
    ppg[19, :30] = 1.0
    ppg[1, 30:60] = 1.0
    ppg[30, 60:] = 1.0
    per_frame, hop = ppg_argmax_per_frame([ppg], None, duration_s=0.9)
    chunks = [SimpleNamespace(start=0.0, end=0.9, text="cat")]
    asr = [SimpleNamespace(text="cat", chunks=chunks, start=0.0, end=0.9)]
    per = ppg_sequence_per_in_window(per_frame, hop, asr, 0.0, 0.9)
    assert per == pytest.approx(0.0)


def test_ppg_sequence_per_high_when_sequences_disagree() -> None:
    """When PPG argmax is all ``<silent>`` but ASR has words → PER == 1.0 (asymmetric)."""
    n_phonemes = 40
    ppg = np.zeros((n_phonemes, 100), dtype=np.float32)
    ppg[39, :] = 1.0  # idx 39 = "<silent>"
    per_frame, hop = ppg_argmax_per_frame([ppg], None, duration_s=1.0)
    chunks = [SimpleNamespace(start=0.0, end=1.0, text="hello world")]
    asr = [SimpleNamespace(text="hello world", chunks=chunks, start=0.0, end=1.0)]
    per = ppg_sequence_per_in_window(per_frame, hop, asr, 0.0, 1.0)
    assert per == pytest.approx(1.0)


def test_ppg_sequence_per_returns_none_when_both_sides_empty() -> None:
    """Empty PPG run-list AND empty ASR phoneme sequence → None (sub-signal dropped)."""
    n_phonemes = 40
    ppg = np.zeros((n_phonemes, 50), dtype=np.float32)
    ppg[39, :] = 1.0  # all <silent>
    per_frame, hop = ppg_argmax_per_frame([ppg], None, duration_s=0.5)
    empty_line = SimpleNamespace(text="", chunks=[], start=None, end=None)
    asr = [empty_line]
    per = ppg_sequence_per_in_window(per_frame, hop, asr, 0.0, 0.5)
    assert per is None


def test_ppg_sequence_per_returns_none_for_empty_frames() -> None:
    """No PPG frames at all → None."""
    assert ppg_sequence_per_in_window([], 0.0, [], 0.0, 1.0) is None


def test_asr_text_in_window_fully_contained_excludes_boundary_words() -> None:
    """``fully_contained=True`` keeps only words whose [start, end] is inside the window."""
    chunks = [
        SimpleNamespace(start=0.10, end=0.40, text="hello"),
        SimpleNamespace(start=0.90, end=1.20, text="world"),
        SimpleNamespace(start=1.50, end=1.90, text="goodbye"),
    ]
    line = SimpleNamespace(text="hello world goodbye", chunks=chunks, start=0.10, end=1.90)
    asr = [line]

    assert "hello" in asr_text_in_window(asr, 0.0, 1.0)
    assert "world" in asr_text_in_window(asr, 0.0, 1.0)

    assert asr_text_in_window(asr, 0.0, 1.0, fully_contained=True) == "hello"
    assert asr_text_in_window(asr, 1.0, 2.0, fully_contained=True) == "goodbye"
    assert asr_text_in_window(asr, 0.5, 1.5, fully_contained=True) == "world"
