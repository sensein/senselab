"""Span proposal: the gate, the peak-anchored onset, the range-relative offset."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.tasks.spans import NoContrast, Span, propose_spans

SR = 16000


def _envelope(events: list[tuple[float, float, float]], seconds: float = 14.0, floor: float = -55.0) -> np.ndarray:
    env = np.full(int(seconds * SR), floor)
    for start, end, peak in events:
        env[int(start * SR) : int(end * SR)] = peak
    return env


def _propose(env: np.ndarray, floor: np.ndarray) -> list[Span] | NoContrast:
    """Run propose_spans with the config's measured values; the fixture supplies the literals."""
    return propose_spans(
        env,
        floor,
        SR,
        k_db=18.0,
        onset_drop_db=15.0,
        offset_fraction=0.7,
        hangover_ms=120,
        min_duration_ms=50,
        min_separation_ms=150,
    )


class TestGate:
    """Only a peak that clears k_db over the local floor proposes a span."""

    def test_a_peak_below_k_is_not_proposed(self) -> None:
        """Of one 10 dB peak and one 35 dB peak, only the one clearing 18 dB becomes a span."""
        env = _envelope([(2.0, 2.5, -45.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert not isinstance(out, NoContrast)
        assert len(out) == 1, "the 10 dB event must be gated out, not merely narrowed"
        (span,) = out
        assert span.start == pytest.approx(8.0, abs=0.05)
        assert span.end == pytest.approx(8.5, abs=0.05)

    def test_a_peak_above_k_is_proposed(self) -> None:
        """A 35 dB event over the floor clears the 18 dB gate."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 1 and isinstance(out[0], Span)

    def test_two_events_far_apart_stay_two_spans(self) -> None:
        """Distant events do not merge."""
        env = _envelope([(2.0, 2.5, -20.0), (8.0, 8.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        assert len(out) == 2


class TestNoContrast:
    """An unmeasurable recording is a distinct value, not a quiet one."""

    def test_no_peak_anywhere_is_no_contrast_not_an_empty_list(self) -> None:
        """A flat envelope 1 dB over its floor yields NoContrast naming the gate."""
        env = np.full(int(3.0 * SR), -30.0)
        out = _propose(env, np.full_like(env, -29.0))
        assert isinstance(out, NoContrast)
        assert "18" in out.reason


class TestKIsRequired:
    """No rule parameter has a default."""

    def test_k_has_no_default(self) -> None:
        """Calling without the rule parameters is a TypeError, not a silent default."""
        env = _envelope([(2.0, 2.5, -20.0)])
        with pytest.raises(TypeError):
            propose_spans(env, np.full_like(env, -55.0), SR)  # type: ignore[call-arg]


class TestSpanCarriesItsContrast:
    """A span reports how far its peak stood above the floor."""

    def test_peak_over_floor_travels_with_the_span(self) -> None:
        """A -20 dB event over a -55 dB floor reports 35 dB of contrast."""
        env = _envelope([(2.0, 2.5, -20.0)])
        out = _propose(env, np.full_like(env, -55.0))
        assert isinstance(out, list)
        (span,) = out
        assert span.peak_over_floor_db == pytest.approx(35.0, abs=0.5)
