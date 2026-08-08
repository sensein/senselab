"""J1 — how many speakers are simultaneously active, from the intact activation channels.

J1 is available now and J4 is not, for one reason worth pinning in a test: `segmentation-3.0`'s
channels are permutation-arbitrary within a window, so channel *k* is not a stable speaker across
the recording (D-7). A *count* of active channels does not care which channel is whom, so it is
well-defined without resolving the speaker↔channel assignment that J4 needs rounds for.

The count is reported as a distribution rather than a number. Entropy needs a distribution, and
"probably one speaker, possibly two" is a different state from "certainly one" even when both round
to the same expected count.
"""

from __future__ import annotations

import numpy as np
import pytest

# ── J2: speaker change points from windowed embeddings ──────────────────────


BAND = {"same_speaker_floor": 0.30, "diff_speaker_floor": 0.70}
"""The calibration band these tests assume. Named here rather than defaulted in the function so
the anchors a test relies on are visible in the test."""


def _windows(vectors: list[list[float]], width_s: float = 2.0, hop_s: float = 0.05) -> list:
    from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding

    return [
        WindowEmbedding(start_s=i * hop_s, end_s=i * hop_s + width_s, vector=np.asarray(v, dtype=np.float64))
        for i, v in enumerate(vectors)
    ]


def test_change_detection_compares_across_a_whole_window_not_adjacent_ones() -> None:
    """D-2: at a 50 ms hop, adjacent 2 s windows share 97.5% of their audio.

    Their distance is therefore dominated by the 2.5% that is new, which is not a speaker-change
    signal at all. The comparison has to span a whole window so the two sides are disjoint spans
    meeting at the boundary — the fine hop buys *localisation*, not independent samples.
    """
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    result = speaker_change_series(_windows([[1.0, 0.0]] * 100, width_s=2.0, hop_s=0.05), **BAND)
    assert result is not None
    assert result["lag_steps"] == 40, "2.0 s window / 0.05 s hop = 40 steps to reach a disjoint span"


def test_a_speaker_change_shows_up_at_the_boundary() -> None:
    """Two speakers back to back: change evidence peaks where they meet."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    a, b = [1.0, 0.0], [0.0, 1.0]
    entries = _windows([a] * 60 + [b] * 60)
    result = speaker_change_series(entries, **BAND)
    assert result is not None
    peak_idx = int(np.argmax(result["p_change"]))
    peak_t = result["times"][peak_idx]
    # Speaker A occupies windows 0-59 (starts 0.00-2.95 s); the first all-B window starts at 3.0 s,
    # so the disjoint-span boundary lands there.
    assert result["p_change"][peak_idx] > 0.9
    assert 2.0 <= peak_t <= 5.0


def test_one_speaker_throughout_yields_no_change_evidence() -> None:
    """A steady speaker must not produce change points, or every recording has them."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    result = speaker_change_series(_windows([[1.0, 0.0]] * 120), **BAND)
    assert result is not None
    assert float(np.max(result["p_change"])) < 0.1
    assert float(np.mean(result["uncertainty"])) < 0.2


def test_change_uncertainty_peaks_where_the_evidence_is_ambiguous() -> None:
    """Uncertainty is the entropy of {change, no change}, so it is highest mid-band.

    A confident change and a confident continuation are both certain; the doubt lives at the
    distances the calibration band cannot resolve.
    """
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    # 60 degrees off, giving cosine similarity 0.5 and therefore distance 0.5 -- the middle of the
    # 0.30-0.70 calibration band. A 45-degree offset would *not* work: distance 0.293 sits below the
    # same-speaker floor, i.e. inside the phonetic noise floor where a small distance is no evidence
    # at all, so it reads as a confident continuation rather than as ambiguity.
    mid = [float(np.cos(np.pi / 3)), float(np.sin(np.pi / 3))]
    ambiguous = speaker_change_series(_windows([[1.0, 0.0]] * 60 + [mid] * 60), **BAND)
    confident = speaker_change_series(_windows([[1.0, 0.0]] * 120), **BAND)
    assert ambiguous is not None and confident is not None
    assert float(np.max(ambiguous["uncertainty"])) > float(np.max(confident["uncertainty"]))


def test_too_few_windows_to_span_a_lag_yields_no_claim() -> None:
    """With less than one window-length of hops there is no disjoint pair to compare."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    assert speaker_change_series(_windows([[1.0, 0.0]] * 10), **BAND) is None
    assert speaker_change_series([], **BAND) is None


def test_change_detection_refuses_to_run_without_calibration() -> None:
    """A pass with no usable calibration band must not get anchors for free.

    The floors are required rather than defaulted precisely so this cannot happen by omission:
    substituting the library defaults would let the signal vote confidently on a pass where the
    embeddings were measured to be uncalibratable, which is the failure FR-007 exists to prevent.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    params = inspect.signature(speaker_change_series).parameters
    for name in ("same_speaker_floor", "diff_speaker_floor"):
        assert params[name].default is inspect.Parameter.empty, f"{name} must not carry a default anchor"
