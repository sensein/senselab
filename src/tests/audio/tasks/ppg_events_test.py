"""Tests for probabilistic phone events and the timing helpers over them."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.ppg import (
    PHONEME_LABELS,
    PPGS_SAMPLE_RATE,
    extract_ppgs_from_audios,
    ppgs_venv_is_provisioned,
    require_posteriorgram,
)
from senselab.audio.tasks.features_extraction.ppg_events import (
    PHONE_CLASSES,
    SILENCE_LABEL,
    filler_support,
    logistic_sigma_from_width,
    phone_class_duration_stats,
    phone_rate_by_speaker,
    ppg_boundary_probability,
    ppg_phone_events,
    turn_transition_timing,
)
from senselab.audio.tasks.features_extraction.sparse_ppg import sparse_ppg_fidelity, sparsify_ppg
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType

RATE = 100.0
INDEX = {label: i for i, label in enumerate(PHONEME_LABELS)}


def _centres(frames: int) -> np.ndarray:
    return (np.arange(frames) + 0.5) / RATE


def _two_phone(frames: int, crossing_s: float, profile: np.ndarray, a: str = "aa", b: str = "s") -> np.ndarray:
    """Frames whose mass moves from ``a`` to ``b`` following ``profile`` (``p_b`` per frame)."""
    dense = np.zeros((frames, len(PHONEME_LABELS)))
    dense[:, INDEX[b]] = profile
    dense[:, INDEX[a]] = 1.0 - profile
    return dense


def _ramp(frames: int, crossing_s: float, half_width_s: float) -> np.ndarray:
    return np.clip(0.5 + (_centres(frames) - crossing_s) / (2.0 * half_width_s), 0.0, 1.0)


def _logistic(frames: int, crossing_s: float, scale_s: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((_centres(frames) - crossing_s) / (2.0 * scale_s)))


def _sequence(labels: List[Tuple[str, int]]) -> np.ndarray:
    """A crisp posteriorgram: each ``(label, frames)`` run holds 0.9 on its phone, the rest spread."""
    rows = []
    for label, count in labels:
        row = np.full(len(PHONEME_LABELS), 0.1 / (len(PHONEME_LABELS) - 1))
        row[INDEX[label]] = 0.9
        rows.extend([row] * count)
    return np.asarray(rows)


@pytest.mark.parametrize("crossing_s", [0.4037, 0.41, 0.4149, 0.45])
def test_onset_recovered_exactly_at_sub_frame_precision(crossing_s: float) -> None:
    """A linear transition's zero crossing is recovered to float32 precision."""
    dense = _two_phone(100, crossing_s, _ramp(100, crossing_s, half_width_s=0.05))
    events = ppg_phone_events(dense)
    assert list(events.labels) == ["aa", "s"]
    assert events.onset_s[1] == pytest.approx(crossing_s, abs=1e-6)
    assert events.offset_s[0] == pytest.approx(crossing_s, abs=1e-6)


def test_width_of_a_linear_transition_is_exact() -> None:
    """With p_b = 0.5 + (t - t0) / 2h, the 0.8 crossings lie 1.2 h apart."""
    half_width = 0.06
    dense = _two_phone(120, 0.5531, _ramp(120, 0.5531, half_width))
    events = ppg_phone_events(dense, level=0.8)
    assert events.onset_width_s[1] == pytest.approx(1.2 * half_width, abs=1e-6)
    assert events.offset_width_s[0] == events.onset_width_s[1]
    assert events.onset_sigma_s[1] == pytest.approx(float(logistic_sigma_from_width(1.2 * half_width, 0.8)))


def test_uncertainty_grows_with_blur_and_matches_the_logistic() -> None:
    """Wider logistic transitions give monotonically wider boundaries, and sigma tracks s·π/√3."""
    widths, sigmas, sharpness = [], [], []
    scales = [0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
    for scale in scales:
        events = ppg_phone_events(_two_phone(400, 2.0034, _logistic(400, 2.0034, scale)))
        assert events.onset_s[1] == pytest.approx(2.0034, abs=0.25 / RATE)
        widths.append(events.onset_width_s[1])
        sigmas.append(events.onset_sigma_s[1])
        sharpness.append(events.onset_sharpness[1])
    assert np.all(np.diff(widths) > 0)
    assert np.all(np.diff(sharpness) < 0)
    for scale, width, sigma in zip(scales[-3:], widths[-3:], sigmas[-3:]):
        assert width == pytest.approx(2 * scale * math.log(4.0), rel=0.03)
        assert sigma == pytest.approx(scale * math.pi / math.sqrt(3.0), rel=0.03)


def test_step_is_maximally_sharp_with_sub_frame_width() -> None:
    """A one-frame swap has sharpness 1 and a width below one frame."""
    profile = (np.arange(50) >= 20).astype(float)
    events = ppg_phone_events(_two_phone(50, 0.2, profile))
    assert events.onset_sharpness[1] == pytest.approx(1.0)
    assert events.onset_s[1] == pytest.approx(0.2, abs=1e-12)
    assert 0 < events.onset_width_s[1] < 1 / RATE


def test_sequence_events_statistics_and_edges() -> None:
    """Runs, labels, per-event statistics and the unobserved recording edges."""
    dense = _sequence([(SILENCE_LABEL, 30), ("hh", 6), ("ah", 12), ("l", 5), ("ow", 15), (SILENCE_LABEL, 20)])
    events = ppg_phone_events(dense)
    assert list(events.labels) == [SILENCE_LABEL, "hh", "ah", "l", "ow", SILENCE_LABEL]
    assert list(events.classes) == ["silence", "fricative", "vowel", "liquid", "diphthong", "silence"]
    np.testing.assert_allclose(events.onset_s[1:], np.cumsum([30, 6, 12, 5, 15]) / RATE)
    np.testing.assert_allclose(events.peak_prob, 0.9)
    np.testing.assert_allclose(events.mean_prob, 0.9)
    assert np.isnan(events.onset_sigma_s[0]) and np.isnan(events.offset_sigma_s[-1])
    assert events.is_interior.tolist() == [False, True, True, True, True, False]
    assert events.durations_s.sum() == pytest.approx(len(dense) / RATE)
    records = events.to_records()
    assert records[2]["phone"] == "ah" and records[2]["phone_class"] == "vowel"


def test_sparse_input_gives_the_same_boundaries() -> None:
    """Events from a top-3 sparse posteriorgram match the dense ones on a two-phone transition."""
    dense = _two_phone(200, 1.0123, _logistic(200, 1.0123, 0.015))
    sparse = sparsify_ppg(dense)
    assert sparse_ppg_fidelity(dense, sparse)["argmax_agreement"] == 1.0
    a, b = ppg_phone_events(dense), ppg_phone_events(sparse)
    np.testing.assert_allclose(a.onset_s, b.onset_s, atol=1e-3)
    np.testing.assert_allclose(a.onset_width_s[1:], b.onset_width_s[1:], atol=1e-3)


def test_level_out_of_range_is_refused() -> None:
    """The transition level must be above one half."""
    with pytest.raises(ValueError, match="level"):
        ppg_phone_events(_sequence([("aa", 5)]), level=0.5)


def test_boundary_probability_is_total_variation() -> None:
    """A full swap scores 1, a constant posterior 0, and a ramp spreads unit mass over its frames."""
    step = ppg_boundary_probability(_two_phone(40, 0.2, (np.arange(40) >= 20).astype(float)))
    assert step[0] == 0 and step[20] == pytest.approx(1.0) and step.sum() == pytest.approx(1.0)
    ramp = ppg_boundary_probability(_two_phone(200, 1.0, _ramp(200, 1.0, 0.1)))
    assert ramp.sum() == pytest.approx(1.0, abs=1e-5)
    assert ramp.max() < 0.1


def test_phone_rate_by_speaker() -> None:
    """Phones are counted per speaker by midpoint; silence inside a span is excluded from articulation."""
    dense = _sequence(
        [(SILENCE_LABEL, 10), ("b", 10), ("aa", 10), (SILENCE_LABEL, 20), ("s", 10), ("iy", 10), ("t", 10)]
    )
    events = ppg_phone_events(dense)
    rates = phone_rate_by_speaker(events, [("A", 0.0, 0.5), ("B", 0.5, 0.8)])
    assert rates["A"]["n_phones"] == 2 and rates["B"]["n_phones"] == 3
    assert rates["A"]["phone_rate"] == pytest.approx(2 / 0.5)
    assert rates["A"]["silence_s"] == pytest.approx(0.1 + 0.2)
    assert rates["A"]["articulation_rate"] == pytest.approx(2 / 0.2)


def test_phone_class_duration_stats_excludes_edges() -> None:
    """Edge events are left out; classes collect their interior events."""
    dense = _sequence([("aa", 7), ("m", 8), ("iy", 12), ("n", 6), ("ae", 9)])
    stats = phone_class_duration_stats(ppg_phone_events(dense))["all"]
    assert stats["nasal"]["count"] == 2
    assert stats["nasal"]["mean_s"] == pytest.approx(0.07)
    assert stats["vowel"]["count"] == 1 and stats["vowel"]["mean_s"] == pytest.approx(0.12)
    by_speaker = phone_class_duration_stats(ppg_phone_events(dense), speaker_spans=[("A", 0.0, 0.2)])
    assert set(by_speaker["A"]) == {"nasal"}


def test_turn_transition_gap_and_propagated_uncertainty() -> None:
    """The gap runs from the last phone's offset to the next speaker's first onset, sigmas in quadrature."""
    frames = 300
    centres = _centres(frames)
    dense = np.zeros((frames, len(PHONEME_LABELS)))
    to_silence = 0.5 * (1.0 + np.tanh((centres - 1.0) / 0.02))
    to_speech = 0.5 * (1.0 + np.tanh((centres - 1.6) / 0.04))
    dense[:, INDEX["n"]] = 1.0 - to_silence
    dense[:, INDEX[SILENCE_LABEL]] = to_silence * (1.0 - to_speech)
    dense[:, INDEX["dh"]] = to_speech
    events = ppg_phone_events(dense)
    [timing, missing] = turn_transition_timing(events, [(1.05, 1.55), (2.2, 2.25)], search_s=0.2)
    assert timing.last_phone == "n" and timing.first_phone == "dh"
    assert timing.gap_s == pytest.approx(0.6, abs=0.01)
    expected_sigma = math.hypot(timing.last_phone_offset_sigma_s or 0.0, timing.first_phone_onset_sigma_s or 0.0)
    assert timing.gap_sigma_s == pytest.approx(expected_sigma)
    assert (timing.last_phone_offset_sigma_s or 0.0) < (timing.first_phone_onset_sigma_s or 0.0)
    assert missing.gap_s is None and missing.last_phone is None


def test_filler_support_separates_vowel_from_fricative() -> None:
    """A held ``ah`` + ``m`` is supported and stable; a fricative span is not."""
    dense = _sequence([("ah", 30), ("m", 15), ("s", 30)])
    um = filler_support(dense, 0.0, 0.45)
    assert um.support == pytest.approx(0.9 + 0.1 * 17 / 39, abs=1e-6)
    assert um.dominant_phone == "ah" and um.n_frames == 45
    assert um.stability > 0.95
    fricative = filler_support(dense, 0.45, 0.75)
    assert fricative.support < 0.1 and fricative.dominant_phone == "s"
    empty = filler_support(dense, 5.0, 6.0)
    assert empty.n_frames == 0 and empty.dominant_phone is None


def test_phone_classes_cover_the_ppgs_inventory() -> None:
    """Every ppgs phone has a class."""
    assert set(PHONEME_LABELS) <= set(PHONE_CLASSES)


CONVERSATION_PATH = Path(__file__).resolve().parents[2] / "data_for_testing" / "english_conversation_higgs_audio_v2.wav"


@pytest.mark.skipif(not ppgs_venv_is_provisioned(), reason="needs the ppgs subprocess venv")
def test_windowed_extraction_matches_whole_file_and_yields_events() -> None:
    """Windowed extraction agrees with one-call extraction and gives plausible phone events."""
    [audio] = resample_audios([Audio(filepath=str(CONVERSATION_PATH))], PPGS_SAMPLE_RATE)
    whole, windowed = (
        require_posteriorgram(extract_ppgs_from_audios([audio], device=DeviceType.CPU, window_s=w)[0])
        for w in (None, 6.0)
    )
    assert whole.shape == windowed.shape == (1, len(PHONEME_LABELS), audio.waveform.shape[-1] // 160)
    whole_np, windowed_np = whole[0].numpy(), windowed[0].numpy()
    np.testing.assert_allclose(windowed_np, whole_np, atol=1e-4)
    events = ppg_phone_events(windowed)
    assert (~events.is_silence).sum() > 100
    assert np.all(np.diff(events.onset_s) > 0)
    assert np.nanmedian(events.onset_sigma_s) < 0.05
    report = sparse_ppg_fidelity(windowed, sparsify_ppg(windowed))
    assert report["argmax_agreement"] > 0.999 and report["mass_retained_mean"] > 0.95
