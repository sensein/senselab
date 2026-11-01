"""Tests for scene quality across the L1/L2 split (features 20260722-175022, 20260728-221507).

L1 (``quality.py``) reports measurements in their native units and does not threshold, clamp, or
rescale. L2 (``degradation.py``) turns those into ``[0, 1]`` degradation scores against calibrated
anchors. The split is load-bearing rather than cosmetic: the previous single-layer version applied
``clip((25 - snr_db) / 20, 0, 1)`` inside L1, which returned ``0.0`` in every bucket of every real
recording because clean speech measures 60-70 dB. The measurement was fine; the clamp destroyed it.

Model-free: Brouhaha frames are constructed directly, so no gated model download is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import senselab.audio.tasks.scene_quality.brouhaha as brouhaha_mod
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames
from senselab.audio.workflows.audio_analysis.degradation import (
    bandwidth_degradation,
    reverb_degradation,
    scene_degradation,
    snr_degradation,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.quality import QUALITY_SIGNALS, harvest_quality_measurements

SR = 16000

_FAKE_SHA = "f" * 40


@pytest.fixture(autouse=True)
def _stub_commit_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these tests offline (Task 5).

    ``quality._provenance`` now resolves Brouhaha's ref to a commit SHA for every window whose
    Brouhaha signals report ``"ok"`` — most tests here supply real (synthetic) Brouhaha frames via
    ``_const_brouhaha``, so without this stub they would make a live Hub call for a gated model.
    """
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: _FAKE_SHA)


def _audio(y: np.ndarray) -> Audio:
    """Wrap a 1-D numpy signal as a mono 16 kHz ``Audio``."""
    wf = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(1, -1)
    return Audio(waveform=wf, sampling_rate=SR)


def _const_brouhaha(duration_s: float, snr_db: float, c50_db: float, hop: float = 0.02) -> BrouhahaFrames:
    """Build constant-valued Brouhaha frames spanning ``duration_s``."""
    n = max(1, int(duration_s / hop))
    return BrouhahaFrames(
        vad=np.ones(n),
        snr_db=np.full(n, snr_db),
        c50_db=np.full(n, c50_db),
        frame_hop_s=hop,
    )


def _white_noise(duration_s: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    """Return broadband white noise (full-band, deterministic)."""
    rng = np.random.default_rng(seed)
    return amp * rng.standard_normal(int(duration_s * SR))


def _lowpass_tones(duration_s: float) -> np.ndarray:
    """Return a band-limited signal (content only ≤ ~1 kHz)."""
    t = np.arange(int(duration_s * SR)) / SR
    return 0.1 * (np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 800 * t))


def _rows(audio: Audio, brouhaha: BrouhahaFrames | None, win: float = 0.5) -> list[dict]:
    return harvest_quality_measurements(audio=audio, brouhaha=brouhaha, grid=BucketGrid(win_length=win, hop_length=win))


# ── L1: measurements in native units ─────────────────────────────────────────


def test_l1_reports_snr_in_db_not_rescaled() -> None:
    """A 70 dB SNR reads 70 dB. This is the regression guard for the clamp that zeroed the column.

    The old L1 emitted ``clip((25 - 70) / 20, 0, 1) == 0.0`` here, indistinguishable from heavy
    noise pinned at the other end of the same scale.
    """
    rows = _rows(_audio(_white_noise(2.0)), _const_brouhaha(2.0, snr_db=70.0, c50_db=59.8))
    assert rows
    for r in rows:
        assert r["snr_brouhaha_db"] == pytest.approx(70.0, abs=1e-6)
        assert r["c50_brouhaha_db"] == pytest.approx(59.8, abs=1e-6)


def test_l1_preserves_full_dynamic_range() -> None:
    """Measured SNRs 75 dB apart stay 75 dB apart — no anchor compresses them."""
    quiet = _rows(_audio(_white_noise(1.0)), _const_brouhaha(1.0, snr_db=-5.0, c50_db=22.9))
    clean = _rows(_audio(_white_noise(1.0)), _const_brouhaha(1.0, snr_db=70.0, c50_db=59.8))
    spread = clean[0]["snr_brouhaha_db"] - quiet[0]["snr_brouhaha_db"]
    assert spread == pytest.approx(75.0, abs=1e-6)


def test_l1_emits_each_snr_estimator_separately() -> None:
    """Estimators are reported side by side, never reduced to one value or to their spread.

    ``primary_snr_db`` (pick brouhaha, else average the DSP metrics) was estimator *selection*,
    and ``quality_uncertainty`` was their standard deviation — but the three quantities use
    different noise-floor definitions, so their spread measured definitional disagreement rather
    than uncertainty and pinned at 1.0 structurally. Both are L2's business, given the parts.
    """
    rows = _rows(_audio(_white_noise(1.0, amp=0.15)), _const_brouhaha(1.0, snr_db=60.0, c50_db=20.0))
    r = rows[0]
    assert r["snr_brouhaha_db"] == pytest.approx(60.0)
    assert r["snr_spectral_gating_db"] is not None
    assert r["snr_peak_db"] is not None
    assert "primary_snr_db" not in r
    assert "quality_uncertainty" not in r


def test_l1_reports_no_degradation_scores() -> None:
    """No ``quality_*`` score is emitted at L1 — those are conclusions, not measurements."""
    rows = _rows(_audio(_white_noise(1.0)), _const_brouhaha(1.0, 20.0, 25.0))
    for r in rows:
        assert not [k for k in r if k.startswith("quality_")], f"L1 emitted degradation scores: {r}"


def test_l1_reports_measurements_on_silence() -> None:
    """Silence gets its measurements, not nulls.

    The old ``rms < 1e-4 -> null everything`` gate was a threshold, and it discarded brouhaha's
    actual reading. Brouhaha reports ~43 dB SNR on digital silence, which is a meaningless answer
    that L2 must be able to see in order to distrust; nulling it hides the evidence.
    """
    rows = _rows(_audio(np.zeros(int(1.0 * SR))), _const_brouhaha(1.0, snr_db=43.5, c50_db=31.1))
    assert rows
    for r in rows:
        assert r["snr_brouhaha_db"] == pytest.approx(43.5, abs=1e-6)
        assert r["rms"] is not None and r["rms"] < 1e-6


def test_l1_bandwidth_is_a_frequency() -> None:
    """Roll-off is reported in Hz, not as an inverted-and-clamped badness score."""
    full = _rows(_audio(_white_noise(1.5)), None)
    band = _rows(_audio(_lowpass_tones(1.5)), None)
    full_hz = np.nanmean([r["rolloff_95_hz"] for r in full if r["rolloff_95_hz"] is not None])
    band_hz = np.nanmean([r["rolloff_95_hz"] for r in band if r["rolloff_95_hz"] is not None])
    assert band_hz < full_hz
    assert band_hz < 2000.0  # content is ≤ 1 kHz
    assert full_hz > 5000.0  # full-band noise rolls off near Nyquist (8 kHz)


def test_l1_clipping_is_a_proportion() -> None:
    """``proportion_clipped`` passes through as measured."""
    clean = _rows(_audio(_white_noise(1.0, amp=0.2)), None)
    clipped = _rows(_audio(np.clip(_white_noise(1.0, amp=2.0), -1.0, 1.0)), None)
    assert max(r["proportion_clipped"] for r in clipped) > max(r["proportion_clipped"] for r in clean)
    for r in clean + clipped:
        assert 0.0 <= r["proportion_clipped"] <= 1.0  # a proportion, by definition


def test_l1_null_when_brouhaha_absent() -> None:
    """FR-023: no Brouhaha → its two columns are null; the DSP measurements still land."""
    rows = _rows(_audio(_white_noise(1.0)), None)
    assert rows
    assert all(r["snr_brouhaha_db"] is None and r["c50_brouhaha_db"] is None for r in rows)
    assert any(r["snr_spectral_gating_db"] is not None for r in rows)
    assert any(r["rolloff_95_hz"] is not None for r in rows)


def test_l1_provenance_declares_units_for_every_signal() -> None:
    """Each measurement carries units, model, and native resolution — the point of the envelope."""
    rows = _rows(_audio(_white_noise(1.0)), _const_brouhaha(1.0, 20.0, 25.0))
    prov = rows[0]["provenance"]
    assert set(prov) == set(QUALITY_SIGNALS)
    expected_units = {
        "snr_brouhaha_db": "dB",
        "c50_brouhaha_db": "dB",
        "snr_spectral_gating_db": "dB",
        "snr_peak_db": "dB",
        "rolloff_95_hz": "hertz",
        "proportion_clipped": "proportion",
        "rms": "arbitrary",
    }
    for name, units in expected_units.items():
        assert prov[name]["units"] == units, f"{name} declared {prov[name]['units']!r}, expected {units!r}"
        assert prov[name]["model"]
        assert prov[name]["resolution_s"] == pytest.approx(0.25)
        assert prov[name]["window_s"] == pytest.approx(0.5)


def test_l1_brouhaha_provenance_carries_the_resolved_commit_not_just_the_ref() -> None:
    """Task 5: a reader must be able to tell the commit that produced the Brouhaha measurements.

    ``revision`` alone (the ref, e.g. "main") cannot distinguish a deliberate pin from a tracked
    ref that happened to resolve there on the day this ran — that ambiguity is exactly what this
    task exists to remove, so both fields must travel together.
    """
    rows = _rows(_audio(_white_noise(1.0)), _const_brouhaha(1.0, 20.0, 25.0))
    prov = rows[0]["provenance"]
    for name in ("snr_brouhaha_db", "c50_brouhaha_db"):
        assert prov[name]["revision"] == "main"
        assert prov[name]["commit_sha"] == _FAKE_SHA
        assert len(prov[name]["commit_sha"]) == 40
    # Non-Brouhaha signals have nothing pinned, so neither field should claim one.
    for name in ("snr_spectral_gating_db", "snr_peak_db", "rolloff_95_hz", "proportion_clipped", "rms"):
        assert prov[name]["revision"] is None
        assert prov[name]["commit_sha"] is None


def test_l1_brouhaha_commit_sha_is_none_when_the_signal_is_unavailable() -> None:
    """No Brouhaha frames → status is "unavailable", so nothing is resolved and nothing is claimed."""
    rows = _rows(_audio(_white_noise(1.0)), None)
    prov = rows[0]["provenance"]
    assert prov["snr_brouhaha_db"]["status"] == "unavailable"
    assert prov["snr_brouhaha_db"]["commit_sha"] is None


# ── L2: degradation from measurements ────────────────────────────────────────


def test_l2_snr_degradation_spans_the_anchors() -> None:
    """SC-001/SC-002 at L2: clean reads ~0, heavy noise saturates, mid-SNR discriminates."""
    assert snr_degradation(70.0) == pytest.approx(0.0)
    assert snr_degradation(30.0) == pytest.approx(0.0)
    assert snr_degradation(3.0) == pytest.approx(1.0)
    assert snr_degradation(15.0) == pytest.approx(0.5)
    assert snr_degradation(None) is None


def test_l2_noised_region_degrades_more_than_clean() -> None:
    """SC-002 end to end: L1 measures dB, L2 separates the noisy half by ≥0.3."""
    duration, hop = 2.0, 0.02
    n = int(duration / hop)
    snr = np.full(n, 30.0)
    snr[: n // 2] = 3.0
    brouhaha = BrouhahaFrames(vad=np.ones(n), snr_db=snr, c50_db=np.full(n, 25.0), frame_hop_s=hop)
    rows = _rows(_audio(_white_noise(duration)), brouhaha)
    scored = [scene_degradation(r, sampling_rate=SR) for r in rows]
    noisy = [s["quality_snr"] for s, r in zip(scored, rows) if r["end"] <= 1.0]
    clean = [s["quality_snr"] for s, r in zip(scored, rows) if r["start"] >= 1.0]
    assert min(noisy) - max(clean) >= 0.3


def test_l2_reverb_and_bandwidth_degradation() -> None:
    """C50 and roll-off map to degradation only at L2, using their own anchors."""
    assert reverb_degradation(59.8) == pytest.approx(0.0)
    assert reverb_degradation(-5.0) == pytest.approx(1.0)
    near_nyquist = bandwidth_degradation(7600.0, sampling_rate=SR)
    telephone = bandwidth_degradation(3400.0, sampling_rate=SR)
    assert near_nyquist is not None and telephone is not None
    assert near_nyquist < 0.1
    assert telephone > 0.5
    assert bandwidth_degradation(None, sampling_rate=SR) is None


def test_l2_calibration_profile_overrides_anchors() -> None:
    """The anchors are calibration, so a fitted profile replaces them at L2."""
    assert snr_degradation(70.0, clean_db=80.0, floor_db=60.0) == pytest.approx(0.5)
    row = {"snr_brouhaha_db": 70.0, "c50_brouhaha_db": None, "rolloff_95_hz": None, "proportion_clipped": None}
    scored = scene_degradation(row, sampling_rate=SR, calibration={"snr_clean_db": 80.0, "snr_floor_db": 60.0})
    assert scored["quality_snr"] == pytest.approx(0.5)


def test_l2_scores_stay_in_unit_range() -> None:
    """Every non-null L2 score is a bounded degradation score."""
    rows = _rows(_audio(_white_noise(2.0)), _const_brouhaha(2.0, 20.0, 25.0))
    for r in rows:
        scored = scene_degradation(r, sampling_rate=SR)
        assert scored["snr_source"] == "snr_brouhaha_db"  # attribution, not a score
        for key, v in scored.items():
            if key == "snr_source":
                continue
            assert v is None or 0.0 <= v <= 1.0, f"{key}={v} out of range"


def test_l2_null_measurement_yields_null_score() -> None:
    """A missing measurement stays missing — it must not become a confident 0.0."""
    row = {"snr_brouhaha_db": None, "c50_brouhaha_db": None, "rolloff_95_hz": None, "proportion_clipped": None}
    scored = scene_degradation(row, sampling_rate=SR)
    assert all(v is None for v in scored.values())


# ── Brouhaha extractor plumbing ──────────────────────────────────────────────


def test_brouhaha_null_safe_when_venv_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-023: if the Brouhaha venv can't be built, extract → [None], no raise."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("uv failed to build the brouhaha venv")

    monkeypatch.setattr(brouhaha_mod, "ensure_venv", _boom)
    assert brouhaha_mod.extract_brouhaha_frames([_audio(_white_noise(0.5))]) == [None]


def test_brouhaha_assembles_frames_from_worker_output(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    """The subprocess results (per-chunk .npy) are loaded + stitched into BrouhahaFrames."""
    npy = tmp_path / "chunk_0_0.npy"  # type: ignore[operator]
    frames = np.stack([np.full(50, 0.9), np.full(50, 22.0), np.full(50, 27.0)], axis=1)  # (50, 3)
    np.save(npy, frames)

    monkeypatch.setattr(brouhaha_mod, "ensure_venv", lambda *a, **k: "/fake/venv")
    monkeypatch.setattr(brouhaha_mod, "venv_python", lambda *a, **k: "/fake/venv/bin/python")
    monkeypatch.setattr(
        brouhaha_mod,
        "parse_subprocess_result",
        lambda *a, **k: {"results": [{"npy": str(npy), "start_s": 0.0, "audio_idx": 0, "hop": 0.02}]},
    )
    monkeypatch.setattr(brouhaha_mod.subprocess, "run", lambda *a, **k: None)

    out = brouhaha_mod.extract_brouhaha_frames([_audio(_white_noise(1.0))])
    assert len(out) == 1
    bf = out[0]
    assert bf is not None
    assert abs(bf.frame_hop_s - 0.02) < 1e-9
    assert np.allclose(bf.vad, 0.9) and np.allclose(bf.snr_db, 22.0) and np.allclose(bf.c50_db, 27.0)


# ── register item 24 / H1: resample onto the reporting grid, don't copy ──────


def _grid_rows(monkeypatch: pytest.MonkeyPatch, per_window: list[dict[str, object]], grid_s: float) -> list[dict]:
    """Drive ``harvest_quality_measurements`` with scripted analysis windows."""
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis import quality as q
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid

    calls = {"i": 0}

    def fake_window(slice_audio, brouhaha, start, end):  # noqa: ANN001, ANN202
        payload = dict.fromkeys(q.QUALITY_SIGNALS)
        payload.update(per_window[min(calls["i"], len(per_window) - 1)])
        calls["i"] += 1
        payload["provenance"] = {
            n: {"units": "decibel", "resolution_s": q.QUALITY_ANALYSIS_HOP_S} for n in q.QUALITY_SIGNALS
        }
        return payload

    monkeypatch.setattr(q, "_analysis_window", fake_window)
    audio = Audio(waveform=torch.zeros(1, 16000 * 2), sampling_rate=16000)
    return q.harvest_quality_measurements(audio=audio, brouhaha=None, grid=BucketGrid(grid_s, grid_s))


def test_a_coarser_bucket_integrates_its_analysis_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finer → coarser is an average, not a pick.

    At the 0.5 s / 0.25 s analysis grid a 1 s reporting bucket covers several windows. Copying the
    nearest one keeps a single measurement and discards the rest, and which one survives is an
    artefact of where the bucket centre happened to fall.
    """
    # Widening gaps on purpose: with an evenly spaced series the average of a run coincides with
    # one of its members, and a copy would be indistinguishable from an integral.
    levels = [10.0, 20.0, 33.0, 47.0, 62.0, 78.0, 95.0, 113.0]
    rows = _grid_rows(monkeypatch, [{"snr_brouhaha_db": v} for v in levels], grid_s=1.0)
    assert rows, "expected reporting buckets"
    first = rows[0]["snr_brouhaha_db"]
    assert first is not None
    assert first not in levels, "a single window's value survived the reduction"
    assert min(levels) < first < max(levels)


def test_a_failed_estimator_in_some_windows_does_not_poison_the_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Averaging must skip the windows that measured nothing, not average in a hole."""
    windows: list[dict[str, object]] = [
        {"snr_brouhaha_db": 20.0},
        {"snr_brouhaha_db": None},
        {"snr_brouhaha_db": 40.0},
        {"snr_brouhaha_db": None},
    ]
    rows = _grid_rows(monkeypatch, windows * 4, grid_s=1.0)
    value = rows[0]["snr_brouhaha_db"]
    assert value is not None and np.isfinite(value)
    assert value == pytest.approx(30.0, abs=6.0)


def test_a_bucket_no_estimator_reached_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No measurement is not a measurement of zero."""
    rows = _grid_rows(monkeypatch, [{"snr_brouhaha_db": None}], grid_s=1.0)
    assert rows[0]["snr_brouhaha_db"] is None


def test_provenance_still_declares_the_analysis_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resampled row must not look like it was measured at the reporting rate.

    On a grid finer than the analysis hop the same measurement is repeated across buckets; the
    declared resolution is what stops a consumer counting those repeats as independent evidence.
    """
    rows = _grid_rows(monkeypatch, [{"snr_brouhaha_db": 12.0}], grid_s=0.1)
    prov = rows[0]["provenance"]["snr_brouhaha_db"]
    from senselab.audio.workflows.audio_analysis import quality as q

    assert prov["resolution_s"] == pytest.approx(q.QUALITY_ANALYSIS_HOP_S)
    assert prov["resolution_s"] > 0.1, "the row must not claim the reporting grid's resolution"
