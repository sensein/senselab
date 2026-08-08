"""US5 (T035): calibration profile round-trip, mapping monotonicity, estimator sweep."""

import json
from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis import calibration as cal_mod
from senselab.audio.workflows.audio_analysis.calibration import (
    DEFAULT_PROFILE,
    PROFILE_VERSION,
    linear_db_to_unit,
    load_calibration_profile,
    profile_to_runtime,
    validate_profile,
)


def test_default_profile_when_no_path_and_no_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Absent profile *and* absent bundle → documented defaults (deep-copied).

    The bundle location is redirected rather than assumed missing: a fitted
    profile now ships with the package, so relying on "no file exists" would
    make this test assert the bundled path instead of the fallback it names.
    """
    monkeypatch.setattr(cal_mod, "BUNDLED_PROFILE_PATH", tmp_path / "absent.json")
    profile = load_calibration_profile(None)
    assert profile["version"] == PROFILE_VERSION
    assert profile["snr"]["clean_db"] == DEFAULT_PROFILE["snr"]["clean_db"]
    profile["snr"]["clean_db"] = -999  # must not leak into the module constant
    assert DEFAULT_PROFILE["snr"]["clean_db"] != -999


def test_bundled_profile_preferred_over_defaults() -> None:
    """The shipped fitted profile is the default when no path is passed.

    Guards the precedence in load_calibration_profile: a package that ships a
    fitted profile must use it, otherwise every install silently falls back to
    the uncalibrated anchors.
    """
    assert cal_mod.BUNDLED_PROFILE_PATH.exists(), "a fitted profile should ship with the package"
    profile = load_calibration_profile(None)
    assert profile["provenance"]["fitted_by"] == "scripts/calibrate_scene_quality.py"
    # Fitted anchors live in the estimator's own output space, which is
    # compressed relative to true dB — so they must differ from the defaults.
    assert profile["snr"]["clean_db"] != DEFAULT_PROFILE["snr"]["clean_db"]


def test_bundled_profile_maps_to_runtime_keys() -> None:
    """The shipped profile survives validation and yields the flat consumer keys."""
    runtime = profile_to_runtime(load_calibration_profile(None))
    for key in ("snr_clean_db", "snr_floor_db", "c50_clean_db", "c50_floor_db"):
        assert isinstance(runtime[key], float)
    assert runtime["snr_clean_db"] > runtime["snr_floor_db"]
    assert runtime["c50_clean_db"] > runtime["c50_floor_db"]


def test_profile_round_trip_and_validation(tmp_path: Path) -> None:
    """T035: profile round-trips load/apply; malformed profiles fail loudly."""
    profile = {
        "version": "1",
        "snr": {"type": "linear_db_to_unit", "clean_db": 22.0, "floor_db": 3.0},
        "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 28.0, "floor_db": -4.0},
        "bandwidth": {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.95},
        "temperature": {"speech_presence": 1.2, "asr": 0.8},
        "token_entropy_reference_nats": 2.5,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    loaded = load_calibration_profile(path)
    assert loaded == profile

    runtime = profile_to_runtime(loaded)
    assert runtime["snr_clean_db"] == 22.0 and runtime["snr_floor_db"] == 3.0
    assert runtime["c50_clean_db"] == 28.0 and runtime["c50_floor_db"] == -4.0
    # Carried through the bridge and validated, but read by no fold today: the aggregators that
    # consumed them had no production caller and are gone, and ``fuse.fuse_axis`` takes no
    # temperature. Asserting the passthrough keeps the fitted values reaching the runtime dict, so
    # wiring them into the fold later is one edit rather than a refit.
    assert runtime["temperature"] == {"speech_presence": 1.2, "asr": 0.8}
    assert runtime["token_entropy_reference_nats"] == 2.5

    with pytest.raises(ValueError, match="version"):
        validate_profile({**profile, "version": "0"})
    with pytest.raises(ValueError, match="clean_db must exceed"):
        validate_profile({**profile, "snr": {"type": "linear_db_to_unit", "clean_db": 3.0, "floor_db": 22.0}})
    with pytest.raises(ValueError, match="temperature"):
        validate_profile({**profile, "temperature": {"speech_presence": 0.0}})


def test_linear_db_to_unit_anchors_and_monotonicity() -> None:
    """0 at clean anchor, 1 at floor anchor, monotone nonincreasing in dB."""
    assert linear_db_to_unit(25.0, 25.0, 5.0) == 0.0
    assert linear_db_to_unit(5.0, 25.0, 5.0) == 1.0
    assert linear_db_to_unit(40.0, 25.0, 5.0) == 0.0  # clipped clean side
    assert linear_db_to_unit(-10.0, 25.0, 5.0) == 1.0  # clipped floor side
    values = [linear_db_to_unit(db, 25.0, 5.0) for db in range(30, -6, -5)]
    assert values == sorted(values)  # degradation rises as dB falls
    with pytest.raises(ValueError):
        linear_db_to_unit(10.0, 5.0, 25.0)


def test_estimator_sweep_monotonic_reported_vs_true() -> None:
    """T035/SC-007 (env-gated): degradation rises monotonically as true SNR falls.

    Crosses the L1/L2 boundary deliberately: L1 measures dB, L2 scores it. Monotonicity has to
    survive both, since either layer could destroy it — L1 by saturating the measurement, L2 by
    picking anchors outside the swept range.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("librosa")
    import numpy as np

    from senselab.audio.data_structures import Audio
    from senselab.audio.workflows.audio_analysis.degradation import scene_degradation
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.quality import harvest_quality_measurements

    rng = np.random.default_rng(0)
    sr = 16000
    # Synthetic "speech": amplitude-modulated tone bursts (voiced-ish energy structure).
    t = np.arange(sr * 3) / sr
    speech = (0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    speech_power = float(np.mean(speech**2))

    reported = []
    for true_snr_db in (30.0, 20.0, 10.0, 0.0):
        noise = rng.normal(0, 1, speech.shape).astype("float32")
        noise *= np.sqrt(speech_power / (10 ** (true_snr_db / 10)) / max(1e-12, float(np.mean(noise**2))))
        mixed = np.clip(speech + noise, -1, 1)
        audio = Audio(waveform=torch.from_numpy(mixed).unsqueeze(0), sampling_rate=sr)
        rows = harvest_quality_measurements(audio=audio, brouhaha=None, grid=BucketGrid(0.5, 0.5))
        scored = [scene_degradation(r, sampling_rate=sr) for r in rows]
        # No Brouhaha here, so the documented fallback should be in use and say so.
        assert {s["snr_source"] for s in scored} == {"snr_spectral_gating_db"}
        vals = [s["quality_snr"] for s in scored if s.get("quality_snr") is not None]
        assert vals, f"no quality_snr at true SNR {true_snr_db}"
        reported.append(float(np.median(vals)))
    assert reported == sorted(reported), f"degradation not monotone vs falling SNR: {reported}"
