"""Audio-variant level provenance (T007, FR-012/FR-017d/FR-019).

Every scene-analysis result must be attributable to the audio variant and gain it was
computed from (SC-006). Three invariants matter most:

- **Gain is capped, and exceeding the cap is an error rather than a silent clamp.** A
  clamped gain would make the recorded provenance wrong, which is worse than failing.
- **Clipping and requantization are reported, never silent.** Amplifying past full scale
  makes the classifiers respond to distortion; amplifying after a lossy write amplifies a
  quantization noise floor that is statistically indistinguishable from real broadband
  noise.
- **The same normalization scalar applies to every variant of one recording.** Independent
  renormalization would corrupt the cross-variant comparison the whole feature rests on.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.level import (
    AudioVariant,
    GainCapExceededError,
    clipped_fraction,
    integrated_lufs,
    loudness_range_lu,
    measure_variant,
    true_peak_dbtp,
    write_level_json,
)


@pytest.fixture(autouse=True)
def _stub_commit_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub HF revision resolution so provenance never touches the network.

    The stage context resolves any Hub-shaped model id to a commit SHA before it
    can record provenance or compute a cache key. Tests here use real-looking ids,
    so without this stub the outcome would depend on whether this machine happens
    to have those repos cached -- verified by running under ``HF_HUB_OFFLINE=1``
    with an empty ``HF_HUB_CACHE``, where the unstubbed form fails outright.
    """
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda repo_id, ref="main", **kw: "f" * 40)


SR = 16000


def _sine(seconds: float = 3.0, freq: float = 997.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2.0 * math.pi * freq * t)).astype(np.float64)


def _noise(seconds: float = 3.0, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float64)


# ── loudness measurement ──────────────────────────────────────────────


def test_full_scale_997hz_sine_reads_near_minus_three_lufs() -> None:
    """BS.1770 pins a full-scale 997 Hz sine at -3.01 LKFS by construction.

    The -0.691 offset in the standard exists precisely to cancel the K-weighting gain at
    997 Hz, so this is a calibration check on the whole measurement path, not a guess.
    """
    assert integrated_lufs(_sine(), SR) == pytest.approx(-3.01, abs=0.2)


def test_halving_amplitude_drops_loudness_by_six_db() -> None:
    """Loudness is a level measure: -6 dB of amplitude is -6 LU."""
    loud = integrated_lufs(_sine(amp=1.0), SR)
    quiet = integrated_lufs(_sine(amp=0.5), SR)
    assert loud - quiet == pytest.approx(6.02, abs=0.2)


def test_silence_returns_negative_infinity_not_a_number() -> None:
    """Digital silence has no loudness; -inf is honest where a large negative is not."""
    assert integrated_lufs(np.zeros(SR, dtype=np.float64), SR) == -math.inf


def test_loudness_range_is_non_negative() -> None:
    """LRA is P95 - P10 of gated short-term loudness, so it cannot be negative."""
    assert loudness_range_lu(_noise(seconds=5.0), SR) >= 0.0


def test_loudness_range_larger_for_varying_material() -> None:
    """A quiet-then-loud signal has a wider loudness range than a steady one."""
    steady = _noise(seconds=6.0, amp=0.1)
    varying = np.concatenate([_noise(seconds=3.0, amp=0.005, seed=1), _noise(seconds=3.0, amp=0.3, seed=2)])
    assert loudness_range_lu(varying, SR) > loudness_range_lu(steady, SR)


# ── true peak and clipping (FR-017d) ──────────────────────────────────


def test_true_peak_of_full_scale_sine_is_about_zero_dbtp() -> None:
    """A full-scale sine peaks at 0 dBTP by definition."""
    assert true_peak_dbtp(_sine(), SR) == pytest.approx(0.0, abs=0.5)


def test_true_peak_detects_inter_sample_overshoot() -> None:
    """Oversampling is the point: a signal under full scale per-sample can still exceed it."""
    assert true_peak_dbtp(_sine(amp=0.99, freq=5000.0), SR) > -1.0


def test_clipped_fraction_zero_when_within_range() -> None:
    """Nothing is clipped when every sample sits inside full scale."""
    assert clipped_fraction(_sine(amp=0.5)) == pytest.approx(0.0)


def test_clipped_fraction_counts_samples_at_or_beyond_full_scale() -> None:
    """A 10x gain on a 0.63-peak signal pins a large share of samples."""
    wav = np.clip(_sine(amp=0.63) * 10.0, -1.0, 1.0)
    assert clipped_fraction(wav) > 0.25


# ── variant measurement and the gain cap (FR-019) ─────────────────────


def test_measure_variant_records_name_and_gain() -> None:
    """The variant record carries the two fields every downstream reference needs."""
    v = measure_variant("unmodified", _noise(), SR, gain_db=0.0, gain_cap_db=10.0)
    assert v.name == "unmodified"
    assert v.gain_db == pytest.approx(0.0)
    assert v.measured_lufs < 0.0


def test_gain_above_cap_raises_rather_than_clamping() -> None:
    """A silently clamped gain makes the recorded provenance a lie."""
    with pytest.raises(GainCapExceededError, match="gain_cap_db"):
        measure_variant("foreground_suppressed", _noise(), SR, gain_db=15.0, gain_cap_db=10.0)


def test_gain_at_cap_accepted() -> None:
    """The cap itself is allowed; only exceeding it is not."""
    v = measure_variant("foreground_suppressed", _noise(amp=0.01), SR, gain_db=10.0, gain_cap_db=10.0)
    assert v.gain_db == pytest.approx(10.0)


def test_requantized_flag_is_carried_through() -> None:
    """A lossy write in the input path must be visible downstream (FR-019b)."""
    v = measure_variant("unmodified", _noise(), SR, gain_db=0.0, gain_cap_db=10.0, requantized=True)
    assert v.requantized is True


def test_per_segment_gain_recorded() -> None:
    """Gain is applied per segment, so provenance is per segment too (FR-019a)."""
    segs = [{"start": 0.0, "end": 1.0, "gain_db": 3.0}, {"start": 1.0, "end": 2.0, "gain_db": 8.0}]
    v = measure_variant("foreground_suppressed", _noise(), SR, gain_db=8.0, gain_cap_db=10.0, per_segment_gain_db=segs)
    assert [s["gain_db"] for s in v.per_segment_gain_db] == [3.0, 8.0]


def test_per_segment_gain_above_cap_raises() -> None:
    """The cap binds per segment, not only on the summary value."""
    segs = [{"start": 0.0, "end": 1.0, "gain_db": 30.0}]
    with pytest.raises(GainCapExceededError):
        measure_variant("foreground_suppressed", _noise(), SR, gain_db=5.0, gain_cap_db=10.0, per_segment_gain_db=segs)


# ── level.json contract (SC-006) ───────────────────────────────────────


def test_write_level_json_shape(tmp_path: Path) -> None:
    """level.json carries the target, the cap, and every field a consumer needs."""
    variants = [
        measure_variant("unmodified", _noise(), SR, gain_db=0.0, gain_cap_db=10.0),
        measure_variant("foreground_suppressed", _noise(amp=0.02), SR, gain_db=8.0, gain_cap_db=10.0),
    ]
    out = write_level_json(tmp_path, target_lufs=-23.0, gain_cap_db=10.0, variants=variants)
    doc = json.loads(out.read_text())
    assert doc["target_lufs"] == pytest.approx(-23.0)
    assert doc["gain_cap_db"] == pytest.approx(10.0)
    assert {v["name"] for v in doc["variants"]} == {"unmodified", "foreground_suppressed"}
    for v in doc["variants"]:
        for key in ("gain_db", "measured_lufs", "lra_lu", "true_peak_dbtp", "clipped_fraction", "requantized"):
            assert key in v, f"{v['name']} missing {key}"


def test_write_level_json_rejects_duplicate_variant_names(tmp_path: Path) -> None:
    """Two variants with one name makes every downstream reference ambiguous."""
    v = measure_variant("unmodified", _noise(), SR, gain_db=0.0, gain_cap_db=10.0)
    with pytest.raises(ValueError, match="duplicate"):
        write_level_json(tmp_path, target_lufs=-23.0, gain_cap_db=10.0, variants=[v, v])


def test_write_level_json_requires_at_least_one_variant(tmp_path: Path) -> None:
    """A level.json with no variants attributes nothing."""
    with pytest.raises(ValueError, match="variant"):
        write_level_json(tmp_path, target_lufs=-23.0, gain_cap_db=10.0, variants=[])


def test_json_is_serializable_with_infinite_loudness(tmp_path: Path) -> None:
    """Silence measures -inf, which is not valid JSON — it must be encoded, not crash."""
    v = measure_variant("unmodified", np.zeros(SR, dtype=np.float64), SR, gain_db=0.0, gain_cap_db=10.0)
    out = write_level_json(tmp_path, target_lufs=-23.0, gain_cap_db=10.0, variants=[v])
    doc = json.loads(out.read_text())
    assert doc["variants"][0]["measured_lufs"] is None


# ── shared normalization scalar (FR-019c) ─────────────────────────────


def test_normalization_gain_for_target_is_deterministic() -> None:
    """Same input and target produce the same scalar every time."""
    from senselab.audio.workflows.audio_analysis.level import normalization_gain_db

    wav = _noise(seconds=4.0)
    assert normalization_gain_db(wav, SR, target_lufs=-23.0) == pytest.approx(
        normalization_gain_db(wav, SR, target_lufs=-23.0)
    )


def test_normalization_gain_moves_measured_loudness_to_target() -> None:
    """Applying the returned gain lands the measured loudness on the target."""
    from senselab.audio.workflows.audio_analysis.level import apply_gain_db, normalization_gain_db

    wav = _noise(seconds=4.0, amp=0.02)
    g = normalization_gain_db(wav, SR, target_lufs=-23.0)
    assert integrated_lufs(apply_gain_db(wav, g), SR) == pytest.approx(-23.0, abs=0.3)


def test_normalization_gain_of_silence_is_zero_not_infinite() -> None:
    """Silence cannot be normalized toward a target; the honest answer is no gain."""
    from senselab.audio.workflows.audio_analysis.level import normalization_gain_db

    assert normalization_gain_db(np.zeros(SR, dtype=np.float64), SR, target_lufs=-23.0) == pytest.approx(0.0)


def test_apply_gain_db_is_exact_roundtrip() -> None:
    """Attenuate-then-amplify is bit-exact in float, which is why gain recovers nothing."""
    from senselab.audio.workflows.audio_analysis.level import apply_gain_db

    wav = _noise(seconds=1.0)
    back = apply_gain_db(apply_gain_db(wav, -40.0), 40.0)
    assert np.max(np.abs(back - wav)) < 1e-12


# ── StageContext variant provenance (T009, FR-012 / SC-006) ────────────


def test_stage_context_defaults_to_unmodified_variant() -> None:
    """A context that never mentions a variant still attributes its results."""
    from senselab.audio.workflows.audio_analysis.stage_context import StageContext

    ctx = StageContext(perturbation="raw", audio_signature="deadbeef")
    assert ctx.variant == "unmodified"
    assert ctx.variant_gain_db == pytest.approx(0.0)


def test_stage_context_records_variant_and_gain_in_provenance() -> None:
    """Every stage outcome carries the variant and gain it was computed from."""
    from senselab.audio.workflows.audio_analysis.stage_context import StageContext

    ctx = StageContext(
        perturbation="suppressed_16k",
        audio_signature="cafe",
        variant="foreground_suppressed",
        variant_gain_db=8.0,
    )
    prov = ctx.provenance_for("ast", "MIT/ast-finetuned-audioset-10-10-0.4593", {"win_length": 10.24})
    assert prov["variant"] == "foreground_suppressed"
    assert prov["variant_gain_db"] == pytest.approx(8.0)


def test_stage_context_rejects_unknown_variant() -> None:
    """A typo'd variant name would make provenance unjoinable, so it fails at construction."""
    from senselab.audio.workflows.audio_analysis.stage_context import StageContext

    with pytest.raises(ValueError, match="variant"):
        StageContext(perturbation="p", audio_signature="s", variant="enhanced")


def test_stage_versions_declare_the_new_stages() -> None:
    """A new stage must declare its own invalidation counter rather than borrow one."""
    from senselab.audio.workflows.audio_analysis.stage_context import STAGE_VERSIONS, stage_code_version

    for stage in ("background_mask", "noise_floor", "background_sources", "level_probe"):
        assert stage in STAGE_VERSIONS, f"{stage} missing from STAGE_VERSIONS"
        assert stage_code_version(stage).startswith(f"{stage}@")


# ── peak-limited gain (FR-019, found on real audio) ───────────────────


def test_high_crest_factor_gain_is_limited_by_true_peak() -> None:
    """A quiet-median recording with peaks at full scale cannot reach a loudness target.

    Found on a real 14 s recording: integrated -30.5 LUFS with true peak already at
    -0.31 dBTP. The +7.5 dB needed to reach -23 LUFS passes the gain cap but would drive
    the peak to +7.2 dBTP. Loudness alone is not a sufficient gain policy.
    """
    from senselab.audio.workflows.audio_analysis.level import peak_limited_gain_db

    # transient at full scale over a quiet bed — the crest-factor shape
    wav = _noise(seconds=4.0, amp=0.004)
    wav[1000:1050] = 0.999
    gain, binding = peak_limited_gain_db(wav, SR, target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, gain_cap_db=10.0)
    assert binding == "true_peak"
    assert gain < 0.0, "a peak already at full scale leaves no headroom to add"


def test_gain_cap_binds_when_headroom_is_ample() -> None:
    """Very quiet audio with clean headroom is limited by policy, not by peak."""
    from senselab.audio.workflows.audio_analysis.level import peak_limited_gain_db

    # amp chosen to stay above BS.1770's -70 LUFS absolute gate: below it loudness is
    # unmeasurable and a different branch applies (see the test below).
    gain, binding = peak_limited_gain_db(
        _noise(seconds=4.0, amp=0.001), SR, target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, gain_cap_db=10.0
    )
    assert binding == "gain_cap"
    assert gain == pytest.approx(10.0)


def test_unmeasurable_loudness_is_reported_as_such() -> None:
    """Below BS.1770's absolute gate there is no target to normalize toward.

    Reporting "target" would claim the material was already at the target when nothing
    could be measured, making an unnormalized variant look like a normalized one.
    """
    from senselab.audio.workflows.audio_analysis.level import peak_limited_gain_db

    _gain, binding = peak_limited_gain_db(
        _noise(seconds=4.0, amp=1e-6), SR, target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, gain_cap_db=10.0
    )
    assert binding == "unmeasurable"


def test_target_binds_for_ordinary_material() -> None:
    """The common case: the loudness target is reachable within both other limits."""
    from senselab.audio.workflows.audio_analysis.level import peak_limited_gain_db

    gain, binding = peak_limited_gain_db(
        _noise(seconds=4.0, amp=0.05), SR, target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, gain_cap_db=20.0
    )
    assert binding == "target"


def test_applying_the_limited_gain_never_exceeds_the_ceiling() -> None:
    """The property that matters: no variant reaches a classifier clipped."""
    from senselab.audio.workflows.audio_analysis.level import apply_gain_db, peak_limited_gain_db

    for amp in (1e-4, 0.004, 0.05, 0.5):
        wav = _noise(seconds=3.0, amp=amp)
        wav[500:520] = min(0.999, amp * 200)
        gain, _ = peak_limited_gain_db(wav, SR, target_lufs=-23.0, true_peak_ceiling_dbtp=-1.0, gain_cap_db=10.0)
        assert true_peak_dbtp(apply_gain_db(wav, gain), SR) <= -1.0 + 1e-6, f"ceiling breached at amp={amp}"
