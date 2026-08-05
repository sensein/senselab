"""Scene-quality calibration profiles (US5, T036 — data-model.md §5).

A ``CalibrationProfile`` is a small versioned JSON mapping raw estimator outputs
(dB) onto the workflow's ``[0, 1]`` degradation scores, plus per-axis
temperatures for the uncertainty aggregators:

```json
{
  "version": "1",
  "snr":        {"type": "linear_db_to_unit", "clean_db": 25.0, "floor_db": 5.0},
  "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": -5.0},
  "bandwidth":  {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.95},
  "temperature": {"speech_presence": 1.0, "asr": 1.0}
}
```

Profiles are fitted by ``scripts/calibrate_scene_quality.py`` from synthetic
mixtures (research.md D9). The dB anchors are consumed at runtime by ``quality.py`` /
``degradation.py`` (flat ``*_clean_db``/``*_floor_db`` keys); :func:`profile_to_runtime` bridges the
versioned on-disk shape to that flat runtime convention. Absent profile →
:data:`DEFAULT_PROFILE`, which mirrors the documented uncalibrated defaults in
``quality.py`` (bounded, not fitted).

**``temperature`` and ``token_entropy_reference_nats`` currently reach no fold.** Their only
consumers were ``aggregate.aggregate_asr`` and ``aggregate.aggregate_speech_presence``, which had no
production caller and are deleted; the run's single fold is ``fuse.fuse_axis``, which takes no
temperature. They stay in the schema, validated, because the *question* they answer is real — two
backends' confidences are not on a common scale — and dropping the fields would lose the fitted
values already on disk. But they are declared-and-unread until ``fuse_axis`` takes them, and
``axes.CALIBRATED_AXES`` names the axes that would receive them; see the note there.

Stdlib-only; safe to import anywhere.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.axes import CALIBRATED_AXES

PROFILE_VERSION = "1"

BUNDLED_PROFILE_PATH = Path(__file__).parent / "data" / "scene_quality_calibration.json"

DEFAULT_PROFILE: dict[str, Any] = {
    "version": PROFILE_VERSION,
    # Anchors mirror quality.py's documented defaults (speech-appropriate SNR window).
    "snr": {"type": "linear_db_to_unit", "clean_db": 25.0, "floor_db": 5.0},
    "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": -5.0},
    "bandwidth": {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.95},
    "temperature": {"speech_presence": 1.0, "asr": 1.0},
}


def linear_db_to_unit(value_db: float, clean_db: float, floor_db: float) -> float:
    """Map a dB value to ``[0, 1]`` degradation (0 at ``clean_db``, 1 at ``floor_db``).

    The pure form of ``quality._linear_db_to_degradation``, exposed for the
    fitting script's round-trip validation. Requires ``clean_db > floor_db``.
    """
    if clean_db <= floor_db:
        raise ValueError(f"clean_db ({clean_db}) must exceed floor_db ({floor_db})")
    return max(0.0, min(1.0, (clean_db - float(value_db)) / (clean_db - floor_db)))


def load_calibration_profile(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate a profile; ``None`` → bundled profile if present, else defaults.

    Raises ``ValueError`` on version mismatch, malformed blocks, or inverted
    anchors — a bad profile must fail loudly rather than silently mis-scale
    every quality column.
    """
    if path is None:
        if BUNDLED_PROFILE_PATH.exists():
            path = BUNDLED_PROFILE_PATH
        else:
            return copy.deepcopy(DEFAULT_PROFILE)
    raw = json.loads(Path(path).read_text())
    return validate_profile(raw, source=str(path))


def validate_profile(profile: dict[str, Any], *, source: str = "<dict>") -> dict[str, Any]:
    """Validate a §5 profile dict (returns it unchanged on success)."""
    if not isinstance(profile, dict):
        raise ValueError(f"calibration profile {source} is not a mapping")
    if str(profile.get("version")) != PROFILE_VERSION:
        raise ValueError(
            f"calibration profile {source} has version {profile.get('version')!r}; expected {PROFILE_VERSION!r}"
        )
    for block_name in ("snr", "reverb_c50"):
        block = profile.get(block_name)
        if not isinstance(block, dict):
            raise ValueError(f"calibration profile {source} missing block {block_name!r}")
        if block.get("type") != "linear_db_to_unit":
            raise ValueError(f"{source}: {block_name}.type must be 'linear_db_to_unit'")
        clean, floor = float(block["clean_db"]), float(block["floor_db"])
        if clean <= floor:
            raise ValueError(f"{source}: {block_name}.clean_db must exceed floor_db ({clean} <= {floor})")
    temperature = profile.get("temperature") or {}
    for axis in CALIBRATED_AXES:
        t = float(temperature.get(axis, 1.0))
        if t <= 0:
            raise ValueError(f"{source}: temperature.{axis} must be > 0 (got {t})")
    return profile


def profile_to_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    """Bridge the versioned §5 profile to the flat runtime dict consumers read.

    ``quality.py`` / ``degradation.py`` read ``snr_clean_db``/``snr_floor_db``/``c50_clean_db``/
    ``c50_floor_db``. ``temperature`` and ``token_entropy_reference_nats`` are carried but
    **currently read by nothing** — see the module docstring. Unknown extra keys pass through
    untouched so future profile fields reach their consumers without another bridge edit.
    """
    runtime: dict[str, Any] = {
        "calibration_version": str(profile.get("version", PROFILE_VERSION)),
        "snr_clean_db": float(profile["snr"]["clean_db"]),
        "snr_floor_db": float(profile["snr"]["floor_db"]),
        "c50_clean_db": float(profile["reverb_c50"]["clean_db"]),
        "c50_floor_db": float(profile["reverb_c50"]["floor_db"]),
        "temperature": dict(profile.get("temperature") or {"speech_presence": 1.0, "asr": 1.0}),
    }
    if isinstance(profile.get("bandwidth"), dict):
        runtime["bandwidth"] = dict(profile["bandwidth"])
    for passthrough in ("token_entropy_reference_nats",):
        if passthrough in profile:
            runtime[passthrough] = profile[passthrough]
    return runtime


# ── detection-margin profile (T006, FR-021d / FR-022 / FR-023) ─────────
#
# A second, independent profile schema lives here rather than in its own module so
# that "where do thresholds come from" has one answer. It shares nothing with the
# scene-quality profile above except the load/validate/stdlib-only convention, so
# every symbol is explicitly namespaced.
#
# What it governs: the 3/6/10 dB margin ladder, the level targets and gain cap, the
# noise-floor estimator settings, the mask geometry limits, and the written
# derivation that makes the margin values auditable.

DETECTION_MARGIN_VERSION = "1"

BUNDLED_DETECTION_MARGIN_DIR = Path(__file__).parent / "data" / "detection_margin"

_VALID_CLAIM_STATUS = ("verified", "provisional")

_MAX_GAIN_CAP_DB = 10.0
"""Measured clipping inflection for the AudioSet classifiers.

Above roughly +10 dB, published gain sweeps attribute the degradation to
clipping-induced harmonic distortion rather than to content, so a larger cap means
measuring distortion. Raising this requires a new derivation entry, not an edit here.
"""

DEFAULT_DETECTION_MARGIN: dict[str, Any] = {
    "profile_version": DETECTION_MARGIN_VERSION,
    "margins_db": {"reject_below": 3.0, "candidate": 3.0, "probable": 6.0, "confident": 10.0},
    "level": {
        "target_lufs": -23.0,
        "true_peak_ceiling_dbtp": -1.0,
        "gain_cap_db": 10.0,
        "reject_below_pregain_dbfs": -45.0,
        "stable_band_dbfs": [-35.0, -15.0],
    },
    "noise_floor": {
        "quantile": 0.10,
        # Long frame: the floor is a long-term percentile and needs frequency
        # resolution, not time resolution. A 25 ms frame cannot resolve third-octave
        # bands below ~140 Hz, which is where mains hum and ventilation live.
        "floor_frame_s": 0.100,
        "window_s": 20.0,
        "max_iterations": 3,
        "event_exclusion_db": 6.0,
        "band_smoothing_bins": 5,
        "top_percentile_cap": 0.95,
        "condition_on_target_activity": True,
        "freeze_inside_events": True,
        "recorder_margin_db": 3.0,
    },
    "guards": {
        "flatness_max": 0.30,
        "min_occupancy": 0.40,
        "min_duration_s": {"default": 0.20},
        "hysteresis": {"trigger_tier": "confident", "extend_tier": "probable"},
        "min_distance_separation_db": 6.0,
        # ECMA-74 / ISO 7779 prominent-discrete-tone criterion. Detects a source that
        # runs through the whole recording -- air conditioning, hum, a music bed -- which
        # is absorbed into its own band floor and so has no excess to measure.
        "prominence_ratio_db": 9.0,
        "quarantined_labels": [
            "White noise",
            "Pink noise",
            "Noise",
            "Static",
            "Environmental noise",
            "Hum",
            "Mains hum",
            "Hiss",
            "Throbbing",
            "Waterfall",
            "Water",
            "Gurgling",
            "Spray",
            "Sine wave",
            "Silence",
            "Inside, small room",
        ],
    },
    "mask": {
        "guard_interval_s": 0.50,
        "negligible_fraction": 0.05,
        "min_region_s": 1.0,
        "max_padding_fraction": 0.50,
        # State thresholds. `target_free` needs BOTH low target confidence and low
        # uncertainty: "probably nothing there, but I cannot tell" is not a region a
        # background claim can rest on.
        "target_active_confidence": 0.60,
        "target_free_confidence": 0.20,
        "max_free_uncertainty": 0.50,
        # Every task includes `speech`: the near-field participant talking is target
        # activity whatever they were asked to do. Omitting it from the cough and breath
        # tasks made their speech target-FREE, so it would have been reported as a
        # background `speech` source -- the same misattribution the task mapping exists to
        # prevent, arriving from the other direction. Seen on a real cough recording where
        # the spoken tail landed in the mask.
        "target_event_types_by_task": {
            "speech": ["speech", "breath", "mouth_noise"],
            "breath": ["speech", "breath"],
            "cough": ["speech", "cough", "throat_clear"],
        },
        "fallback_target_event_types": ["speech", "breath", "cough", "mouth_noise"],
    },
    "derivation": {
        "human_basis": [
            {"claim": "minimum measurability ~+3 dB", "source": "ISO 1996-2:2017", "status": "verified"},
            {
                "claim": "10 dB octave / 13 dB third-octave above masked threshold",
                "source": "ISO 7731",
                "status": "verified",
            },
            {"claim": "+5 adverse / +10 significant over background", "source": "BS 4142", "status": "verified"},
            {
                "claim": "TNR >= 8 dB, PR >= 9 dB for tone prominence",
                "source": "ECMA-74 17th ed.",
                "status": "verified",
            },
            {
                "claim": "partial-masking transition -3 to +15 dB",
                "source": "Moore/Glasberg partial loudness",
                "status": "provisional",
                "note": "Surfaced in indexed text; primary source paywalled. Corroborates the ladder, does not set it.",
            },
        ],
        "machine_basis": [
            {
                "claim": "short-window reliable floor 5-10 dB SNR",
                "source": "level probe (measured)",
                "status": "verified",
            },
            {
                "claim": "long-window reliable floor 15-20 dB SNR non-speech",
                "source": "level probe (measured)",
                "status": "verified",
            },
            {
                "claim": "noise-family label contamination from ~20 dB SNR",
                "source": "level probe (measured)",
                "status": "verified",
            },
            {
                "claim": "short-window silence floor ~-60 dBFS, learned not arithmetic",
                "source": "level probe (measured)",
                "status": "verified",
            },
        ],
        "agreement_note": (
            "Human criteria place confident identification near +10 dB; the short-window "
            "classifier's measured reliable floor is 5-10 dB SNR. The +10 dB confident tier "
            "satisfies both simultaneously (SC-017)."
        ),
        "derived_statistics_status": "provisional",
        "derived_statistics_note": (
            "The exponential-periodogram bias correction 1/(-ln(1-q)), the per-bin sigma of "
            "5.57 dB, and the patch-variance collapse are straightforward chi-squared results "
            "but were not found stated in this form in the literature. Validate on synthetic "
            "noise before relying on them."
        ),
    },
}


def quantile_bias_correction_db(quantile: float) -> float:
    """DB to add to a ``quantile``-th percentile band-power floor to recover the mean.

    For a noise-only STFT bin the periodogram is exponentially distributed, so the
    ``q``-quantile of power sits a factor ``-ln(1-q)`` below the mean. The correction is
    therefore ``10*log10(1 / -ln(1-q))`` — about **9.77 dB for the conventional 10th
    percentile**.

    This is computed, never stored (FR-021d). A stored copy could drift from the quantile
    it was derived from, and a drifted correction makes every relative-dB gate silently
    that much more permissive — the failure would look like generosity, not like a bug.

    Args:
        quantile: Floor quantile, in ``(0, 0.5]``.

    Returns:
        Positive dB offset to add to the raw quantile estimate.

    Raises:
        ValueError: If ``quantile`` is outside ``(0, 0.5]``.
    """
    q = float(quantile)
    if not 0.0 < q <= 0.5:
        raise ValueError(f"noise-floor quantile must be in (0, 0.5]; got {q}")
    return 10.0 * math.log10(1.0 / (-math.log1p(-q)))


def _validate_basis(basis: Any, *, name: str, source: str) -> None:  # noqa: ANN401 — validating unknown shape
    """Require a non-empty basis list with at least one verified, well-formed claim."""
    if not isinstance(basis, list) or not basis:
        raise ValueError(f"{source}: derivation.{name} must be a non-empty list")
    verified = 0
    for claim in basis:
        if not isinstance(claim, dict):
            raise ValueError(f"{source}: derivation.{name} entries must be mappings")
        status = claim.get("status")
        if status not in _VALID_CLAIM_STATUS:
            raise ValueError(
                f"{source}: derivation.{name} claim status {status!r} must be one of {_VALID_CLAIM_STATUS}"
            )
        if status == "provisional" and not str(claim.get("note") or "").strip():
            raise ValueError(
                f"{source}: provisional claim {claim.get('claim')!r} in derivation.{name} requires a 'note' "
                "explaining why it is provisional"
            )
        if status == "verified":
            verified += 1
    if verified == 0:
        raise ValueError(
            f"{source}: derivation.{name} has no verified claim — a provisional figure may corroborate a "
            "margin but may never set one"
        )


def validate_detection_margin_profile(profile: dict[str, Any], *, source: str = "<dict>") -> dict[str, Any]:
    """Validate a detection-margin profile and return a runtime copy with derived fields.

    The input is not mutated; the returned copy carries the computed
    ``noise_floor.bias_correction_db``.

    Args:
        profile: Profile mapping, shaped like :data:`DEFAULT_DETECTION_MARGIN`.
        source: Label used in error messages (a path, usually).

    Returns:
        A validated deep copy with derived fields filled in.

    Raises:
        ValueError: On version mismatch, a non-monotone ladder, an out-of-range quantile,
            a gain cap above the measured clipping inflection, or a derivation that cannot
            support its own margins.
    """
    if not isinstance(profile, dict):
        raise ValueError(f"detection-margin profile {source} is not a mapping")
    if str(profile.get("profile_version")) != DETECTION_MARGIN_VERSION:
        raise ValueError(
            f"detection-margin profile {source} has version {profile.get('profile_version')!r}; "
            f"expected {DETECTION_MARGIN_VERSION!r}"
        )
    runtime = copy.deepcopy(profile)

    margins = runtime.get("margins_db")
    if not isinstance(margins, dict):
        raise ValueError(f"{source}: missing 'margins_db' block")
    try:
        ladder = [float(margins[k]) for k in ("reject_below", "candidate", "probable", "confident")]
    except KeyError as exc:
        raise ValueError(f"{source}: margins_db missing key {exc.args[0]!r}") from exc
    if any(b < a for a, b in zip(ladder, ladder[1:])):
        raise ValueError(
            f"{source}: margins_db must be monotone non-decreasing "
            f"(reject_below <= candidate <= probable <= confident); got {ladder}"
        )

    level = runtime.get("level")
    if not isinstance(level, dict):
        raise ValueError(f"{source}: missing 'level' block")
    gain_cap = float(level.get("gain_cap_db", _MAX_GAIN_CAP_DB))
    if gain_cap > _MAX_GAIN_CAP_DB:
        raise ValueError(
            f"{source}: level.gain_cap_db {gain_cap} exceeds the measured clipping inflection "
            f"({_MAX_GAIN_CAP_DB} dB); above it the classifiers respond to distortion, not content"
        )

    noise_floor = runtime.get("noise_floor")
    if not isinstance(noise_floor, dict):
        raise ValueError(f"{source}: missing 'noise_floor' block")
    # Raises on an out-of-range quantile, and overwrites any stored correction.
    noise_floor["bias_correction_db"] = quantile_bias_correction_db(noise_floor.get("quantile", 0.10))

    derivation = runtime.get("derivation")
    if not isinstance(derivation, dict):
        raise ValueError(f"{source}: missing 'derivation' block")
    _validate_basis(derivation.get("human_basis"), name="human_basis", source=source)
    _validate_basis(derivation.get("machine_basis"), name="machine_basis", source=source)
    if not str(derivation.get("agreement_note") or "").strip():
        raise ValueError(
            f"{source}: derivation.agreement_note is required — SC-017 asks for the human and "
            "machine bases to be shown to agree, not merely both present"
        )
    return runtime


def load_detection_margin_profile(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate a detection-margin profile.

    Args:
        path: Profile path, or ``None`` for the bundled default (falling back to
            :data:`DEFAULT_DETECTION_MARGIN` when no bundled file is present).

    Returns:
        The validated runtime profile.

    Raises:
        FileNotFoundError: If ``path`` names a file that does not exist — a named-but-absent
            profile is an operator error, not a reason to silently use defaults.
        ValueError: If the profile fails validation.
    """
    if path is None:
        bundled = sorted(BUNDLED_DETECTION_MARGIN_DIR.glob("*.json"))
        if not bundled:
            return validate_detection_margin_profile(DEFAULT_DETECTION_MARGIN, source="<default>")
        path = bundled[-1]
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"detection-margin profile not found: {p}")
    return validate_detection_margin_profile(json.loads(p.read_text()), source=str(p))
