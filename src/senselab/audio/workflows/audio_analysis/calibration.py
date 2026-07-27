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
  "temperature": {"presence": 1.0, "utterance": 1.0}
}
```

Profiles are fitted by ``scripts/calibrate_scene_quality.py`` from synthetic
mixtures (research.md D9) and consumed at runtime by ``quality.py`` (flat
``*_clean_db``/``*_floor_db`` keys) and ``aggregate.py`` (``temperature``,
``token_entropy_reference_nats``): :func:`profile_to_runtime` bridges the
versioned on-disk shape to that flat runtime convention. Absent profile →
:data:`DEFAULT_PROFILE`, which mirrors the documented uncalibrated defaults in
``quality.py`` (bounded, not fitted).

Stdlib-only; safe to import anywhere.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

PROFILE_VERSION = "1"

BUNDLED_PROFILE_PATH = Path(__file__).parent / "data" / "scene_quality_calibration.json"

DEFAULT_PROFILE: dict[str, Any] = {
    "version": PROFILE_VERSION,
    # Anchors mirror quality.py's documented defaults (speech-appropriate SNR window).
    "snr": {"type": "linear_db_to_unit", "clean_db": 25.0, "floor_db": 5.0},
    "reverb_c50": {"type": "linear_db_to_unit", "clean_db": 30.0, "floor_db": -5.0},
    "bandwidth": {"nyquist_ref_hz": 8000.0, "rolloff_pct": 0.95},
    "temperature": {"presence": 1.0, "utterance": 1.0},
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
    for axis in ("presence", "utterance"):
        t = float(temperature.get(axis, 1.0))
        if t <= 0:
            raise ValueError(f"{source}: temperature.{axis} must be > 0 (got {t})")
    return profile


def profile_to_runtime(profile: dict[str, Any]) -> dict[str, Any]:
    """Bridge the versioned §5 profile to the flat runtime dict consumers read.

    ``quality.py`` reads ``snr_clean_db``/``snr_floor_db``/``c50_clean_db``/
    ``c50_floor_db``; ``aggregate.py`` reads ``temperature`` (per axis) and
    ``token_entropy_reference_nats``. Unknown extra keys pass through untouched
    so future profile fields reach their consumers without another bridge edit.
    """
    runtime: dict[str, Any] = {
        "calibration_version": str(profile.get("version", PROFILE_VERSION)),
        "snr_clean_db": float(profile["snr"]["clean_db"]),
        "snr_floor_db": float(profile["snr"]["floor_db"]),
        "c50_clean_db": float(profile["reverb_c50"]["clean_db"]),
        "c50_floor_db": float(profile["reverb_c50"]["floor_db"]),
        "temperature": dict(profile.get("temperature") or {"presence": 1.0, "utterance": 1.0}),
    }
    if isinstance(profile.get("bandwidth"), dict):
        runtime["bandwidth"] = dict(profile["bandwidth"])
    for passthrough in ("token_entropy_reference_nats",):
        if passthrough in profile:
            runtime[passthrough] = profile[passthrough]
    return runtime
