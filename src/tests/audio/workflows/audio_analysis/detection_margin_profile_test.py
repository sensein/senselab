"""Detection-margin profile loading and validation (T005, FR-021d/FR-022/FR-023).

The profile carries the 3/6/10 dB margin ladder, the level targets, the noise-floor
estimator settings, and the written derivation that makes the margin values auditable.

Two invariants carry most of the weight:

- **The bias correction is computed from the quantile, never stored.** A low-percentile
  estimate of band power sits a calculable amount below the true mean noise power
  (research.md D4). Storing it alongside ``quantile`` would let the two drift apart, and
  a drifted correction silently makes every relative-dB gate more permissive.
- **A provisional derivation claim may corroborate a margin but may never set one.** The
  partial-loudness figures and the derived chi-squared statistics could not be verified
  against primary sources, so a profile resting on them alone is rejected rather than
  shipped with an unauditable basis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.calibration import (
    DEFAULT_DETECTION_MARGIN,
    DETECTION_MARGIN_VERSION,
    load_detection_margin_profile,
    quantile_bias_correction_db,
    validate_detection_margin_profile,
)


def _profile(**overrides: Any) -> dict[str, Any]:  # noqa: ANN401 — heterogeneous profile fields
    """A valid profile, deep-copied, with top-level sections overridden."""
    prof = json.loads(json.dumps(DEFAULT_DETECTION_MARGIN))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(prof.get(key), dict):
            prof[key].update(value)
        else:
            prof[key] = value
    return prof


# ── the default profile is itself valid ───────────────────────────────


def test_default_profile_validates() -> None:
    """The bundled default must pass its own validator."""
    assert validate_detection_margin_profile(DEFAULT_DETECTION_MARGIN) is not None


def test_default_profile_carries_the_corroborated_ladder() -> None:
    """3 / 6 / 10 dB — the values three independent lines of evidence converged on."""
    margins = DEFAULT_DETECTION_MARGIN["margins_db"]
    assert margins["candidate"] == pytest.approx(3.0)
    assert margins["probable"] == pytest.approx(6.0)
    assert margins["confident"] == pytest.approx(10.0)
    assert DEFAULT_DETECTION_MARGIN["profile_version"] == DETECTION_MARGIN_VERSION


# ── margin ladder ordering ────────────────────────────────────────────


def test_non_monotone_ladder_rejected() -> None:
    """Probable below candidate is incoherent, not merely unusual."""
    bad = _profile(margins_db={"candidate": 6.0, "probable": 3.0})
    with pytest.raises(ValueError, match="monotone"):
        validate_detection_margin_profile(bad)


def test_confident_below_probable_rejected() -> None:
    """The top tier cannot be easier to clear than the middle one."""
    bad = _profile(margins_db={"probable": 10.0, "confident": 6.0})
    with pytest.raises(ValueError, match="monotone"):
        validate_detection_margin_profile(bad)


def test_equal_tiers_allowed() -> None:
    """Monotone is non-strict — collapsing two tiers is a policy choice, not an error."""
    ok = _profile(margins_db={"candidate": 6.0, "probable": 6.0})
    assert validate_detection_margin_profile(ok) is not None


# ── quantile range and bias correction (FR-021d) ──────────────────────


@pytest.mark.parametrize("q", [0.0, -0.1, 0.6, 1.0])
def test_quantile_outside_range_rejected(q: float) -> None:
    """A floor quantile must sit in (0, 0.5]; above the median it is not a floor."""
    with pytest.raises(ValueError, match="quantile"):
        validate_detection_margin_profile(_profile(noise_floor={"quantile": q}))


@pytest.mark.parametrize(
    ("q", "expected_db"),
    [
        (0.10, 9.7733),  # the research.md D4 figure
        (0.20, 6.5144),
        (0.50, 1.5917),
    ],
)
def test_bias_correction_matches_exponential_derivation(q: float, expected_db: float) -> None:
    """``10*log10(1/(-ln(1-q)))`` — the exponential-periodogram correction."""
    assert quantile_bias_correction_db(q) == pytest.approx(expected_db, abs=1e-3)


def test_bias_correction_is_positive_for_every_valid_quantile() -> None:
    """A percentile floor always underestimates, so the correction always adds."""
    for q in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        assert quantile_bias_correction_db(q) > 0.0


def test_bias_correction_derived_not_stored() -> None:
    """A stored correction is ignored — the computed value wins (FR-021d).

    Storing it would let it drift from ``quantile``; a drifted correction makes every
    relative-dB gate silently more permissive.
    """
    prof = _profile(noise_floor={"quantile": 0.10, "bias_correction_db": 0.0})
    runtime = validate_detection_margin_profile(prof)
    assert runtime["noise_floor"]["bias_correction_db"] == pytest.approx(9.7733, abs=1e-3)


def test_bias_correction_tracks_a_changed_quantile() -> None:
    """Changing q alone changes the correction — the two cannot desynchronize."""
    a = validate_detection_margin_profile(_profile(noise_floor={"quantile": 0.10}))
    b = validate_detection_margin_profile(_profile(noise_floor={"quantile": 0.20}))
    assert a["noise_floor"]["bias_correction_db"] > b["noise_floor"]["bias_correction_db"]


def test_bias_correction_rejects_invalid_quantile_directly() -> None:
    """The helper guards itself, not only the validator."""
    with pytest.raises(ValueError, match="quantile"):
        quantile_bias_correction_db(0.0)


# ── derivation provenance (FR-022) ────────────────────────────────────


def test_provisional_claim_without_note_rejected() -> None:
    """A provisional figure must say why it is provisional."""
    bad = _profile(
        derivation={
            "human_basis": [{"claim": "x", "source": "y", "status": "provisional"}],
            "machine_basis": DEFAULT_DETECTION_MARGIN["derivation"]["machine_basis"],
            "agreement_note": "n",
        }
    )
    with pytest.raises(ValueError, match="provisional"):
        validate_detection_margin_profile(bad)


def test_provisional_claim_with_note_accepted() -> None:
    """Provisional is allowed as corroboration when it is labelled as such."""
    ok = _profile(
        derivation={
            "human_basis": [
                {"claim": "verified thing", "source": "ISO 1996-2:2017", "status": "verified"},
                {"claim": "corroborating", "source": "paywalled", "status": "provisional", "note": "not primary"},
            ],
        }
    )
    assert validate_detection_margin_profile(ok) is not None


def test_all_provisional_human_basis_rejected() -> None:
    """A margin resting only on unverified figures has no auditable basis."""
    bad = _profile(
        derivation={
            "human_basis": [{"claim": "c", "source": "s", "status": "provisional", "note": "n"}],
        }
    )
    with pytest.raises(ValueError, match="verified"):
        validate_detection_margin_profile(bad)


def test_missing_machine_basis_rejected() -> None:
    """The threshold must balance human and machine evidence, so both are required."""
    bad = _profile(derivation={"machine_basis": []})
    with pytest.raises(ValueError, match="machine_basis"):
        validate_detection_margin_profile(bad)


def test_missing_agreement_note_rejected() -> None:
    """SC-017 requires the two bases to be *shown* to agree, not merely both present."""
    prof = _profile()
    del prof["derivation"]["agreement_note"]
    with pytest.raises(ValueError, match="agreement_note"):
        validate_detection_margin_profile(prof)


@pytest.mark.parametrize("status", ["unknown", "", "VERIFIED"])
def test_unrecognized_status_rejected(status: str) -> None:
    """Only ``verified`` and ``provisional`` are meaningful."""
    bad = _profile(
        derivation={"human_basis": [{"claim": "c", "source": "s", "status": status}]},
    )
    with pytest.raises(ValueError, match="status"):
        validate_detection_margin_profile(bad)


# ── level and gain limits ─────────────────────────────────────────────


def test_gain_cap_above_measured_clipping_inflection_rejected() -> None:
    """+10 dB is where measured classifier behavior starts reflecting clipping."""
    with pytest.raises(ValueError, match="gain_cap_db"):
        validate_detection_margin_profile(_profile(level={"gain_cap_db": 20.0}))


def test_gain_cap_at_the_limit_accepted() -> None:
    """The cap itself is allowed; only exceeding it is not."""
    assert validate_detection_margin_profile(_profile(level={"gain_cap_db": 10.0})) is not None


def test_quantifying_thresholds_present() -> None:
    """The three keys that exist so vague spec language cannot become code literals."""
    assert "recorder_margin_db" in DEFAULT_DETECTION_MARGIN["noise_floor"]
    assert "min_distance_separation_db" in DEFAULT_DETECTION_MARGIN["guards"]
    assert "min_region_s" in DEFAULT_DETECTION_MARGIN["mask"]
    assert "max_padding_fraction" in DEFAULT_DETECTION_MARGIN["mask"]


# ── loading ───────────────────────────────────────────────────────────


def test_load_without_path_returns_default() -> None:
    """Zero configuration is a valid configuration."""
    assert load_detection_margin_profile(None)["margins_db"]["confident"] == pytest.approx(10.0)


def test_load_from_path_roundtrips(tmp_path: Path) -> None:
    """A written profile loads back with its overrides intact."""
    prof = _profile(margins_db={"confident": 12.0})
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(prof))
    assert load_detection_margin_profile(p)["margins_db"]["confident"] == pytest.approx(12.0)


def test_load_validates(tmp_path: Path) -> None:
    """Loading is not a bypass for validation."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_profile(level={"gain_cap_db": 99.0})))
    with pytest.raises(ValueError, match="gain_cap_db"):
        load_detection_margin_profile(p)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    """A named-but-absent profile is an error, not a silent fallback to defaults."""
    with pytest.raises(FileNotFoundError):
        load_detection_margin_profile(tmp_path / "nope.json")


def test_validation_does_not_mutate_input() -> None:
    """The caller's dict is left alone; the runtime copy carries derived fields."""
    prof = _profile()
    before = json.dumps(prof, sort_keys=True)
    validate_detection_margin_profile(prof)
    assert json.dumps(prof, sort_keys=True) == before


def test_math_reference_is_exact() -> None:
    """Guard the derivation itself, independent of the implementation."""
    q = 0.1
    assert quantile_bias_correction_db(q) == pytest.approx(10.0 * math.log10(1.0 / (-math.log1p(-q))), abs=1e-12)
