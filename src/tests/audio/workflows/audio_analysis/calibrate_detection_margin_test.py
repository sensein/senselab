"""The detection-margin calibration script's refusals (T070, FR-022, SC-017).

The margin ladder decides which background sources a run will report, so the script's value
is in what it refuses to emit rather than in the file it writes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_detection_margin",
    Path(__file__).resolve().parents[5] / "scripts" / "calibrate_detection_margin.py",
)
assert _SPEC and _SPEC.loader
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)


def _verdicts(*floors: tuple[str, float | None]) -> dict[str, Any]:
    return {
        "host": "test-host",
        "verdicts": [{"classifier": name, "reliable_floor_snr_db": f} for name, f in floors],
    }


def _base() -> dict[str, Any]:
    from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile

    return load_detection_margin_profile()


def test_measured_floors_become_the_machine_basis() -> None:
    """FR-022: the machine side of the ladder must be measured, not asserted."""
    claims = cal.machine_basis_from_verdicts(_verdicts(("yamnet", 7.5), ("ast", 18.0)))
    assert [c["status"] for c in claims] == ["verified", "verified"]
    assert "7.5 dB SNR" in claims[0]["claim"]
    assert claims[0]["measured_on"] == "test-host"


def test_a_classifier_the_probe_skipped_contributes_no_claim() -> None:
    """An absent measurement and a measurement of zero license opposite conclusions."""
    claims = cal.machine_basis_from_verdicts(_verdicts(("yamnet", 7.5), ("skipped", None)))
    assert len(claims) == 1


def test_a_profile_with_no_measurements_is_refused() -> None:
    """Emitting it would claim a machine basis the run does not have."""
    with pytest.raises(ValueError, match="machine basis"):
        cal.build_profile(_base(), _verdicts(("skipped", None)), calibrated_as="test")


def test_a_confident_tier_no_classifier_can_reach_is_refused() -> None:
    """SC-017: the ladder is only defensible if the human criterion and the measured machine
    floor are satisfied at once. A confident tier above every measured floor is a threshold
    already known not to work on the probed host, and shipping it would hide that."""
    with pytest.raises(ValueError, match="no classifier reaches this tier"):
        cal.build_profile(_base(), _verdicts(("ast", 25.0)), calibrated_as="test")


def test_a_reachable_tier_is_accepted() -> None:
    """The guard must not reject the ladder the project actually ships."""
    profile = cal.build_profile(_base(), _verdicts(("yamnet", 7.5), ("ast", 18.0)), calibrated_as="test")
    assert profile["calibrated_as"] == "test"
    assert len(profile["derivation"]["machine_basis"]) == 2


def test_the_emitted_profile_passes_the_loader_that_will_read_it(tmp_path: Path) -> None:
    """A profile rejected at read time must never be written in the first place."""
    from senselab.audio.workflows.audio_analysis.calibration import load_detection_margin_profile

    verdicts_path = tmp_path / "level-verdicts.json"
    verdicts_path.write_text(json.dumps(_verdicts(("yamnet", 7.5))))
    out = tmp_path / "2026-08-01.json"
    assert cal.main(["--level-verdicts", str(verdicts_path), "--out", str(out)]) == 0
    reloaded = load_detection_margin_profile(out)
    assert reloaded["calibrated_as"] == "2026-08-01"


def test_an_unreadable_verdicts_file_exits_without_writing(tmp_path: Path) -> None:
    """Failing loudly beats emitting a profile from nothing."""
    out = tmp_path / "profile.json"
    assert cal.main(["--level-verdicts", str(tmp_path / "missing.json"), "--out", str(out)]) == 2
    assert not out.exists()


def test_an_unmarked_provisional_figure_is_refused(tmp_path: Path) -> None:
    """FR-022: a provisional number without a note reads as settled.

    Telling settled from unsettled is the entire purpose of the derivation block, so the
    script refuses rather than emitting a profile that quietly overstates its evidence.
    """
    base = _base()
    base["derivation"]["human_basis"] = [
        {"claim": "minimum measurability ~+3 dB", "source": "ISO 1996-2:2017", "status": "verified"},
        {"claim": "confident identification ~+10 dB", "source": "unpublished", "status": "provisional"},
    ]
    verdicts_path = tmp_path / "v.json"
    verdicts_path.write_text(json.dumps(_verdicts(("yamnet", 7.5))))
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    out = tmp_path / "out.json"
    assert cal.main(["--level-verdicts", str(verdicts_path), "--out", str(out), "--base", str(base_path)]) != 0
    assert not out.exists()
