"""The `calibration` profile section, its load-time rules, the run config's trust override, and the format tags."""

import re
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel import formats
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.execution_settings import CalibrationRuleError, CalibrationSettings
from senselab_core.runtime.profiles import ExecutionProfile, UnknownProfileKey

PROFILES = Path(__file__).parents[2] / "src/senselab_core/runtime/data/profiles"
NAMES = ["laptop", "orcd"]
TRUST = {"laptop": "own", "orcd": "group"}
PROBE_ITEMS = {"laptop": 50, "orcd": 200}


def _calibration(overrides: dict[str, Any]) -> dict[str, Any]:
    return {"calibration": overrides}


@pytest.mark.logic
@pytest.mark.parametrize("name", NAMES)
def test_packaged_profiles_carry_the_designed_calibration(name: str) -> None:
    """Each packaged profile holds the section's values of the design, and its trust default."""
    c = ExecutionProfile.load(name).calibration
    assert (c.trust, c.probe_items) == (TRUST[name], PROBE_ITEMS[name])
    assert (c.warmup_requests, c.length_strata, c.min_requests, c.single_item_jobs, c.batched_sizes) == (2, 2, 3, 4, 1)
    assert (c.probe_s, c.confidence, c.drift_window, c.drift_confidence) == (90, 1.645, 6, 2.576)
    assert (c.peer_wait_s, c.observations_kept, c.unmeasured_workers) == (390, 3, 1)
    assert len(c.measures_roots) == (2 if name == "orcd" else 1)


@pytest.mark.logic
def test_profile_refuses_probe_s_not_load_share_of_target_wall() -> None:
    """`probe_s` is `sizing.load_share × sizing.target_wall_s`; a value off it or a target moved from it is refused."""
    with pytest.raises(CalibrationRuleError, match=r"probe_s 91.*load_share 0\.05.*target_wall_s 1800"):
        ExecutionProfile.load("orcd", _calibration({"probe_s": 91}))
    with pytest.raises(CalibrationRuleError, match="probe_s 90"):
        ExecutionProfile.load("orcd", {"sizing": {"target_wall_s": 3600}})
    moved = ExecutionProfile.load(
        "orcd", {"sizing": {"target_wall_s": 3600}, **_calibration({"probe_s": 180, "peer_wait_s": 480})}
    )
    assert moved.calibration.probe_s == 180


@pytest.mark.logic
def test_profile_refuses_peer_wait_s_not_probe_s_plus_lease_stale_bound() -> None:
    """`peer_wait_s` is `probe_s + lease.stale_after_s`."""
    with pytest.raises(CalibrationRuleError, match=r"peer_wait_s 391.*probe_s 90.*stale_after_s 300"):
        ExecutionProfile.load("laptop", _calibration({"peer_wait_s": 391}))
    with pytest.raises(CalibrationRuleError, match="peer_wait_s 390"):
        ExecutionProfile.load("laptop", {"lease": {"stale_after_s": 600}})


@pytest.mark.logic
def test_profile_refuses_drift_window_not_strata_times_min_requests() -> None:
    """`drift_window` is `length_strata × min_requests`."""
    with pytest.raises(CalibrationRuleError, match=r"drift_window 7.*length_strata 2.*min_requests 3"):
        ExecutionProfile.load("laptop", _calibration({"drift_window": 7}))
    table = _calibration({"length_strata": 3, "drift_window": 9, "probe_items": 2 + 3 * 3 * 101 + 12})
    assert ExecutionProfile.load("laptop", table).calibration.length_strata == 3


@pytest.mark.logic
@pytest.mark.parametrize("items", [49, 51, 13])
def test_profile_refuses_probe_items_that_are_not_the_minimal_protocol_at_a_target_batch(items: int) -> None:
    """`probe_items` is `warmup + strata × min × (1 + B) + min × single_item_jobs` for a whole B of at least 1."""
    with pytest.raises(CalibrationRuleError, match=f"probe_items {items}"):
        ExecutionProfile.load("laptop", _calibration({"probe_items": items}))


@pytest.mark.logic
@pytest.mark.parametrize(
    "bad",
    [
        {"trust": "everyone"},
        {"warmup_requests": -1},
        {"length_strata": 0},
        {"min_requests": 0},
        {"single_item_jobs": 1},
        {"batched_sizes": 3},
        {"confidence": 0},
        {"drift_confidence": -1},
        {"probe_s": 0},
        {"observations_kept": 0},
        {"unmeasured_workers": 0},
        {"measures_roots": []},
    ],
)
def test_calibration_values_outside_their_domain_are_refused(bad: dict[str, Any]) -> None:
    """Each key's domain is checked when the section is read."""
    with pytest.raises(ValueError, match=next(iter(bad))):
        CalibrationSettings.model_validate({**ExecutionProfile.load("laptop").calibration.model_dump(), **bad})


@pytest.mark.logic
def test_an_unknown_calibration_key_suggests_the_closest() -> None:
    """A misspelt key of the section is named with its closest key."""
    with pytest.raises(UnknownProfileKey, match=r"calibration\.warmup_requets.*warmup_requests"):
        ExecutionProfile.load("laptop", _calibration({"warmup_requets": 2}))


@pytest.mark.logic
@pytest.mark.parametrize("name", NAMES)
def test_run_config_trust_overrides_the_profile(name: str) -> None:
    """The run config's `calibration.trust` replaces the profile's; with none given the profile's stands."""
    settings = ExecutionProfile.load(name).calibration
    assert RunCalibration().apply(settings).trust == TRUST[name]
    for trust in ("own", "group"):
        run = RunCalibration.model_validate({"trust": trust})
        assert run.apply(settings).trust == trust
        assert run.apply(settings).model_copy(update={"trust": settings.trust}) == settings
    with pytest.raises(ValueError, match="trust"):
        RunCalibration.model_validate({"trust": "all"})
    with pytest.raises(ValueError, match="extra|forbid|probe_s"):
        RunCalibration.model_validate({"probe_s": 1})


@pytest.mark.logic
@pytest.mark.parametrize("name", NAMES)
def test_every_calibration_key_has_a_derivation(name: str) -> None:
    """The section carries exactly the model's keys, each under a `#` comment line, and no prose value."""
    lines = (PROFILES / f"{name}.yaml").read_text().splitlines()
    start = lines.index("calibration:")
    body = []
    for line in lines[start + 1 :]:
        if line[:1].isalpha():
            break
        body.append(line)
    keys = [(i, m[1]) for i, line in enumerate(body) if (m := re.match(r"^  (\w+):", line))]
    assert {k for _, k in keys} == set(CalibrationSettings.model_fields)
    assert [k for i, k in keys if i == 0 or not body[i - 1].lstrip().startswith("# ")] == []
    assert lines[start - 1].startswith("# ")


@pytest.mark.logic
def test_measures_and_placement_formats_are_registered() -> None:
    """Every format of the calibration design is a well-formed tag, and its reader accepts later minors only."""
    tags = [
        formats.PLACEMENT_REQUEST_FORMAT,
        formats.PLACEMENT_ANSWER_FORMAT,
        formats.MEASURE_UNIT_FORMAT,
        formats.MEASURE_OBSERVATIONS_FORMAT,
        formats.MEASURE_DRIFT_FORMAT,
        formats.MEASURE_SUPERSEDE_FORMAT,
        formats.MEASURE_REGIME_FORMAT,
        formats.MEASURE_OOM_FORMAT,
        formats.RESOURCE_TABLE_2_FORMAT,
    ]
    assert len(set(tags)) == len(tags)
    for tag in tags:
        assert re.fullmatch(formats.FORMAT_PATTERN, tag)
        assert formats.check(tag, tag.split(".")[0] + ".9") is formats.FormatVerdict.ACCEPTED
        assert formats.check(tag, tag.split("@")[0] + "@99.0") is formats.FormatVerdict.UNKNOWN
    assert formats.MEASURES_LAYOUT_TAG == "measures@1"
    assert formats.MEASURE_UNIT_TAG == "measure-unit@1"
