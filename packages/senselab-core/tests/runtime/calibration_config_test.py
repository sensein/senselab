"""The `calibration` profile section, its derived values, the run config's trust override, and the format tags."""

import re
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel import formats
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.config import expand
from senselab_core.runtime.execution_settings import CalibrationSettings
from senselab_core.runtime.profiles import ExecutionProfile, UnknownProfileKey

CLUSTER = (
    next(p for p in Path(__file__).resolve().parents if p.name == "packages")
    / "senselab-core/tests/data/profiles/cluster.yaml"
)

PROFILES = Path(__file__).parents[2] / "src/senselab_core/runtime/data/profiles"
NAMES = ["laptop", "slurm"]
SOURCES: dict[str, str | Path] = {"laptop": "laptop", "cluster": CLUSTER}
TRUST = {"laptop": "own", "cluster": "group"}


def _calibration(overrides: dict[str, Any]) -> dict[str, Any]:
    return {"calibration": overrides}


@pytest.mark.logic
@pytest.mark.parametrize("name", SOURCES)
def test_profiles_carry_their_trust_and_roots(name: str) -> None:
    """The laptop profile trusts its own measures; the cluster fixture pools the group root before its own."""
    c = ExecutionProfile.load(SOURCES[name]).calibration
    assert c.trust == TRUST[name]
    assert len(c.measures_roots) == (2 if name == "cluster" else 1)


@pytest.mark.logic
def test_probe_s_peer_wait_s_and_drift_window_are_derived_from_the_profile() -> None:
    """The three are properties, so a moved `sizing`, `lease` or section key moves them and they are never keys."""
    profile = ExecutionProfile.load(CLUSTER)
    assert (profile.probe_s, profile.peer_wait_s, profile.calibration.drift_window) == (90, 390, 6)
    moved = ExecutionProfile.load(CLUSTER, {"sizing": {"target_wall_s": 3600}, "lease": {"stale_after_s": 600}})
    assert (moved.probe_s, moved.peer_wait_s) == (180, 780)
    table = ExecutionProfile.load("laptop", _calibration({"length_strata": 3}))
    assert table.calibration.drift_window == 9
    for key in ("probe_s", "peer_wait_s", "drift_window"):
        assert key not in CalibrationSettings.model_fields
        with pytest.raises(UnknownProfileKey, match=key):
            ExecutionProfile.load("laptop", _calibration({key: 1}))


@pytest.mark.logic
def test_probe_items_is_not_a_profile_key() -> None:
    """The probe's job budget is derived from the protocol and the pool's quantum, so a profile cannot set it."""
    assert "probe_items" not in CalibrationSettings.model_fields
    with pytest.raises(UnknownProfileKey, match="probe_items"):
        ExecutionProfile.load("laptop", _calibration({"probe_items": 50}))


@pytest.mark.logic
def test_the_group_measures_root_is_the_variable_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first cluster root has no fallback: an unset variable is no root, never a root of each member's own."""
    first = ExecutionProfile.load(CLUSTER).calibration.measures_roots[0]
    assert first == "${SENSELAB_MEASURES_ROOT}"
    monkeypatch.delenv("SENSELAB_MEASURES_ROOT", raising=False)
    with pytest.raises(ValueError, match="not set"):
        expand(first)
    monkeypatch.setenv("SENSELAB_MEASURES_ROOT", "/group/measures")
    assert expand(first) == "/group/measures"


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
        {"observations_kept": 0},
        {"unmeasured_workers": 0},
        {"resolve_s": 0},
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
@pytest.mark.parametrize("name", SOURCES)
def test_run_config_trust_overrides_the_profile(name: str) -> None:
    """The run config's `calibration.trust` replaces the profile's; with none given the profile's stands."""
    settings = ExecutionProfile.load(SOURCES[name]).calibration
    assert RunCalibration().apply(settings).trust == TRUST[name]
    for trust in ("own", "group"):
        run = RunCalibration.model_validate({"trust": trust})
        assert run.apply(settings).trust == trust
        assert run.apply(settings).model_copy(update={"trust": settings.trust}) == settings
    with pytest.raises(ValueError, match="trust"):
        RunCalibration.model_validate({"trust": "all"})
    with pytest.raises(ValueError, match="extra|forbid|probe_items"):
        RunCalibration.model_validate({"probe_items": 1})


@pytest.mark.logic
@pytest.mark.parametrize("name", NAMES)
def test_every_calibration_key_is_described(name: str) -> None:
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
    ]
    assert len(set(tags)) == len(tags)
    for tag in tags:
        assert re.fullmatch(formats.FORMAT_PATTERN, tag)
        assert formats.check(tag, tag.split(".")[0] + ".9") is formats.FormatVerdict.ACCEPTED
        assert formats.check(tag, tag.split("@")[0] + "@99.0") is formats.FormatVerdict.UNKNOWN
    assert formats.MEASURES_LAYOUT_TAG == "measures@1"
    assert formats.MEASURE_UNIT_TAG == "measure-unit@1"
