"""The run config file: one `Formatted` value holding its `thresholds` and `calibration` sections."""

from pathlib import Path

import pydantic
import pytest
from senselab_core.kernel.formats import RUN_CONFIG_FORMAT
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.call import RunConfig
from senselab_core.runtime.compare import RunConfigFile, UnknownThreshold
from senselab_core.runtime.profiles import ExecutionProfile


@pytest.mark.logic
def test_one_run_config_file_validates_thresholds_and_calibration(tmp_path: Path) -> None:
    """A run config file's `calibration.trust` reaches `RunCalibration.apply`; its thresholds merge over packaged."""
    config = tmp_path / "run.yaml"
    config.write_text("format: run-config@2.1\nthresholds:\n  embedding:\n    atol: 2\ncalibration:\n  trust: own\n")
    loaded = RunConfigFile.load(config)
    assert loaded.calibration == RunCalibration(trust="own")
    assert loaded.calibration.apply(ExecutionProfile.load("orcd").calibration).trust == "own"
    assert loaded.thresholds.by_name["embedding"].atol == 2.0
    assert RunConfigFile.load().calibration == RunCalibration()
    with pytest.raises(UnknownThreshold, match="did you mean 'trust'"):
        RunConfigFile.load({"calibration": {"trus": "own"}})
    with pytest.raises(UnknownThreshold, match="did you mean 'calibration'"):
        RunConfigFile.load({"calibrations": {}})
    with pytest.raises(ValueError, match="trust"):
        RunConfigFile.load({"calibration": {"trust": "all"}})
    assert RunConfigFile.load().format == RUN_CONFIG_FORMAT == "run-config@2.1"


@pytest.mark.logic
@pytest.mark.parametrize("tag", ["run-config@2.0", "run-config@2.1", "run-config@2.9"])
def test_a_run_config_of_the_major_reads_at_any_minor(tag: str) -> None:
    """Every minor of the major is read, a later one included."""
    assert RunConfigFile.load({"format": tag}).calibration == RunCalibration()


@pytest.mark.logic
@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ("run-config@1.0", "reads run-config@2.1 and its minor versions, got run-config@1.0"),
        ("run-config@3.0", "reads run-config@2.1 and its minor versions, got run-config@3.0"),
        ("thresholds@2.0", "reads run-config@2.1 and its minor versions, got thresholds@2.0"),
        (7, "format\n  Input should be a valid string"),
    ],
)
def test_a_run_config_of_another_name_or_major_is_refused(bad: object, match: str) -> None:
    """A tag of another name or major is refused naming it; a value that is no tag is a typed error at `format`."""
    with pytest.raises(pydantic.ValidationError, match=match):
        RunConfigFile.load({"format": bad})


@pytest.mark.logic
@pytest.mark.parametrize(
    ("section", "value"), [("calibration", "own"), ("calibration", 5), ("thresholds", "x"), ("thresholds", [1])]
)
def test_a_section_that_is_not_a_mapping_is_refused_naming_it(section: str, value: object) -> None:
    """A scalar or list where a section's mapping belongs is a typed error naming the section."""
    with pytest.raises(pydantic.ValidationError, match=rf"{section}\n.*built from a mapping of its fields"):
        RunConfigFile.load({section: value})


@pytest.mark.logic
def test_a_run_config_file_reaches_the_run_whole(tmp_path: Path) -> None:
    """`RunConfig` is built from one run config file; its thresholds and calibration are that file's sections."""
    config = tmp_path / "run.yaml"
    config.write_text("thresholds:\n  embedding:\n    atol: 2\ncalibration:\n  trust: own\n")
    run = RunConfig(config=RunConfigFile.load(config))
    assert run.calibration == RunCalibration(trust="own")
    assert run.thresholds.by_name["embedding"].atol == 2.0
    assert RunConfig().thresholds == RunConfigFile.load().thresholds
    assert RunConfig().calibration == RunCalibration()
    with pytest.raises(TypeError, match="thresholds"):
        RunConfig(thresholds=run.thresholds)  # type: ignore[call-arg]
