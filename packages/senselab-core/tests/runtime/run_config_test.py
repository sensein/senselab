"""The run config file: one reader validates its `thresholds` and `calibration` sections."""

from pathlib import Path

import pytest
from senselab_core.kernel.formats import RUN_CONFIG_FORMAT
from senselab_core.runtime.calibration import RunCalibration
from senselab_core.runtime.compare import Thresholds, UnknownThreshold, load_run_config
from senselab_core.runtime.profiles import ExecutionProfile


@pytest.mark.logic
def test_one_run_config_reader_validates_thresholds_and_calibration(tmp_path: Path) -> None:
    """A run config file's `calibration.trust` reaches `RunCalibration.apply`; `Thresholds.load` reads the same file."""
    config = tmp_path / "run.yaml"
    config.write_text("format: run-config@2.1\nthresholds:\n  embedding:\n    atol: 2\ncalibration:\n  trust: own\n")
    loaded = load_run_config(config)
    assert loaded.calibration == RunCalibration(trust="own")
    assert loaded.calibration.apply(ExecutionProfile.load("orcd").calibration).trust == "own"
    assert loaded.thresholds == Thresholds.load(config)
    assert loaded.thresholds.by_name["embedding"].atol == 2.0
    assert load_run_config().calibration == RunCalibration()
    with pytest.raises(UnknownThreshold, match="did you mean 'trust'"):
        load_run_config({"calibration": {"trus": "own"}})
    with pytest.raises(UnknownThreshold, match="did you mean 'calibration'"):
        load_run_config({"calibrations": {}})
    with pytest.raises(ValueError, match="trust"):
        load_run_config({"calibration": {"trust": "all"}})
    assert RUN_CONFIG_FORMAT == "run-config@2.1"
