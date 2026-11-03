"""`parity_test.py` skips without a campaign and fails on an incomplete one."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

PARITY = Path(__file__).with_name("parity_test.py")


def _run(env_extra: dict[str, str]) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SENSELAB_PARITY")} | env_extra
    done = subprocess.run(  # noqa: S603 - pytest under this interpreter on the repo's own test file
        [
            sys.executable,
            "-m",
            "pytest",
            str(PARITY),
            "-o",
            "addopts=",
            "-m",
            "limits",
            "-rsf",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=250,
        check=False,
    )
    return done.stdout


@pytest.mark.logic
def test_the_campaign_tests_skip_when_no_parity_directory_is_set() -> None:
    """With `SENSELAB_PARITY_DIR` unset, each test that needs outputs skips and none fails."""
    out = _run({})
    assert "SENSELAB_PARITY_DIR is unset" in out and "failed" not in out
    assert "[3]" in out and "4 skipped" in out


@pytest.mark.logic
def test_the_campaign_tests_fail_when_the_directory_holds_no_outputs(tmp_path: Path) -> None:
    """With the directory set but empty, the tests that need outputs fail rather than skip."""
    out = _run({"SENSELAB_PARITY_DIR": str(tmp_path)})
    assert "FAILED" in out and "no parity outputs" in out
