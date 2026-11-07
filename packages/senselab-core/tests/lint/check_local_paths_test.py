"""The local-paths check: home directories and home-relative paths outside the generic conventions."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_local_paths.py"
_spec = importlib.util.spec_from_file_location("check_local_paths", _SCRIPT)
assert _spec and _spec.loader
lp = importlib.util.module_from_spec(_spec)
sys.modules["check_local_paths"] = lp
_spec.loader.exec_module(lp)

HOME = "~" + "/"


@pytest.mark.parametrize("rest", ["software/x", "Downloads/a.wav", ".claude/jobs/x/", "orcd/other/x"])
def test_a_personal_home_relative_path_fails(rest: str) -> None:
    """A `~/<name>/` path outside the conventions is found."""
    assert lp.kinds(f"see {HOME}{rest}") == ["home-relative path"]


@pytest.mark.parametrize(
    "text",
    [
        f"{HOME}.cache/huggingface/token",
        f"{HOME}.local/bin",
        f"{HOME}ffmpeg/lib",
        f"{HOME}orcd/scratch",
        f"{HOME}orcd/scratch/run",
        f"{HOME}auto",
        "a~/b/",
        "/home/user/a.wav",
    ],
)
def test_conventions_and_generic_accounts_pass(text: str) -> None:
    """Tool and cluster conventions, a bare `~/name` and the generic accounts pass."""
    assert lp.kinds(text) == []
