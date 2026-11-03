"""No tracked file may carry a local filesystem path or a personal Slurm partition."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard = _module("check_local_paths", ROOT / "scripts" / "check_local_paths.py")

# Built by concatenation so this file never contains a matching literal itself.
_USERS = "/" + "Users" + "/"
_HOME = "/" + "home" + "/"
_ORCD = "/" + "orcd" + "/"
_PART = "p" + "i_"


def test_no_tracked_file_carries_a_local_path() -> None:
    """Every tracked path and file body is free of home paths, ORCD paths and personal partitions."""
    hits = guard.findings(guard.tracked_files(ROOT), ROOT)
    assert not hits, "local paths in tracked files:\n" + "\n".join(hits[:50])


def test_patterns_flag_real_shapes_and_pass_placeholders() -> None:
    """Home, ORCD and partition shapes match; placeholders, generic CI accounts and ``~/orcd`` do not."""
    flagged = {
        _USERS + "alice/software/x": "user home path",
        _HOME + "alice/.cache": "user home path",
        "cd " + _ORCD + "scratch/lab/alice/run": "ORCD path",
        "sbatch -p " + _PART + "alice job.sh": "personal Slurm partition",
        "--partition=" + _PART + "lab": "personal Slurm partition",
    }
    for text, kind in flagged.items():
        assert guard.kinds(text) == [kind], text
    clean = [
        "${ORCD_DATA}/datasets",
        "${AGENT_SCRATCH}/run",
        "~" + _ORCD + "scratch",
        _USERS + "runner/work/repo",
        _HOME + "ec2-user",
        _HOME + "user/.cache",
        "--partition=${CPU_PARTITION}",
        "api_module " + _PART[1:] + "x",
    ]
    assert not any(guard.kinds(s) for s in clean)


def test_findings_never_echo_the_value(tmp_path: Path) -> None:
    """A finding names file, line and kind but not the matched text, and the local values file is skipped."""
    secret = "alice"
    (tmp_path / "notes.md").write_text(f"fine\nsee {_USERS}{secret}/data\n")
    (tmp_path / "references.local.yaml").write_text(f"X: {_USERS}{secret}/data\n")
    out = guard.findings(["notes.md", "references.local.yaml"], tmp_path)
    assert out == ["notes.md:2: user home path"]
    assert secret not in "".join(out)
