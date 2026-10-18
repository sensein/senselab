"""No tracked file may carry a research-participant identifier."""

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


guard = _module("check_participant_ids", ROOT / "scripts" / "check_participant_ids.py")

# Built by concatenation so this file never contains a matching literal itself.
_HEX8 = "0f1e2d3c"
_UUID = _HEX8 + "-4b5a-6978-8796-a5b4c3d2e1f0"


def test_no_tracked_file_carries_a_participant_id() -> None:
    """Every tracked path and file body is free of sub-/ses- UUIDs and 8-hex prefixes."""
    hits = guard.findings(guard.tracked_files(ROOT), ROOT)
    assert not hits, "participant identifiers in tracked files:\n" + "\n".join(hits[:50])


def test_pattern_flags_real_shapes_and_passes_synthetic_labels() -> None:
    """Full UUIDs and 8-hex prefixes match in either case; placeholders and short labels do not."""
    flagged = [
        "sub-" + _UUID,
        "ses-" + _UUID.upper(),
        "sub-" + _HEX8,
        "path/sub-" + _HEX8 + "_ses-" + _HEX8 + "_task-x.json",
    ]
    clean = ["sub-A", "sub-B", "sub-C", "sub-SUBJ01", "ses-SES01", "ses-01", "sub-fresh", "sub-" + _HEX8 + "ab"]
    assert all(guard.PARTICIPANT_ID_RE.search(s) for s in flagged)
    assert not any(guard.PARTICIPANT_ID_RE.search(s) for s in clean)


def test_findings_never_echo_the_identifier(tmp_path: Path) -> None:
    """A finding names file, line and shape but masks the value, including in the path."""
    name = "sub-" + _HEX8 + "_notes.md"
    (tmp_path / name).write_text("x\nseen in sub-" + _UUID + " and ses-" + _HEX8 + "\n")
    hits = guard.findings([name], tmp_path)
    assert len(hits) == 3
    assert not any(_HEX8 in h for h in hits)
    assert hits[1].endswith(":2: sub- full UUID")
