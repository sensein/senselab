"""The cross-node lease probe's verdict: what it passes and what it fails, on planted event logs."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "probe_lease_cross_node.py"
_spec = importlib.util.spec_from_file_location("probe_lease_cross_node", _SCRIPT)
assert _spec and _spec.loader
probe = importlib.util.module_from_spec(_spec)
sys.modules["probe_lease_cross_node"] = probe
_spec.loader.exec_module(probe)


def _write(root: Path, events: list[dict[str, Any]]) -> None:
    for event in events:
        with open(root / f"events-{event['role']}.jsonl", "a") as handle:
            handle.write(json.dumps(event) + "\n")


def _run(hosts: tuple[str, str, str] = ("a", "b", "c"), *, release_b: bool = True) -> list[dict[str, Any]]:
    events = [
        {"role": "A", "host": hosts[0], "event": "held", "storage_t": 0.0},
        {"role": "B", "host": hosts[1], "event": "refused", "storage_t": 10.0},
        {"role": "C", "host": hosts[2], "event": "refused", "storage_t": 11.0},
        {"role": "A", "host": hosts[0], "event": "kill", "storage_t": 360.0},
        {"role": "B", "host": hosts[1], "event": "took", "storage_t": 470.0},
        {"role": "C", "host": hosts[2], "event": "took", "storage_t": 720.0},
        {"role": "C", "host": hosts[2], "event": "released", "storage_t": 960.0},
    ]
    if release_b:
        events.append({"role": "B", "host": hosts[1], "event": "released", "storage_t": 710.0})
    return events


def test_three_hosts_with_a_takeover_inside_the_bound_pass(tmp_path: Path) -> None:
    """A takeover 110 s after the kill on a 120 s bound, with three hosts and no overlap, passes."""
    _write(tmp_path, _run())
    assert probe.verdict(tmp_path, 120, 75)


def test_a_take_without_a_release_fails_rather_than_crashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A taker that logged `took` and never `released` is a failed verdict that counts it, not a `StopIteration`."""
    _write(tmp_path, _run(release_b=False))
    assert not probe.verdict(tmp_path, 120, 75)
    assert json.loads(capsys.readouterr().out)["unreleased_takes"] == 1


def test_fewer_than_three_hosts_fail(tmp_path: Path) -> None:
    """A run whose roles shared a node is not a cross-node check, and fails."""
    _write(tmp_path, _run(("a", "b", "b")))
    assert not probe.verdict(tmp_path, 120, 75)
