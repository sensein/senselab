"""The summary arithmetic of the CrisperWhisper confidence-cost measurement, over fake timings."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "confidence_cost", ROOT / "benchmarks" / "crisperwhisper_confidence_cost.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cost = _script()


def _item(windows: int, decode: float, confidence: float, off_decode: float, duration: float = 1.0) -> dict[str, Any]:
    return {
        "duration_s": duration,
        "on": {"windows": windows, "decode_s": decode, "confidence_s": confidence},
        "off": {"windows": windows, "decode_s": off_decode, "confidence_s": 0.0},
    }


@pytest.mark.logic
def test_summary_divides_by_windows_and_gives_both_shares() -> None:
    """Per-window median and p90, and the confidence share by its own seconds and by the on-off difference."""
    items = [_item(1, 1.0, 0.25, 1.0), _item(2, 4.0, 1.0, 4.0), _item(1, 3.0, 0.75, 3.0), _item(4, 8.0, 2.0, 8.0)]
    summary = cost.summarize(items)
    assert summary["items"] == 4 and summary["windows"] == 8
    assert summary["per_window_s"]["decode_on"]["median"] == pytest.approx(2.0)
    assert summary["per_window_s"]["confidence_on"]["median"] == pytest.approx(0.5)
    assert summary["per_window_s"]["confidence_on"]["p90"] == pytest.approx(0.75)
    assert summary["per_window_s"]["total_on"]["median"] == pytest.approx(2.5)
    assert summary["share_of_total"] == pytest.approx(4.0 / 20.0)
    assert summary["share_of_decode"] == pytest.approx(4.0 / 16.0)
    assert summary["share_by_difference"] == pytest.approx(4.0 / 20.0)


@pytest.mark.logic
def test_difference_share_sees_time_the_pass_does_not_claim() -> None:
    """If `on` is slower than `off` by more than `confidence_s`, the difference share is the larger."""
    summary = cost.summarize([_item(1, 2.0, 1.0, 1.5)])
    assert summary["share_of_total"] == pytest.approx(1.0 / 3.0)
    assert summary["share_by_difference"] == pytest.approx(0.5)


@pytest.mark.logic
def test_nearest_rank_percentile() -> None:
    """The 90th percentile of ten values is the ninth."""
    assert cost._percentile(list(range(1, 11)), 0.9) == 9
    assert cost._percentile([5.0], 0.9) == 5.0


@pytest.mark.logic
def test_duration_terciles_and_resummarising_a_result_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Shares are given for the shortest, middle and longest thirds by duration, and `--summarize` re-reads a file."""
    items = [_item(1, 4.0, 1.0, 4.0, d) for d in (9.0, 8.0, 7.0)] + [
        _item(1, 4.0, 2.0, 4.0, d) for d in (3.0, 2.0, 1.0)
    ]
    items += [_item(1, 2.0, 1.0, 2.0, d) for d in (6.0, 5.0, 4.0)]
    terciles = cost.summarize(items)["by_duration"]
    assert [terciles[k]["items"] for k in ("short", "middle", "long")] == [3, 3, 3]
    assert terciles["short"]["duration_s"] == [1.0, 3.0] and terciles["long"]["duration_s"] == [7.0, 9.0]
    assert terciles["short"]["share_of_decode"] == pytest.approx(0.5)
    assert terciles["middle"]["share_of_decode"] == pytest.approx(0.5)
    assert terciles["long"]["share_of_decode"] == pytest.approx(0.25)
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"format": cost.FORMAT, "items": items}))
    cost.main(["--summarize", str(path)])
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(cost.summarize(items)))
