"""Behavioural tests for the corpus measurement distributions.

A threshold is argued against a distribution, so the distribution has to be right about which
readings it saw, which family produced them, and what a non-numeric reading is.
"""

from __future__ import annotations

import json
from pathlib import Path

from senselab.audio.workflows.triage.measure_stats import (
    Distribution,
    collect,
    readings,
    render_markdown,
)


def _store(root: Path, stem: str, family: str, measurements: list[tuple[str, object]]) -> Path:
    """Write one store carrying a VERDICT family and some measurements."""
    path = root / stem / "run" / "store.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "record": "entity",
                "id": "v",
                "prov_type": "verdict",
                "attributes": {"node": "VERDICT", "declared_family": family},
            }
        )
    ]
    for index, (name, value) in enumerate(measurements):
        lines.append(
            json.dumps(
                {
                    "record": "entity",
                    "id": f"m{index}",
                    "prov_type": "measurement",
                    "attributes": {"name": name, "value": value},
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


class TestReadingTheCorpus:
    """Every measurement is attributed to the family whose recording produced it."""

    def test_a_reading_is_tagged_with_its_declared_family(self, tmp_path: Path) -> None:
        """Without the family a distribution cannot say which task a bound would suit."""
        _store(tmp_path, "a", "diadochokinesis-pa", [("rate_hz", 6.2)])
        _store(tmp_path, "b", "prolonged-vowel", [("rate_hz", 1.1)])
        stats = collect(readings(tmp_path))
        assert sorted(stats["rate_hz"]) == ["diadochokinesis-pa", "prolonged-vowel"]

    def test_a_recording_declaring_no_family_is_still_counted(self, tmp_path: Path) -> None:
        """Dropping them would bias every distribution toward the declared corpus."""
        _store(tmp_path, "a", "", [("rate_hz", 3.0)])
        assert "(undeclared)" in collect(readings(tmp_path))["rate_hz"]

    def test_a_measurement_with_no_value_is_not_a_reading(self, tmp_path: Path) -> None:
        """An absence is not a zero, and must not pull a quantile toward one."""
        _store(tmp_path, "a", "loudness", [("rate_hz", None), ("other", 1.0)])
        stats = collect(readings(tmp_path))
        assert "rate_hz" not in stats
        assert stats["other"]["loudness"].n == 1

    def test_an_unreadable_store_does_not_stop_the_scan(self, tmp_path: Path) -> None:
        """One truncated store must not cost the other 62,008 their readings."""
        _store(tmp_path, "a", "loudness", [("rate_hz", 2.0)])
        bad = tmp_path / "zz" / "run" / "store.jsonl"
        bad.parent.mkdir(parents=True)
        bad.write_text("{not json\n")
        assert collect(readings(tmp_path))["rate_hz"]["loudness"].n == 1


class TestTheDistribution:
    """Quantiles for numbers, counts for everything else."""

    def test_quantiles_come_from_the_readings(self) -> None:
        """The median is what a bound is usually argued against."""
        d = Distribution()
        for v in range(1, 101):
            d.add(float(v))
        s = d.summary()
        assert s["n"] == 100 and s["min"] == 1.0 and s["max"] == 100.0
        assert s["p50"] == 51.0 and s["p5"] == 6.0 and s["p95"] == 96.0

    def test_a_boolean_is_counted_not_averaged(self) -> None:
        """True is not 1.0: averaging flags would invent a quantile with no meaning."""
        d = Distribution()
        for v in (True, True, False):
            d.add(v)
        s = d.summary()
        assert s["values"] == {"True": 2, "False": 1}
        assert "p50" not in s

    def test_a_non_finite_reading_is_recorded_and_kept_out_of_the_quantiles(self) -> None:
        """A NaN in the numbers would silently poison every quantile above it."""
        d = Distribution()
        d.add(1.0)
        d.add(float("nan"))
        d.add(float("inf"))
        s = d.summary()
        assert s["numeric_n"] == 1
        assert s["values"]["non-finite"] == 2

    def test_a_string_reading_is_counted(self) -> None:
        """Vocabulary terms are the other thing a measurement carries."""
        d = Distribution()
        d.add("mouth")
        d.add("nose")
        d.add("mouth")
        assert d.summary()["values"] == {"mouth": 2, "nose": 1}


class TestRendering:
    """The report names the reading, the family and the spread."""

    def test_a_numeric_measure_renders_its_quantiles(self, tmp_path: Path) -> None:
        """Rendered so the tails are visible, since that is where a bound sits."""
        for index in range(10):
            _store(tmp_path, f"s{index}", "diadochokinesis-pa", [("rate_hz", float(index))])
        rendered = render_markdown(collect(readings(tmp_path)), tmp_path)
        assert "`rate_hz`" in rendered and "median" in rendered
        assert "diadochokinesis-pa" in rendered

    def test_a_non_numeric_measure_renders_its_counts(self, tmp_path: Path) -> None:
        """A route or a flag has no median and must not be given one."""
        _store(tmp_path, "a", "respiration-and-cough-v2-breath", [("route", "mouth")])
        rendered = render_markdown(collect(readings(tmp_path)), tmp_path)
        assert "`mouth`×1" in rendered
