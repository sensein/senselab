"""Tests for the ``seed_band`` expectation: the band, its verdicts and the chain-1 sub-verdict."""

from __future__ import annotations

import statistics
from typing import Any

import pytest

from senselab.utils.reproduce.bands import evaluate_band
from senselab.utils.reproduce.spec import Expect

CHAINS = ["c1", "c2", "c3"]


def _expect(**band: Any) -> Expect:  # noqa: ANN401
    spec: dict[str, Any] = {"unit": "fraction", "floor": 0.5, "floor_source": "expected_floor", "chains": CHAINS}
    spec.update(band)
    return Expect.model_validate({"metric": "balanced_accuracy", "kind": "seed_band", "band": spec})


def _metrics(value: float, floor: float = 0.5, unit: str = "fraction") -> dict[str, Any]:
    return {
        "headline": [{"name": "balanced_accuracy", "value": value, "floor": floor, "lift": value - floor, "unit": unit}]
    }


def _run(
    expect: Expect,
    chains: list[float],
    own: float | None = None,
    **kw: Any,  # noqa: ANN401
) -> tuple[dict[str, Any], list[str]]:
    unit = expect.band.unit if expect.band is not None else "fraction"
    names = expect.band.chains if expect.band is not None and expect.band.chains else CHAINS
    upstream = {name: _metrics(value, unit=unit, **kw) for name, value in zip(names, chains)}
    mean = statistics.fmean(chains) if own is None else own
    return evaluate_band(expect, _metrics(mean, unit=unit, **kw), upstream)


def test_pass_inside_band_with_informative_width() -> None:
    """Chains around the stored value with a narrow spread pass; ``h = max(a_min, k*sd, rho*lift)``."""
    result, codes = _run(_expect(stored=0.70), [0.69, 0.70, 0.71])
    assert codes == []
    assert result["verdict"] == "pass" and result["pass"] is True
    assert result["tolerance"] == pytest.approx(max(0.002, 2 * 0.01, 0.02)) == pytest.approx(0.02)
    assert result["target"] == 0.70 and result["delta"] == pytest.approx(0.0)
    assert result["floor_used"] == 0.5 and result["lift_used"] == pytest.approx(0.2)
    assert result["sd"] == pytest.approx(0.01) and "chain1_pass" not in result


def test_fail_outside_band() -> None:
    """A chain mean further than ``h`` from the stored value fails."""
    result, codes = _run(_expect(stored=0.70), [0.60, 0.61, 0.62])
    assert codes == []
    assert result["verdict"] == "fail" and result["pass"] is False
    assert result["delta"] == pytest.approx(-0.09)


def test_uninformative_when_width_exceeds_quarter_lift() -> None:
    """A band wider than ``max_width_frac * lift`` is uninformative, not a pass."""
    result, _ = _run(_expect(stored=0.70), [0.55, 0.70, 0.85])
    assert result["tolerance"] > 0.25 * 0.2
    assert result["verdict"] == "uninformative" and result["pass"] is False


def test_below_chance_is_characterise_at_defaults() -> None:
    """A stored value below the floor has a negative lift and is ``characterise`` at the default ``a_min``."""
    result, _ = _run(_expect(stored=0.471), [0.47, 0.48, 0.46])
    assert result["lift_used"] == pytest.approx(-0.029)
    assert result["verdict"] == "characterise" and result["pass"] is False


def test_just_above_chance_is_characterise_at_defaults() -> None:
    """``lift <= a_min / max_width_frac`` (0.008 at the defaults) is ``characterise``."""
    result, _ = _run(_expect(stored=0.505), [0.50, 0.505, 0.51])
    assert result["lift_used"] == pytest.approx(0.005)
    assert result["verdict"] == "characterise"


def test_above_the_chance_threshold_is_not_characterise() -> None:
    """Past the chance threshold the band decides: here the mean is far off, so the row fails."""
    result, _ = _run(_expect(stored=0.509), [0.60, 0.61, 0.62])
    assert result["lift_used"] == pytest.approx(0.009)
    assert result["verdict"] == "fail"


def test_characterise_override_wins() -> None:
    """A declared ``characterise: true`` overrides a passing band."""
    result, _ = _run(_expect(stored=0.70, characterise=True), [0.69, 0.70, 0.71])
    assert result["verdict"] == "characterise" and result["pass"] is False


def test_order_characterise_before_fail() -> None:
    """A row at chance whose chain mean is far off is ``characterise``, not ``fail``."""
    result, _ = _run(_expect(stored=0.50), [0.80, 0.81, 0.82])
    assert result["verdict"] == "characterise"


def test_pp_units_and_default_a_min() -> None:
    """``unit: pp`` defaults ``a_min`` to 0.2, and pp values give the fraction verdicts times 100."""
    pp = _expect(stored=70.0, unit="pp", floor=50.0)
    assert pp.band is not None and pp.band.a_min == 0.2
    fraction = _expect(stored=0.70)
    assert fraction.band is not None and fraction.band.a_min == 0.002
    for chains in ([0.69, 0.70, 0.71], [0.60, 0.61, 0.62], [0.55, 0.70, 0.85]):
        fr, _ = _run(fraction, chains)
        pr, _ = _run(pp, [100 * c for c in chains], floor=50.0)
        assert pr["verdict"] == fr["verdict"]
        assert pr["tolerance"] == pytest.approx(100 * fr["tolerance"])


def test_chain1_subverdict_does_not_fail_row() -> None:
    """Chain 1 far from the stored value fails its sub-verdict only; the row's verdict comes from the mean."""
    expect = _expect(stored=0.70, chains=["c1", "c2", "c3", "c4"], chain1="c1", rho=0.2)
    result, _ = _run(expect, [0.80, 0.66, 0.67, 0.68])
    assert result["chain1_pass"] is False
    assert result["verdict"] != "fail"
    close, _ = _run(expect, [0.70, 0.69, 0.70, 0.71])
    assert close["chain1_pass"] is True
    short, _ = _run(_expect(stored=0.70, chain1="c1"), [0.70, 0.69, 0.71])
    assert short["chain1_pass"] is True
    two = _expect(stored=0.70, chains=["c1", "c2"], chain1="c1")
    result, _ = _run(two, [0.70, 0.71])
    assert "chain1_pass" not in result


def test_floor_from_entry_reads_the_metric_floor() -> None:
    """``floor_from: entry`` takes the floor from the step's own metric entry."""
    spec = {"unit": "fraction", "floor_from": "entry", "floor_source": "expected_floor", "chains": CHAINS}
    expect = Expect.model_validate(
        {"metric": "balanced_accuracy", "kind": "seed_band", "band": {**spec, "stored": 0.70}}
    )
    result, _ = _run(expect, [0.69, 0.70, 0.71], floor=0.25)
    assert result["floor_used"] == 0.25 and result["lift_used"] == pytest.approx(0.45)
    assert result["verdict"] == "pass"


def test_mean_mismatch_is_e_band_mean() -> None:
    """The step's own value must be the chain mean, else ``E_BAND_MEAN``."""
    _, codes = _run(_expect(stored=0.70), [0.65, 0.66, 0.67], own=0.70)
    assert codes == ["E_BAND_MEAN"]
    _, clean = _run(_expect(stored=0.70), [0.65, 0.66, 0.67], own=0.66 * (1 + 1e-12))
    assert clean == []
