"""``packages/conftest.py`` fails collection for a test with no tier marker or with two."""

from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[3] / "conftest.py"
MARKERS = "[pytest]\nmarkers =\n    logic\n    contract\n    limits\n    usecase\n    env\n    network\n"


def _project(pytester: pytest.Pytester, body: str) -> None:
    pytester.makeini(MARKERS)
    pytester.makeconftest(HOOK.read_text())
    pytester.makepyfile(test_sample=body)


@pytest.mark.logic
def test_untiered_test_fails_collection(pytester: pytest.Pytester) -> None:
    """A test with no tier marker is a usage error naming its node id; nothing runs."""
    _project(
        pytester, "import pytest\n\ndef test_bare():\n    pass\n\n@pytest.mark.network\ndef test_net():\n    pass\n"
    )
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    err = result.stderr.str()
    assert "test_sample.py::test_bare" in err and "test_sample.py::test_net" in err


@pytest.mark.logic
def test_two_tiers_fail_collection(pytester: pytest.Pytester) -> None:
    """A test with two tier markers, on the function or split between class and function, fails collection."""
    _project(
        pytester,
        "import pytest\n\n"
        "@pytest.mark.logic\n@pytest.mark.env\ndef test_two():\n    pass\n\n"
        "@pytest.mark.contract\nclass TestSplit:\n    @pytest.mark.limits\n    def test_split(self):\n        pass\n",
    )
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    err = result.stderr.str()
    assert "test_sample.py::test_two" in err and "TestSplit::test_split" in err


@pytest.mark.logic
def test_exactly_one_tier_collects_and_runs(pytester: pytest.Pytester) -> None:
    """One tier marker, alone or with a non-tier marker, collects and passes."""
    _project(
        pytester,
        "import pytest\n\n@pytest.mark.logic\ndef test_a():\n    pass\n\n"
        "@pytest.mark.env\n@pytest.mark.network\ndef test_b():\n    pass\n",
    )
    result = pytester.runpytest()
    assert result.ret == pytest.ExitCode.OK
    result.assert_outcomes(passed=2)


@pytest.mark.logic
def test_hook_applies_only_under_packages(pytester: pytest.Pytester) -> None:
    """A test outside the directory holding the hook is not checked."""
    pytester.makeini(MARKERS)
    pytester.mkdir("packages")
    (pytester.path / "packages" / "conftest.py").write_text(HOOK.read_text())
    pytester.makepyfile(test_outside="def test_bare():\n    pass\n")
    assert pytester.runpytest().ret == pytest.ExitCode.OK
