"""The fixed-parameter check: suffixed numeric constants and literal waits in package source (P-34)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_fixed_parameters.py"
_spec = importlib.util.spec_from_file_location("check_fixed_parameters", _SCRIPT)
assert _spec and _spec.loader
fp = importlib.util.module_from_spec(_spec)
sys.modules["check_fixed_parameters"] = fp
_spec.loader.exec_module(fp)

PATH = "packages/x/src/x/m.py"
NONE: frozenset[tuple[str, str]] = frozenset()


def _problems(
    source: str, invariants: frozenset[tuple[str, str]] = NONE, pending: dict[tuple[str, str], int] | None = None
) -> list[str]:
    found: list[str] = fp.problems(fp.sites(PATH, source), invariants, pending or {})
    return found


@pytest.mark.parametrize(
    "source",
    [
        "POLL_S = 0.05\n",
        "COMMIT_ATTEMPTS: int = 3\n",
        "WINDOW_BYTES = 16 * 2**20\n",
        "LEASE_DIR_MODE = 0o2770\n",
        "if True:\n    POLL_S = 0.05\n",
        "try:\n    import x\nexcept ImportError:\n    POLL_S = 1\n",
        "import time\ntime.sleep(0.5)\n",
        "def f(e):\n    e.wait(2)\n",
        "def f(t):\n    t.join(-1.5)\n",
        "def f(open):\n    open('u', timeout=60)\n",
    ],
)
def test_a_fixed_parameter_fails(source: str) -> None:
    """A suffixed numeric constant or a literal wait, unlisted, fails."""
    assert _problems(source)


@pytest.mark.parametrize(
    "source",
    [
        "POLL_S = config.poll_s\n",
        "LIMIT = 3\n",
        "def f():\n    POLL_S = 1\n",
        "class C:\n    POLL_S = config.poll_s\n",
        "def f(e, s):\n    e.wait(s)\n    e.wait(timeout=0)\n",
        "','.join(['a'])\n",
    ],
)
def test_config_reads_unsuffixed_names_and_zero_waits_pass(source: str) -> None:
    """A value read from config, an unsuffixed or function-local name and a zero wait are not fixed parameters."""
    assert not _problems(source)


def test_a_listed_declared_invariant_passes_and_an_undeclared_one_fails() -> None:
    """A listed name passes only when its line declares it an invariant."""
    listed = frozenset({(PATH, "_NS_PER_S")})
    assert not _problems("_NS_PER_S: Final = 10**9  # invariant: nanoseconds in a second\n", listed)
    assert _problems("_NS_PER_S: Final = 10**9\n", listed)


def test_a_pending_site_passes_and_a_stale_entry_fails() -> None:
    """A pending site passes; an entry matching no site is reported."""
    assert not _problems("POLL_S = 0.05\n", pending={(PATH, "POLL_S"): 1})
    assert _problems("", pending={(PATH, "POLL_S"): 1})


def test_a_pending_call_form_admits_only_its_count() -> None:
    """A pending `timeout=` entry admits its count of sites in that file; one more fails."""
    one = "def f(o):\n    o('u', timeout=60)\n"
    two = one + "    o('v', timeout=5)\n"
    assert not _problems(one, pending={(PATH, "timeout="): 1})
    assert _problems(two, pending={(PATH, "timeout="): 1})
    assert not _problems(two, pending={(PATH, "timeout="): 2})


@pytest.mark.parametrize(
    ("section", "entry"),
    [
        ("invariants", "  reason: ''\n"),
        ("pending", "  register: ' '\n"),
        ("pending", "  reason: r\n"),
        ("pending", "  register: r\n  count: 0\n"),
    ],
)
def test_an_entry_without_its_reason_is_refused(tmp_path: Path, section: str, entry: str) -> None:
    """An invariant without a reason, or a pending site without a register entry, refuses the file."""
    path = tmp_path / "invariants.yaml"
    path.write_text(f"format: invariants@1.0\n{section}:\n- path: {PATH}\n  name: POLL_S\n{entry}")
    with pytest.raises(ValueError, match="needs a|count"):
        fp.load_allowed(path)


def test_the_tree_passes() -> None:
    """Every fixed parameter under packages/*/src is a declared invariant or pending."""
    assert fp.main() == 0
