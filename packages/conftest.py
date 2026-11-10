"""Tier hook: every test under ``packages/`` carries exactly one of the five tier markers.

The `model` sub-marker (loads weights or needs a backend env, ORCD only) goes with `limits`, `usecase` or `env`.
A `gpu` test is skipped on a host with no GPU. A test with no `timeout` mark gets its tier's budget from
`senselab/tests/data/tier_budgets.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

TIERS = frozenset({"logic", "contract", "limits", "usecase", "env"})
MODEL_TIERS = frozenset({"limits", "usecase", "env"})
PACKAGES = Path(__file__).resolve().parent
BUDGETS = PACKAGES / "senselab" / "tests" / "data" / "tier_budgets.json"
LOCK = PACKAGES.parent / "uv.lock"
LOCK_DIGEST = "senselab-lock.sha"
"""Beside the venv's interpreter prefix: the digest of the uv.lock it was synced from (scripts/sync_local_venv.sh)."""


def venv_status(prefix: Path, lock: Path) -> str:
    """`STALE VENV ...` when the venv at `prefix` was synced from another `lock`; else whether its digest matches."""
    recorded = prefix / LOCK_DIGEST
    if not recorded.is_file():
        return f"venv lock digest: unrecorded ({recorded} is missing)"
    want = hashlib.sha256(lock.read_bytes()).hexdigest()
    have = recorded.read_text().strip()
    if have != want:
        return f"STALE VENV: synced from uv.lock {have[:12]}, this tree's is {want[:12]}; results are not evidence"
    return f"venv lock digest: {want[:12]}"


def pytest_report_header() -> str:
    """The venv's lock status, first line of every run."""
    return venv_status(Path(sys.prefix), LOCK)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Repeat a stale venv at the end of the run, beside the result."""
    status = venv_status(Path(sys.prefix), LOCK)
    if status.startswith("STALE VENV"):
        terminalreporter.write_line(status, red=True)


def tier_budgets() -> dict[str, float]:
    """Return the time budget in seconds of one test of each tier, from `tier_budgets.json`."""
    document = json.loads(BUDGETS.read_text())
    return {tier: float(row["budget_s"]) for tier, row in document["tiers"].items()}


@pytest.fixture(scope="session")
def tier_names() -> frozenset[str]:
    """The five tier marker names."""
    return TIERS


@pytest.fixture(scope="session")
def budgets_file() -> Path:
    """The file holding each tier's time budget."""
    return BUDGETS


@pytest.fixture(scope="session")
def tier_budget_s() -> dict[str, float]:
    """The time budget in seconds of one test of each tier."""
    return tier_budgets()


@pytest.fixture(scope="session", autouse=True)
def real_cache_untouched() -> Iterator[None]:
    """Fail the session if a test kept a local model's identity under the user's real cache root."""
    try:
        from senselab_core.runtime.models import LOCAL_IDENTITIES_DIR  # noqa: PLC0415 - core may not be installed
        from senselab_core.runtime.revision import cache_root  # noqa: PLC0415
    except ImportError:
        yield
        return
    kept = cache_root() / LOCAL_IDENTITIES_DIR
    before = set(kept.iterdir()) if kept.is_dir() else set()
    yield
    written = (set(kept.iterdir()) if kept.is_dir() else set()) - before
    assert not written, f"tests wrote {len(written)} file(s) under the real cache root {kept}; give them tmp_path"


PROFILE_VARIABLE = "SENSELAB_TEST_PROFILE"
CLUSTER_FIXTURE = PACKAGES / "senselab-core" / "tests" / "data" / "profiles" / "cluster.yaml"


@pytest.fixture(scope="session")
def _session_env_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("envs")


@pytest.fixture
def test_profile(
    _session_env_root: Path,
) -> Callable[[Mapping[str, Any] | None], Any]:
    """Return `load(override=None)`, the test profile: `$SENSELAB_TEST_PROFILE` untouched, else the cluster fixture.

    The fixture's env roots are `${SENSELAB_ENV_ROOT}` with no fallback; when that variable is unset they are pointed,
    by a mapping override, at a session temp directory. The process environment is never changed.
    """
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - core may not be installed

    def load(override: Mapping[str, Any] | None = None) -> Any:  # noqa: ANN401
        site = os.environ.get(PROFILE_VARIABLE)
        if site:
            return ExecutionProfile.load(Path(site), override)
        roots = (
            {}
            if os.environ.get("SENSELAB_ENV_ROOT")
            else {"env_roots": {"isolated": str(_session_env_root), "server": str(_session_env_root / "servers")}}
        )
        return ExecutionProfile.load(CLUSTER_FIXTURE, {**roots, **(override or {})})

    return load


@pytest.fixture
def site_profile(test_profile: Callable[[Mapping[str, Any] | None], Any]) -> Callable[[Mapping[str, Any] | None], Any]:
    """`test_profile` for a model or GPU test, skipped unless `$SENSELAB_TEST_PROFILE` names a site profile."""
    if not os.environ.get(PROFILE_VARIABLE):
        pytest.skip(f"set {PROFILE_VARIABLE} to a site profile")
    return test_profile


def _slack_document(request: pytest.FixtureRequest) -> dict[str, Any]:
    import yaml  # noqa: PLC0415 - the root conftest imports the stdlib and pytest only

    tests = next(p for p in Path(str(request.path)).resolve().parents if p.name == "tests")
    path = tests / "data" / "test_slack.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: this package has no test slack file")
    document: dict[str, Any] = yaml.safe_load(path.read_text())
    return document


@pytest.fixture
def slack(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The profile override of the requesting test's package slack file, `<package>/tests/data/test_slack.yaml`.

    Test slack is a profile override (P-34); its derivations are in config-derivations.md. The `tests` section is
    not part of the override: `slack_tests` returns it.
    """
    return {k: v for k, v in _slack_document(request).items() if k != "tests"}


@pytest.fixture
def slack_tests(request: pytest.FixtureRequest) -> dict[str, Any]:
    """The `tests` section of the package slack file: values only tests read (P-34)."""
    section: dict[str, Any] = _slack_document(request)["tests"]
    return section


@pytest.fixture
def tier_bound(request: pytest.FixtureRequest) -> float:
    """The seconds the requesting test's tier allows it, which bounds every wait on a sentinel in it."""
    marker = request.node.get_closest_marker("timeout")
    assert marker is not None
    return float(marker.args[0])


def _apply_budgets(items: list[pytest.Item]) -> None:
    if not items:
        return
    budgets = tier_budgets()
    for item in items:
        tier = next((m.name for m in item.iter_markers() if m.name in TIERS), None)
        if tier is not None and not item.get_closest_marker("timeout"):
            item.add_marker(pytest.mark.timeout(budgets[tier]))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Fail the session, listing the node ids, when a test under ``packages/`` has zero or several tier marks."""
    offenders = []
    owned = [item for item in items if PACKAGES in Path(str(item.path)).resolve().parents]
    for item in owned:
        tiers = sorted(m.name for m in item.iter_markers() if m.name in TIERS)
        if len(set(tiers)) != 1:
            offenders.append(f"{item.nodeid}: {sorted(set(tiers)) or 'no tier'}")
        elif item.get_closest_marker("model") and tiers[0] not in MODEL_TIERS:
            offenders.append(f"{item.nodeid}: `model` goes with one of {sorted(MODEL_TIERS)}, not {tiers[0]!r}")
    if offenders:
        raise pytest.UsageError(
            f"each test under packages/ needs exactly one tier marker of {sorted(TIERS)}, "
            f"and `model` only beside {sorted(MODEL_TIERS)}:\n" + "\n".join(offenders)
        )
    _apply_budgets(owned)
    gpu = [item for item in items if item.get_closest_marker("gpu")]
    if gpu:
        from senselab_core.runtime.hardware import (
            probe_hardware,  # noqa: PLC0415 - probed only when a gpu test is collected
        )

        if not probe_hardware().gpus:
            for item in gpu:
                item.add_marker(pytest.mark.skip(reason="needs a GPU; this host has none"))
