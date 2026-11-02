"""The env-grouping proposal: pair verdicts, greedy groups and what a mismatch of Python or CUDA ceiling refuses."""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from senselab_core.runtime.envs import LockInputs

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "env_grouping.py"
_spec = importlib.util.spec_from_file_location("env_grouping", _SCRIPT)
assert _spec and _spec.loader
eg = importlib.util.module_from_spec(_spec)
sys.modules["env_grouping"] = eg
_spec.loader.exec_module(eg)


def _env(name: str, *requirements: str, python: str = "3.12", cuda: tuple[int, int] | None = None) -> LockInputs:
    return LockInputs(name, requirements, python, "2026-10-01T00:00:00Z", max_cuda=cuda)


def _resolver(clashes: set[frozenset[str]]) -> Callable[[LockInputs], str]:
    """A resolver that pins each requirement `name==1` and fails when a clashing pair is present."""

    def resolve(inputs: LockInputs) -> str:
        names = {r.split("==")[0] for r in inputs.requirements}
        if any(c <= names for c in clashes):
            raise RuntimeError("no solution\n  because a conflicts with b")
        return "".join(f"{n}==1 \\\n    --hash=sha256:aa\n" for n in sorted(names))

    return resolve


ENVS = {n: _env(n, n, "shared") for n in ("a", "b", "c")}


def test_pairs_record_compatibility_and_the_packages_a_union_saves() -> None:
    """A resolving pair carries its package count and the packages shared; a conflict carries its reason."""
    got = eg.propose(ENVS, _resolver({frozenset({"a", "c"})}))
    pairs = {tuple(p["envs"]): p for p in got["pairs"]}
    assert pairs["a", "b"] == {"envs": ["a", "b"], "compatible": True, "packages": 3, "saved_packages": 1}
    assert pairs["a", "c"]["compatible"] is False
    assert pairs["a", "c"]["conflict"] == "no solution because a conflicts with b"


def test_groups_are_greedy_in_name_order_and_each_member_resolves_with_the_whole_group() -> None:
    """`c` is refused by the group `a, b` because `a` and `c` clash, so it starts its own group."""
    got = eg.propose(ENVS, _resolver({frozenset({"a", "c"})}))
    assert [g["envs"] for g in got["groups"]] == [["a", "b"], ["c"]]
    assert got["groups"][0] == {"envs": ["a", "b"], "packages": 3, "separate_packages": 4}


@pytest.mark.parametrize(
    ("other", "reason"),
    [(_env("b", "b", python="3.13"), "python differs"), (_env("b", "b", cuda=(12, 8)), "max_cuda differs")],
)
def test_envs_on_another_python_or_cuda_ceiling_never_share_an_env(other: LockInputs, reason: str) -> None:
    """The union is refused before any resolution, so the resolver is never asked."""
    envs = {"a": _env("a", "a"), "b": other}
    got = eg.propose(envs, _resolver(set()))
    assert not got["pairs"][0]["compatible"]
    assert got["pairs"][0]["conflict"].startswith(reason)
