"""Every decorated task's key must exist in the matrix.

`check_compatibility` returns True for a key it does not know (`compatibility.py`, the
`entry is None` branch). That is reasonable for a caller probing an arbitrary key, but it means a
decorated task whose key has drifted -- a typo, or a module that moved -- **silently skips both its
dependency check and its isolated-venv provisioning**. Nothing raises at import and nothing raises
at call; the task simply runs without the environment it declared, and fails somewhere less
obvious or not at all.

Found when audio-visual target-speaker extraction moved from `audio/tasks/` to `video/tasks/`: the
matrix key is a dotted path matched as a string, so a stale key would have been invisible.

The derivation is in `specs/20260906-compatibility-key-drift/`.
"""

import re
from pathlib import Path

import pytest

from senselab.utils.compatibility import COMPATIBILITY_MATRIX, requires_compatibility

_SOURCE = Path(__file__).resolve().parents[2] / "senselab"
_DECORATED = re.compile(r"@requires_compatibility\(\s*[\"']([^\"']+)[\"']")


def _decorated_keys() -> list[tuple[str, Path]]:
    """Every key passed to the decorator anywhere in the package.

    Returns:
        ``(key, file)`` pairs, in path order.
    """
    found: list[tuple[str, Path]] = []
    for path in sorted(_SOURCE.rglob("*.py")):
        for match in _DECORATED.finditer(path.read_text()):
            found.append((match.group(1), path))
    return found


class TestEveryDecoratedKeyIsInTheMatrix:
    """A decorated key the matrix does not hold would skip its checks in silence."""

    def test_the_package_decorates_something(self) -> None:
        """Guards the regex: a scan that silently finds nothing would pass every other test."""
        assert len(_decorated_keys()) >= 15

    @pytest.mark.parametrize("key,path", _decorated_keys(), ids=lambda v: v if isinstance(v, str) else "")
    def test_the_key_resolves(self, key: str, path: Path) -> None:
        """Each decorated key names a real matrix entry."""
        assert key in COMPATIBILITY_MATRIX, (
            f"{key} is decorated in {path} but absent from COMPATIBILITY_MATRIX, so its dependency "
            "check and isolated-venv provisioning would be skipped without any error"
        )


class TestTheDecoratorRefusesAnUnknownKey:
    """The scan above cannot see a key built at runtime; the decorator itself must refuse one."""

    def test_an_unknown_key_raises_at_decoration_time(self) -> None:
        """Import time, not call time -- the point is that it never reaches a caller."""
        with pytest.raises(KeyError, match="not in COMPATIBILITY_MATRIX"):
            requires_compatibility("audio.tasks.no_such_task.do_nothing")

    def test_a_near_miss_is_suggested(self) -> None:
        """A moved module's old key differs by one segment; naming the near match saves the search."""
        real = next(k for k in COMPATIBILITY_MATRIX if k.count(".") >= 2)
        wrong = (
            real.replace("audio.", "video.", 1) if real.startswith("audio.") else real.replace("video.", "audio.", 1)
        )
        if wrong == real:
            pytest.skip("no key with a swappable leading segment")
        with pytest.raises(KeyError, match="Did you mean"):
            requires_compatibility(wrong)

    def test_a_known_key_is_accepted(self) -> None:
        """The guard must not refuse the keys the package actually uses."""
        known = next(iter(COMPATIBILITY_MATRIX))
        assert callable(requires_compatibility(known))
